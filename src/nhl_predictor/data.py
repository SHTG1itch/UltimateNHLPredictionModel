"""Download reproducible game results from the public NHL web API."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Current and relocated/defunct clubs cover the modern seasons useful for
# training. Bad team/season combinations return 404 and are safely skipped.
TEAM_CODES = (
    "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH",
    "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL",
    "TOR", "UTA", "VAN", "VGK", "WPG", "WSH", "ATL", "PHX",
)
API = "https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
STATS_API = "https://api.nhle.com/stats/rest/en/team/{report}"
SCHEDULE_API = "https://api-web.nhle.com/v1/schedule/{date}"
# Each report is league-wide and game-grained, avoiding one API request per
# boxscore. Values are post-game observations and enter features only after the
# corresponding prediction has been recorded.
REPORT_FIELDS = {
    "summary": {
        "shotsForPerGame": "shots",
        "faceoffWinPct": "faceoff_pct",
    },
    "powerplay": {
        "powerPlayGoalsFor": "pp_goals",
        "ppOpportunities": "pp_opportunities",
    },
    "penaltykill": {
        "ppGoalsAgainst": "pk_goals_against",
        "timesShorthanded": "times_shorthanded",
    },
    "realtime": {
        "totalShotAttempts": "shot_attempts",
    },
}
TEAM_STAT_FIELDS = tuple(
    f"{side}_{field}"
    for side in ("home", "away")
    for field in (
        "shots", "faceoff_pct", "pp_goals", "pp_opportunities",
        "pk_goals_against", "times_shorthanded", "shot_attempts",
    )
)
FIELDS = (
    "game_id", "season", "game_type", "game_date", "start_time_utc",
    "home_team", "away_team", "home_score", "away_score", "overtime",
    *TEAM_STAT_FIELDS,
)


def _fetch_team(team: str, season: int, timeout: float) -> list[dict]:
    """Fetch one team schedule; unsupported historical abbreviations are empty."""

    request = Request(
        API.format(team=team, season=season),
        headers={"User-Agent": "UltimateNHLPredictionModel/0.1"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS host
            return json.load(response).get("games", [])
    except HTTPError as error:
        if error.code == 404:
            return []
        raise


def _fetch_report(season: int, game_type: int, report: str, timeout: float) -> list[dict]:
    """Fetch every team's per-game rows for one compact NHL Stats report."""

    query = urlencode(
        {
            "isAggregate": "false",
            "isGame": "true",
            "start": 0,
            "limit": -1,
            "cayenneExp": f"seasonId={season} and gameTypeId={game_type}",
        }
    )
    request = Request(
        f"{STATS_API.format(report=report)}?{query}",
        headers={"User-Agent": "UltimateNHLPredictionModel/0.2"},
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS host
        return json.load(response).get("data", [])


def _parse_schedule(payload: dict, target_date: str) -> list[dict]:
    """Extract only the requested date from the API's surrounding game week."""

    games = []
    for day in payload.get("gameWeek", []):
        for game in day.get("games", []):
            game_date = game.get("gameDate", day.get("date"))
            if game_date != target_date or game.get("gameType") not in {2, 3}:
                continue
            games.append(
                {
                    "game_id": int(game["id"]),
                    "season": int(game["season"]),
                    "game_type": int(game["gameType"]),
                    "game_date": game_date,
                    "start_time_utc": game["startTimeUTC"],
                    "home_team": game["homeTeam"]["abbrev"],
                    "away_team": game["awayTeam"]["abbrev"],
                }
            )
    return sorted(games, key=lambda game: (game["start_time_utc"], game["game_id"]))


def fetch_schedule(target_date: str, *, timeout: float = 30.0) -> list[dict]:
    """Fetch the NHL schedule for one exact ISO date."""

    date.fromisoformat(target_date)  # Reject ambiguous/non-ISO dates at the CLI boundary.
    request = Request(
        SCHEDULE_API.format(date=target_date),
        headers={"User-Agent": "UltimateNHLPredictionModel/0.2"},
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS host
        return _parse_schedule(json.load(response), target_date)


def download_seasons(
    seasons: list[int],
    output: str | Path,
    *,
    workers: int = 8,
    timeout: float = 30.0,
) -> int:
    """Download completed regular-season/playoff games and write one CSV row each."""

    if not seasons or any(len(str(season)) != 8 for season in seasons):
        raise ValueError("seasons must use the YYYYYYYY format, e.g. 20242025")

    games: dict[int, dict] = {}
    # Team schedules overlap by design. Concurrent requests reduce download
    # time; the game ID below gives deterministic league-wide deduplication.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_team, team, season, timeout): (team, season)
            for season in seasons
            for team in TEAM_CODES
        }
        for future in as_completed(futures):
            for game in future.result():
                if game.get("gameState") in {"FINAL", "OFF"}:
                    games[int(game["id"])] = game

    # Merge report columns by official game ID and H/R orientation. Every game
    # must have exactly one home and one road team in each requested report.
    stats: dict[tuple[int, str], dict[str, float]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_report, season, game_type, report, timeout): report
            for season in seasons
            for game_type in (2, 3)
            for report in REPORT_FIELDS
        }
        for future in as_completed(futures):
            report = futures[future]
            for record in future.result():
                values = stats.setdefault((int(record["gameId"]), record["homeRoad"]), {})
                for source, destination in REPORT_FIELDS[report].items():
                    values[destination] = float(record.get(source) or 0.0)

    rows = []
    for game in games.values():
        # Preseason (type 1) is excluded because rosters and incentives differ.
        if game.get("gameType") not in {2, 3}:
            continue
        period = game.get("gameOutcome", {}).get("lastPeriodType", "REG")
        row = {
                "game_id": int(game["id"]),
                "season": int(game["season"]),
                "game_type": int(game["gameType"]),
                "game_date": game["gameDate"],
                "start_time_utc": game["startTimeUTC"],
                "home_team": game["homeTeam"]["abbrev"],
                "away_team": game["awayTeam"]["abbrev"],
                "home_score": int(game["homeTeam"]["score"]),
                "away_score": int(game["awayTeam"]["score"]),
                "overtime": int(period in {"OT", "SO"}),
        }
        for side, orientation in (("home", "H"), ("away", "R")):
            game_stats = stats.get((row["game_id"], orientation))
            if game_stats is None or set(game_stats) != {
                field.removeprefix(f"{side}_") for field in TEAM_STAT_FIELDS if field.startswith(side)
            }:
                raise ValueError(f"incomplete NHL Stats reports for game {row['game_id']} ({side})")
            row.update({f"{side}_{field}": value for field, value in game_stats.items()})
        rows.append(row)

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["start_time_utc"], row["game_id"])))
    return len(rows)


def read_games(path: str | Path) -> list[dict]:
    """Read and validate the compact CSV schema consumed by the model."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = set(FIELDS) - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
    for row in rows:
        for field in ("game_id", "season", "game_type", "home_score", "away_score", "overtime"):
            row[field] = int(row[field])
        for field in TEAM_STAT_FIELDS:
            row[field] = float(row[field])
    return sorted(rows, key=lambda row: (row["start_time_utc"], row["game_id"]))
