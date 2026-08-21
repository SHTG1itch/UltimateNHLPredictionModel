"""Download reproducible game results from the public NHL web API."""

from __future__ import annotations

import csv
import json
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from io import BytesIO, TextIOWrapper
from math import isfinite
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

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
GOALIE_STATS_API = "https://api.nhle.com/stats/rest/en/goalie/summary"
SKATER_STATS_API = "https://api.nhle.com/stats/rest/en/skater/summary"
SCHEDULE_API = "https://api-web.nhle.com/v1/schedule/{date}"
MONEYPUCK_XG_API = "https://peter-tanner.com/moneypuck/downloads/shots_{year}.zip"
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
        "missedShots": "missed_shots",
        "blockedShots": "blocked_shots",
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
XG_FIELDS = ("home_xg", "away_xg")
GOALIE_FIELDS = tuple(
    f"{side}_starter_{field}"
    for side in ("home", "away")
    for field in ("goalie_id", "time_on_ice", "goals_against", "shots_against")
)
SKATER_FIELDS = ("home_skaters", "away_skaters")
FIELDS = (
    "game_id", "season", "game_type", "game_date", "start_time_utc",
    "home_team", "away_team", "home_score", "away_score", "overtime",
    *TEAM_STAT_FIELDS,
    *XG_FIELDS,
    *SKATER_FIELDS,
    *GOALIE_FIELDS,
    "xg_available",
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


def _fetch_goalies(season: int, game_type: int, timeout: float) -> list[dict]:
    """Fetch league-wide per-game goalie rows for starter identification."""

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
        f"{GOALIE_STATS_API}?{query}",
        headers={"User-Agent": "UltimateNHLPredictionModel/0.3"},
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS host
        return json.load(response).get("data", [])


def _fetch_skaters(
    season: int,
    game_type: int,
    year: int,
    month: int,
    timeout: float,
) -> list[dict]:
    """Fetch one month of game-grain skater rows below the API's 10k cap."""

    start = date(year, month, 1)
    end = date(year, month, monthrange(year, month)[1])
    query = urlencode(
        {
            "isAggregate": "false",
            "isGame": "true",
            "start": 0,
            "limit": -1,
            "cayenneExp": (
                f'seasonId={season} and gameTypeId={game_type} '
                f'and gameDate>="{start}" and gameDate<="{end}"'
            ),
        }
    )
    request = Request(
        f"{SKATER_STATS_API}?{query}",
        headers={"User-Agent": "UltimateNHLPredictionModel/0.4"},
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS host
        payload = json.load(response)
    rows = payload.get("data", [])
    if payload.get("total", len(rows)) >= 10_000:
        raise ValueError(f"skater report hit API cap for {year}-{month:02d}")
    return rows


def _encode_skaters(rows: list[dict]) -> str:
    """Pack one lineup's IDs, usage, and raw effectiveness into one CSV cell."""

    players = []
    for row in rows:
        players.append(
            tuple(
                int(round(float(row.get(field) or 0)))
                for field in (
                    "playerId", "timeOnIcePerGame", "goals", "assists", "plusMinus", "shots"
                )
            )
        )
    if len({player[0] for player in players}) != len(players):
        raise ValueError("duplicate player in skater lineup")
    return ";".join(":".join(map(str, player)) for player in sorted(players))


def _decode_skaters(value: str) -> tuple[tuple[int, ...], ...]:
    """Validate and unpack a compact skater lineup from the CSV boundary."""

    if not value:
        return ()
    try:
        players = tuple(tuple(map(int, player.split(":"))) for player in value.split(";"))
    except ValueError as error:
        raise ValueError("invalid skater lineup") from error
    if any(len(player) != 6 or player[0] <= 0 or player[1] < 0 for player in players):
        raise ValueError("invalid skater lineup")
    if len({player[0] for player in players}) != len(players):
        raise ValueError("duplicate player in skater lineup")
    return players


def _complete_shot_attempts(stats: dict[tuple[int, str], dict[str, float | None]]) -> None:
    """Reconstruct older null SAT totals from shots, misses, and opponent blocks."""

    for (game_id, orientation), values in stats.items():
        if values.get("shot_attempts") is not None:
            continue
        opponent = stats.get((game_id, "R" if orientation == "H" else "H"))
        if opponent is None:
            raise ValueError(f"missing opponent realtime row for game {game_id}")
        values["shot_attempts"] = (
            float(values["shots"])
            + float(values["missed_shots"])
            + float(opponent["blocked_shots"])
        )


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


def _aggregate_xg(rows: object) -> dict[tuple[int, str], float]:
    """Collapse MoneyPuck shots into expected goals by game and team side."""

    totals: dict[tuple[int, str], float] = {}
    for row in rows:
        xg = float(row["xGoal"])
        if not isfinite(xg) or not 0.0 <= xg <= 1.0:
            raise ValueError("MoneyPuck xGoal must be a finite probability")
        # MoneyPuck stores the start year plus a six-digit NHL game suffix.
        game_id = int(f"{int(row['season'])}{int(float(row['game_id'])):06d}")
        orientation = "H" if float(row["isHomeTeam"]) == 1.0 else "R"
        key = (game_id, orientation)
        totals[key] = totals.get(key, 0.0) + xg
    return totals


def _fetch_xg(season: int, timeout: float) -> dict[tuple[int, str], float]:
    """Stream one listed MoneyPuck season ZIP into compact game-level totals."""

    year = int(str(season)[:4])
    request = Request(
        MONEYPUCK_XG_API.format(year=year),
        headers={"User-Agent": "Mozilla/5.0 UltimateNHLPredictionModel/0.3"},
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS host
        # A hard cap prevents a changed endpoint from exhausting memory. Listed
        # season archives are currently about 20 MB compressed.
        compressed = response.read(100_000_001)
    if len(compressed) > 100_000_000:
        raise ValueError(f"MoneyPuck season {year} exceeded the 100 MB safety cap")

    with ZipFile(BytesIO(compressed)) as archive:
        expected_name = f"shots_{year}.csv"
        if (
            archive.namelist() != [expected_name]
            or archive.getinfo(expected_name).file_size > 1_000_000_000
        ):
            raise ValueError(f"unexpected MoneyPuck archive structure for season {year}")
        with archive.open(expected_name) as raw:
            reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            required = {"season", "game_id", "isHomeTeam", "xGoal"}
            if not required <= set(reader.fieldnames or ()):
                raise ValueError(f"MoneyPuck season {year} is missing required columns")
            return _aggregate_xg(reader)


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

    # MoneyPuck publishes these exact season archives for non-commercial use.
    # Only two game-level sums survive; raw shot ZIPs are never written to disk.
    expected_goals: dict[tuple[int, str], float] = {}
    for season in seasons:
        expected_goals.update(_fetch_xg(season, timeout))

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
                    value = record.get(source)
                    values[destination] = (
                        None if report == "realtime" and value is None else float(value or 0.0)
                    )

    _complete_shot_attempts(stats)

    starters: dict[tuple[int, str], dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_fetch_goalies, season, game_type, timeout)
            for season in seasons
            for game_type in (2, 3)
        ]
        for future in as_completed(futures):
            for record in future.result():
                if int(record.get("gamesStarted") or 0) != 1:
                    continue
                key = (int(record["gameId"]), record["homeRoad"])
                if key in starters:
                    raise ValueError(f"multiple starting goalies for game {key[0]} ({key[1]})")
                starters[key] = {
                    "goalie_id": int(record["playerId"]),
                    "time_on_ice": int(record["timeOnIce"]),
                    "goals_against": int(record["goalsAgainst"]),
                    "shots_against": int(record["shotsAgainst"]),
                }

    skaters: dict[tuple[int, str], list[dict]] = {}
    months = {
        (
            int(game["season"]),
            int(game["gameType"]),
            int(game["gameDate"][:4]),
            int(game["gameDate"][5:7]),
        )
        for game in games.values()
        if game.get("gameType") in {2, 3}
    }
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_fetch_skaters, season, game_type, year, month, timeout)
            for season, game_type, year, month in months
        ]
        for future in as_completed(futures):
            for record in future.result():
                skaters.setdefault(
                    (int(record["gameId"]), record["homeRoad"]), []
                ).append(record)

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
        xg_by_side: dict[str, float | None] = {}
        for side, orientation in (("home", "H"), ("away", "R")):
            game_stats = stats.get((row["game_id"], orientation))
            required_stats = {
                field.removeprefix(f"{side}_") for field in TEAM_STAT_FIELDS if field.startswith(side)
            }
            if game_stats is None or not required_stats <= game_stats.keys():
                raise ValueError(f"incomplete NHL Stats reports for game {row['game_id']} ({side})")
            row.update({f"{side}_{field}": game_stats[field] for field in required_stats})
            xg_by_side[side] = expected_goals.get((row["game_id"], orientation))
            starter = starters.get((row["game_id"], orientation))
            if starter is None:
                raise ValueError(f"missing starting goalie for game {row['game_id']} ({side})")
            row.update({f"{side}_starter_{field}": value for field, value in starter.items()})
            lineup = skaters.get((row["game_id"], orientation), [])
            if not lineup:
                raise ValueError(f"missing skater lineup for game {row['game_id']} ({side})")
            row[f"{side}_skaters"] = _encode_skaters(lineup)
        if (xg_by_side["home"] is None) != (xg_by_side["away"] is None):
            raise ValueError(f"partial MoneyPuck expected goals for game {row['game_id']}")
        row["xg_available"] = int(xg_by_side["home"] is not None)
        for side in ("home", "away"):
            # Zero is a storage sentinel only when the explicit availability
            # flag is false; downstream form updates skip the whole xG pair.
            row[f"{side}_xg"] = xg_by_side[side] or 0.0
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
        for field in (
            "game_id", "season", "game_type", "home_score", "away_score",
            "overtime", "xg_available",
            *GOALIE_FIELDS,
        ):
            row[field] = int(row[field])
        for field in (*TEAM_STAT_FIELDS, *XG_FIELDS):
            row[field] = float(row[field])
        for field in SKATER_FIELDS:
            row[field] = _decode_skaters(row[field])
    return sorted(rows, key=lambda row: (row["start_time_utc"], row["game_id"]))
