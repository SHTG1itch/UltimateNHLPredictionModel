"""Download reproducible game results from the public NHL web API."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
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
FIELDS = (
    "game_id", "season", "game_type", "game_date", "start_time_utc",
    "home_team", "away_team", "home_score", "away_score", "overtime",
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

    rows = []
    for game in games.values():
        # Preseason (type 1) is excluded because rosters and incentives differ.
        if game.get("gameType") not in {2, 3}:
            continue
        period = game.get("gameOutcome", {}).get("lastPeriodType", "REG")
        rows.append(
            {
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
        )

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
    return sorted(rows, key=lambda row: (row["start_time_utc"], row["game_id"]))

