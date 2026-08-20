"""Command-line entry points for downloading data and walk-forward evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .data import download_seasons, fetch_schedule, read_games
from .model import backtest, forecast


def _file_metadata(path: str) -> dict[str, int | str]:
    """Return reproducibility metadata without loading the whole file in memory."""

    digest = hashlib.sha256()
    source = Path(path)
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": source.stat().st_size, "sha256": digest.hexdigest()}


def _write_predictions(rows: list[dict], path: str) -> None:
    """Write the backtest ledger so every forecast remains auditable."""

    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse the deliberately small public CLI."""

    parser = argparse.ArgumentParser(prog="nhl-predict")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download completed NHL games")
    download.add_argument("--seasons", nargs="+", type=int, required=True)
    download.add_argument("--output", default="data/games.csv")

    evaluate = commands.add_parser("backtest", help="run chronological pre-game predictions")
    evaluate.add_argument("--input", default="data/games.csv")
    evaluate.add_argument("--output", default="artifacts/predictions.csv")
    evaluate.add_argument("--evaluate-from", type=int, help="score this season and later")

    predict = commands.add_parser("predict", help="forecast one future NHL date")
    predict.add_argument("--date", required=True, help="date in YYYY-MM-DD format")
    predict.add_argument("--input", default="data/games.csv")
    predict.add_argument("--output", default="artifacts/forecast.csv")

    arguments = parser.parse_args()
    if arguments.command == "download":
        count = download_seasons(arguments.seasons, arguments.output)
        print(
            json.dumps(
                {
                    "games": count,
                    "output": arguments.output,
                    **_file_metadata(arguments.output),
                },
                indent=2,
            )
        )
    elif arguments.command == "backtest":
        predictions, metrics = backtest(
            read_games(arguments.input),
            evaluation_start_season=arguments.evaluate_from,
        )
        _write_predictions(predictions, arguments.output)
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        predictions = forecast(read_games(arguments.input), fetch_schedule(arguments.date))
        _write_predictions(predictions, arguments.output)
        print(json.dumps({"games": len(predictions), "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
