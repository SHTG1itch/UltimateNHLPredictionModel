"""Command-line entry points for downloading data and walk-forward evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import download_seasons, read_games
from .model import backtest


def _write_predictions(rows: list[dict], path: str) -> None:
    """Write the backtest ledger so every forecast remains auditable."""

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

    arguments = parser.parse_args()
    if arguments.command == "download":
        count = download_seasons(arguments.seasons, arguments.output)
        print(json.dumps({"games": count, "output": arguments.output}, indent=2))
    else:
        predictions, metrics = backtest(read_games(arguments.input))
        _write_predictions(predictions, arguments.output)
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

