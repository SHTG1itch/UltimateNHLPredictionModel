from __future__ import annotations

import csv
from datetime import date, timedelta

import numpy as np

from nhl_predictor.data import FIELDS, TEAM_STAT_FIELDS, read_games
from nhl_predictor.model import BayesianPlackettLuce, backtest
from nhl_predictor.random_matrix import clean_covariance


def test_repeated_wins_raise_probability() -> None:
    model = BayesianPlackettLuce(["A", "B"], daily_process_variance=1e-6)
    day = date(2025, 1, 1)
    before = model.predict("A", "B", day)
    for offset in range(12):
        played_on = day + timedelta(days=offset)
        model.update("A", "B", played_on, True)
    after = model.predict("A", "B", day + timedelta(days=12))
    assert before == 0.5
    assert before < after < 1.0


def test_backtest_predicts_before_learning_result() -> None:
    games = [
        {
            "game_id": i,
            "start_time_utc": f"2025-01-{i:02d}T00:00:00Z",
            "game_date": f"2025-01-{i:02d}",
            "home_team": "A",
            "away_team": "B",
            "home_score": 3,
            "away_score": 1,
        }
        for i in range(1, 5)
    ]
    predictions, metrics = backtest(games)
    assert predictions[0]["home_win_probability"] == 0.5
    assert predictions[-1]["home_win_probability"] > predictions[1]["home_win_probability"]
    assert metrics["games"] == 4
    assert set(("home_rate_log_loss", "elo_log_loss")) <= metrics.keys()


def test_mp_cleaner_is_positive_definite_and_detects_signal() -> None:
    rng = np.random.default_rng(7)
    common = rng.normal(size=(500, 1))
    observations = rng.normal(size=(500, 8))
    observations[:, :2] += 2.5 * common
    covariance, diagnostics = clean_covariance(observations)
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert diagnostics.signal_eigenvalues >= 1
    assert diagnostics.noise_eigenvalues + diagnostics.signal_eigenvalues == 8


def test_enriched_game_csv_is_typed(tmp_path) -> None:
    path = tmp_path / "games.csv"
    row = {field: 0 for field in FIELDS}
    row.update(
        game_id=1,
        season=20242025,
        game_type=2,
        game_date="2025-01-01",
        start_time_utc="2025-01-01T00:00:00Z",
        home_team="A",
        away_team="B",
        home_score=3,
        away_score=2,
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)

    parsed = read_games(path)[0]
    assert isinstance(parsed["game_id"], int)
    assert all(isinstance(parsed[field], float) for field in TEAM_STAT_FIELDS)
