from __future__ import annotations

import csv
import hashlib
from datetime import date, timedelta

import numpy as np

from nhl_predictor.data import (
    FIELDS,
    GOALIE_FIELDS,
    TEAM_STAT_FIELDS,
    XG_FIELDS,
    _aggregate_xg,
    _parse_schedule,
    read_games,
)
from nhl_predictor.cli import _file_metadata
from nhl_predictor.model import BayesianPlackettLuce, backtest, forecast
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


def test_overtime_result_has_smaller_posterior_update() -> None:
    regulation = BayesianPlackettLuce(["A", "B"])
    overtime = BayesianPlackettLuce(["A", "B"])
    played_on = date(2025, 1, 1)
    regulation.update("A", "B", played_on, True, {"overtime": 0})
    overtime.update("A", "B", played_on, True, {"overtime": 1})
    assert regulation.mean[regulation.index["A"]] > overtime.mean[overtime.index["A"]]


def test_backtest_predicts_before_learning_result() -> None:
    games = [
        {
            "game_id": i,
            "season": 20242025,
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
    assert (
        predictions[0]["home_win_epistemic_lower"]
        < predictions[0]["home_win_probability"]
        < predictions[0]["home_win_epistemic_upper"]
    )
    assert predictions[-1]["home_win_probability"] > predictions[1]["home_win_probability"]
    assert metrics["games"] == 4
    assert set(("home_rate_log_loss", "elo_log_loss")) <= metrics.keys()
    assert "season_20242025_log_loss" in metrics


def test_mp_cleaner_is_positive_definite_and_detects_signal() -> None:
    rng = np.random.default_rng(7)
    common = rng.normal(size=(500, 1))
    observations = rng.normal(size=(500, 8))
    observations[:, :2] += 2.5 * common
    covariance, diagnostics = clean_covariance(observations)
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert diagnostics.signal_eigenvalues >= 1
    assert diagnostics.noise_eigenvalues + diagnostics.signal_eigenvalues == 8


def test_backtest_exposes_spectral_diagnostics_after_enough_weeks() -> None:
    games = [
        {
            "game_id": offset,
            "season": 20242025,
            "start_time_utc": f"2025-01-{1 + 7 * offset:02d}T00:00:00Z",
            "game_date": f"2025-01-{1 + 7 * offset:02d}",
            "home_team": "A",
            "away_team": "B",
            "home_score": 3,
            "away_score": 2,
        }
        for offset in range(5)
    ]
    _, metrics = backtest(games)
    assert metrics["mp_noise_eigenvalues"] + metrics["mp_signal_eigenvalues"] == 2


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
        xg_available=1,
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)

    parsed = read_games(path)[0]
    assert isinstance(parsed["game_id"], int)
    assert isinstance(parsed["xg_available"], int)
    assert all(isinstance(parsed[field], int) for field in GOALIE_FIELDS)
    assert all(isinstance(parsed[field], float) for field in (*TEAM_STAT_FIELDS, *XG_FIELDS))

    metadata = _file_metadata(str(path))
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["bytes"] == path.stat().st_size


def test_shots_are_compacted_to_game_team_expected_goals() -> None:
    rows = [
        {"season": "2024", "game_id": "20001", "isHomeTeam": "1.0", "xGoal": "0.2"},
        {"season": "2024", "game_id": "20001", "isHomeTeam": "1.0", "xGoal": "0.3"},
        {"season": "2024", "game_id": "20001", "isHomeTeam": "0.0", "xGoal": "0.1"},
    ]
    assert _aggregate_xg(rows) == {
        (2024020001, "H"): 0.5,
        (2024020001, "R"): 0.1,
    }


def test_current_boxscore_enters_form_only_after_update() -> None:
    model = BayesianPlackettLuce(["A", "B"])
    played_on = date(2025, 1, 1)
    before = model.form.copy()
    model.predict("A", "B", played_on)
    assert np.array_equal(model.form, before)

    game = {
        "home_score": 4,
        "away_score": 2,
        "home_shots": 35,
        "away_shots": 25,
        "home_pp_goals": 1,
        "home_pp_opportunities": 3,
        "home_pk_goals_against": 0,
        "home_times_shorthanded": 2,
        "home_shot_attempts": 65,
        "away_pp_goals": 0,
        "away_pp_opportunities": 2,
        "away_pk_goals_against": 1,
        "away_times_shorthanded": 3,
        "away_shot_attempts": 50,
        "home_xg": 3.2,
        "away_xg": 2.1,
        "xg_available": 1,
    }
    model.update("A", "B", played_on, True, game)
    assert model.form[model.index["A"], 0] == 4
    assert model.form[model.index["B"], 0] == 2
    assert model.form[model.index["A"], 10] == 3.2
    assert model.form[model.index["A"], 12] == 4


def test_future_schedule_parser_and_forecast() -> None:
    payload = {
        "gameWeek": [
            {
                "date": "2025-10-10",
                "games": [
                    {
                        "id": 2,
                        "season": 20252026,
                        "gameType": 2,
                        "startTimeUTC": "2025-10-10T23:00:00Z",
                        "homeTeam": {"abbrev": "A"},
                        "awayTeam": {"abbrev": "B"},
                    },
                    {"id": 99, "gameType": 1, "gameDate": "2025-10-11"},
                ]
            }
        ]
    }
    schedule = _parse_schedule(payload, "2025-10-10")
    history = [
        {
            "game_id": 1,
            "season": 20242025,
            "game_date": "2025-04-01",
            "start_time_utc": "2025-04-01T23:00:00Z",
            "home_team": "A",
            "away_team": "B",
            "home_score": 3,
            "away_score": 2,
        }
    ]
    predictions = forecast(history, schedule)
    assert predictions[0]["game_id"] == 2
    assert 0.0 < predictions[0]["home_win_probability"] < 1.0


def test_forecast_rejects_historical_schedule() -> None:
    history = [
        {
            "game_id": 1,
            "game_date": "2025-01-02",
            "start_time_utc": "2025-01-02T00:00:00Z",
            "home_team": "A",
            "away_team": "B",
            "home_score": 3,
            "away_score": 2,
        }
    ]
    schedule = [
        {
            "game_id": 2,
            "game_date": "2025-01-01",
            "start_time_utc": "2025-01-01T00:00:00Z",
            "home_team": "A",
            "away_team": "B",
        }
    ]
    with np.testing.assert_raises_regex(ValueError, "after every training result"):
        forecast(history, schedule)
