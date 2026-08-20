"""Sequential Bayesian Plackett-Luce model and leakage-safe backtest."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from math import exp, log, pi, sqrt
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .random_matrix import SpectralDiagnostics, clean_covariance


def _sigmoid(value: float) -> float:
    """Numerically stable logistic link used by the two-item Plackett-Luce model."""

    return 1.0 / (1.0 + exp(-value)) if value >= 0 else exp(value) / (1.0 + exp(value))


def _logit(probability: float) -> float:
    """Map a smoothed rate to log-odds without numerical infinities."""

    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return log(probability / (1.0 - probability))


class BayesianPlackettLuce:
    """Online Gaussian posterior over team strength and pre-game context.

    A two-team Plackett-Luce ranking is the Bradley-Terry logistic likelihood.
    Each result receives a one-step Laplace/assumed-density update, so every
    forecast is made from the posterior containing only earlier games.
    """

    def __init__(
        self,
        teams: Iterable[str],
        *,
        team_prior_variance: float = 1.0,
        context_prior_variance: float = 2.0,
        daily_process_variance: float = 0.00001,
        covariance_window_weeks: int = 104,
        form_decay: float = 0.96,
        offseason_retention: float = 0.60,
        offseason_variance: float = 0.025,
    ) -> None:
        unique_teams = sorted(set(teams))
        if len(unique_teams) < 2:
            raise ValueError("at least two teams are required")
        if min(
            team_prior_variance,
            context_prior_variance,
            daily_process_variance,
            offseason_variance,
        ) <= 0:
            raise ValueError("all variances must be positive")
        if not 0.0 < form_decay < 1.0 or not 0.0 <= offseason_retention <= 1.0:
            raise ValueError("decay and retention must be probabilities")

        self.teams = unique_teams
        self.index = {team: i for i, team in enumerate(unique_teams)}
        self.team_count = len(unique_teams)
        # Context coefficients learn home ice, rest, and six rolling team-form
        # contrasts: goals, shots, finishing, save rate, special teams, and SAT.
        self.context_count = 8
        dimension = self.team_count + self.context_count
        self.mean = np.zeros(dimension)
        variances = (
            [team_prior_variance] * self.team_count
            + [context_prior_variance] * self.context_count
        )
        self.covariance = np.diag(variances).astype(float)
        self.daily_process_variance = daily_process_variance
        self.diffusion = np.eye(self.team_count) * daily_process_variance
        self.covariance_window_weeks = covariance_window_weeks
        self.form_decay = form_decay
        self.offseason_retention = offseason_retention
        self.offseason_variance = offseason_variance
        # Columns hold GF, GA, SF, SA, PP goals/opportunities, PP goals allowed,
        # times shorthanded, and shot attempts for/against. They are decayed
        # sufficient statistics, not current-game inputs.
        self.form = np.zeros((self.team_count, 10))
        self.last_date: date | None = None
        self.last_played: dict[str, date] = {}
        self.current_week: tuple[int, int] | None = None
        self.weekly_residual = np.zeros(self.team_count)
        self.weekly_innovations: list[NDArray[np.float64]] = []
        self.spectral_diagnostics: SpectralDiagnostics | None = None

    def _rest_feature(self, home: str, away: str, played_on: date) -> float:
        """Return bounded home-minus-away rest; unknown/opening rest is neutral."""

        def days(team: str) -> int:
            previous = self.last_played.get(team)
            return 3 if previous is None else min(max((played_on - previous).days - 1, 0), 3)

        return (days(home) - days(away)) / 3.0

    def _form_rates(self, team: str) -> NDArray[np.float64]:
        """Return Bayesian-smoothed rates from only the team's previous games."""

        (
            gf, ga, sf, sa, pp_goals, pp_chances, pk_goals,
            pk_chances, sat_for, sat_against,
        ) = self.form[self.index[team]]
        # Pseudo-counts encode conservative league-average priors and disappear
        # as evidence accumulates. This stabilizes percentages early each season.
        return np.array(
            [
                (gf + 30.0) / (gf + ga + 60.0),
                (sf + 300.0) / (sf + sa + 600.0),
                (gf + 4.75) / (sf + 50.0),
                1.0 - (ga + 4.75) / (sa + 50.0),
                (pp_goals + 2.0) / (pp_chances + 10.0),
                1.0 - (pk_goals + 2.0) / (pk_chances + 10.0),
                (sat_for + 600.0) / (sat_for + sat_against + 1200.0),
            ]
        )

    def _form_features(self, home: str, away: str) -> NDArray[np.float64]:
        """Construct bounded home-minus-away log-odds form contrasts."""

        home_rates = self._form_rates(home)
        away_rates = self._form_rates(away)
        differences = np.array([
            _logit(home_rate) - _logit(away_rate)
            for home_rate, away_rate in zip(home_rates, away_rates)
        ])
        # PP and PK are combined into one team special-teams strength feature;
        # keeping both separately would add two highly collinear coefficients.
        special_teams = (differences[4] + differences[5]) / 2.0
        selected = np.r_[differences[:4], special_teams, differences[6]]
        return np.clip(selected, -2.0, 2.0)

    def _features(self, home: str, away: str, played_on: date) -> NDArray[np.float64]:
        """Build the sparse comparison vector for one game."""

        if home not in self.index or away not in self.index or home == away:
            raise ValueError("home and away must be distinct known teams")
        features = np.zeros(self.mean.size)
        features[self.index[home]] = 1.0
        features[self.index[away]] = -1.0
        features[self.team_count] = 1.0  # Constant home-ice log-worth advantage.
        features[self.team_count + 1] = self._rest_feature(home, away, played_on)
        features[self.team_count + 2 :] = self._form_features(home, away)
        return features

    def start_new_season(self) -> None:
        """Partially pool last season toward league average after roster churn."""

        self.mean[: self.team_count] *= self.offseason_retention
        team_slice = np.s_[: self.team_count, : self.team_count]
        self.covariance[team_slice] += np.eye(self.team_count) * self.offseason_variance
        self.form *= self.offseason_retention
        # Offseason duration is not meaningful game rest, so opening-day rest
        # contrasts begin neutral for every team.
        self.last_played.clear()

    def _advance_time(self, played_on: date) -> None:
        """Diffuse uncertain team strengths between game dates."""

        if self.last_date is not None:
            elapsed = (played_on - self.last_date).days
            if elapsed < 0:
                raise ValueError("games must be processed chronologically")
            if elapsed:
                self.covariance[: self.team_count, : self.team_count] += elapsed * self.diffusion
        self.last_date = played_on

    def _roll_week(self, played_on: date) -> None:
        """Aggregate innovations weekly and periodically learn correlated drift."""

        week = played_on.isocalendar()[:2]
        if self.current_week is None:
            self.current_week = week
            return
        if week == self.current_week:
            return

        self.weekly_innovations.append(self.weekly_residual.copy())
        self.weekly_residual.fill(0.0)
        self.current_week = week
        # At least two observations per dimension avoids treating a profoundly
        # undersampled spectrum as reliable structure.
        if len(self.weekly_innovations) < 2 * self.team_count:
            return
        observations = np.asarray(self.weekly_innovations[-self.covariance_window_weeks :])
        cleaned, self.spectral_diagnostics = clean_covariance(observations)
        mean_variance = float(np.diag(cleaned).mean())
        self.diffusion = cleaned * (self.daily_process_variance / max(mean_variance, 1e-12))

    def predict(self, home: str, away: str, played_on: date) -> float:
        """Return posterior predictive probability that the home team wins."""

        self._advance_time(played_on)
        self._roll_week(played_on)
        features = self._features(home, away, played_on)
        margin_mean = float(features @ self.mean)
        margin_variance = max(float(features @ self.covariance @ features), 0.0)
        # MacKay's logistic-Gaussian approximation integrates parameter
        # uncertainty instead of pretending the posterior mean is exact.
        adjusted_margin = margin_mean / sqrt(1.0 + pi * margin_variance / 8.0)
        return _sigmoid(adjusted_margin)

    def _update_form(self, home: str, away: str, game: dict) -> None:
        """Assimilate post-game boxscore sufficient statistics after prediction."""

        required = {
            f"{side}_{field}"
            for side in ("home", "away")
            for field in (
                "shots", "pp_goals", "pp_opportunities", "pk_goals_against",
                "times_shorthanded", "shot_attempts",
            )
        }
        if not required <= game.keys():
            return  # Hand-built callers may use the ranking model without boxscores.

        home_index, away_index = self.index[home], self.index[away]
        self.form[[home_index, away_index]] *= self.form_decay
        home_stats = np.array(
            [
                game["home_score"], game["away_score"],
                game["home_shots"], game["away_shots"],
                game["home_pp_goals"], game["home_pp_opportunities"],
                game["home_pk_goals_against"], game["home_times_shorthanded"],
                game["home_shot_attempts"], game["away_shot_attempts"],
            ],
            dtype=float,
        )
        away_stats = np.array(
            [
                home_stats[1], home_stats[0], home_stats[3], home_stats[2],
                game["away_pp_goals"], game["away_pp_opportunities"],
                game["away_pk_goals_against"], game["away_times_shorthanded"],
                home_stats[9], home_stats[8],
            ],
            dtype=float,
        )
        self.form[home_index] += home_stats
        self.form[away_index] += away_stats

    def update(
        self,
        home: str,
        away: str,
        played_on: date,
        home_win: bool,
        game: dict | None = None,
    ) -> None:
        """Assimilate one observed ranking into the Gaussian posterior."""

        self._advance_time(played_on)
        self._roll_week(played_on)
        features = self._features(home, away, played_on)
        probability = _sigmoid(float(features @ self.mean))
        residual = float(home_win) - probability
        projected = self.covariance @ features
        curvature = max(probability * (1.0 - probability), 1e-9)

        # Sherman-Morrison applies the rank-one Hessian update without a costly
        # matrix inverse on every game.
        denominator = 1.0 + curvature * float(features @ projected)
        self.covariance -= curvature * np.outer(projected, projected) / denominator
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.mean += (self.covariance @ features) * residual

        # Only strength differences are identifiable, so recentering removes
        # harmless numerical drift without changing any game probability.
        self.mean[: self.team_count] -= self.mean[: self.team_count].mean()
        self.weekly_residual[self.index[home]] += residual
        self.weekly_residual[self.index[away]] -= residual
        self.last_played[home] = played_on
        self.last_played[away] = played_on
        if game is not None:
            self._update_form(home, away, game)


def backtest(games: list[dict], **model_options: float) -> tuple[list[dict], dict[str, float]]:
    """Generate one strictly out-of-sample prediction per chronological game."""

    if not games:
        raise ValueError("games cannot be empty")
    teams = {row[side] for row in games for side in ("home_team", "away_team")}
    model = BayesianPlackettLuce(teams, **model_options)
    predictions = []
    losses = []
    home_rate_losses = []
    elo_losses = []
    correct = 0
    # Beta(11, 9) is a weak 55% home-win prior; it prevents an undefined or
    # overconfident empirical rate during the first games of the data set.
    home_wins, games_seen = 11.0, 20.0
    # This intentionally plain Elo is a credible sequential baseline, not a
    # tuned competitor: 35 rating points for home ice and K=20 after each game.
    elo: defaultdict[str, float] = defaultdict(lambda: 1500.0)
    current_season: int | None = None

    for row in sorted(games, key=lambda game: (game["start_time_utc"], game["game_id"])):
        if current_season is not None and row.get("season") != current_season:
            model.start_new_season()
        current_season = row.get("season", current_season)
        played_on = date.fromisoformat(row["game_date"])
        probability = model.predict(row["home_team"], row["away_team"], played_on)
        outcome = int(row["home_score"] > row["away_score"])
        home_rate_probability = home_wins / games_seen
        elo_probability = 1.0 / (
            1.0
            + 10.0
            ** ((elo[row["away_team"]] - elo[row["home_team"]] - 35.0) / 400.0)
        )
        clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
        losses.append(-(outcome * log(clipped) + (1 - outcome) * log(1 - clipped)))
        home_rate_losses.append(
            -(outcome * log(home_rate_probability) + (1 - outcome) * log(1 - home_rate_probability))
        )
        elo_losses.append(
            -(outcome * log(elo_probability) + (1 - outcome) * log(1 - elo_probability))
        )
        correct += int((probability >= 0.5) == bool(outcome))
        predictions.append(
            {
                **row,
                "home_win_probability": probability,
                "home_rate_probability": home_rate_probability,
                "elo_probability": elo_probability,
                "home_win": outcome,
                "log_loss": losses[-1],
            }
        )
        # The result becomes available only after its prediction is recorded.
        model.update(row["home_team"], row["away_team"], played_on, bool(outcome), row)
        home_wins += outcome
        games_seen += 1.0
        elo_change = 20.0 * (outcome - elo_probability)
        elo[row["home_team"]] += elo_change
        elo[row["away_team"]] -= elo_change

    probabilities = np.array([row["home_win_probability"] for row in predictions])
    outcomes = np.array([row["home_win"] for row in predictions])
    metrics = {
        "games": float(len(games)),
        "accuracy": correct / len(games),
        "log_loss": float(np.mean(losses)),
        "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
        "home_rate_log_loss": float(np.mean(home_rate_losses)),
        "elo_log_loss": float(np.mean(elo_losses)),
    }
    # Flat per-season keys keep the CLI JSON easy to consume while exposing
    # regime drift that one aggregate score can hide.
    if all("season" in row for row in predictions):
        for season in sorted({row["season"] for row in predictions}):
            season_rows = [row for row in predictions if row["season"] == season]
            season_probabilities = np.array(
                [row["home_win_probability"] for row in season_rows]
            )
            season_outcomes = np.array([row["home_win"] for row in season_rows])
            metrics[f"season_{season}_accuracy"] = float(
                np.mean((season_probabilities >= 0.5) == season_outcomes)
            )
            metrics[f"season_{season}_brier_score"] = float(
                np.mean((season_probabilities - season_outcomes) ** 2)
            )
            metrics[f"season_{season}_log_loss"] = float(
                np.mean([row["log_loss"] for row in season_rows])
            )
            metrics[f"season_{season}_elo_log_loss"] = float(
                np.mean(
                    [
                        -(
                            row["home_win"] * log(row["elo_probability"])
                            + (1 - row["home_win"]) * log(1 - row["elo_probability"])
                        )
                        for row in season_rows
                    ]
                )
            )
    return predictions, metrics


def fit(
    games: list[dict],
    *,
    extra_teams: Iterable[str] = (),
    **model_options: float,
) -> tuple[BayesianPlackettLuce, int | None]:
    """Fit the sequential posterior to all completed games for live forecasts."""

    if not games:
        raise ValueError("games cannot be empty")
    teams = {row[side] for row in games for side in ("home_team", "away_team")}
    model = BayesianPlackettLuce(teams | set(extra_teams), **model_options)
    current_season: int | None = None
    for row in sorted(games, key=lambda game: (game["start_time_utc"], game["game_id"])):
        if current_season is not None and row.get("season") != current_season:
            model.start_new_season()
        current_season = row.get("season", current_season)
        played_on = date.fromisoformat(row["game_date"])
        model.update(
            row["home_team"],
            row["away_team"],
            played_on,
            row["home_score"] > row["away_score"],
            row,
        )
    return model, current_season


def forecast(games: list[dict], schedule: list[dict], **model_options: float) -> list[dict]:
    """Fit completed history and forecast an exact future schedule."""

    if not schedule:
        return []
    latest_result = max(row["start_time_utc"] for row in games)
    if min(row["start_time_utc"] for row in schedule) <= latest_result:
        raise ValueError("scheduled games must occur after every training result")

    scheduled_teams = {
        row[side] for row in schedule for side in ("home_team", "away_team")
    }
    model, current_season = fit(games, extra_teams=scheduled_teams, **model_options)
    predictions = []
    for row in sorted(schedule, key=lambda game: (game["start_time_utc"], game["game_id"])):
        if current_season is not None and row.get("season") != current_season:
            model.start_new_season()
        current_season = row.get("season", current_season)
        probability = model.predict(
            row["home_team"], row["away_team"], date.fromisoformat(row["game_date"])
        )
        predictions.append({**row, "home_win_probability": probability})
    return predictions
