"""Sequential Bayesian Plackett-Luce model and leakage-safe backtest."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from math import exp, log, pi, sqrt
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .random_matrix import SpectralDiagnostics, clean_covariance

FORM_FEATURES = (
    "goal_share",
    "shot_share",
    "finishing",
    "save",
    "special_teams",
    "sat_share",
    "xg_share",
    "xg_finishing",
    "xg_saving",
)
DEFAULT_FORM_FEATURES = (
    "xg_share",
    "shot_share",
    "special_teams",
    "sat_share",
    "xg_finishing",
    "xg_saving",
)


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
        context_prior_variance: float = 1.0,
        daily_process_variance: float = 0.00001,
        covariance_window_weeks: int = 104,
        form_decay: float = 0.96,
        offseason_retention: float = 0.60,
        offseason_variance: float = 0.025,
        form_features: tuple[str, ...] = DEFAULT_FORM_FEATURES,
        overtime_weight: float = 0.70,
        overtime_credit: float = 1.0,
        margin_scale: float = 0.0,
        james_stein_scale: float = 0.0,
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
        if not 0.0 < overtime_weight <= 1.0:
            raise ValueError("overtime_weight must be in (0, 1]")
        if not 0.5 <= overtime_credit <= 1.0:
            raise ValueError("overtime_credit must be in [0.5, 1]")
        if margin_scale < 0.0:
            raise ValueError("margin_scale cannot be negative")
        if james_stein_scale < 0.0:
            raise ValueError("james_stein_scale cannot be negative")
        if not form_features or len(set(form_features)) != len(form_features):
            raise ValueError("form_features must contain unique feature names")
        unknown_features = set(form_features) - set(FORM_FEATURES)
        if unknown_features:
            raise ValueError(f"unknown form features: {', '.join(sorted(unknown_features))}")

        self.teams = unique_teams
        self.index = {team: i for i, team in enumerate(unique_teams)}
        self.team_count = len(unique_teams)
        # Every selected form contrast gets one Bayesian coefficient after the
        # fixed home-ice and rest terms.
        self.form_features = form_features
        self.context_count = 2 + len(form_features)
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
        self.overtime_weight = overtime_weight
        self.overtime_credit = overtime_credit
        self.margin_scale = margin_scale
        self.james_stein_scale = james_stein_scale
        # Columns hold GF, GA, SF, SA, PP goals/opportunities, PP goals allowed,
        # times shorthanded, shot attempts for/against, xGF/xGA, and goals
        # for/against in xG-covered games. They are decayed sufficient stats.
        self.form = np.zeros((self.team_count, 14))
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

    def _form_rates(self, team: str) -> dict[str, float]:
        """Return Bayesian-smoothed rates from only the team's previous games."""

        (
            gf, ga, sf, sa, pp_goals, pp_chances, pk_goals,
            pk_chances, sat_for, sat_against, xgf, xga, xg_gf, xg_ga,
        ) = self.form[self.index[team]]
        # Pseudo-counts encode conservative league-average priors and disappear
        # as evidence accumulates. This stabilizes percentages early each season.
        return {
            "goal_share": (gf + 30.0) / (gf + ga + 60.0),
            "shot_share": (sf + 300.0) / (sf + sa + 600.0),
            "finishing": (gf + 4.75) / (sf + 50.0),
            "save": 1.0 - (ga + 4.75) / (sa + 50.0),
            "power_play": (pp_goals + 2.0) / (pp_chances + 10.0),
            "penalty_kill": 1.0 - (pk_goals + 2.0) / (pk_chances + 10.0),
            "sat_share": (sat_for + 600.0) / (sat_for + sat_against + 1200.0),
            "xg_share": (xgf + 30.0) / (xgf + xga + 60.0),
            # Pearson-style residuals distinguish finishing and goaltending
            # from chance quality while shrinking small xG samples toward zero.
            "xg_finishing": (xg_gf - xgf) / sqrt(xgf + 10.0),
            "xg_saving": (xga - xg_ga) / sqrt(xga + 10.0),
        }

    def _form_features(self, home: str, away: str) -> NDArray[np.float64]:
        """Construct bounded home-minus-away log-odds form contrasts."""

        home_rates = self._form_rates(home)
        away_rates = self._form_rates(away)
        rate_names = ("goal_share", "shot_share", "finishing", "save", "sat_share", "xg_share")
        features = {
            name: _logit(home_rates[name]) - _logit(away_rates[name])
            for name in rate_names
        }
        # PP and PK are combined into one team special-teams strength feature;
        # keeping both separately would add two highly collinear coefficients.
        features["special_teams"] = (
            _logit(home_rates["power_play"])
            - _logit(away_rates["power_play"])
            + _logit(home_rates["penalty_kill"])
            - _logit(away_rates["penalty_kill"])
        ) / 2.0
        for name in ("xg_finishing", "xg_saving"):
            features[name] = home_rates[name] - away_rates[name]
        return np.clip([features[name] for name in self.form_features], -2.0, 2.0)

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

    def predict_details(self, home: str, away: str, played_on: date) -> dict[str, float]:
        """Return win probability and posterior uncertainty for one matchup."""

        self._advance_time(played_on)
        self._roll_week(played_on)
        features = self._features(home, away, played_on)
        team_mean = self.mean[: self.team_count]
        squared_signal = float(team_mean @ team_mean)
        pooled_variance = float(np.trace(
            self.covariance[: self.team_count, : self.team_count]
        )) / self.team_count
        # Positive-part James-Stein estimator for p >= 3 team effects. Scaling
        # zero cleanly disables it for the validation ablation.
        james_stein_factor = (
            max(
                0.0,
                1.0
                - self.james_stein_scale
                * (self.team_count - 2)
                * pooled_variance
                / max(squared_signal, 1e-12),
            )
            if self.team_count >= 3
            else 1.0
        )
        margin_mean = float(features @ self.mean) + (james_stein_factor - 1.0) * (
            team_mean[self.index[home]] - team_mean[self.index[away]]
        )
        margin_variance = max(float(features @ self.covariance @ features), 0.0)
        margin_sd = sqrt(margin_variance)
        # MacKay's logistic-Gaussian approximation integrates parameter
        # uncertainty instead of pretending the posterior mean is exact.
        adjusted_margin = margin_mean / sqrt(1.0 + pi * margin_variance / 8.0)
        # Transforming Gaussian margin quantiles gives an epistemic interval for
        # the underlying win propensity. It is not a frequentist game-outcome CI.
        return {
            "home_win_probability": _sigmoid(adjusted_margin),
            "latent_margin_mean": margin_mean,
            "latent_margin_sd": margin_sd,
            "james_stein_factor": james_stein_factor,
            "home_win_epistemic_lower": _sigmoid(margin_mean - 1.96 * margin_sd),
            "home_win_epistemic_upper": _sigmoid(margin_mean + 1.96 * margin_sd),
        }

    def predict(self, home: str, away: str, played_on: date) -> float:
        """Return posterior predictive probability that the home team wins."""

        return self.predict_details(home, away, played_on)["home_win_probability"]

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
        xg_available = bool(game.get("xg_available", 0))
        home_stats = np.array(
            [
                game["home_score"], game["away_score"],
                game["home_shots"], game["away_shots"],
                game["home_pp_goals"], game["home_pp_opportunities"],
                game["home_pk_goals_against"], game["home_times_shorthanded"],
                game["home_shot_attempts"], game["away_shot_attempts"],
                game.get("home_xg", 0.0) if xg_available else 0.0,
                game.get("away_xg", 0.0) if xg_available else 0.0,
                game["home_score"] if xg_available else 0.0,
                game["away_score"] if xg_available else 0.0,
            ],
            dtype=float,
        )
        away_stats = np.array(
            [
                home_stats[1], home_stats[0], home_stats[3], home_stats[2],
                game["away_pp_goals"], game["away_pp_opportunities"],
                game["away_pk_goals_against"], game["away_times_shorthanded"],
                home_stats[9], home_stats[8],
                home_stats[11], home_stats[10], home_stats[13], home_stats[12],
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
        target = float(home_win)
        if game is not None and game.get("overtime"):
            target = self.overtime_credit if home_win else 1.0 - self.overtime_credit
        residual = target - probability
        projected = self.covariance @ features
        # An OT/SO game is a near-tie ranking, so a fractional power likelihood
        # can keep it from moving strength as much as a regulation result.
        observation_weight = (
            self.overtime_weight if game is not None and game.get("overtime") else 1.0
        )
        if game is not None and self.margin_scale and {
            "home_score", "away_score"
        } <= game.keys():
            observation_weight *= 1.0 + self.margin_scale * log(
                max(abs(game["home_score"] - game["away_score"]), 1)
            )
        curvature = max(observation_weight * probability * (1.0 - probability), 1e-9)

        # Sherman-Morrison applies the rank-one Hessian update without a costly
        # matrix inverse on every game.
        denominator = 1.0 + curvature * float(features @ projected)
        self.covariance -= curvature * np.outer(projected, projected) / denominator
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.mean += (self.covariance @ features) * residual * observation_weight

        # Only strength differences are identifiable, so recentering removes
        # harmless numerical drift without changing any game probability.
        self.mean[: self.team_count] -= self.mean[: self.team_count].mean()
        self.weekly_residual[self.index[home]] += residual
        self.weekly_residual[self.index[away]] -= residual
        self.last_played[home] = played_on
        self.last_played[away] = played_on
        if game is not None:
            self._update_form(home, away, game)


class BayesianModelAverage:
    """Online Bayesian average of two validation-selected PL specifications."""

    def __init__(
        self,
        teams: Iterable[str],
        *,
        forgetting: float = 0.50,
        likelihood_power: float = 1.0,
        **model_options: float,
    ) -> None:
        if not 0.0 < forgetting <= 1.0 or not 0.0 < likelihood_power <= 1.0:
            raise ValueError("BMA forgetting and likelihood power must be in (0, 1]")
        teams = tuple(teams)
        specifications = (
            {
                "form_features": ("xg_share", "shot_share", "sat_share"),
                "form_decay": 0.94,
            },
            {
                "form_decay": 0.70,
                "james_stein_scale": 0.15,
                "overtime_weight": 0.40,
                "overtime_credit": 0.90,
                "margin_scale": 0.20,
            },
        )
        self.names = ("possession", "fast_james_stein")
        self.members = [
            BayesianPlackettLuce(teams, **(specification | model_options))
            for specification in specifications
        ]
        self.log_weights = np.full(len(self.members), -log(len(self.members)))
        self.forgetting = forgetting
        self.likelihood_power = likelihood_power
        self._pending: tuple[str, str, date, NDArray[np.float64]] | None = None

    @property
    def spectral_diagnostics(self) -> SpectralDiagnostics | None:
        """Expose the first member's identically scheduled spectral audit."""

        return self.members[0].spectral_diagnostics

    @property
    def weights(self) -> NDArray[np.float64]:
        """Return normalized posterior model probabilities."""

        return np.exp(self.log_weights)

    def start_new_season(self) -> None:
        """Apply the same offseason transition to every candidate model."""

        for member in self.members:
            member.start_new_season()
        self._pending = None

    def predict_details(self, home: str, away: str, played_on: date) -> dict[str, float]:
        """Average predictive distributions using pre-game model weights."""

        member_details = [
            member.predict_details(home, away, played_on) for member in self.members
        ]
        probabilities = np.array(
            [details["home_win_probability"] for details in member_details]
        )
        weights = self.weights
        margin_means = np.array(
            [details["latent_margin_mean"] for details in member_details]
        )
        margin_variances = np.array(
            [details["latent_margin_sd"] ** 2 for details in member_details]
        )
        margin_mean = float(weights @ margin_means)
        margin_sd = sqrt(max(float(
            weights @ (margin_variances + margin_means**2) - margin_mean**2
        ), 0.0))
        self._pending = (home, away, played_on, probabilities)
        result = {
            "home_win_probability": float(weights @ probabilities),
            "latent_margin_mean": margin_mean,
            "latent_margin_sd": margin_sd,
            "home_win_epistemic_lower": _sigmoid(margin_mean - 1.96 * margin_sd),
            "home_win_epistemic_upper": _sigmoid(margin_mean + 1.96 * margin_sd),
            "james_stein_factor": float(
                weights @ np.array([
                    details["james_stein_factor"] for details in member_details
                ])
            ),
        }
        for index, name in enumerate(self.names):
            result[f"{name}_probability"] = float(probabilities[index])
            result[f"{name}_weight"] = float(weights[index])
        return result

    def predict(self, home: str, away: str, played_on: date) -> float:
        """Return the model-averaged home-win probability."""

        return self.predict_details(home, away, played_on)["home_win_probability"]

    def update(
        self,
        home: str,
        away: str,
        played_on: date,
        home_win: bool,
        game: dict | None = None,
    ) -> None:
        """Update model probabilities from the held-out result, then each member."""

        if self._pending is None or self._pending[:3] != (home, away, played_on):
            self.predict_details(home, away, played_on)
        assert self._pending is not None
        probabilities = self._pending[3]
        likelihoods = probabilities if home_win else 1.0 - probabilities
        self.log_weights = (
            self.forgetting * self.log_weights
            + self.likelihood_power * np.log(np.clip(likelihoods, 1e-12, 1.0))
        )
        self.log_weights -= np.logaddexp.reduce(self.log_weights)
        for member in self.members:
            member.update(home, away, played_on, home_win, game)
        self._pending = None


def backtest(
    games: list[dict],
    *,
    model_average: bool = True,
    **model_options: float,
) -> tuple[list[dict], dict[str, float]]:
    """Generate one strictly out-of-sample prediction per chronological game."""

    if not games:
        raise ValueError("games cannot be empty")
    teams = {row[side] for row in games for side in ("home_team", "away_team")}
    model = (
        BayesianModelAverage(teams, **model_options)
        if model_average
        else BayesianPlackettLuce(teams, **model_options)
    )
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
        details = model.predict_details(row["home_team"], row["away_team"], played_on)
        probability = details["home_win_probability"]
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
                **details,
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
    if model.spectral_diagnostics is not None:
        spectral = model.spectral_diagnostics
        metrics.update(
            {
                "mp_upper_edge": spectral.mp_upper_edge,
                "mp_noise_eigenvalues": float(spectral.noise_eigenvalues),
                "mp_signal_eigenvalues": float(spectral.signal_eigenvalues),
                "wigner_radius": spectral.wigner_radius,
            }
        )
    if isinstance(model, BayesianModelAverage):
        metrics.update({
            f"final_{name}_weight": float(weight)
            for name, weight in zip(model.names, model.weights, strict=True)
        })
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
    model_average: bool = True,
    **model_options: float,
) -> tuple[BayesianPlackettLuce | BayesianModelAverage, int | None]:
    """Fit the sequential posterior to all completed games for live forecasts."""

    if not games:
        raise ValueError("games cannot be empty")
    teams = {row[side] for row in games for side in ("home_team", "away_team")}
    model = (
        BayesianModelAverage(teams | set(extra_teams), **model_options)
        if model_average
        else BayesianPlackettLuce(teams | set(extra_teams), **model_options)
    )
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


def forecast(
    games: list[dict],
    schedule: list[dict],
    *,
    model_average: bool = True,
    **model_options: float,
) -> list[dict]:
    """Fit completed history and forecast an exact future schedule."""

    if not schedule:
        return []
    latest_result = max(row["start_time_utc"] for row in games)
    if min(row["start_time_utc"] for row in schedule) <= latest_result:
        raise ValueError("scheduled games must occur after every training result")

    scheduled_teams = {
        row[side] for row in schedule for side in ("home_team", "away_team")
    }
    model, current_season = fit(
        games,
        extra_teams=scheduled_teams,
        model_average=model_average,
        **model_options,
    )
    predictions = []
    for row in sorted(schedule, key=lambda game: (game["start_time_utc"], game["game_id"])):
        if current_season is not None and row.get("season") != current_season:
            model.start_new_season()
        current_season = row.get("season", current_season)
        details = model.predict_details(
            row["home_team"], row["away_team"], date.fromisoformat(row["game_date"])
        )
        predictions.append({**row, **details})
    return predictions
