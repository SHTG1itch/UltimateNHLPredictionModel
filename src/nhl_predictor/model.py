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
TEAM_ALIASES = {"ARI": "UTA"}


def _franchise(team: str) -> str:
    """Keep inherited hockey history under the active team abbreviation."""

    return TEAM_ALIASES.get(team, team)


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
        team_prior_variance: float = 0.20,
        context_prior_variance: float = 1.0,
        daily_process_variance: float = 0.0000001,
        covariance_window_weeks: int = 104,
        form_decay: float = 0.96,
        offseason_retention: float = 0.60,
        offseason_variance: float = 0.025,
        form_features: tuple[str, ...] = DEFAULT_FORM_FEATURES,
        overtime_weight: float = 0.70,
        overtime_credit: float = 1.0,
        margin_scale: float = 0.0,
        james_stein_scale: float = 0.0,
        matchup_decay: float = 0.50,
        matchup_prior: float = 2.0,
        matchup_scale: float = 0.03,
        player_k: float = 8.0,
        raw_effect_scale: float = 0.50,
        chemistry_k: float = 0.75,
        player_scale: float = 1.0,
        chemistry_scale: float = 1.0,
        sat_scale: float = 0.44,
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
        if not 0.0 < matchup_decay <= 1.0 or matchup_prior <= 0.0 or matchup_scale < 0.0:
            raise ValueError("matchup decay and prior must be positive")
        if min(player_k, raw_effect_scale, chemistry_k, player_scale, chemistry_scale, sat_scale) < 0.0:
            raise ValueError("feature scales cannot be negative")
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
        self.context_count = 5 + len(form_features)
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
        self.matchup_decay = matchup_decay
        self.matchup_prior = matchup_prior
        self.matchup_scale = matchup_scale
        self.matchups: dict[tuple[str, str], list[float]] = {}
        self.player_k = player_k
        self.raw_effect_scale = raw_effect_scale
        self.chemistry_k = chemistry_k
        self.player_scale = player_scale
        self.chemistry_scale = chemistry_scale
        self.sat_scale = sat_scale
        self.player_elos: defaultdict[int, float] = defaultdict(lambda: 1500.0)
        self.chemistry_elos: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.lineups: dict[str, tuple[tuple[tuple[int, float], ...], int | None]] = {}
        # Columns hold GF, GA, SF, SA, PP goals/opportunities, PP goals allowed,
        # times shorthanded, shot attempts for/against, xGF/xGA, and goals
        # for/against in xG-covered games. They are decayed sufficient stats.
        self.form = np.zeros((self.team_count, 14))
        self.last_date: date | None = None
        self.last_played: dict[str, date] = {}
        self.recent_games: defaultdict[str, list[date]] = defaultdict(list)
        self.current_week: tuple[int, int] | None = None
        self.weekly_residual = np.zeros(self.team_count)
        self.weekly_innovations: list[NDArray[np.float64]] = []
        self.spectral_diagnostics: SpectralDiagnostics | None = None

    def _rest_feature(self, home: str, away: str, played_on: date) -> float:
        """Return away-minus-home fatigue from back-to-backs and congestion."""

        def fatigue(team: str) -> float:
            previous = self.last_played.get(team)
            back_to_back = previous is not None and (played_on - previous).days == 1
            # ponytail: NHL season lists are tiny; use deques if schedules grow.
            crowded = max(
                sum((played_on - game).days <= 5 for game in self.recent_games[team]) - 1,
                0,
            )
            return float(back_to_back) + 0.25 * crowded

        return fatigue(away) - fatigue(home)

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
        features["sat_share"] *= self.sat_scale
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

    def _matchup_feature(self, home: str, away: str) -> float:
        """Return prior-smoothed head-to-head log-odds from earlier meetings."""

        first, second = sorted((home, away))
        residual, meetings = self.matchups.get((first, second), (0.0, 0.0))
        signal = residual / (meetings + self.matchup_prior)
        return self.matchup_scale * (signal if home == first else -signal)

    def _player_context(self, team: str) -> tuple[float, float]:
        """Return last-known lineup Elo and pair chemistry without current-game data."""

        lineup = self.lineups.get(team)
        if lineup is None:
            return 1500.0, 0.0
        skaters, goalie = lineup
        skater_elo = sum(weight * self.player_elos[player] for player, weight in skaters)
        player_elo = 0.8 * skater_elo + 0.2 * self.player_elos[goalie] if goalie else skater_elo
        players = [player for player, _ in skaters] + ([goalie] if goalie else [])
        # ponytail: NHL lineups are ~19; use sparse recent pairs if rosters grow.
        pairs = [tuple(sorted((first, second))) for i, first in enumerate(players) for second in players[i + 1 :]]
        chemistry = sum(self.chemistry_elos[pair] for pair in pairs) / len(pairs) if pairs else 0.0
        return player_elo, chemistry

    def _features(self, home: str, away: str, played_on: date) -> NDArray[np.float64]:
        """Build the sparse comparison vector for one game."""

        if home not in self.index or away not in self.index or home == away:
            raise ValueError("home and away must be distinct known teams")
        features = np.zeros(self.mean.size)
        features[self.index[home]] = 1.0
        features[self.index[away]] = -1.0
        features[self.team_count] = 1.0  # Constant home-ice log-worth advantage.
        features[self.team_count + 1] = self._rest_feature(home, away, played_on)
        features[self.team_count + 2] = self._matchup_feature(home, away)
        home_player, home_chemistry = self._player_context(home)
        away_player, away_chemistry = self._player_context(away)
        elo_to_logit = log(10.0) / 400.0
        features[self.team_count + 3] = self.player_scale * elo_to_logit * (
            home_player - away_player
        )
        features[self.team_count + 4] = self.chemistry_scale * elo_to_logit * (
            home_chemistry - away_chemistry
        )
        features[self.team_count + 5 :] = self._form_features(home, away)
        return features

    def start_new_season(self) -> None:
        """Partially pool last season toward league average after roster churn."""

        self.mean[: self.team_count] *= self.offseason_retention
        team_slice = np.s_[: self.team_count, : self.team_count]
        self.covariance[team_slice] += np.eye(self.team_count) * self.offseason_variance
        self.form *= self.offseason_retention
        for history in self.matchups.values():
            history[0] *= self.offseason_retention
            history[1] *= self.offseason_retention
        for player, rating in self.player_elos.items():
            self.player_elos[player] = 1500.0 + self.offseason_retention * (rating - 1500.0)
        for pair in self.chemistry_elos:
            self.chemistry_elos[pair] *= self.offseason_retention
        # Offseason duration is not meaningful game rest, so opening-day rest
        # contrasts begin neutral for every team.
        self.last_played.clear()
        self.recent_games.clear()

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
        home_player, home_chemistry = self._player_context(home)
        away_player, away_chemistry = self._player_context(away)
        elo_per_logit = 400.0 / log(10.0)
        player_coefficient = self.mean[self.team_count + 3]
        chemistry_coefficient = self.mean[self.team_count + 4]
        home_team_elo = 1500.0 + elo_per_logit * team_mean[self.index[home]]
        away_team_elo = 1500.0 + elo_per_logit * team_mean[self.index[away]]
        # Transforming Gaussian margin quantiles gives an epistemic interval for
        # the underlying win propensity. It is not a frequentist game-outcome CI.
        return {
            "home_win_probability": _sigmoid(adjusted_margin),
            "latent_margin_mean": margin_mean,
            "latent_margin_sd": margin_sd,
            "james_stein_factor": james_stein_factor,
            "matchup_logit": self._matchup_feature(home, away),
            "home_player_elo": home_player,
            "away_player_elo": away_player,
            "home_chemistry_elo": home_chemistry,
            "away_chemistry_elo": away_chemistry,
            "home_team_elo": home_team_elo,
            "away_team_elo": away_team_elo,
            "home_overall_elo": home_team_elo
            + player_coefficient * self.player_scale * (home_player - 1500.0)
            + chemistry_coefficient * self.chemistry_scale * home_chemistry,
            "away_overall_elo": away_team_elo
            + player_coefficient * self.player_scale * (away_player - 1500.0)
            + chemistry_coefficient * self.chemistry_scale * away_chemistry,
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

    def _update_players(self, home: str, away: str, residual: float, game: dict) -> None:
        """Update individual Elo, raw effectiveness, and pair chemistry postgame."""

        if not all(f"{side}_skaters" in game for side in ("home", "away")):
            return
        for side, team, direction in (("home", home, 1.0), ("away", away, -1.0)):
            skaters = game[f"{side}_skaters"]
            if not skaters:
                continue
            total_toi = sum(player[1] for player in skaters)
            if total_toi <= 0:
                continue
            weights = tuple((player[0], player[1] / total_toi) for player in skaters)
            raw_scores = np.array(
                [player[2] + 0.7 * player[3] + 0.08 * player[5] + 0.15 * player[4] for player in skaters]
            )
            expected_scores = raw_scores.sum() * np.array([weight for _, weight in weights])
            for (player, weight), raw_residual in zip(
                weights, raw_scores - expected_scores, strict=True
            ):
                self.player_elos[player] += (
                    self.player_k * direction * residual * len(weights) * weight
                    + self.raw_effect_scale * float(np.clip(raw_residual, -2.0, 2.0))
                )

            goalie = int(game.get(f"{side}_starter_goalie_id") or 0) or None
            if goalie is not None:
                shots = float(game.get(f"{side}_starter_shots_against", 0))
                goals = float(game.get(f"{side}_starter_goals_against", 0))
                saves_above_average = 0.095 * shots - goals
                self.player_elos[goalie] += (
                    1.5 * self.player_k * direction * residual
                    + self.raw_effect_scale * float(np.clip(saves_above_average, -3.0, 3.0))
                )

            players = [player for player, _ in weights] + ([goalie] if goalie else [])
            if self.chemistry_k:
                for i, first in enumerate(players):
                    for second in players[i + 1 :]:
                        self.chemistry_elos[tuple(sorted((first, second)))] += (
                            self.chemistry_k * direction * residual
                        )
            self.lineups[team] = (weights, goalie)

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
        self.recent_games[home].append(played_on)
        self.recent_games[away].append(played_on)
        first, second = sorted((home, away))
        history = self.matchups.setdefault((first, second), [0.0, 0.0])
        first_residual = residual if home == first else -residual
        history[0] = history[0] * self.matchup_decay + first_residual
        history[1] = history[1] * self.matchup_decay + 1.0
        if game is not None:
            self._update_players(home, away, residual, game)
            self._update_form(home, away, game)


class BayesianModelAverage:
    """Validation-weighted average of two Bayesian PL specifications."""

    def __init__(
        self,
        teams: Iterable[str],
        *,
        probability_temperature: float = 1.10,
        **model_options: float,
    ) -> None:
        if probability_temperature <= 0.0:
            raise ValueError("probability_temperature must be positive")
        teams = tuple(teams)
        specifications = (
            {
                "form_features": ("xg_share", "shot_share", "sat_share"),
                "form_decay": 0.94,
            },
            {
                "form_decay": 0.80,
                "james_stein_scale": 0.22,
                "overtime_weight": 0.40,
                "overtime_credit": 0.90,
                "margin_scale": 0.35,
            },
        )
        self.names = ("possession", "fast_james_stein")
        self.members = [
            BayesianPlackettLuce(teams, **(specification | model_options))
            for specification in specifications
        ]
        self.weights = np.array([0.55, 0.45])
        self.probability_temperature = probability_temperature

    @property
    def spectral_diagnostics(self) -> SpectralDiagnostics | None:
        """Expose the first member's identically scheduled spectral audit."""

        return self.members[0].spectral_diagnostics

    def start_new_season(self) -> None:
        """Apply the same offseason transition to every candidate model."""

        for member in self.members:
            member.start_new_season()

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
        uncalibrated_probability = float(weights @ probabilities)
        result = {
            "home_win_probability": _sigmoid(
                self.probability_temperature * _logit(uncalibrated_probability)
            ),
            "uncalibrated_probability": uncalibrated_probability,
            "probability_temperature": self.probability_temperature,
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
        for key in (
            "matchup_logit",
            "home_player_elo", "away_player_elo",
            "home_chemistry_elo", "away_chemistry_elo",
            "home_team_elo", "away_team_elo",
            "home_overall_elo", "away_overall_elo",
        ):
            result[key] = float(weights @ np.array([details[key] for details in member_details]))
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
        """Update both member posteriors after the held-out result."""

        for member in self.members:
            member.update(home, away, played_on, home_win, game)


def backtest(
    games: list[dict],
    *,
    evaluation_start_season: int | None = None,
    model_average: bool = True,
    **model_options: float,
) -> tuple[list[dict], dict[str, float]]:
    """Generate one strictly out-of-sample prediction per chronological game."""

    if not games:
        raise ValueError("games cannot be empty")
    teams = {_franchise(row[side]) for row in games for side in ("home_team", "away_team")}
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
        home, away = _franchise(row["home_team"]), _franchise(row["away_team"])
        details = model.predict_details(home, away, played_on)
        probability = details["home_win_probability"]
        outcome = int(row["home_score"] > row["away_score"])
        home_rate_probability = home_wins / games_seen
        elo_probability = 1.0 / (
            1.0
            + 10.0
            ** ((elo[away] - elo[home] - 35.0) / 400.0)
        )
        if evaluation_start_season is None or current_season >= evaluation_start_season:
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
        model.update(home, away, played_on, bool(outcome), row)
        home_wins += outcome
        games_seen += 1.0
        elo_change = 20.0 * (outcome - elo_probability)
        elo[home] += elo_change
        elo[away] -= elo_change

    if not predictions:
        raise ValueError("evaluation start season excludes every game")
    probabilities = np.array([row["home_win_probability"] for row in predictions])
    outcomes = np.array([row["home_win"] for row in predictions])
    metrics = {
        "games": float(len(predictions)),
        "accuracy": correct / len(predictions),
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
        metrics["final_player_elos"] = float(len(model.members[0].player_elos))
        metrics["final_chemistry_pairs"] = float(len(model.members[0].chemistry_elos))
        metrics["final_probability_temperature"] = model.probability_temperature
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
    teams = {_franchise(row[side]) for row in games for side in ("home_team", "away_team")}
    model = (
        BayesianModelAverage(teams | {_franchise(team) for team in extra_teams}, **model_options)
        if model_average
        else BayesianPlackettLuce(teams | {_franchise(team) for team in extra_teams}, **model_options)
    )
    current_season: int | None = None
    for row in sorted(games, key=lambda game: (game["start_time_utc"], game["game_id"])):
        if current_season is not None and row.get("season") != current_season:
            model.start_new_season()
        current_season = row.get("season", current_season)
        played_on = date.fromisoformat(row["game_date"])
        model.update(
            _franchise(row["home_team"]),
            _franchise(row["away_team"]),
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
        _franchise(row[side]) for row in schedule for side in ("home_team", "away_team")
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
            _franchise(row["home_team"]),
            _franchise(row["away_team"]),
            date.fromisoformat(row["game_date"]),
        )
        predictions.append({**row, **details})
    return predictions
