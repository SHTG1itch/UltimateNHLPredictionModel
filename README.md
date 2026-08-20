# Ultimate NHL Prediction Model

A reproducible Python model for pre-game NHL win probabilities. It combines a
sequential Bayesian Plackett-Luce likelihood, strict walk-forward evaluation,
and random-matrix covariance cleaning. "Most accurate" is treated as a measured
benchmark outcome, not something guaranteed by naming sophisticated methods.

## Statistical model

For home team \(h\) and away team \(a\), the two-item Plackett-Luce likelihood
(the Bradley-Terry special case) is

\[
P(h \succ a) = \sigma(s_h - s_a + \boldsymbol{\beta}^{T}\mathbf{x}_{ha}).
\]

The context vector contains home ice, differential rest, and rolling contrasts
for expected-goal share, shot share, special teams, total-shot-attempt share
(SAT/Corsi), finishing above xG, and goals saved above xG. Every rolling rate
uses conservative league-average pseudo-counts and exponentially decayed
sufficient statistics. With decay 0.96, a game's weight halves after about 17
subsequent team games. Current-game statistics never enter its own prediction.

The latent team strengths and context coefficients have a multivariate Gaussian
posterior. If \(\mathbf{m}\) and \(\mathbf{C}\) are the prior mean and covariance,
the local logistic curvature is \(w=p(1-p)\), and one online Laplace step uses

\[
\mathbf{C}' = \mathbf{C} -
\frac{w(\mathbf{C}\mathbf{x})(\mathbf{C}\mathbf{x})^T}
{1 + w\mathbf{x}^T\mathbf{C}\mathbf{x}}, \qquad
\mathbf{m}' = \mathbf{m} + \mathbf{C}'\mathbf{x}(y-p).
\]

This rank-one Sherman-Morrison form avoids a matrix inverse per game. Posterior
predictions integrate uncertainty with MacKay's logistic-Gaussian approximation
\(\sigma(\mu/\sqrt{1+\pi\sigma^2/8})\). Team ratings partially pool 60% toward
their previous value between seasons, while covariance increases to represent
roster turnover.

Every ledger row also reports the latent log-odds margin mean and standard
deviation. The 95% epistemic bounds are
\(\sigma(\mu-1.96\sigma_m)\) and \(\sigma(\mu+1.96\sigma_m)\). These describe
parameter uncertainty in the underlying win propensity, not a confidence
interval for the binary outcome of one inherently random game.

Overtime and shootout results use a 0.70 power likelihood: both still count as
wins for the moneyline target, but their score curvature and residual update
are multiplied by 0.70 because they are near-ties. The weight was selected on
the validation seasons and reduces posterior overreaction.

## Walk-forward protocol

Evaluation follows the information order of a live prediction:

1. Sort games by official UTC start time.
2. Predict game \(t\) using only results and boxscores before \(t\).
3. Record the probability, baselines, and proper scoring rules.
4. Only then update the posterior and rolling form with game \(t\).

This prevents future leakage. Same-day games use scheduled UTC ordering. Binary
log loss is the primary selection metric because it rewards calibration and
penalizes unjustified certainty; Brier score and winner accuracy are also
reported. The prediction ledger makes every forecast auditable.

## Marchenko-Pastur and Wigner

Weekly team innovations form an observations-by-teams matrix. Once there are at
least two observations per team, its correlation eigenvalues are compared with
the Marchenko-Pastur null ceiling

\[
\lambda_+ = (1 + \sqrt{p/n})^2,
\]

where \(p\) is the number of teams and \(n\) the weekly samples. Eigenvalues in
the noise bulk are replaced by their mean, preserving trace, while signal modes
above the ceiling survive. The positive-definite cleaned covariance becomes the
correlated diffusion structure for changing team strengths.

Wigner's 1955 semicircle law came from nuclear-physics random matrices, but it
describes symmetric random matrices, not sample covariance matrices. The code
reports a Wigner radius as a spectral diagnostic and deliberately does not mix
it into the Marchenko-Pastur covariance cutoff. Marchenko and Pastur's sample
covariance result was published in 1967, not 1955.

On the current five-season run, the final 104-week window classifies all 33
team modes as noise. That is a valid empirical result: the cleaner reduces the
diffusion toward isotropic noise instead of inventing unsupported correlations.
If a later window produces eigenvalues above \(\lambda_+\), those modes are
retained automatically.

## Data pipeline

The downloader gathers completed regular-season and playoff games from the NHL
web schedule API. It then fetches league-wide per-game `summary`, `powerplay`,
`penaltykill`, `realtime`, and goalie reports from the NHL Stats API. Reports are
joined by official game ID and home/road orientation; incomplete joins fail.

The compact CSV contains scores, shots, shot attempts, faceoffs, power-play
goals/opportunities, penalty-kill goals allowed/times shorthanded, and MoneyPuck
expected goals. Listed shot ZIPs are aggregated in memory and discarded. Only
two xG totals and a coverage flag survive per game. See
[data/README.md](data/README.md) for attribution, terms, schema, and storage.
Actual starter IDs and workload are retained strictly as post-game history.

## Quick start

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

nhl-predict download --seasons 20212022 20222023 20232024 20242025 20252026
nhl-predict backtest
nhl-predict predict --date 2026-10-10
python -m pytest
```

Downloaded data is ignored by Git. The downloader stores one compact CSV rather
than raw API responses, and generated ledgers go under ignored `artifacts/`.
The complete five-season development workspace is measured during validation
and remains several orders of magnitude below the hard 10 GB ceiling.

### Forecast an upcoming date

`nhl-predict predict` fits the posterior to every completed row in
`data/games.csv`, downloads one exact NHL schedule date, prints home-win
probabilities, latent margin uncertainty, and 95% epistemic bounds, then writes
`artifacts/forecast.csv`. It rejects any schedule at or before the latest
training result, preventing an accidental historical forecast from using future
games. An unpublished or game-free date returns an empty prediction list
without creating a misleading artifact.

The command predicts regular-season and playoff games only. New teams present
in a future schedule receive the same zero-centered prior as every other unseen
team. Forecasts currently use information through the latest completed game;
they do not pretend that unrecorded starting-goalie or lineup news is known.

## Verified benchmark

`nhl-predict backtest` writes model and baseline probabilities to
`artifacts/predictions.csv`. A verified run over 6,993 games from 2021-22
through 2025-26 produced:

```json
{
  "accuracy": 0.5945945945945946,
  "brier_score": 0.2359000529347083,
  "elo_log_loss": 0.6754472619256587,
  "games": 6993.0,
  "home_rate_log_loss": 0.6908971193256509,
  "log_loss": 0.6642990021795636
}
```

Season-level results expose drift hidden by the aggregate:

| Season role | Games | Model log loss | Elo log loss | Brier | Accuracy |
|---|---:|---:|---:|---:|---:|
| 2021-22 warm-up | 1,401 | 0.66125 | 0.65977 | 0.23381 | 61.81% |
| 2022-23 validation | 1,400 | 0.66196 | 0.66988 | 0.23472 | 60.64% |
| 2023-24 validation | 1,400 | 0.65393 | 0.67198 | 0.23114 | 60.64% |
| 2024-25 parameter holdout | 1,398 | 0.66106 | 0.67660 | 0.23466 | 58.87% |
| 2025-26 prospective check | 1,394 | 0.68338 | 0.69912 | 0.24521 | 55.31% |

The defaults minimize mean log loss across the 2022-23 and 2023-24 validation
seasons. The explored grid covered team prior variance 0.5-3.0, context variance
0.1-3.0, daily diffusion \(10^{-6}\)-\(10^{-3}\), form decay 0.85-0.985,
offseason retention 0.5-1.0, and offseason variance 0.01-0.4. Those values were
frozen before the later-season checks under the selected defaults.

Shot-quality feature selection used the same validation seasons. Eight fixed
ablations compared the previous score/shot percentages with xG share and xG
residuals. The retained six-feature set improved validation mean log loss from
0.658493 to 0.658196. It also improves the five-season aggregate modestly; raw
goal share and actual shooting/save percentages were removed as redundant,
noisier versions of the xG decomposition.

An overtime power-likelihood grid (0.40, 0.55, 0.70, 0.85, 1.00) selected 0.70,
improving validation mean log loss from 0.658196 to 0.657943. The later 2024-25
check also improved; 2025-26 was essentially flat and slightly worse.

The model beats Elo on probability quality in every post-warm-up season, but
2024-25 winner accuracy is slightly lower than Elo. Log loss remains the model
selection objective because changing a classification threshold to chase one
season's hit rate would not improve the probability forecast.

## Reproducibility and limitations

- Schedules/results come from `api-web.nhle.com`; per-game reports come from
  `api.nhle.com/stats/rest`. The public schemas are not versioned, so validation
  fails instead of silently imputing missing values.
- Hyperparameters are fixed defaults in `BayesianPlackettLuce`; the benchmark
  command requires no hidden notebook or private configuration.
- Current rolling features include xG, shots, SAT, rest, and special teams.
  A time-zone travel feature was researched but rejected: it worsened validation
  mean log loss from 0.658196 to 0.658442, and a strongly shrunk travel prior did
  not beat exclusion. A future candidate needs actual distance and circadian
  direction. A prior-start goalie mixture was also rejected: validation log
  loss worsened from 0.657943 to 0.658201, while a strongly shrunk goalie prior
  only approached exclusion (0.657946). Actual starter history is collected,
  but confirmed pregame starters, injuries, lineups, and market odds require
  timestamped historical snapshots;
  final rosters or later revisions leak.
- "Most accurate" remains a benchmark target. Every model change must beat
  fixed baselines on later seasons rather than only fitting historical games.

## Research basis

- [Guiver & Snelson (2009), *Bayesian inference for Plackett-Luce ranking
  models*](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/plackett.pdf):
  Bayesian regularization and uncertainty for sparse rankings.
- [Plackett (1975), *The Analysis of
  Permutations*](https://doi.org/10.2307/2346567): the ranking likelihood.
- [Marchenko & Pastur (1967), *Distribution of eigenvalues for some sets of
  random matrices*](https://doi.org/10.1070/SM1967v001n04ABEH001994): the
  sample-covariance noise spectrum.
- [Wigner (1955), *Characteristic vectors of bordered matrices with infinite
  dimensions*](https://doi.org/10.2307/1970079): the nuclear-physics
  random-matrix lineage and semicircle law.
- [Macdonald (2012), *Evaluating NHL Goalies, Skaters, and Teams Using Weighted
  Shots*](https://arxiv.org/abs/1205.1746): shot quality, goaltending, and team
  evaluation motivate the rolling shot/finishing/save decomposition.
- [MoneyPuck Download Data](https://moneypuck.com/data.htm): published shot
  probabilities used under its attributed non-commercial data terms.
- [Charest et al. (2022), *Associations between time zone changes, travel
  distance and performance*](https://doi.org/10.1016/j.jsams.2022.10.005):
  motivates the audited travel candidate and its acclimation assumption.
- Forward validation preserves temporal order; this implementation enforces it
  by writing each forecast before updating any posterior or form statistic.
