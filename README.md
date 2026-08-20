# Ultimate NHL Prediction Model

A reproducible Python model for pre-game NHL win probabilities. It combines
sequential Bayesian Plackett-Luce likelihoods with validation-weighted model
averaging, positive-part James-Stein shrinkage, strict walk-forward evaluation,
and random-matrix covariance cleaning. "Most accurate" is treated as a measured
benchmark outcome, not something guaranteed by naming sophisticated methods.

## Statistical model

For home team \(h\) and away team \(a\), the two-item Plackett-Luce likelihood
(the Bradley-Terry special case) is

\[
P(h \succ a) = \sigma(s_h - s_a + \boldsymbol{\beta}^{T}\mathbf{x}_{ha}).
\]

The context vector contains home ice, schedule fatigue, and rolling contrasts
for expected-goal share, shot share, special teams, total-shot-attempt share
(SAT/Corsi), finishing above xG, and goals saved above xG. Every rolling rate
uses conservative league-average pseudo-counts and exponentially decayed
sufficient statistics. Current-game statistics never enter its own prediction.

Schedule fatigue is computed entirely from prior game dates. If \(B_i\) marks a
back-to-back and \(N_{i,5}\) is team \(i\)'s number of prior games in the last
five days, its scalar cost is

\[
F_i=B_i+0.25\max(N_{i,5}-1,0),
\]

and the model receives \(F_a-F_h\). Validation rejected treating every extra
off-day as distinct; the back-to-back plus congestion representation was both
simpler and more accurate.

Arizona and Utah share one internal franchise key so Utah inherits the prior
hockey evidence instead of starting from zero. This matches the NHL's transfer
of the Coyotes' full roster, reserve list, draft picks, and hockey operations to
the Utah franchise in 2024 ([NHL Public Relations](https://www.nhl.com/utah/news/nhl-bog-approves-establishment-of-new-franchise-in-utah-x8485)).

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
roster turnover. The initial team prior variance is 0.20 and daily diffusion is
\(10^{-7}\), both selected conservatively to avoid early-season overconfidence
and unsupported day-to-day drift.

### James-Stein shrinkage and model averaging

The CLI averages two complementary specifications rather than assuming that one
feature horizon is certainly correct:

1. `possession` uses xG share, shot share, and SAT share with 0.94 form decay,
   a half-life of about 11 team games.
2. `fast_james_stein` uses the full six-feature context with 0.80 form decay,
   a half-life of about three team games, tempered James-Stein team effects, and
   conservative score-margin evidence.

For the second member, let \(p\) be the number of teams, \(\mathbf{s}\) their
posterior means, and
\(\bar v=\operatorname{tr}(\mathbf{C}_{s})/p\) their pooled posterior variance.
The positive-part estimator used at prediction time is

\[
\kappa = \max\!\left(0, 1-
\rho\frac{(p-2)\bar v}{\sum_i(s_i-\bar s)^2}\right), \qquad
\tilde s_i=\bar s+\kappa(s_i-\bar s),
\]

with \(\rho=0.22\). The theorem's \(p\ge3\) boundary is enforced, and
\(\rho=0\) is the exact no-shrinkage ablation. Shrinkage changes forecasts but
does not destructively overwrite the learned posterior. The classical
\(\rho=1\) coefficient was far too strong for correlated, evolving NHL ratings;
it remains an audited rejection rather than being retained for its name.

Let \(p_{k,t}\) be member \(k\)'s pre-game probability. The reported forecast is

\[
\hat p_t=0.55p_{\mathrm{possession},t}
+0.45p_{\mathrm{fast},t}.
\]

The fixed validation blend beat discounted per-game weight updates on both the
declared validation seasons and the later-season mean. Each ledger row records
both member probabilities and the fixed weights.

Every ledger row also reports the latent log-odds margin mean and standard
deviation. The 95% epistemic bounds are
\(\sigma(\mu-1.96\sigma_m)\) and \(\sigma(\mu+1.96\sigma_m)\). These describe
parameter uncertainty in the underlying win propensity, not a confidence
interval for the binary outcome of one inherently random game.

Overtime and shootout results are near-ties: the possession member uses a 0.70
power likelihood and the fast member uses 0.40. In the fast James-Stein member,
the rating target is also softened to 0.90 for the winner and 0.10 for the
loser, a conservative Davidson-inspired tie treatment; the moneyline evaluation
target remains the actual winner. That member's regulation evidence is
multiplied by

\[
1 + 0.35\log(\max(|G_h-G_a|,1)),
\]

so a blowout carries somewhat more rating information without growing linearly
with score margin. The possession member omits this term, allowing model
averaging to protect forecasts when margin is misleading.

## Walk-forward protocol

The 2019-20 and 2020-21 seasons are unscored warm-up history. They initialize
the sequential state before the fixed five-season evaluation window; they are
never included in its metrics. Evaluation then follows the information order
of a live prediction:

1. Sort games by official UTC start time.
2. Predict game \(t\) using only results and boxscores before \(t\).
3. Record the mixture probability, member probabilities and weights, baselines,
   and proper scoring rules.
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

On the current seven-season training run, the final 104-week window classifies
31 of 32 team modes as noise and retains one signal mode above
\(\lambda_+=2.4171\).
The cleaner therefore suppresses the bulk without discarding the supported
cross-team direction; later windows are classified again from their own data.

The CLI reports `mp_upper_edge`, `mp_noise_eigenvalues`,
`mp_signal_eigenvalues`, and `wigner_radius` from the final window, making this
decision auditable for every new data refresh.

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

nhl-predict download --seasons 20192020 20202021 20212022 20222023 20232024 20242025 20252026
nhl-predict backtest --evaluate-from 20212022
nhl-predict predict --date 2026-10-10
python -m pytest
```

Downloaded data is ignored by Git. The downloader stores one compact CSV rather
than raw API responses, and generated ledgers go under ignored `artifacts/`.
The complete seven-season development workspace is measured during validation
and remains several orders of magnitude below the hard 10 GB ceiling.
The download command prints the exact CSV byte size and SHA-256 digest so a
benchmark can be tied to the data bytes that produced it.

### Forecast an upcoming date

`nhl-predict predict` fits both posteriors to every
completed row in `data/games.csv`, downloads one exact NHL schedule date, prints home-win
probabilities, member weights, latent margin uncertainty, and 95% epistemic
bounds, then writes `artifacts/forecast.csv`. It rejects any schedule at or
before the latest training result, preventing an accidental historical forecast
from using future games. An unpublished or game-free date returns an empty
prediction list without creating a misleading artifact.

The command predicts regular-season and playoff games only. New teams present
in a future schedule receive the same zero-centered prior as every other unseen
team. Forecasts currently use information through the latest completed game;
they do not pretend that unrecorded starting-goalie or lineup news is known.

## Verified benchmark

`nhl-predict backtest --evaluate-from 20212022` writes model and baseline
probabilities to `artifacts/predictions.csv`. A verified run trained on 9,157
games from 2019-20 through 2025-26 and scored the final 6,993 games. It used a
1,900,396-byte `data/games.csv` with SHA-256
`a5150384d8c3d60f3bb71cc4dd5fd84119b288608093dd1e167fa8d7cc430da4` and
produced:

```json
{
  "accuracy": 0.6021736021736022,
  "brier_score": 0.23322995648123016,
  "elo_log_loss": 0.6748934351602023,
  "final_fast_james_stein_weight": 0.45,
  "final_possession_weight": 0.55,
  "games": 6993.0,
  "home_rate_log_loss": 0.6906025254572833,
  "log_loss": 0.6583178286894047,
  "mp_noise_eigenvalues": 31.0,
  "mp_signal_eigenvalues": 1.0
}
```

Season-level results expose drift hidden by the aggregate:

| Season role | Games | Model log loss | Elo log loss | Brier | Accuracy |
|---|---:|---:|---:|---:|---:|
| 2021-22 evaluation | 1,401 | 0.64381 | 0.65655 | 0.22611 | 63.45% |
| 2022-23 validation | 1,400 | 0.65736 | 0.67030 | 0.23273 | 61.00% |
| 2023-24 validation | 1,400 | 0.65359 | 0.67205 | 0.23098 | 61.43% |
| 2024-25 parameter holdout | 1,398 | 0.65658 | 0.67657 | 0.23255 | 60.44% |
| 2025-26 prospective check | 1,394 | 0.68035 | 0.69912 | 0.24382 | 54.73% |

Every season's log loss improves over the preceding model-averaged benchmark.
Across all scored games, log loss falls from 0.663269 to 0.658318, Brier score
falls from 0.235414 to 0.233230, and accuracy rises from 59.70% to 60.22%.
The historical warm-up and retuning add four correct winners over the preceding
0.659890 benchmark while improving every season's log loss.

The defaults improve mean log loss across the 2022-23 and 2023-24 validation
seasons while requiring later-season robustness and no aggregate accuracy loss.
The explored grid covered team prior variance 0.05-3.0, context
variance 0.1-3.0, daily diffusion \(10^{-8}\)-\(10^{-3}\), form decay
0.30-0.985, offseason retention 0.3-1.0, and offseason variance 0.005-0.4.
Later seasons were retained as explicit robustness checks rather than folded
into one randomly shuffled training score.

Shot-quality feature selection used the same validation seasons. Eight fixed
ablations compared the previous score/shot percentages with xG share and xG
residuals. The retained six-feature set improved validation mean log loss from
0.658493 to 0.658196. It also improves the five-season aggregate modestly; raw
goal share and actual shooting/save percentages were removed as redundant,
noisier versions of the xG decomposition.

An overtime power-likelihood grid (0.40, 0.55, 0.70, 0.85, 1.00) selected 0.70,
improving validation mean log loss from 0.658196 to 0.657943. The later 2024-25
check also improved; 2025-26 was essentially flat and slightly worse.

The advanced-model experiments used the 2022-23 and 2023-24 mean as the primary
selection score and reported the next two seasons separately:

| Candidate | Validation mean log loss | Later-season mean | All games | Decision |
|---|---:|---:|---:|---|
| Previous model average | 0.657417 | 0.671099 | 0.663269 | Reference |
| Revised horizons and shrinkage | 0.657275 | 0.670532 | 0.662982 | Retain |
| Stronger prior + lower drift | 0.657031 | 0.670378 | 0.661729 | Retain |
| Back-to-back status | 0.656207 | 0.669811 | 0.660636 | Retain |
| Five-day congestion | 0.655876 | 0.668870 | 0.660110 | Retain |
| Previous ensemble retuning | 0.655805 | 0.668858 | 0.660047 | Retain |
| Arizona/Utah continuity | 0.655792 | 0.668756 | 0.660000 | Retain |
| Fixed 54/46 robust blend | 0.655770 | 0.668753 | 0.659962 | Retain |
| Margin/shrinkage retuning | 0.655722 | 0.668655 | 0.659890 | Retain |
| Two-season warm-up + 55/45 blend | **0.655477** | **0.668463** | **0.658318** | Retain |

Fixed temperature/intercept scaling, a separate online Bayesian logit
calibrator, faceoff and starter-save form, separate fatigue coefficients,
seasonal home-ice resets, dynamic model weights, schedule-sequence corrections,
and extra model-average members were rejected because they failed to improve
the full validation and later-season gate. The simplest validated model remains
the shipped model.

The model beats Elo on probability quality and winner accuracy in every season.
Log loss remains the primary selection objective because changing a
classification threshold to chase one season's hit rate would not improve the
probability forecast.

## Reproducibility and limitations

- Schedules/results come from `api-web.nhle.com`; per-game reports come from
  `api.nhle.com/stats/rest`. The public schemas are not versioned, so validation
  fails instead of silently imputing missing values.
- Hyperparameters are fixed defaults in `BayesianPlackettLuce` and
  `BayesianModelAverage`; the benchmark command requires no hidden notebook or
  private configuration.
- Current rolling features include xG, shots, SAT, schedule fatigue, and special
  teams.
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
- [James & Stein (1961), *Estimation with quadratic
  loss*](https://digicoll.lib.berkeley.edu/record/112898): the multivariate
  normal shrinkage result and its dimension-three boundary.
- [Davidson (1970), *On Extending the Bradley-Terry Model to Accommodate Ties
  in Paired Comparison Experiments*](https://doi.org/10.1080/01621459.1970.10481082):
  motivates treating overtime results as near-ties for rating updates.
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
