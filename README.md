# Ultimate NHL Prediction Model

A reproducible Python baseline for pre-game NHL win probabilities. The model is
Bayesian, evaluates every forecast with walk-forward chronology, and uses
random-matrix theory to suppress finite-sample covariance noise. It is an
honest baseline, not a claim that a named method is automatically “the most
accurate”: accuracy is established only by comparisons on unseen future games.

## What is implemented

For home team \(h\) and away team \(a\), the two-item Plackett-Luce likelihood
is

\[
P(h \succ a) = \sigma(s_h - s_a + \beta_{home} + \beta_{rest}\Delta rest).
\]

The latent team strengths and context coefficients have a Gaussian posterior.
After each result, a rank-one Laplace update changes its mean and covariance.
Predictions integrate posterior uncertainty with the logistic-Gaussian
approximation rather than treating ratings as known constants. Between dates,
team uncertainty diffuses, allowing strength to move over time.

The evaluation is strict expanding-window walk-forward prediction:

1. Sort games by official UTC start time.
2. Predict game \(t\) using only results before \(t\).
3. Record the probability and proper scoring rules.
4. Only then update the posterior with game \(t\).

This order prevents future leakage. Accuracy, binary log loss, and Brier score
are reported; log loss is the primary model-selection metric because it rewards
well-calibrated probabilities and strongly penalizes unjustified certainty.

## Marchenko-Pastur and Wigner

Weekly team innovations form an observations-by-teams matrix. Once there are at
least two observations per team, its correlation eigenvalues are compared with
the Marchenko-Pastur noise ceiling

\[
\lambda_+ = (1 + \sqrt{p/n})^2,
\]

where \(p\) is the number of teams and \(n\) the weekly samples. Eigenvalues in
the noise bulk are replaced by their mean, preserving trace, while signal modes
above the ceiling survive. The cleaned positive-definite covariance becomes the
correlated diffusion structure for changing team strengths.

Wigner's 1955 semicircle law came from nuclear-physics random matrices, but it
describes symmetric random matrices—not sample covariance matrices. The code
reports a Wigner radius as a spectral diagnostic and deliberately does **not**
mix it into the Marchenko-Pastur covariance cutoff. Marchenko and Pastur's sample
covariance result was published in 1967, not 1955.

## Quick start

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

nhl-predict download --seasons 20212022 20222023 20232024 20242025
nhl-predict backtest
pytest
```

Downloaded data is ignored by Git. Four modern seasons are only a few hundred
kilobytes, and the downloader stores one compact CSV rather than large API
responses. Generated prediction ledgers go under ignored `artifacts/`, keeping
the repository far below 10 GB.

## Output and interpretation

`nhl-predict backtest` writes every model and baseline probability to
`artifacts/predictions.csv`. A verified run over 5,599 games from 2021–22
through 2024–25 produced:

```json
{
  "accuracy": 0.601893195213431,
  "brier_score": 0.23659122654549175,
  "elo_log_loss": 0.6695535207204005,
  "games": 5599.0,
  "home_rate_log_loss": 0.6903190771508779,
  "log_loss": 0.6662372463436382
}
```

This expanding-window development run beats the included home-rate and fixed
Elo baselines on log loss, but it is not a final unbiased performance estimate:
hyperparameters were not selected on a separate validation period. A final
claim requires freezing the model, then scoring a later untouched season.

## Reproducibility and limitations

- The source is the public NHL web API. The endpoint is public but its schema is
  not versioned; validation fails loudly when required CSV fields disappear.
- Same-day games are ordered by scheduled UTC time. Each result is learned only
  after that game's prediction.
- The current features are team strength, home ice, and capped rest advantage.
  Starting goalies, injuries, travel, shots/xG, special teams, and market odds
  are not yet present. They should be added only with timestamped, historically
  available data so the backtest cannot see information unavailable pre-game.
- “Most accurate” is a benchmark outcome, not an architectural property. Model
  changes belong behind later-season comparisons against fixed baselines.

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
- Forward validation preserves temporal order; the implementation makes that
  constraint explicit by logging each forecast before its posterior update.
