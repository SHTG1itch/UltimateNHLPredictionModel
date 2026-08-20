"""Random-matrix denoising for finite-sample covariance estimates.

Marchenko-Pastur describes eigenvalues of a *sample covariance* under an
independent-noise null. Wigner's 1955 semicircle law concerns a different
object: symmetric random matrices. They are related historically through
random-matrix theory, but combining their cutoffs would be mathematically
incorrect. This module therefore uses the Marchenko-Pastur upper edge for
covariance cleaning and exposes the Wigner radius only as a noise diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SpectralDiagnostics:
    """Values needed to audit what the spectral cleaner changed."""

    samples: int
    features: int
    mp_upper_edge: float
    noise_eigenvalues: int
    signal_eigenvalues: int
    wigner_radius: float


def clean_covariance(
    observations: ArrayLike,
    *,
    floor: float = 1e-8,
) -> tuple[NDArray[np.float64], SpectralDiagnostics]:
    """Return a positive-definite covariance with MP noise modes averaged.

    Rows are observations and columns are variables. The function cleans the
    correlation spectrum so units cannot determine which modes survive, then
    restores the original variances. Replacing bulk eigenvalues by their mean
    preserves the trace while suppressing sampling noise in their directions.
    """

    values = np.asarray(observations, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("observations must be a 2-D array with at least 2 rows and columns")
    if not np.isfinite(values).all():
        raise ValueError("observations must contain only finite values")

    samples, features = values.shape
    centered = values - values.mean(axis=0)
    # Constant columns have no empirical scale; floor keeps the result invertible.
    scales = np.maximum(centered.std(axis=0, ddof=1), np.sqrt(floor))
    standardized = centered / scales
    correlation = standardized.T @ standardized / (samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)

    # For X with shape samples x features and unit noise variance, the null
    # covariance spectrum ends at (1 + sqrt(features / samples))**2.
    mp_upper = float((1.0 + np.sqrt(features / samples)) ** 2)
    noise = eigenvalues <= mp_upper
    cleaned_values = np.maximum(eigenvalues, floor)
    if noise.any():
        cleaned_values[noise] = max(float(eigenvalues[noise].mean()), floor)

    cleaned_correlation = (eigenvectors * cleaned_values) @ eigenvectors.T
    # Eigenvalue averaging can move the diagonal slightly; correlation must
    # have unit diagonal before original measurement scales are restored.
    diagonal = np.sqrt(np.maximum(np.diag(cleaned_correlation), floor))
    cleaned_correlation /= np.outer(diagonal, diagonal)
    covariance = cleaned_correlation * np.outer(scales, scales)
    covariance = (covariance + covariance.T) / 2
    covariance += np.eye(features) * floor

    # Wigner's radius is a diagnostic for the centered symmetric off-diagonal
    # noise, not the covariance cutoff used above.
    off_diagonal = correlation[np.triu_indices(features, 1)]
    sigma = float(off_diagonal.std(ddof=1)) if off_diagonal.size > 1 else 0.0
    diagnostics = SpectralDiagnostics(
        samples=samples,
        features=features,
        mp_upper_edge=mp_upper,
        noise_eigenvalues=int(noise.sum()),
        signal_eigenvalues=int((~noise).sum()),
        wigner_radius=2.0 * sigma * np.sqrt(features),
    )
    return covariance, diagnostics

