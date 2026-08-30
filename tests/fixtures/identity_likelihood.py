"""Literal well-conditioned fixture for identity-XD likelihood tests.

The values are intentionally written out rather than generated from a random
number stream.  This keeps the Phase 1 likelihood and responsibility evidence
stable across NumPy and JAX releases.
"""

from __future__ import annotations

import numpy as np


FIXTURE_ID = "xd-ip-likelihood-001"
FIXTURE_VERSION = 1


OBSERVATIONS = (
    (-1.80, 0.10),
    (-1.25, 0.65),
    (-0.85, -0.45),
    (-0.35, 0.95),
    (0.05, -0.80),
    (0.45, -0.10),
    (0.80, 0.55),
    (1.10, 1.35),
    (1.55, 0.35),
    (1.95, 1.05),
    (0.25, 1.70),
)


MEASUREMENT_COVARIANCES = (
    ((0.08, 0.01), (0.01, 0.12)),
    ((0.18, -0.03), (-0.03, 0.10)),
    ((0.11, 0.02), (0.02, 0.20)),
    ((0.24, 0.05), (0.05, 0.16)),
    ((0.09, -0.01), (-0.01, 0.07)),
    ((0.14, 0.04), (0.04, 0.21)),
    ((0.07, 0.00), (0.00, 0.13)),
    ((0.20, -0.04), (-0.04, 0.26)),
    ((0.12, 0.03), (0.03, 0.09)),
    ((0.16, 0.02), (0.02, 0.18)),
    ((0.22, -0.05), (-0.05, 0.28)),
)


WEIGHTS = (0.22, 0.48, 0.30)


MEANS = (
    (-1.10, 0.35),
    (0.25, -0.55),
    (1.40, 0.85),
)


LATENT_COVARIANCES = (
    ((0.72, 0.16), (0.16, 0.58)),
    ((0.83, -0.11), (-0.11, 0.67)),
    ((0.61, 0.09), (0.09, 0.91)),
)


def likelihood_fixture(
    dtype: np.dtype | type[np.floating] = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return independent arrays for the literal ``N=11, K=3, D=2`` case."""

    selected = np.dtype(dtype)
    return (
        np.asarray(OBSERVATIONS, dtype=selected),
        np.asarray(MEASUREMENT_COVARIANCES, dtype=selected),
        np.asarray(WEIGHTS, dtype=selected),
        np.asarray(MEANS, dtype=selected),
        np.asarray(LATENT_COVARIANCES, dtype=selected),
    )


__all__ = [
    "FIXTURE_ID",
    "FIXTURE_VERSION",
    "LATENT_COVARIANCES",
    "MEANS",
    "MEASUREMENT_COVARIANCES",
    "OBSERVATIONS",
    "WEIGHTS",
    "likelihood_fixture",
]

