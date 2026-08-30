"""Permutation-invariant metrics for statistical mixture recovery evidence.

This module is not an EM oracle.  It never evaluates an update, stopping rule,
or implementation endpoint.  It aligns an already fitted mixture to known
generating components and measures the fitted latent distribution against
stored truth and an independent latent holdout sample.  Keeping this helper
separate prevents final-fit statistical recovery from being mislabeled as
algorithm/reference parity.

The exhaustive assignment is intentionally limited to the small mixtures used
by scientific-validation fixtures.  A production label-alignment API is not
implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ComponentAlignment:
    """Candidate parameters reordered to correspond to reference components."""

    permutation: tuple[int, ...]
    total_symmetric_gaussian_kl: float
    weights: FloatArray
    means: FloatArray
    covariances: FloatArray


@dataclass(frozen=True)
class RecoveryMetrics:
    """Label-invariant finite-sample diagnostics for one fitted mixture."""

    alignment: ComponentAlignment
    max_absolute_weight_error: float
    max_mean_mahalanobis_error: float
    max_relative_covariance_frobenius_error: float
    latent_log_density_rms_error: float
    latent_log_density_mean_gap: float
    mixture_mean_l2_error: float
    mixture_covariance_relative_frobenius_error: float


def _validated_mixture(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    *,
    name: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    alpha = np.asarray(weights, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    covariance = np.asarray(covariances, dtype=np.float64)
    if alpha.ndim != 1:
        raise ValueError(f"{name} weights must have shape (K,)")
    n_components = alpha.shape[0]
    if not 1 <= n_components <= 8:
        raise ValueError("exhaustive recovery alignment requires 1 <= K <= 8")
    if mu.ndim != 2 or mu.shape[0] != n_components:
        raise ValueError(f"{name} means must have shape (K,D)")
    expected_covariance_shape = (
        n_components,
        mu.shape[1],
        mu.shape[1],
    )
    if covariance.shape != expected_covariance_shape:
        raise ValueError(
            f"{name} covariances must have shape {expected_covariance_shape}"
        )
    if (
        not np.all(np.isfinite(alpha))
        or not np.all(np.isfinite(mu))
        or not np.all(np.isfinite(covariance))
    ):
        raise ValueError(f"{name} mixture must be finite")
    if np.any(alpha <= 0.0) or not np.isclose(
        alpha.sum(), 1.0, rtol=0.0, atol=1e-8
    ):
        raise ValueError(f"{name} weights must be positive and normalized")
    for component in range(n_components):
        if not np.allclose(
            covariance[component],
            covariance[component].T,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(f"{name} covariance {component} is asymmetric")
        try:
            np.linalg.cholesky(covariance[component])
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"{name} covariance {component} is not positive definite"
            ) from error
    return alpha, mu, covariance


def _symmetric_gaussian_kl(
    reference_mean: FloatArray,
    reference_covariance: FloatArray,
    candidate_mean: FloatArray,
    candidate_covariance: FloatArray,
) -> float:
    """Return the average of both directed Gaussian KL divergences."""

    dimension = reference_mean.shape[0]
    difference = candidate_mean - reference_mean
    trace_terms = np.trace(
        np.linalg.solve(candidate_covariance, reference_covariance)
    ) + np.trace(np.linalg.solve(reference_covariance, candidate_covariance))
    mean_term = difference @ (
        np.linalg.solve(candidate_covariance, difference)
        + np.linalg.solve(reference_covariance, difference)
    )
    value = 0.25 * (trace_terms + mean_term - 2.0 * dimension)
    # The exact expression is nonnegative.  Roundoff at identical inputs can
    # produce a tiny negative number, which has no assignment meaning.
    return float(max(0.0, value))


def align_components(
    reference_weights: npt.ArrayLike,
    reference_means: npt.ArrayLike,
    reference_covariances: npt.ArrayLike,
    candidate_weights: npt.ArrayLike,
    candidate_means: npt.ArrayLike,
    candidate_covariances: npt.ArrayLike,
) -> ComponentAlignment:
    """Align candidate labels by minimum total symmetric Gaussian KL.

    The returned permutation satisfies
    ``aligned_reference_component[j] == candidate[permutation[j]]``.  Mixture
    weights do not influence the assignment, so a weight error cannot hide a
    closer match in the Gaussian component distributions.
    """

    reference = _validated_mixture(
        reference_weights,
        reference_means,
        reference_covariances,
        name="reference",
    )
    candidate = _validated_mixture(
        candidate_weights,
        candidate_means,
        candidate_covariances,
        name="candidate",
    )
    reference_alpha, reference_mu, reference_covariance = reference
    candidate_alpha, candidate_mu, candidate_covariance = candidate
    if reference_mu.shape != candidate_mu.shape:
        raise ValueError("reference and candidate mixtures must share K and D")

    n_components = len(reference_alpha)
    pair_cost = np.empty((n_components, n_components), dtype=np.float64)
    for reference_component in range(n_components):
        for candidate_component in range(n_components):
            pair_cost[reference_component, candidate_component] = (
                _symmetric_gaussian_kl(
                    reference_mu[reference_component],
                    reference_covariance[reference_component],
                    candidate_mu[candidate_component],
                    candidate_covariance[candidate_component],
                )
            )

    best_cost = np.inf
    best_permutation: tuple[int, ...] | None = None
    for permutation in permutations(range(n_components)):
        cost = float(
            sum(pair_cost[index, permutation[index]] for index in range(n_components))
        )
        if cost < best_cost:
            best_cost = cost
            best_permutation = tuple(permutation)
    if best_permutation is None:  # Defensive; validation already requires K >= 1.
        raise RuntimeError("component alignment produced no permutation")

    indices = np.asarray(best_permutation, dtype=np.int64)
    return ComponentAlignment(
        permutation=best_permutation,
        total_symmetric_gaussian_kl=best_cost,
        weights=candidate_alpha[indices].copy(),
        means=candidate_mu[indices].copy(),
        covariances=candidate_covariance[indices].copy(),
    )


def mixture_log_density(
    samples: npt.ArrayLike,
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
) -> FloatArray:
    """Evaluate a full-covariance latent GMM with explicit NumPy solves."""

    alpha, mu, covariance = _validated_mixture(
        weights, means, covariances, name="mixture"
    )
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != mu.shape[1]:
        raise ValueError(
            f"samples must have shape (N,{mu.shape[1]}); received {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")

    n_samples = values.shape[0]
    n_components, dimension = mu.shape
    log_joint = np.empty((n_samples, n_components), dtype=np.float64)
    normalizer = dimension * np.log(2.0 * np.pi)
    for component in range(n_components):
        factor = np.linalg.cholesky(covariance[component])
        residual = values - mu[component]
        whitened = np.linalg.solve(factor, residual.T).T
        squared_distance = np.einsum("nd,nd->n", whitened, whitened)
        log_determinant = 2.0 * np.log(np.diag(factor)).sum()
        log_joint[:, component] = np.log(alpha[component]) - 0.5 * (
            normalizer + log_determinant + squared_distance
        )
    maximum = np.max(log_joint, axis=1)
    return maximum + np.log(
        np.sum(np.exp(log_joint - maximum[:, None]), axis=1)
    )


def mixture_moments(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return a mixture's exact latent mean and covariance."""

    alpha, mu, covariance = _validated_mixture(
        weights, means, covariances, name="mixture"
    )
    mixture_mean = np.einsum("k,kd->d", alpha, mu)
    centered = mu - mixture_mean
    mixture_covariance = np.einsum(
        "k,kde->de",
        alpha,
        covariance + centered[:, :, None] * centered[:, None, :],
    )
    mixture_covariance = 0.5 * (
        mixture_covariance + mixture_covariance.T
    )
    return mixture_mean, mixture_covariance


def recovery_metrics(
    reference_weights: npt.ArrayLike,
    reference_means: npt.ArrayLike,
    reference_covariances: npt.ArrayLike,
    candidate_weights: npt.ArrayLike,
    candidate_means: npt.ArrayLike,
    candidate_covariances: npt.ArrayLike,
    latent_holdout: npt.ArrayLike,
) -> RecoveryMetrics:
    """Measure an already fitted candidate against generating truth."""

    reference_alpha, reference_mu, reference_covariance = _validated_mixture(
        reference_weights,
        reference_means,
        reference_covariances,
        name="reference",
    )
    alignment = align_components(
        reference_alpha,
        reference_mu,
        reference_covariance,
        candidate_weights,
        candidate_means,
        candidate_covariances,
    )

    weight_error = float(
        np.max(np.abs(alignment.weights - reference_alpha))
    )
    mean_errors = []
    covariance_errors = []
    for component in range(len(reference_alpha)):
        difference = alignment.means[component] - reference_mu[component]
        mean_errors.append(
            float(
                np.sqrt(
                    max(
                        0.0,
                        difference
                        @ np.linalg.solve(
                            reference_covariance[component], difference
                        ),
                    )
                )
            )
        )
        covariance_errors.append(
            float(
                np.linalg.norm(
                    alignment.covariances[component]
                    - reference_covariance[component],
                    ord="fro",
                )
                / np.linalg.norm(reference_covariance[component], ord="fro")
            )
        )

    reference_log_density = mixture_log_density(
        latent_holdout,
        reference_alpha,
        reference_mu,
        reference_covariance,
    )
    candidate_log_density = mixture_log_density(
        latent_holdout,
        alignment.weights,
        alignment.means,
        alignment.covariances,
    )
    density_difference = reference_log_density - candidate_log_density

    reference_mixture_mean, reference_mixture_covariance = mixture_moments(
        reference_alpha, reference_mu, reference_covariance
    )
    candidate_mixture_mean, candidate_mixture_covariance = mixture_moments(
        alignment.weights, alignment.means, alignment.covariances
    )
    return RecoveryMetrics(
        alignment=alignment,
        max_absolute_weight_error=weight_error,
        max_mean_mahalanobis_error=max(mean_errors),
        max_relative_covariance_frobenius_error=max(covariance_errors),
        latent_log_density_rms_error=float(
            np.sqrt(np.mean(np.square(density_difference)))
        ),
        latent_log_density_mean_gap=float(np.mean(density_difference)),
        mixture_mean_l2_error=float(
            np.linalg.norm(candidate_mixture_mean - reference_mixture_mean)
        ),
        mixture_covariance_relative_frobenius_error=float(
            np.linalg.norm(
                candidate_mixture_covariance - reference_mixture_covariance,
                ord="fro",
            )
            / np.linalg.norm(reference_mixture_covariance, ord="fro")
        ),
    )


__all__ = [
    "ComponentAlignment",
    "RecoveryMetrics",
    "align_components",
    "mixture_log_density",
    "mixture_moments",
    "recovery_metrics",
]
