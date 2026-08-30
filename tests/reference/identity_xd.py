"""Clear NumPy reference for the identity-projection XD equations.

This module intentionally favors explicit loops and independent linear algebra
over sharing structure with the future JAX implementation. It is test evidence,
not a performance implementation or public package API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class ReferenceEStep:
    """Independent E-step and posterior quantities."""

    component_log_density: FloatArray
    component_log_joint: FloatArray
    score_samples: FloatArray
    responsibilities: FloatArray
    conditional_mean: FloatArray
    conditional_covariance: FloatArray


@dataclass(frozen=True)
class ReferenceSufficientStatistics:
    """Component sufficient statistics for one exact M-step."""

    mass: FloatArray
    first_moment: FloatArray
    second_moment: FloatArray


@dataclass(frozen=True)
class ReferenceParameters:
    """Mixture parameters returned by the independent M-step."""

    weights: FloatArray
    means: FloatArray
    covariances: FloatArray


def _as_float64(value: npt.ArrayLike) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


def _logsumexp(values: FloatArray, axis: int) -> FloatArray:
    maximum = np.max(values, axis=axis, keepdims=True)
    reduced = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(reduced, axis=axis)


def identity_e_step(
    observations: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    *,
    factor_jitter: float = 0.0,
) -> ReferenceEStep:
    """Evaluate density, responsibilities, and latent posterior in float64."""

    x = _as_float64(observations)
    noise = _as_float64(measurement_covariances)
    alpha = _as_float64(weights)
    mu = _as_float64(means)
    latent_covariance = _as_float64(covariances)

    n_samples, dimension = x.shape
    n_components = alpha.shape[0]
    identity = np.eye(dimension, dtype=np.float64)

    log_density = np.empty((n_samples, n_components), dtype=np.float64)
    conditional_mean = np.empty(
        (n_samples, n_components, dimension), dtype=np.float64
    )
    conditional_covariance = np.empty(
        (n_samples, n_components, dimension, dimension), dtype=np.float64
    )

    for sample in range(n_samples):
        effective_noise = noise[sample] + factor_jitter * identity
        for component in range(n_components):
            total = latent_covariance[component] + effective_noise
            factor = np.linalg.cholesky(total)
            residual = x[sample] - mu[component]

            whitened = np.linalg.solve(factor, residual)
            log_determinant = 2.0 * np.log(np.diag(factor)).sum()
            log_density[sample, component] = -0.5 * (
                dimension * np.log(2.0 * np.pi)
                + log_determinant
                + whitened @ whitened
            )

            first_solve = np.linalg.solve(factor, latent_covariance[component].T)
            inverse_times_v_transpose = np.linalg.solve(factor.T, first_solve)
            gain = inverse_times_v_transpose.T

            conditional_mean[sample, component] = (
                mu[component] + gain @ residual
            )
            residual_operator = identity - gain
            posterior_covariance = (
                residual_operator
                @ latent_covariance[component]
                @ residual_operator.T
                + gain @ effective_noise @ gain.T
            )
            conditional_covariance[sample, component] = 0.5 * (
                posterior_covariance + posterior_covariance.T
            )

    component_log_joint = log_density + np.log(alpha)[None, :]
    score_samples = _logsumexp(component_log_joint, axis=1)
    responsibilities = np.exp(component_log_joint - score_samples[:, None])

    return ReferenceEStep(
        component_log_density=log_density,
        component_log_joint=component_log_joint,
        score_samples=score_samples,
        responsibilities=responsibilities,
        conditional_mean=conditional_mean,
        conditional_covariance=conditional_covariance,
    )


def marginalized_posterior(
    e_step: ReferenceEStep,
) -> tuple[FloatArray, FloatArray]:
    """Return component-marginalized latent posterior mean and covariance."""

    weights = e_step.responsibilities
    component_means = e_step.conditional_mean
    posterior_mean = np.einsum("nk,nkd->nd", weights, component_means)
    centered = component_means - posterior_mean[:, None, :]
    posterior_covariance = np.einsum(
        "nk,nkde->nde",
        weights,
        e_step.conditional_covariance
        + centered[:, :, :, None] * centered[:, :, None, :],
    )
    posterior_covariance = 0.5 * (
        posterior_covariance + np.swapaxes(posterior_covariance, -1, -2)
    )
    return posterior_mean, posterior_covariance


def sufficient_statistics(e_step: ReferenceEStep) -> ReferenceSufficientStatistics:
    """Accumulate the exact component statistics ``(n, h, G)``."""

    q = e_step.responsibilities
    b = e_step.conditional_mean
    second_conditional_moment = (
        e_step.conditional_covariance
        + b[:, :, :, None] * b[:, :, None, :]
    )
    return ReferenceSufficientStatistics(
        mass=q.sum(axis=0),
        first_moment=np.einsum("nk,nkd->kd", q, b),
        second_moment=np.einsum("nk,nkde->kde", q, second_conditional_moment),
    )


def identity_m_step(
    e_step: ReferenceEStep,
    *,
    covariance_ridge: float = 0.0,
) -> tuple[ReferenceParameters, ReferenceSufficientStatistics]:
    """Perform one independent, centered two-pass M-step."""

    statistics = sufficient_statistics(e_step)
    if np.any(~np.isfinite(statistics.mass)) or np.any(statistics.mass <= 0.0):
        raise FloatingPointError("collapsed component in reference M-step")

    weights = statistics.mass / statistics.mass.sum()
    means = statistics.first_moment / statistics.mass[:, None]
    centered = e_step.conditional_mean - means[None, :, :]
    covariances = np.einsum(
        "nk,nkde->kde",
        e_step.responsibilities,
        e_step.conditional_covariance
        + centered[:, :, :, None] * centered[:, :, None, :],
    ) / statistics.mass[:, None, None]

    dimension = means.shape[-1]
    covariances = covariances + covariance_ridge * np.eye(dimension)[None, :, :]
    covariances = 0.5 * (covariances + np.swapaxes(covariances, -1, -2))
    return ReferenceParameters(weights, means, covariances), statistics


def identity_em_step(
    observations: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    *,
    factor_jitter: float = 0.0,
    covariance_ridge: float = 0.0,
) -> tuple[ReferenceParameters, ReferenceEStep, ReferenceSufficientStatistics]:
    """Run one independent identity-projection E/M update."""

    e_step = identity_e_step(
        observations,
        measurement_covariances,
        weights,
        means,
        covariances,
        factor_jitter=factor_jitter,
    )
    parameters, statistics = identity_m_step(
        e_step, covariance_ridge=covariance_ridge
    )
    return parameters, e_step, statistics

