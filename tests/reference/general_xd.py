"""Independent NumPy oracle for fixed-observed-dimension general XD.

This module is test evidence for the model

``z_i ~ sum_k alpha_k N(mu_k, V_k)`` and
``x_i = R_i z_i + epsilon_i``, ``epsilon_i ~ N(0, S_i)``.

It deliberately uses explicit Python loops, ``numpy.linalg.slogdet``, and
general linear solves.  It does not import the JAX implementation and is not a
performance implementation or public package API.  The equations follow Bovy,
Hogg, and Roweis (2011), equations 7, 8, and 29--30.

Only a common observed dimension ``M`` is represented here.  A future grouped
or ragged missing-coordinate adapter can call the oracle once per group, but
that adapter is outside this reference module.  For ``M == 0``, the oracle
implements the contract's mixture-prior inference result and excludes those
rows from fitting statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class ReferenceGeneralEStep:
    """Observed densities, responsibilities, and latent conditional moments."""

    component_log_density: FloatArray
    component_log_joint: FloatArray
    score_samples: FloatArray
    responsibilities: FloatArray
    conditional_mean: FloatArray
    conditional_covariance: FloatArray
    observed_dimension: int


@dataclass(frozen=True)
class ReferenceGeneralSufficientStatistics:
    """Weighted component statistics ``(n, h, G)`` for one M-step."""

    mass: FloatArray
    first_moment: FloatArray
    second_moment: FloatArray


@dataclass(frozen=True)
class ReferenceGeneralParameters:
    """Mixture parameters returned by the independent general-XD M-step."""

    weights: FloatArray
    means: FloatArray
    covariances: FloatArray


@dataclass(frozen=True)
class ReferenceGroupedGeneralSufficientStatistics:
    """Stable global reductions for a sequence of fixed-``M`` groups.

    The first three fields are the raw contracted ``(n, h, Q)`` statistics.
    ``component_mean`` and ``centered_covariance`` are accumulated in a second
    pass over every informative row, so the reference M-step never obtains a
    covariance by subtracting two large raw moments.
    """

    mass: FloatArray
    first_moment: FloatArray
    second_moment: FloatArray
    log_mass: FloatArray
    component_mean: FloatArray
    centered_covariance: FloatArray
    weighted_log_likelihood: np.float64
    informative_weight: np.float64
    objective: np.float64


def _as_float64(value: npt.ArrayLike) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


def _logsumexp(values: FloatArray, axis: int) -> FloatArray:
    maximum = np.max(values, axis=axis, keepdims=True)
    reduced = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(reduced, axis=axis)


def _observation_weights(
    sample_weight: npt.ArrayLike | None,
    n_samples: int,
) -> FloatArray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)

    values = _as_float64(sample_weight)
    if values.shape != (n_samples,):
        raise ValueError(
            "sample_weight must have shape "
            f"{(n_samples,)}; received {values.shape}"
        )
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("sample_weight must be finite and nonnegative")
    if not np.any(values > 0.0):
        raise ValueError("sample_weight must contain positive total mass")
    return values


def general_e_step(
    observations: npt.ArrayLike,
    projection_matrices: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    *,
    factor_jitter: float = 0.0,
) -> ReferenceGeneralEStep:
    """Evaluate the general-projection XD E-step in NumPy float64.

    Canonical shapes are ``x: (N,M)``, ``R: (N,M,D)``, ``S: (N,M,M)``,
    ``alpha: (K,)``, ``mu: (K,D)``, and ``V: (K,D,D)``.  Inputs are assumed
    to have already passed the future public validation boundary.

    ``slogdet`` and a general solve are intentionally used instead of mirroring
    the planned JAX Cholesky kernel.  No covariance inverse is formed.
    """

    x = _as_float64(observations)
    projection = _as_float64(projection_matrices)
    noise = _as_float64(measurement_covariances)
    alpha = _as_float64(weights)
    mu = _as_float64(means)
    latent_covariance = _as_float64(covariances)

    n_samples, observed_dimension = x.shape
    n_components, latent_dimension = mu.shape

    expected_projection_shape = (
        n_samples,
        observed_dimension,
        latent_dimension,
    )
    if projection.shape != expected_projection_shape:
        raise ValueError(
            "projection_matrices must have canonical shape "
            f"{expected_projection_shape}; received {projection.shape}"
        )
    expected_noise_shape = (
        n_samples,
        observed_dimension,
        observed_dimension,
    )
    if noise.shape != expected_noise_shape:
        raise ValueError(
            "measurement_covariances must have canonical shape "
            f"{expected_noise_shape}; received {noise.shape}"
        )
    if alpha.shape != (n_components,):
        raise ValueError(
            f"weights must have shape {(n_components,)}; received {alpha.shape}"
        )
    expected_covariance_shape = (
        n_components,
        latent_dimension,
        latent_dimension,
    )
    if latent_covariance.shape != expected_covariance_shape:
        raise ValueError(
            "covariances must have shape "
            f"{expected_covariance_shape}; received {latent_covariance.shape}"
        )

    if observed_dimension == 0:
        log_density = np.zeros((n_samples, n_components), dtype=np.float64)
        component_log_joint = np.broadcast_to(
            np.log(alpha), (n_samples, n_components)
        ).copy()
        # With canonical normalized alpha, an empty-dimensional Gaussian has
        # density one and the mixture density is exactly one by contract.
        score_samples = np.zeros(n_samples, dtype=np.float64)
        responsibilities = np.broadcast_to(
            alpha, (n_samples, n_components)
        ).copy()
        conditional_mean = np.broadcast_to(
            mu, (n_samples, n_components, latent_dimension)
        ).copy()
        conditional_covariance = np.broadcast_to(
            latent_covariance,
            (
                n_samples,
                n_components,
                latent_dimension,
                latent_dimension,
            ),
        ).copy()
        return ReferenceGeneralEStep(
            component_log_density=log_density,
            component_log_joint=component_log_joint,
            score_samples=score_samples,
            responsibilities=responsibilities,
            conditional_mean=conditional_mean,
            conditional_covariance=conditional_covariance,
            observed_dimension=0,
        )

    observed_identity = np.eye(observed_dimension, dtype=np.float64)
    log_density = np.empty((n_samples, n_components), dtype=np.float64)
    conditional_mean = np.empty(
        (n_samples, n_components, latent_dimension), dtype=np.float64
    )
    conditional_covariance = np.empty(
        (
            n_samples,
            n_components,
            latent_dimension,
            latent_dimension,
        ),
        dtype=np.float64,
    )

    for sample in range(n_samples):
        row_projection = projection[sample]
        effective_noise = noise[sample] + factor_jitter * observed_identity
        for component in range(n_components):
            component_covariance = latent_covariance[component]
            projected_covariance = (
                row_projection @ component_covariance @ row_projection.T
            )
            total_covariance = projected_covariance + effective_noise
            sign, log_determinant = np.linalg.slogdet(total_covariance)
            if sign <= 0.0:
                raise np.linalg.LinAlgError(
                    "effective observed covariance is not positive definite"
                )

            residual = x[sample] - row_projection @ mu[component]
            solved_residual = np.linalg.solve(total_covariance, residual)
            log_density[sample, component] = -0.5 * (
                observed_dimension * np.log(2.0 * np.pi)
                + log_determinant
                + residual @ solved_residual
            )

            # Solve T X = R V, then transpose: X.T = V R.T T^-1.
            solved_projection_covariance = np.linalg.solve(
                total_covariance,
                row_projection @ component_covariance,
            )
            gain = solved_projection_covariance.T
            conditional_mean[sample, component] = (
                mu[component] + gain @ residual
            )

            posterior_covariance = (
                component_covariance
                - gain @ row_projection @ component_covariance
            )
            conditional_covariance[sample, component] = 0.5 * (
                posterior_covariance + posterior_covariance.T
            )

    component_log_joint = log_density + np.log(alpha)[None, :]
    score_samples = _logsumexp(component_log_joint, axis=1)
    responsibilities = np.exp(component_log_joint - score_samples[:, None])

    return ReferenceGeneralEStep(
        component_log_density=log_density,
        component_log_joint=component_log_joint,
        score_samples=score_samples,
        responsibilities=responsibilities,
        conditional_mean=conditional_mean,
        conditional_covariance=conditional_covariance,
        observed_dimension=observed_dimension,
    )


def general_objective(
    e_step: ReferenceGeneralEStep,
    sample_weight: npt.ArrayLike | None = None,
    *,
    normalized: bool = False,
) -> np.float64:
    """Return the weighted observed-data objective.

    The unnormalized result is ``sum_i w_i log p(x_i)``.  With
    ``normalized=True`` it is divided by ``sum_i w_i``; this is the quantity
    whose scale is invariant under a common positive rescaling of all weights.
    """

    weights = _observation_weights(sample_weight, e_step.score_samples.shape[0])
    objective = np.dot(weights, e_step.score_samples)
    if normalized:
        if e_step.observed_dimension == 0:
            raise ValueError(
                "normalized fitting objective requires positive-weight "
                "observed rows"
            )
        objective = objective / weights.sum()
    return np.float64(objective)


def marginalized_posterior(
    e_step: ReferenceGeneralEStep,
) -> tuple[FloatArray, FloatArray]:
    """Return component-marginalized latent posterior mean and covariance."""

    q = e_step.responsibilities
    component_means = e_step.conditional_mean
    n_samples, n_components, latent_dimension = component_means.shape
    posterior_mean = np.empty((n_samples, latent_dimension), dtype=np.float64)
    posterior_covariance = np.empty(
        (n_samples, latent_dimension, latent_dimension), dtype=np.float64
    )

    for sample in range(n_samples):
        mean = np.zeros(latent_dimension, dtype=np.float64)
        for component in range(n_components):
            mean += q[sample, component] * component_means[sample, component]
        posterior_mean[sample] = mean

        covariance = np.zeros(
            (latent_dimension, latent_dimension), dtype=np.float64
        )
        for component in range(n_components):
            centered = component_means[sample, component] - mean
            covariance += q[sample, component] * (
                e_step.conditional_covariance[sample, component]
                + np.outer(centered, centered)
            )
        posterior_covariance[sample] = 0.5 * (
            covariance + covariance.T
        )

    return posterior_mean, posterior_covariance


def general_sufficient_statistics(
    e_step: ReferenceGeneralEStep,
    sample_weight: npt.ArrayLike | None = None,
) -> ReferenceGeneralSufficientStatistics:
    """Accumulate weighted complete-data statistics using explicit loops."""

    q = e_step.responsibilities
    b = e_step.conditional_mean
    n_samples, n_components, latent_dimension = b.shape
    weights = _observation_weights(sample_weight, n_samples)
    mass = np.zeros(n_components, dtype=np.float64)
    first_moment = np.zeros(
        (n_components, latent_dimension), dtype=np.float64
    )
    second_moment = np.zeros(
        (n_components, latent_dimension, latent_dimension), dtype=np.float64
    )

    # A fully missing fixed-M group is inference-valid but carries no fitting
    # information.  Its supplied weights are validated above, then excluded.
    if e_step.observed_dimension == 0:
        return ReferenceGeneralSufficientStatistics(
            mass=mass,
            first_moment=first_moment,
            second_moment=second_moment,
        )

    for sample in range(n_samples):
        for component in range(n_components):
            effective_responsibility = weights[sample] * q[sample, component]
            component_mean = b[sample, component]
            mass[component] += effective_responsibility
            first_moment[component] += (
                effective_responsibility * component_mean
            )
            second_moment[component] += effective_responsibility * (
                e_step.conditional_covariance[sample, component]
                + np.outer(component_mean, component_mean)
            )

    return ReferenceGeneralSufficientStatistics(
        mass=mass,
        first_moment=first_moment,
        second_moment=second_moment,
    )


def general_m_step(
    e_step: ReferenceGeneralEStep,
    *,
    sample_weight: npt.ArrayLike | None = None,
    covariance_ridge: float = 0.0,
) -> tuple[ReferenceGeneralParameters, ReferenceGeneralSufficientStatistics]:
    """Perform one independent weighted, centered two-pass M-step."""

    statistics = general_sufficient_statistics(e_step, sample_weight)
    if np.any(~np.isfinite(statistics.mass)) or np.any(
        statistics.mass <= 0.0
    ):
        raise FloatingPointError("collapsed component in general reference M-step")

    n_samples, n_components, latent_dimension = e_step.conditional_mean.shape
    observation_weights = _observation_weights(sample_weight, n_samples)
    weights = statistics.mass / statistics.mass.sum()
    means = statistics.first_moment / statistics.mass[:, None]
    covariances = np.zeros(
        (n_components, latent_dimension, latent_dimension), dtype=np.float64
    )

    for sample in range(n_samples):
        for component in range(n_components):
            effective_responsibility = (
                observation_weights[sample]
                * e_step.responsibilities[sample, component]
            )
            centered = (
                e_step.conditional_mean[sample, component] - means[component]
            )
            covariances[component] += effective_responsibility * (
                e_step.conditional_covariance[sample, component]
                + np.outer(centered, centered)
            )

    latent_identity = np.eye(latent_dimension, dtype=np.float64)
    for component in range(n_components):
        covariances[component] /= statistics.mass[component]
        covariances[component] += covariance_ridge * latent_identity
        covariances[component] = 0.5 * (
            covariances[component] + covariances[component].T
        )

    return (
        ReferenceGeneralParameters(weights, means, covariances),
        statistics,
    )


def general_em_step(
    observations: npt.ArrayLike,
    projection_matrices: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    *,
    sample_weight: npt.ArrayLike | None = None,
    factor_jitter: float = 0.0,
    covariance_ridge: float = 0.0,
) -> tuple[
    ReferenceGeneralParameters,
    ReferenceGeneralEStep,
    ReferenceGeneralSufficientStatistics,
]:
    """Run one independent weighted general-projection E/M update."""

    e_step = general_e_step(
        observations,
        projection_matrices,
        measurement_covariances,
        weights,
        means,
        covariances,
        factor_jitter=factor_jitter,
    )
    parameters, statistics = general_m_step(
        e_step,
        sample_weight=sample_weight,
        covariance_ridge=covariance_ridge,
    )
    return parameters, e_step, statistics


def _group_sample_weight(
    sample_weight: npt.ArrayLike | None,
    n_samples: int,
) -> FloatArray:
    """Validate one group's weights without requiring group-local mass.

    A group may contain only zero-weight rows while another group supplies the
    global informative mass, so the fixed-``M`` oracle's local-positive-mass
    requirement is intentionally not reused here.
    """

    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)
    values = _as_float64(sample_weight)
    if values.shape != (n_samples,):
        raise ValueError(
            "sample_weight must have shape "
            f"{(n_samples,)}; received {values.shape}"
        )
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("sample_weight must be finite and nonnegative")
    return values


def general_grouped_m_step(
    e_steps: Sequence[ReferenceGeneralEStep],
    sample_weights: Sequence[npt.ArrayLike | None],
    *,
    covariance_ridge: float = 0.0,
) -> tuple[
    ReferenceGeneralParameters,
    ReferenceGroupedGeneralSufficientStatistics,
]:
    """Perform one stable global M-step over fixed-``M`` NumPy results.

    Groups with ``M == 0`` are validated but excluded from the objective,
    informative weight, and every fit statistic. All informative groups are
    reduced before one component-collapse check and one M-step.
    """

    if len(e_steps) == 0:
        raise ValueError("at least one grouped E-step is required")
    if len(e_steps) != len(sample_weights):
        raise ValueError(
            "e_steps and sample_weights must contain the same number of groups"
        )
    if not np.isfinite(covariance_ridge) or covariance_ridge < 0.0:
        raise ValueError("covariance_ridge must be finite and nonnegative")

    first = e_steps[0]
    if first.conditional_mean.ndim != 3:
        raise ValueError("conditional_mean must have shape (N,K,D)")
    _, n_components, latent_dimension = first.conditional_mean.shape
    mass = np.zeros(n_components, dtype=np.float64)
    first_moment = np.zeros(
        (n_components, latent_dimension), dtype=np.float64
    )
    second_moment = np.zeros(
        (n_components, latent_dimension, latent_dimension), dtype=np.float64
    )
    weighted_log_likelihood = np.float64(0.0)
    informative_weight = np.float64(0.0)
    prepared_weights: list[FloatArray] = []

    for e_step, supplied_weight in zip(e_steps, sample_weights, strict=True):
        n_samples, group_components, group_dimension = (
            e_step.conditional_mean.shape
        )
        if (group_components, group_dimension) != (
            n_components,
            latent_dimension,
        ):
            raise ValueError("every group must use the same K and D")
        weights = _group_sample_weight(supplied_weight, n_samples)
        prepared_weights.append(weights)
        if e_step.observed_dimension == 0:
            continue

        informative_weight += np.sum(weights, dtype=np.float64)
        weighted_log_likelihood += np.dot(weights, e_step.score_samples)
        for sample in range(n_samples):
            for component in range(n_components):
                effective_weight = (
                    weights[sample] * e_step.responsibilities[sample, component]
                )
                component_mean = e_step.conditional_mean[sample, component]
                mass[component] += effective_weight
                first_moment[component] += effective_weight * component_mean
                second_moment[component] += effective_weight * (
                    e_step.conditional_covariance[sample, component]
                    + np.outer(component_mean, component_mean)
                )

    if not np.isfinite(informative_weight) or informative_weight <= 0.0:
        raise ValueError("no_informative_weight")
    if not np.isfinite(weighted_log_likelihood):
        raise FloatingPointError("nonfinite grouped weighted log likelihood")
    if (
        np.any(~np.isfinite(mass))
        or np.any(~np.isfinite(first_moment))
        or np.any(~np.isfinite(second_moment))
        or np.any(mass <= 0.0)
    ):
        raise FloatingPointError("collapsed component in grouped reference M-step")

    component_mean = first_moment / mass[:, None]
    centered_covariance = np.zeros_like(second_moment)
    for e_step, weights in zip(e_steps, prepared_weights, strict=True):
        if e_step.observed_dimension == 0:
            continue
        n_samples = e_step.conditional_mean.shape[0]
        for sample in range(n_samples):
            for component in range(n_components):
                effective_weight = (
                    weights[sample] * e_step.responsibilities[sample, component]
                )
                centered = (
                    e_step.conditional_mean[sample, component]
                    - component_mean[component]
                )
                centered_covariance[component] += effective_weight * (
                    e_step.conditional_covariance[sample, component]
                    + np.outer(centered, centered)
                )

    centered_covariance /= mass[:, None, None]
    latent_identity = np.eye(latent_dimension, dtype=np.float64)
    covariances = centered_covariance + covariance_ridge * latent_identity
    covariances = 0.5 * (covariances + covariances.swapaxes(-1, -2))
    parameters = ReferenceGeneralParameters(
        weights=mass / mass.sum(),
        means=component_mean,
        covariances=covariances,
    )
    objective = np.float64(weighted_log_likelihood / informative_weight)
    statistics = ReferenceGroupedGeneralSufficientStatistics(
        mass=mass,
        first_moment=first_moment,
        second_moment=second_moment,
        log_mass=np.log(mass),
        component_mean=component_mean,
        centered_covariance=centered_covariance,
        weighted_log_likelihood=np.float64(weighted_log_likelihood),
        informative_weight=np.float64(informative_weight),
        objective=objective,
    )
    return parameters, statistics


def general_grouped_objective(
    e_steps: Sequence[ReferenceGeneralEStep],
    sample_weights: Sequence[npt.ArrayLike | None],
) -> tuple[np.float64, np.float64, np.float64]:
    """Return ``(raw, informative_weight, normalized)`` for dense groups."""

    if len(e_steps) == 0 or len(e_steps) != len(sample_weights):
        raise ValueError(
            "e_steps and sample_weights must contain the same nonzero number "
            "of groups"
        )
    raw = np.float64(0.0)
    informative_weight = np.float64(0.0)
    all_empty = True
    for e_step, supplied_weight in zip(e_steps, sample_weights, strict=True):
        weights = _group_sample_weight(
            supplied_weight, e_step.score_samples.shape[0]
        )
        if e_step.observed_dimension == 0:
            continue
        all_empty = False
        informative_weight += np.sum(weights, dtype=np.float64)
        raw += np.dot(weights, e_step.score_samples)

    if all_empty:
        return np.float64(0.0), np.float64(0.0), np.float64(0.0)
    if not np.isfinite(informative_weight) or informative_weight <= 0.0:
        raise ValueError("no_informative_weight")
    if not np.isfinite(raw):
        raise FloatingPointError("nonfinite grouped weighted log likelihood")
    return raw, informative_weight, np.float64(raw / informative_weight)
