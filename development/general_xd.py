"""Temporary pure-JAX kernels for fixed-``M`` general-projection XD.

This module implements the dense numerical leaf described by
``docs/general-model-contract.md``.  It is deliberately outside the future
installable package namespace and is not a public API.  Callers pass canonical,
already-validated arrays with one fixed observed dimension per invocation.

Each observation/component pair uses one Cholesky factor for its density,
generalized gain, and Joseph-form posterior covariance.  No covariance inverse
is formed.  The special ``M == 0`` path returns the mixture prior without
attempting a zero-dimensional factorization.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .identity_xd import (
    EMStepResult,
    EStep,
    Params,
    _nonnegative_conversion_is_valid,
    _real_scalar_control,
)


Array = jax.Array


class GeneralSufficientStatistics(NamedTuple):
    """Weighted general-XD moments with authoritative device failure status.

    ``failed_pairs`` contains only failures attached to positive-weight rows.
    A failed zero-weight row remains visible on the accompanying ``EStep`` but
    has no fitting effect. ``numerical_failure`` also covers invalid weights,
    invalid jitter, and nonfinite arithmetic or total mass.
    """

    mass: Array
    first_moment: Array
    second_moment: Array
    numerical_failure: Array
    failed_pairs: Array


def _general_scalar_control(
    value: object,
    *,
    dtype: jnp.dtype,
    name: str,
) -> tuple[Array, Array]:
    """Validate a scalar control while preserving eager NumPy source values.

    Python numeric scalars, NumPy scalars, and rank-zero ndarrays receive sign,
    finiteness, overflow, and nonzero-to-zero conversion checks before JAX can
    canonicalize a float64 source under disabled x64. Traced values use the
    identity kernel's device-resident selected-dtype validation because no
    wider source remains.
    """

    numpy_source_is_valid: bool | None = None
    eager_source = isinstance(value, (float, int, np.ndarray, np.generic))
    if eager_source and not isinstance(value, (bool, np.bool_)):
        source = np.asarray(value)
        source_is_real_numeric = np.issubdtype(
            source.dtype, np.integer
        ) or np.issubdtype(source.dtype, np.floating)
        if source.ndim == 0 and source_is_real_numeric:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                selected_source = source.astype(np.dtype(dtype), copy=False)
            numpy_source_is_valid = bool(
                np.isfinite(source)
                and source >= 0
                and np.isfinite(selected_source)
                and not (source != 0 and selected_source == 0)
            )

    converted, is_valid = _real_scalar_control(value, dtype=dtype, name=name)
    if numpy_source_is_valid is not None:
        is_valid = is_valid & jnp.asarray(numpy_source_is_valid)
    return converted, is_valid


def _canonical_general_inputs(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
) -> tuple[Params, Array, Array, Array, tuple[int, ...], int, int, int]:
    """Coerce arrays to the parameter dtype and check canonical static shapes."""

    means = jnp.asarray(params.means)
    if means.ndim != 2 or means.shape[0] < 1 or means.shape[1] < 1:
        raise ValueError(
            "means must have shape (K, D) with K,D >= 1; " f"received {means.shape}"
        )
    if not jnp.issubdtype(means.dtype, jnp.floating):
        raise TypeError(
            "parameter means must have a floating dtype; " f"received {means.dtype}"
        )

    dtype = means.dtype
    n_components, latent_dimension = means.shape
    weights = jnp.asarray(params.weights, dtype=dtype)
    covariances = jnp.asarray(params.covariances, dtype=dtype)
    if weights.shape != (n_components,):
        raise ValueError(
            f"weights must have shape {(n_components,)}; received {weights.shape}"
        )
    expected_covariance_shape = (
        n_components,
        latent_dimension,
        latent_dimension,
    )
    if covariances.shape != expected_covariance_shape:
        raise ValueError(
            "parameter covariances must have shape "
            f"{expected_covariance_shape}; received {covariances.shape}"
        )

    x = jnp.asarray(observations, dtype=dtype)
    if x.ndim < 1:
        raise ValueError(
            "observations must have shape B + (M,); " f"received {x.shape}"
        )
    batch_shape = x.shape[:-1]
    observed_dimension = x.shape[-1]

    projection = jnp.asarray(projection_matrices, dtype=dtype)
    expected_projection_shape = batch_shape + (
        observed_dimension,
        latent_dimension,
    )
    if projection.shape != expected_projection_shape:
        raise ValueError(
            "projection_matrices must have canonical shape "
            f"{expected_projection_shape}; received {projection.shape}"
        )

    noise = jnp.asarray(measurement_covariances, dtype=dtype)
    expected_noise_shape = batch_shape + (
        observed_dimension,
        observed_dimension,
    )
    if noise.shape != expected_noise_shape:
        raise ValueError(
            "measurement_covariances must have canonical shape "
            f"{expected_noise_shape}; received {noise.shape}"
        )

    canonical_params = Params(weights, means, covariances)
    return (
        canonical_params,
        x,
        projection,
        noise,
        batch_shape,
        n_components,
        latent_dimension,
        observed_dimension,
    )


def _canonical_sample_weight(
    sample_weight: Array,
    *,
    n_samples: int,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Convert weights and retain a device-resident validity flag.

    An eager NumPy ndarray receives a source-dtype sign, finiteness, overflow,
    and positive-underflow guard before JAX conversion. A traced JAX value has
    already lost any wider host representation, so the compiled numerical path
    can validate only the dtype and values presented to it.
    """

    # Preserve NumPy source precision at this eager boundary.  With JAX x64
    # disabled, ``jnp.asarray`` canonicalizes a float64 ndarray before device
    # code can distinguish a tiny positive value from zero or a tiny negative
    # value from ``-0.0``.  Traced JAX arrays deliberately skip this host check;
    # their already-selected dtype is all a compiled leaf can observe.
    numpy_source_is_valid: bool | None = None
    if isinstance(sample_weight, np.ndarray):
        source = sample_weight
        source_is_real_numeric = np.issubdtype(
            source.dtype, np.integer
        ) or np.issubdtype(source.dtype, np.floating)
        if source_is_real_numeric:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                selected_source = source.astype(np.dtype(dtype), copy=False)
            numpy_source_is_valid = bool(
                np.all(
                    np.isfinite(source)
                    & (source >= 0)
                    & np.isfinite(selected_source)
                    & ~((source > 0) & (selected_source == 0))
                )
            )

    try:
        original = jnp.asarray(sample_weight)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "sample_weight must be a real numeric array with shape (N,)"
        ) from error
    if original.shape != (n_samples,):
        raise ValueError(
            "sample_weight must have canonical shape "
            f"{(n_samples,)}; received {original.shape}"
        )
    is_real_numeric = jnp.issubdtype(original.dtype, jnp.integer) or jnp.issubdtype(
        original.dtype, jnp.floating
    )
    if not is_real_numeric:
        raise TypeError(
            "sample_weight must have a real numeric dtype; "
            f"received {original.dtype}"
        )

    converted = jnp.asarray(original, dtype=dtype)
    is_valid = jnp.all(
        _nonnegative_conversion_is_valid(original, converted)
    )
    if numpy_source_is_valid is not None:
        is_valid = is_valid & jnp.asarray(numpy_source_is_valid)
    return converted, is_valid


def _component_posterior_general(
    observation: Array,
    projection: Array,
    effective_noise: Array,
    mean: Array,
    covariance: Array,
    latent_identity: Array,
    log_two_pi: Array,
) -> tuple[Array, Array, Array, Array]:
    """Evaluate one general-projection component with one reused factor."""

    projected_covariance = projection @ covariance @ projection.T
    total_covariance = projected_covariance + effective_noise
    factor = jax.lax.linalg.cholesky(total_covariance, symmetrize_input=False)
    residual = observation - projection @ mean

    whitened = jsp.linalg.solve_triangular(factor, residual, lower=True)
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
    log_density = -0.5 * (
        observation.shape[-1] * log_two_pi
        + log_determinant
        + jnp.vdot(whitened, whitened)
    )

    # Solve T X = R V and transpose: X.T = V R.T T^-1.
    projection_times_covariance = projection @ covariance
    first_solve = jsp.linalg.solve_triangular(
        factor, projection_times_covariance, lower=True
    )
    inverse_times_projection_covariance = jsp.linalg.solve_triangular(
        factor.T, first_solve, lower=False
    )
    gain = inverse_times_projection_covariance.T

    conditional_mean = mean + gain @ residual

    # Generalized Joseph covariance.  A value-dependent identity-projection
    # specialization would make the derivative with respect to ``R``
    # discontinuous.  The only exact branch below depends on effective noise
    # and static dimensions: a successful square, noiseless observation fully
    # determines the latent value, so its posterior covariance is exactly zero
    # for every invertible square projection.
    residual_operator = latent_identity - gain @ projection
    conditional_covariance = (
        residual_operator @ covariance @ residual_operator.T
        + gain @ effective_noise @ gain.T
    )
    half_conditional_covariance = jax.lax.optimization_barrier(
        0.5 * conditional_covariance
    )
    conditional_covariance = (
        half_conditional_covariance + half_conditional_covariance.T
    )
    if projection.shape[-2] == projection.shape[-1]:
        zero_effective_noise = jnp.all(effective_noise == 0.0)
        conditional_covariance = jnp.where(
            zero_effective_noise,
            jnp.zeros_like(conditional_covariance),
            conditional_covariance,
        )

    pair_is_valid = (
        jnp.all(jnp.isfinite(factor))
        & jnp.all(jnp.diag(factor) > 0.0)
        & jnp.isfinite(log_density)
        & jnp.all(jnp.isfinite(gain))
        & jnp.all(jnp.isfinite(conditional_mean))
        & jnp.all(jnp.isfinite(conditional_covariance))
    )
    return (
        log_density,
        conditional_mean,
        conditional_covariance,
        ~pair_is_valid,
    )


def _empty_observation_posterior(
    params: Params,
    *,
    batch_shape: tuple[int, ...],
    jitter_is_valid: Array,
) -> EStep:
    """Return exact mixture-prior inference for the static ``M == 0`` path."""

    n_components, latent_dimension = params.means.shape
    dtype = params.means.dtype
    component_shape = batch_shape + (n_components,)
    failed_pairs = jnp.broadcast_to(~jitter_is_valid, component_shape)
    component_log_density = jnp.where(
        failed_pairs,
        -jnp.inf,
        jnp.zeros(component_shape, dtype=dtype),
    )
    log_mixture_weight = _log_positive_with_gradual_underflow(params.weights)
    component_log_joint = component_log_density + jnp.broadcast_to(
        log_mixture_weight, component_shape
    )
    score_samples = jnp.where(
        jnp.any(failed_pairs, axis=-1),
        -jnp.inf,
        jnp.zeros(batch_shape, dtype=dtype),
    )
    responsibilities = jnp.broadcast_to(params.weights, component_shape)
    conditional_mean = jnp.broadcast_to(
        params.means, batch_shape + (n_components, latent_dimension)
    )
    conditional_covariance = jnp.broadcast_to(
        params.covariances,
        batch_shape + (n_components, latent_dimension, latent_dimension),
    )
    conditional_mean = jnp.where(failed_pairs[..., None], 0.0, conditional_mean)
    conditional_covariance = jnp.where(
        failed_pairs[..., None, None], 0.0, conditional_covariance
    )
    return EStep(
        component_log_density=component_log_density,
        component_log_joint=component_log_joint,
        score_samples=score_samples,
        responsibilities=responsibilities,
        conditional_mean=conditional_mean,
        conditional_covariance=conditional_covariance,
        numerical_failure=jnp.any(failed_pairs) | (~jitter_is_valid),
        failed_pairs=failed_pairs,
    )


def posterior_components_general(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> EStep:
    """Return fixed-``M`` general-XD densities and latent posterior moments.

    Inputs use canonical per-item projection and full-covariance shapes.  An
    inference batch may have any leading shape ``B``; no shared-array
    broadcasting or transpose inference occurs inside this numerical leaf.
    """

    (
        canonical_params,
        x,
        projection,
        noise,
        batch_shape,
        n_components,
        latent_dimension,
        observed_dimension,
    ) = _canonical_general_inputs(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
    )
    dtype = canonical_params.means.dtype
    jitter, jitter_is_valid = _general_scalar_control(
        factor_jitter, dtype=dtype, name="factor_jitter"
    )
    if observed_dimension == 0:
        return _empty_observation_posterior(
            canonical_params,
            batch_shape=batch_shape,
            jitter_is_valid=jitter_is_valid,
        )

    safe_jitter = jnp.where(jitter_is_valid, jitter, 0.0)
    observed_identity = jnp.eye(observed_dimension, dtype=dtype)
    latent_identity = jnp.eye(latent_dimension, dtype=dtype)
    log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=dtype))

    flat_x = jnp.reshape(x, (-1, observed_dimension))
    flat_projection = jnp.reshape(
        projection, (-1, observed_dimension, latent_dimension)
    )
    flat_noise = jnp.reshape(noise, (-1, observed_dimension, observed_dimension))

    def evaluate_observation(
        observation: Array,
        row_projection: Array,
        measurement_covariance: Array,
    ) -> tuple[Array, Array, Array, Array]:
        effective_noise = measurement_covariance + safe_jitter * observed_identity

        def evaluate_component(
            mean: Array, covariance: Array
        ) -> tuple[Array, Array, Array, Array]:
            return _component_posterior_general(
                observation,
                row_projection,
                effective_noise,
                mean,
                covariance,
                latent_identity,
                log_two_pi,
            )

        return jax.vmap(evaluate_component)(
            canonical_params.means, canonical_params.covariances
        )

    (
        flat_log_density,
        flat_conditional_mean,
        flat_conditional_covariance,
        flat_failed_pairs,
    ) = jax.vmap(evaluate_observation)(flat_x, flat_projection, flat_noise)
    raw_component_log_density = jnp.reshape(
        flat_log_density, batch_shape + (n_components,)
    )
    raw_conditional_mean = jnp.reshape(
        flat_conditional_mean,
        batch_shape + (n_components, latent_dimension),
    )
    raw_conditional_covariance = jnp.reshape(
        flat_conditional_covariance,
        batch_shape + (n_components, latent_dimension, latent_dimension),
    )
    failed_pairs = jnp.reshape(flat_failed_pairs, batch_shape + (n_components,)) | (
        ~jitter_is_valid
    )
    numerical_failure = jnp.any(failed_pairs) | (~jitter_is_valid)

    component_log_density = jnp.where(failed_pairs, -jnp.inf, raw_component_log_density)
    conditional_mean = jnp.where(failed_pairs[..., None], 0.0, raw_conditional_mean)
    conditional_covariance = jnp.where(
        failed_pairs[..., None, None], 0.0, raw_conditional_covariance
    )

    log_mixture_weight = _log_positive_with_gradual_underflow(
        canonical_params.weights
    )
    component_log_joint = component_log_density + log_mixture_weight
    every_pair_failed = jnp.all(failed_pairs, axis=-1)
    fallback_log_joint = jnp.broadcast_to(
        log_mixture_weight, component_log_joint.shape
    )
    normalization_log_joint = jnp.where(
        every_pair_failed[..., None], fallback_log_joint, component_log_joint
    )
    normalization_score = jsp.special.logsumexp(normalization_log_joint, axis=-1)
    score_samples = jnp.where(every_pair_failed, -jnp.inf, normalization_score)
    normalized_responsibilities = _exp_with_gradual_underflow(
        normalization_log_joint - normalization_score[..., None]
    )
    responsibilities = jnp.where(
        every_pair_failed[..., None],
        jnp.broadcast_to(canonical_params.weights, normalized_responsibilities.shape),
        normalized_responsibilities,
    )

    return EStep(
        component_log_density=component_log_density,
        component_log_joint=component_log_joint,
        score_samples=score_samples,
        responsibilities=responsibilities,
        conditional_mean=conditional_mean,
        conditional_covariance=conditional_covariance,
        numerical_failure=numerical_failure,
        failed_pairs=failed_pairs,
    )


def _weighted_statistics(
    e_step: EStep,
    sample_weight: Array,
    *,
    observed_dimension: int,
    sample_weight_is_valid: Array,
    jitter_is_valid: Array,
) -> GeneralSufficientStatistics:
    """Accumulate weighted ``(n, h, Q)`` for one fixed-``M`` group."""

    n_components = e_step.responsibilities.shape[-1]
    latent_dimension = e_step.conditional_mean.shape[-1]
    dtype = e_step.conditional_mean.dtype
    positive_sample_weight = (
        jnp.isfinite(sample_weight)
        & (~jnp.signbit(sample_weight))
        & _has_nonzero_floating_magnitude(sample_weight)
    )
    active_failed_pairs = e_step.failed_pairs & positive_sample_weight[:, None]
    base_numerical_failure = (
        (~sample_weight_is_valid)
        | (~jitter_is_valid)
        | jnp.any(active_failed_pairs)
    )
    if observed_dimension == 0:
        return GeneralSufficientStatistics(
            mass=jnp.zeros((n_components,), dtype=dtype),
            first_moment=jnp.zeros((n_components, latent_dimension), dtype=dtype),
            second_moment=jnp.zeros(
                (n_components, latent_dimension, latent_dimension),
                dtype=dtype,
            ),
            numerical_failure=base_numerical_failure,
            failed_pairs=active_failed_pairs,
        )

    (
        log_mass,
        normalized_log_weight,
        _,
    ) = _component_log_weight_reductions(e_step, sample_weight)
    active_effective_weight = jnp.isfinite(normalized_log_weight)
    active_conditional_mean = jnp.where(
        active_effective_weight[..., None], e_step.conditional_mean, 0.0
    )
    active_conditional_covariance = jnp.where(
        active_effective_weight[..., None, None],
        e_step.conditional_covariance,
        0.0,
    )
    weighted_conditional_mean = _multiply_by_log_scale_with_gradual_underflow(
        active_conditional_mean, normalized_log_weight[..., None]
    )
    weighted_conditional_covariance = (
        _multiply_by_log_scale_with_gradual_underflow(
            active_conditional_covariance,
            normalized_log_weight[..., None, None],
        )
    )
    sqrt_weighted_conditional_mean = (
        _multiply_by_log_scale_with_gradual_underflow(
            active_conditional_mean,
            0.5 * normalized_log_weight[..., None],
        )
    )
    normalized_first_moment = jnp.sum(weighted_conditional_mean, axis=0)
    normalized_second_moment = jnp.sum(
        weighted_conditional_covariance
        + sqrt_weighted_conditional_mean[..., :, None]
        * sqrt_weighted_conditional_mean[..., None, :],
        axis=0,
    )
    mass = _exp_with_gradual_underflow(log_mass)
    first_moment = _multiply_by_log_scale_with_gradual_underflow(
        normalized_first_moment, log_mass[:, None]
    )
    second_moment = _multiply_by_log_scale_with_gradual_underflow(
        normalized_second_moment, log_mass[:, None, None]
    )
    total_sample_mass = jnp.sum(sample_weight)
    total_component_mass = jnp.sum(mass)
    statistics_are_finite = (
        jnp.all(jnp.isfinite(mass))
        & jnp.all(jnp.isfinite(first_moment))
        & jnp.all(jnp.isfinite(second_moment))
        & jnp.isfinite(total_sample_mass)
        & jnp.isfinite(total_component_mass)
    )
    return GeneralSufficientStatistics(
        mass=mass,
        first_moment=first_moment,
        second_moment=second_moment,
        numerical_failure=base_numerical_failure | (~statistics_are_finite),
        failed_pairs=active_failed_pairs,
    )


def _component_log_weight_reductions(
    e_step: EStep, sample_weight: Array
) -> tuple[Array, Array, Array]:
    """Return log component masses and within-component normalized weights.

    Effective weights are formed as ``log(w_i) + log(q_ik)`` from the
    component log joints, so neither an exponentiated tail responsibility nor
    division by one global sample-weight maximum can discard representable
    component mass.
    """

    positive_sample_weight = (
        jnp.isfinite(sample_weight)
        & (~jnp.signbit(sample_weight))
        & _has_nonzero_floating_magnitude(sample_weight)
    )
    log_sample_weight = _log_positive_with_gradual_underflow(sample_weight)
    raw_log_responsibility = (
        e_step.component_log_joint - e_step.score_samples[:, None]
    )
    active_pair = positive_sample_weight[:, None] & (~e_step.failed_pairs)
    log_effective_weight = jnp.where(
        active_pair,
        log_sample_weight[:, None] + raw_log_responsibility,
        -jnp.inf,
    )
    log_mass = jsp.special.logsumexp(log_effective_weight, axis=0)
    finite_log_mass = jnp.isfinite(log_mass)
    safe_log_mass = jnp.where(finite_log_mass, log_mass, 0.0)
    normalized_log_weight = jnp.where(
        active_pair & finite_log_mass[None, :],
        log_effective_weight - safe_log_mass[None, :],
        -jnp.inf,
    )
    normalized_effective_weight = _exp_with_gradual_underflow(
        normalized_log_weight
    )
    return log_mass, normalized_log_weight, normalized_effective_weight


def _log_positive_with_gradual_underflow(value: Array) -> Array:
    """Take a positive log without losing an IEEE subnormal input."""

    value = jnp.asarray(value)
    ordinary_log = jnp.log(value)
    if value.dtype == jnp.dtype(jnp.float64):
        bits = jax.lax.bitcast_convert_type(value, jnp.uint64)
        fraction = bits & jnp.asarray(0x000FFFFFFFFFFFFF, dtype=jnp.uint64)
        exponent_bits = (bits >> jnp.asarray(52, dtype=jnp.uint64)) & jnp.asarray(
            0x7FF, dtype=jnp.uint64
        )
        is_subnormal = (
            (~jnp.signbit(value))
            & (exponent_bits == 0)
            & (fraction != 0)
        )
        safe_fraction = jnp.where(
            is_subnormal, fraction, jnp.asarray(1, dtype=jnp.uint64)
        )
        reconstructed_log = jnp.log(
            safe_fraction.astype(jnp.float64)
        ) - jnp.asarray(1074.0, dtype=jnp.float64) * jnp.log(
            jnp.asarray(2.0, dtype=jnp.float64)
        )
    elif value.dtype == jnp.dtype(jnp.float32):
        bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
        fraction = bits & jnp.asarray(0x007FFFFF, dtype=jnp.uint32)
        exponent_bits = (bits >> jnp.asarray(23, dtype=jnp.uint32)) & jnp.asarray(
            0xFF, dtype=jnp.uint32
        )
        is_subnormal = (
            (~jnp.signbit(value))
            & (exponent_bits == 0)
            & (fraction != 0)
        )
        safe_fraction = jnp.where(
            is_subnormal, fraction, jnp.asarray(1, dtype=jnp.uint32)
        )
        reconstructed_log = jnp.log(
            safe_fraction.astype(jnp.float32)
        ) - jnp.asarray(149.0, dtype=jnp.float32) * jnp.log(
            jnp.asarray(2.0, dtype=jnp.float32)
        )
    else:  # The canonical boundary currently permits only float32/float64.
        return ordinary_log
    return jnp.where(is_subnormal, reconstructed_log, ordinary_log)


def _exp_with_gradual_underflow(log_value: Array) -> Array:
    """Exponentiate while constructing representable subnormals by bits."""

    log_value = jnp.asarray(log_value)
    ordinary_exp = jnp.exp(log_value)
    log_two = jnp.log(jnp.asarray(2.0, dtype=log_value.dtype))
    if log_value.dtype == jnp.dtype(jnp.float64):
        min_normal_log = jnp.asarray(-1022.0, dtype=log_value.dtype) * log_two
        use_gradual_exp = log_value < min_normal_log
        shifted_log = jnp.where(
            use_gradual_exp,
            log_value + jnp.asarray(1074.0, dtype=log_value.dtype) * log_two,
            -jnp.inf,
        )
        subnormal_units = jnp.asarray(
            jnp.rint(jnp.exp(shifted_log)), dtype=jnp.uint64
        )
        gradual_exp = jax.lax.bitcast_convert_type(subnormal_units, jnp.float64)
    elif log_value.dtype == jnp.dtype(jnp.float32):
        min_normal_log = jnp.asarray(-126.0, dtype=log_value.dtype) * log_two
        use_gradual_exp = log_value < min_normal_log
        shifted_log = jnp.where(
            use_gradual_exp,
            log_value + jnp.asarray(149.0, dtype=log_value.dtype) * log_two,
            -jnp.inf,
        )
        subnormal_units = jnp.asarray(
            jnp.rint(jnp.exp(shifted_log)), dtype=jnp.uint32
        )
        gradual_exp = jax.lax.bitcast_convert_type(subnormal_units, jnp.float32)
    else:  # The canonical boundary currently permits only float32/float64.
        return ordinary_exp
    return jnp.where(use_gradual_exp, gradual_exp, ordinary_exp)


def _multiply_by_log_scale_with_gradual_underflow(
    value: Array, log_scale: Array
) -> Array:
    """Multiply by a positive log-scale without losing subnormal precision.

    Raw moments must use the unrounded component log mass: multiplying by its
    already-rounded subnormal representation can lose a material fraction of
    the result. Ordinary finite-normal products retain the native path.
    """

    value = jnp.asarray(value)
    log_scale = jnp.asarray(log_scale, dtype=value.dtype)
    scale = _exp_with_gradual_underflow(log_scale)
    absolute_value = jnp.abs(value)
    ordinary_product = value * scale
    value_is_nonzero = _has_nonzero_floating_magnitude(absolute_value)
    finite_nonzero_inputs = (
        jnp.isfinite(value)
        & jnp.isfinite(log_scale)
        & value_is_nonzero
    )
    safe_absolute_value = jnp.where(
        value_is_nonzero,
        absolute_value,
        jnp.ones_like(absolute_value),
    )
    log_absolute_value = _log_positive_with_gradual_underflow(
        safe_absolute_value
    )
    log_absolute_product = log_absolute_value + log_scale

    if value.dtype == jnp.dtype(jnp.float64):
        min_normal_log = jnp.asarray(-1022.0, dtype=value.dtype) * jnp.log(
            jnp.asarray(2.0, dtype=value.dtype)
        )
    elif value.dtype == jnp.dtype(jnp.float32):
        min_normal_log = jnp.asarray(-126.0, dtype=value.dtype) * jnp.log(
            jnp.asarray(2.0, dtype=value.dtype)
        )
    else:  # The canonical boundary currently permits only float32/float64.
        return ordinary_product

    use_log_product = finite_nonzero_inputs & (
        (log_absolute_value < min_normal_log)
        | (log_scale < min_normal_log)
        | (log_absolute_product < min_normal_log)
    )
    gradual_magnitude = _exp_with_gradual_underflow(log_absolute_product)
    gradual_product = jnp.where(
        jnp.signbit(value),
        -gradual_magnitude,
        gradual_magnitude,
    )
    return jnp.where(use_log_product, gradual_product, ordinary_product)


def _has_nonzero_floating_magnitude(value: Array) -> Array:
    """Test floating nonzero from IEEE bits so subnormals are not flushed."""

    if value.dtype == jnp.dtype(jnp.float64):
        bits = jax.lax.bitcast_convert_type(value, jnp.uint64)
        magnitude = bits & jnp.asarray(0x7FFFFFFFFFFFFFFF, dtype=jnp.uint64)
    elif value.dtype == jnp.dtype(jnp.float32):
        bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
        magnitude = bits & jnp.asarray(0x7FFFFFFF, dtype=jnp.uint32)
    else:
        return value != 0.0
    return magnitude != 0


def _canonical_fitting_inputs(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    sample_weight: Array,
) -> tuple[Params, Array, Array, Array, Array, Array, int, int, int]:
    """Prepare the exact leading-``N`` canonical representation for fitting."""

    (
        canonical_params,
        x,
        projection,
        noise,
        _,
        n_components,
        latent_dimension,
        observed_dimension,
    ) = _canonical_general_inputs(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
    )
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(
            "general fitting observations must have shape (N, M) with N >= 1; "
            f"received {x.shape}"
        )
    weights, sample_weight_is_valid = _canonical_sample_weight(
        sample_weight,
        n_samples=x.shape[0],
        dtype=canonical_params.means.dtype,
    )
    return (
        canonical_params,
        x,
        projection,
        noise,
        weights,
        sample_weight_is_valid,
        n_components,
        latent_dimension,
        observed_dimension,
    )


def sufficient_statistics_general(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    sample_weight: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> GeneralSufficientStatistics:
    """Return weighted sufficient statistics for one canonical fixed-``M`` group."""

    (
        canonical_params,
        x,
        projection,
        noise,
        weights,
        sample_weight_is_valid,
        _,
        _,
        observed_dimension,
    ) = _canonical_fitting_inputs(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        sample_weight,
    )
    e_step = posterior_components_general(
        canonical_params,
        x,
        projection,
        noise,
        factor_jitter=factor_jitter,
    )
    _, jitter_is_valid = _general_scalar_control(
        factor_jitter,
        dtype=canonical_params.means.dtype,
        name="factor_jitter",
    )
    safe_weights = jnp.where(sample_weight_is_valid, weights, jnp.zeros_like(weights))
    return _weighted_statistics(
        e_step,
        safe_weights,
        observed_dimension=observed_dimension,
        sample_weight_is_valid=sample_weight_is_valid,
        jitter_is_valid=jitter_is_valid,
    )


def one_em_step_general(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    sample_weight: Array,
    *,
    factor_jitter: float | Array = 0.0,
    covariance_ridge: float | Array = 0.0,
) -> EMStepResult:
    """Run one weighted general-projection update with whole-state rollback."""

    (
        canonical_params,
        x,
        projection,
        noise,
        weights,
        sample_weight_is_valid,
        n_components,
        latent_dimension,
        observed_dimension,
    ) = _canonical_fitting_inputs(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        sample_weight,
    )
    ridge, ridge_is_valid = _general_scalar_control(
        covariance_ridge,
        dtype=canonical_params.means.dtype,
        name="covariance_ridge",
    )
    safe_ridge = jnp.where(ridge_is_valid, ridge, 0.0)
    e_step = posterior_components_general(
        canonical_params,
        x,
        projection,
        noise,
        factor_jitter=factor_jitter,
    )
    _, jitter_is_valid = _general_scalar_control(
        factor_jitter,
        dtype=canonical_params.means.dtype,
        name="factor_jitter",
    )
    safe_weights = jnp.where(sample_weight_is_valid, weights, jnp.zeros_like(weights))
    statistics = _weighted_statistics(
        e_step,
        safe_weights,
        observed_dimension=observed_dimension,
        sample_weight_is_valid=sample_weight_is_valid,
        jitter_is_valid=jitter_is_valid,
    )

    # An empty observed space has no informative mass.  Return before forming
    # or factoring any candidate covariance: the temporary result schema can
    # represent this only as whole-state component collapse (the future public
    # boundary will use the contract's ``no_informative_weight`` validation
    # error).
    if observed_dimension == 0:
        numerical_failure = statistics.numerical_failure | (~ridge_is_valid)
        collapsed_components = jnp.where(
            numerical_failure,
            jnp.zeros((n_components,), dtype=bool),
            jnp.ones((n_components,), dtype=bool),
        )
        return EMStepResult(
            parameters=canonical_params,
            e_step=e_step,
            statistics=statistics,
            collapsed=jnp.any(collapsed_components),
            collapsed_components=collapsed_components,
            numerical_failure=numerical_failure,
        )

    (
        log_mass,
        normalized_log_weight,
        _,
    ) = _component_log_weight_reductions(e_step, safe_weights)
    mass_is_valid = (
        jnp.isfinite(log_mass)
        & _has_nonzero_floating_magnitude(statistics.mass)
    )
    finite_log_mass = jnp.where(mass_is_valid, log_mass, -jnp.inf)
    log_total_mass = jsp.special.logsumexp(finite_log_mass)
    safe_log_total_mass = jnp.where(jnp.isfinite(log_total_mass), log_total_mass, 0.0)
    candidate_weights = _exp_with_gradual_underflow(
        jnp.where(
            mass_is_valid,
            log_mass - safe_log_total_mass,
            -jnp.inf,
        )
    )

    active_effective_weight = jnp.isfinite(normalized_log_weight)
    active_conditional_mean = jnp.where(
        active_effective_weight[..., None], e_step.conditional_mean, 0.0
    )
    candidate_means = jnp.sum(
        _multiply_by_log_scale_with_gradual_underflow(
            active_conditional_mean, normalized_log_weight[..., None]
        ),
        axis=0,
    )
    centered = jnp.where(
        active_effective_weight[..., None],
        active_conditional_mean - candidate_means[None, :, :],
        0.0,
    )
    active_conditional_covariance = jnp.where(
        active_effective_weight[..., None, None],
        e_step.conditional_covariance,
        0.0,
    )
    weighted_conditional_covariance = (
        _multiply_by_log_scale_with_gradual_underflow(
            active_conditional_covariance,
            normalized_log_weight[..., None, None],
        )
    )
    weighted_centered = _multiply_by_log_scale_with_gradual_underflow(
        centered, 0.5 * normalized_log_weight[..., None]
    )
    candidate_covariances = jnp.sum(
        weighted_conditional_covariance
        + weighted_centered[..., :, None] * weighted_centered[..., None, :],
        axis=0,
    )

    latent_identity = jnp.eye(latent_dimension, dtype=canonical_params.means.dtype)
    candidate_covariances = candidate_covariances + safe_ridge * latent_identity
    half_candidate_covariances = jax.lax.optimization_barrier(
        0.5 * candidate_covariances
    )
    candidate_covariances = (
        half_candidate_covariances
        + half_candidate_covariances.swapaxes(-1, -2)
    )
    candidate_factors = jax.lax.linalg.cholesky(
        candidate_covariances, symmetrize_input=False
    )
    finite_parameters = (
        jnp.isfinite(candidate_weights)
        & (~jnp.signbit(candidate_weights))
        & _has_nonzero_floating_magnitude(candidate_weights)
        & jnp.all(jnp.isfinite(candidate_means), axis=-1)
        & jnp.all(jnp.isfinite(candidate_covariances), axis=(-2, -1))
        & jnp.all(jnp.isfinite(candidate_factors), axis=(-2, -1))
        & jnp.all(
            jnp.diagonal(candidate_factors, axis1=-2, axis2=-1) > 0.0,
            axis=-1,
        )
    )

    numerical_failure = statistics.numerical_failure | (~ridge_is_valid)
    proposed_collapsed_components = (~mass_is_valid) | (~finite_parameters)
    collapsed_components = jnp.where(
        numerical_failure,
        jnp.zeros((n_components,), dtype=bool),
        proposed_collapsed_components,
    )
    collapsed = jnp.any(collapsed_components)
    rollback = numerical_failure | collapsed

    candidate_params = Params(candidate_weights, candidate_means, candidate_covariances)
    returned_params = Params(
        weights=jnp.where(rollback, canonical_params.weights, candidate_params.weights),
        means=jnp.where(rollback, canonical_params.means, candidate_params.means),
        covariances=jnp.where(
            rollback,
            canonical_params.covariances,
            candidate_params.covariances,
        ),
    )
    return EMStepResult(
        parameters=returned_params,
        e_step=e_step,
        statistics=statistics,
        collapsed=collapsed,
        collapsed_components=collapsed_components,
        numerical_failure=numerical_failure,
    )


__all__ = [
    "GeneralSufficientStatistics",
    "one_em_step_general",
    "posterior_components_general",
    "sufficient_statistics_general",
]
