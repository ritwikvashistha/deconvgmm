"""Temporary pure-JAX kernels for identity-projection Extreme Deconvolution.

This module is a development target for the versioned mathematical contract. It
is deliberately outside the future installable package namespace and is not a
public API.  Callers are expected to pass canonical, already-validated arrays;
only static shape checks that protect the numerical equations live here.

The implementation uses one Cholesky factor per observation/component pair for
the component log density and posterior solves.  It never forms a covariance
inverse explicitly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp


Array = jax.Array


class Params(NamedTuple):
    """Canonical identity-XD mixture parameters.

    Fields have shapes ``(K,)``, ``(K, D)``, and ``(K, D, D)``.  A
    ``NamedTuple`` is used so the temporary container is immutable and is a JAX
    PyTree without registration or host conversion.
    """

    weights: Array
    means: Array
    covariances: Array


class EStep(NamedTuple):
    """Density, posterior quantities, and device-resident failure status.

    ``failed_pairs`` has shape ``B + (K,)`` and identifies an observation /
    component factorization or arithmetic failure. ``numerical_failure`` is its
    scalar reduction and is suitable for a compiled caller's control flow.
    """

    component_log_density: Array
    component_log_joint: Array
    score_samples: Array
    responsibilities: Array
    conditional_mean: Array
    conditional_covariance: Array
    numerical_failure: Array
    failed_pairs: Array


class SufficientStatistics(NamedTuple):
    """Component effective masses and first/second conditional moments."""

    mass: Array
    first_moment: Array
    second_moment: Array


class EMStepResult(NamedTuple):
    """One proposed exact EM update and its failure statuses.

    If ``collapsed`` or ``numerical_failure`` is true, ``parameters`` is the
    unchanged input state. This whole-step rollback prevents a partial or
    numerically invalid update from being mistaken for success. Collapse marks
    invalid component mass/proposals; numerical failure marks factorization,
    pair arithmetic, or scalar-configuration failure.
    """

    parameters: Params
    e_step: EStep
    statistics: SufficientStatistics
    collapsed: Array
    collapsed_components: Array
    numerical_failure: Array


def _validate_variances_eager(values: Array, *, representation: str) -> None:
    """Validate adapter values at an eager, host-controlled input boundary."""

    if bool(jnp.any(~jnp.isfinite(values) | (values < 0.0))):
        raise ValueError(
            f"{representation} variances must be finite and nonnegative"
        )


def isotropic_covariance(variances: Array, dimension: int) -> Array:
    """Expand explicit isotropic variances onto covariance diagonals only.

    This temporary adapter is an eager boundary helper, not a JIT-kernel
    guarantee. It accepts either one scalar or a one-dimensional fitting batch;
    the result appends ``(dimension, dimension)``. Values are variances, not
    standard deviations. Rank greater than one, including the deliberately
    ambiguous fitting shape ``(N, 1)``, is rejected.
    """

    values = jnp.asarray(variances)
    if dimension < 1:
        raise ValueError(f"dimension must be positive; received {dimension}")
    if values.ndim > 1:
        raise ValueError(
            "isotropic variances must be a scalar or have fitting shape (N,); "
            "received ambiguous shape "
            f"{values.shape}"
        )
    _validate_variances_eager(values, representation="isotropic")
    identity = jnp.eye(dimension, dtype=values.dtype)
    return values[..., None, None] * identity


def diagonal_covariance(variances: Array) -> Array:
    """Expand per-coordinate variances at an eager, non-JIT input boundary."""

    values = jnp.asarray(variances)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError(
            "diagonal variances must have shape B + (D,) with D >= 1; "
            f"received {values.shape}"
        )
    _validate_variances_eager(values, representation="diagonal")
    dimension = values.shape[-1]
    identity = jnp.eye(dimension, dtype=values.dtype)
    return values[..., :, None] * identity


def full_covariance(covariances: Array) -> Array:
    """Return an explicitly supplied full-covariance representation.

    This is an eager input-boundary helper, not a JIT-kernel guarantee. It
    deliberately performs no broadcasting and no value repair, and currently
    rejects only a statically nonsquare trailing matrix shape.
    """

    values = jnp.asarray(covariances)
    if (
        values.ndim < 2
        or values.shape[-2] < 1
        or values.shape[-2] != values.shape[-1]
    ):
        raise ValueError(
            "full covariances must have shape B + (D, D) with D >= 1; "
            f"received {values.shape}"
        )
    return values


def _canonical_inputs(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
) -> tuple[Params, Array, Array, tuple[int, ...], int, int]:
    """Coerce arrays to the parameter dtype and check canonical static shapes."""

    means = jnp.asarray(params.means)
    if means.ndim != 2 or means.shape[0] < 1 or means.shape[1] < 1:
        raise ValueError(
            "means must have shape (K, D) with K,D >= 1; "
            f"received {means.shape}"
        )
    if not jnp.issubdtype(means.dtype, jnp.floating):
        raise TypeError(
            "parameter means must have a floating dtype; "
            f"received {means.dtype}"
        )

    dtype = means.dtype
    weights = jnp.asarray(params.weights, dtype=dtype)
    covariances = jnp.asarray(params.covariances, dtype=dtype)
    n_components, dimension = means.shape
    if weights.shape != (n_components,):
        raise ValueError(
            f"weights must have shape {(n_components,)}; received {weights.shape}"
        )
    expected_covariance_shape = (n_components, dimension, dimension)
    if covariances.shape != expected_covariance_shape:
        raise ValueError(
            "parameter covariances must have shape "
            f"{expected_covariance_shape}; received {covariances.shape}"
        )

    x = jnp.asarray(observations, dtype=dtype)
    noise = jnp.asarray(measurement_covariances, dtype=dtype)
    if x.ndim < 1 or x.shape[-1] != dimension:
        raise ValueError(
            f"observations must have shape B + {(dimension,)}; received {x.shape}"
        )
    batch_shape = x.shape[:-1]
    expected_noise_shape = batch_shape + (dimension, dimension)
    if noise.shape != expected_noise_shape:
        raise ValueError(
            "measurement covariances must have canonical shape "
            f"{expected_noise_shape}; received {noise.shape}"
        )

    canonical_params = Params(weights, means, covariances)
    return canonical_params, x, noise, batch_shape, n_components, dimension


def _real_scalar_control(
    value: object,
    *,
    dtype: jnp.dtype,
    name: str,
) -> tuple[Array, Array]:
    """Convert a scalar control and report whether its value survived safely.

    Shape and dtype errors are static failures.  The returned device-resident
    flag also checks sign and finiteness in the control's original dtype, then
    rejects nonzero values that become zero (or nonfinite) in the selected
    computation dtype.  This keeps the check available under JIT tracing.
    """

    try:
        original = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be a real numeric rank-zero scalar"
        ) from error
    if original.ndim != 0:
        raise ValueError(
            f"{name} must be a rank-zero scalar; received shape {original.shape}"
        )
    is_real_numeric = jnp.issubdtype(
        original.dtype, jnp.integer
    ) or jnp.issubdtype(original.dtype, jnp.floating)
    if not is_real_numeric:
        raise TypeError(
            f"{name} must be a real numeric scalar; received dtype "
            f"{original.dtype}"
        )
    converted = jnp.asarray(original, dtype=dtype)
    is_valid = _nonnegative_conversion_is_valid(original, converted)
    return converted, is_valid


def _has_nonzero_scalar_magnitude(value: Array) -> Array:
    """Test scalar floating magnitude from bits so FTZ cannot hide it."""

    if value.dtype == jnp.dtype(jnp.float64):
        bits = jax.lax.bitcast_convert_type(value, jnp.uint64)
        magnitude = bits & jnp.asarray(0x7FFFFFFFFFFFFFFF, dtype=jnp.uint64)
    elif value.dtype == jnp.dtype(jnp.float32):
        bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
        magnitude = bits & jnp.asarray(0x7FFFFFFF, dtype=jnp.uint32)
    else:
        return value != jnp.asarray(0, dtype=value.dtype)
    return magnitude != 0


def _nonnegative_conversion_is_valid(original: Array, converted: Array) -> Array:
    """Validate sign and positive-to-zero conversion without FTZ compares."""

    def scalar_is_valid(pair: tuple[Array, Array]) -> Array:
        source, selected = pair
        source_is_nonzero = _has_nonzero_scalar_magnitude(source)
        selected_is_nonzero = _has_nonzero_scalar_magnitude(selected)
        source_is_negative = jax.lax.cond(
            source_is_nonzero,
            lambda _: jnp.signbit(source),
            lambda _: jnp.asarray(False),
            operand=None,
        )
        conversion_underflow = jax.lax.cond(
            source_is_nonzero,
            lambda _: jax.lax.cond(
                selected_is_nonzero,
                lambda _: jnp.asarray(False),
                lambda _: jnp.asarray(True),
                operand=None,
            ),
            lambda _: jnp.asarray(False),
            operand=None,
        )
        return (
            jnp.isfinite(source)
            & (~source_is_negative)
            & jnp.isfinite(selected)
            & (~conversion_underflow)
        )

    if original.ndim == 0:
        return scalar_is_valid((original, converted))
    flat_valid = jax.lax.map(
        scalar_is_valid,
        (jnp.ravel(original), jnp.ravel(converted)),
    )
    return jnp.reshape(flat_valid, original.shape)


def _component_posterior(
    observation: Array,
    effective_noise: Array,
    mean: Array,
    covariance: Array,
    identity: Array,
    log_two_pi: Array,
) -> tuple[Array, Array, Array, Array]:
    """Evaluate one component and expose failed factor/arithmetic as status."""

    total_covariance = covariance + effective_noise
    # Inputs are already canonical and symmetric.  Disabling JAX's default
    # ``(A + A.T) / 2`` repair avoids overflowing finite matrices near the
    # selected dtype limit while retaining the boundary's symmetry policy.
    factor = jax.lax.linalg.cholesky(
        total_covariance, symmetrize_input=False
    )
    residual = observation - mean

    whitened = jsp.linalg.solve_triangular(factor, residual, lower=True)
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
    log_density = -0.5 * (
        observation.shape[-1] * log_two_pi
        + log_determinant
        + jnp.vdot(whitened, whitened)
    )

    # Solve T X = V.T and transpose the result to obtain K = V T^{-1}.
    first_solve = jsp.linalg.solve_triangular(
        factor, jnp.swapaxes(covariance, -1, -2), lower=True
    )
    inverse_times_v_transpose = jsp.linalg.solve_triangular(
        jnp.swapaxes(factor, -1, -2), first_solve, lower=False
    )
    gain = jnp.swapaxes(inverse_times_v_transpose, -1, -2)
    gain = jnp.where(jnp.all(effective_noise == 0.0), identity, gain)

    conditional_mean = mean + gain @ residual

    # Evaluate I - K as S_eff T_eff^{-1}, rather than subtracting a gain near
    # identity.  The latter can turn roundoff into an enormous absolute Joseph
    # residual when V is close to the largest finite float32 scale.  These
    # additional solves reuse the same factor and give exact zero for exact
    # zero measurement noise.
    first_noise_solve = jsp.linalg.solve_triangular(
        factor, jnp.swapaxes(effective_noise, -1, -2), lower=True
    )
    inverse_times_noise_transpose = jsp.linalg.solve_triangular(
        jnp.swapaxes(factor, -1, -2), first_noise_solve, lower=False
    )
    residual_operator = jnp.swapaxes(
        inverse_times_noise_transpose, -1, -2
    )
    conditional_covariance = (
        residual_operator
        @ covariance
        @ jnp.swapaxes(residual_operator, -1, -2)
        + gain @ effective_noise @ jnp.swapaxes(gain, -1, -2)
    )
    conditional_covariance = 0.5 * (
        conditional_covariance
        + jnp.swapaxes(conditional_covariance, -1, -2)
    )
    pair_is_valid = (
        jnp.all(jnp.isfinite(factor))
        & jnp.all(jnp.diag(factor) > 0.0)
        & jnp.isfinite(log_density)
        & jnp.all(jnp.isfinite(conditional_mean))
        & jnp.all(jnp.isfinite(conditional_covariance))
    )
    return (
        log_density,
        conditional_mean,
        conditional_covariance,
        ~pair_is_valid,
    )


def posterior_components(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> EStep:
    """Return component densities, probabilities, and latent posterior moments.

    Inputs use the canonical full-covariance representation.  ``observations``
    may have shape ``(D,)`` or a batch shape ``B + (D,)``; outputs retain ``B``.
    Responsibility normalization is performed entirely in log space.
    """

    (
        canonical_params,
        x,
        noise,
        batch_shape,
        n_components,
        dimension,
    ) = _canonical_inputs(params, observations, measurement_covariances)
    dtype = canonical_params.means.dtype
    jitter, jitter_is_valid = _real_scalar_control(
        factor_jitter, dtype=dtype, name="factor_jitter"
    )
    safe_jitter = jnp.where(jitter_is_valid, jitter, 0.0)
    identity = jnp.eye(dimension, dtype=dtype)
    log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=dtype))

    flat_x = jnp.reshape(x, (-1, dimension))
    flat_noise = jnp.reshape(noise, (-1, dimension, dimension))

    def evaluate_observation(
        observation: Array, measurement_covariance: Array
    ) -> tuple[Array, Array, Array, Array]:
        effective_noise = measurement_covariance + safe_jitter * identity

        def evaluate_component(
            mean: Array, covariance: Array
        ) -> tuple[Array, Array, Array, Array]:
            return _component_posterior(
                observation,
                effective_noise,
                mean,
                covariance,
                identity,
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
    ) = jax.vmap(evaluate_observation)(flat_x, flat_noise)
    raw_component_log_density = jnp.reshape(
        flat_log_density, batch_shape + (n_components,)
    )
    raw_conditional_mean = jnp.reshape(
        flat_conditional_mean, batch_shape + (n_components, dimension)
    )
    raw_conditional_covariance = jnp.reshape(
        flat_conditional_covariance,
        batch_shape + (n_components, dimension, dimension),
    )
    failed_pairs = jnp.reshape(
        flat_failed_pairs, batch_shape + (n_components,)
    ) | (~jitter_is_valid)
    numerical_failure = jnp.any(failed_pairs)

    # Failed pairs use explicit sentinels for safe downstream reduction. Status
    # remains authoritative: these placeholders are never reported as success.
    component_log_density = jnp.where(
        failed_pairs, -jnp.inf, raw_component_log_density
    )
    conditional_mean = jnp.where(
        failed_pairs[..., None], 0.0, raw_conditional_mean
    )
    conditional_covariance = jnp.where(
        failed_pairs[..., None, None], 0.0, raw_conditional_covariance
    )

    component_log_joint = component_log_density + jnp.log(
        canonical_params.weights
    )
    every_pair_failed = jnp.all(failed_pairs, axis=-1)
    fallback_log_joint = jnp.broadcast_to(
        jnp.log(canonical_params.weights), component_log_joint.shape
    )
    normalization_log_joint = jnp.where(
        every_pair_failed[..., None], fallback_log_joint, component_log_joint
    )
    normalization_score = jsp.special.logsumexp(
        normalization_log_joint, axis=-1
    )
    score_samples = jnp.where(
        every_pair_failed, -jnp.inf, normalization_score
    )
    log_responsibilities = (
        normalization_log_joint - normalization_score[..., None]
    )
    responsibilities = jnp.exp(log_responsibilities)

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


def marginalized_posterior(e_step: EStep) -> tuple[Array, Array]:
    """Marginalize component posterior moments by total expectation/covariance."""

    posterior_mean = jnp.sum(
        e_step.responsibilities[..., :, None] * e_step.conditional_mean,
        axis=-2,
    )
    centered = e_step.conditional_mean - posterior_mean[..., None, :]
    second_centered = (
        e_step.conditional_covariance
        + centered[..., :, :, None] * centered[..., :, None, :]
    )
    posterior_covariance = jnp.sum(
        e_step.responsibilities[..., :, None, None] * second_centered,
        axis=-3,
    )
    posterior_covariance = 0.5 * (
        posterior_covariance + jnp.swapaxes(posterior_covariance, -1, -2)
    )
    return posterior_mean, posterior_covariance


def sufficient_statistics(e_step: EStep) -> SufficientStatistics:
    """Accumulate ``(n, h, G)`` across every leading observation batch axis."""

    n_components = e_step.responsibilities.shape[-1]
    dimension = e_step.conditional_mean.shape[-1]
    responsibilities = jnp.reshape(
        e_step.responsibilities, (-1, n_components)
    )
    conditional_mean = jnp.reshape(
        e_step.conditional_mean, (-1, n_components, dimension)
    )
    conditional_covariance = jnp.reshape(
        e_step.conditional_covariance,
        (-1, n_components, dimension, dimension),
    )

    conditional_second_moment = (
        conditional_covariance
        + conditional_mean[..., :, None] * conditional_mean[..., None, :]
    )
    mass = jnp.sum(responsibilities, axis=0)
    first_moment = jnp.sum(
        responsibilities[..., None] * conditional_mean, axis=0
    )
    second_moment = jnp.sum(
        responsibilities[..., None, None] * conditional_second_moment, axis=0
    )
    return SufficientStatistics(mass, first_moment, second_moment)


def em_step(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
    covariance_ridge: float | Array = 0.0,
) -> EMStepResult:
    """Run one centered two-pass M-step with whole-step collapse rollback.

    Rank-invalid scalar controls raise during static tracing. Device-dependent
    nonfinite/negative controls, E-step failures, nonpositive mass, and invalid
    proposed parameters instead set their distinct status fields and return the
    input parameters unchanged. The host API will later turn these statuses into
    the contract's default actionable errors.
    """

    canonical_params, _, _, _, n_components, dimension = _canonical_inputs(
        params, observations, measurement_covariances
    )
    ridge, ridge_is_valid = _real_scalar_control(
        covariance_ridge,
        dtype=canonical_params.means.dtype,
        name="covariance_ridge",
    )
    safe_ridge = jnp.where(ridge_is_valid, ridge, 0.0)
    e_step = posterior_components(
        canonical_params,
        observations,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    statistics = sufficient_statistics(e_step)

    mass_is_valid = jnp.isfinite(statistics.mass) & (statistics.mass > 0.0)
    safe_mass = jnp.where(mass_is_valid, statistics.mass, 1.0)
    finite_mass = jnp.where(mass_is_valid, statistics.mass, 0.0)
    total_mass = jnp.sum(finite_mass)
    safe_total_mass = jnp.where(
        jnp.isfinite(total_mass) & (total_mass > 0.0), total_mass, 1.0
    )

    candidate_weights = finite_mass / safe_total_mass
    candidate_means = statistics.first_moment / safe_mass[:, None]

    conditional_mean = jnp.reshape(
        e_step.conditional_mean, (-1, n_components, dimension)
    )
    conditional_covariance = jnp.reshape(
        e_step.conditional_covariance,
        (-1, n_components, dimension, dimension),
    )
    responsibilities = jnp.reshape(
        e_step.responsibilities, (-1, n_components)
    )
    centered = conditional_mean - candidate_means[None, :, :]
    centered_second_moment = (
        conditional_covariance
        + centered[..., :, None] * centered[..., None, :]
    )
    candidate_covariances = jnp.sum(
        responsibilities[..., None, None] * centered_second_moment, axis=0
    ) / safe_mass[:, None, None]

    identity = jnp.eye(dimension, dtype=canonical_params.means.dtype)
    candidate_covariances = candidate_covariances + safe_ridge * identity
    candidate_covariances = 0.5 * (
        candidate_covariances
        + jnp.swapaxes(candidate_covariances, -1, -2)
    )

    candidate_factors = jax.lax.linalg.cholesky(
        candidate_covariances, symmetrize_input=False
    )
    finite_parameters = (
        jnp.isfinite(candidate_weights)
        & (candidate_weights > 0.0)
        & jnp.all(jnp.isfinite(candidate_means), axis=-1)
        & jnp.all(jnp.isfinite(candidate_covariances), axis=(-2, -1))
        & jnp.all(jnp.isfinite(candidate_factors), axis=(-2, -1))
        & jnp.all(jnp.diagonal(candidate_factors, axis1=-2, axis2=-1) > 0.0, axis=-1)
    )
    numerical_failure = e_step.numerical_failure | (~ridge_is_valid)
    proposed_collapsed_components = (~mass_is_valid) | (~finite_parameters)
    collapsed_components = jnp.where(
        numerical_failure,
        jnp.zeros_like(proposed_collapsed_components),
        proposed_collapsed_components,
    )
    collapsed = jnp.any(collapsed_components)
    rollback = numerical_failure | collapsed

    candidate_params = Params(
        candidate_weights, candidate_means, candidate_covariances
    )
    returned_params = Params(
        jnp.where(rollback, canonical_params.weights, candidate_params.weights),
        jnp.where(rollback, canonical_params.means, candidate_params.means),
        jnp.where(
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


# Descriptive aliases make notebooks readable without creating a second
# implementation.  They remain temporary for the same reason as the base names.
IdentityXDParameters = Params
IdentityXDEstep = EStep
IdentityXDSufficientStatistics = SufficientStatistics
IdentityXDEMStepResult = EMStepResult
identity_e_step = posterior_components
identity_em_step = em_step


__all__ = [
    "EMStepResult",
    "EStep",
    "IdentityXDEMStepResult",
    "IdentityXDEstep",
    "IdentityXDParameters",
    "IdentityXDSufficientStatistics",
    "Params",
    "SufficientStatistics",
    "diagonal_covariance",
    "em_step",
    "full_covariance",
    "identity_e_step",
    "identity_em_step",
    "isotropic_covariance",
    "marginalized_posterior",
    "posterior_components",
    "sufficient_statistics",
]
