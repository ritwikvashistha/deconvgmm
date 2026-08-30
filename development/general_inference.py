"""Temporary scoring and posterior conveniences for general-projection XD.

The fixed-``M`` functions in this module are pure JAX numerical leaves over
canonical arrays.  Their eager grouped counterparts operate on the deterministic
mask groups prepared by :mod:`development.general_validation`.  Neither surface
is a public API.

Detailed factorization status remains authoritative in
:func:`development.general_xd.posterior_components_general` and
:func:`development.general_grouped.posterior_components_grouped`.  These
conveniences make any failed row unmistakable by returning NaN row values and
the prediction label ``-1``.  Weighted reductions exclude failed rows only when
their stored weight is exactly zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from .general_grouped import (
    GroupedPosteriorResult,
    posterior_components_grouped,
)
from .general_validation import GroupedGeneralInputs
from .general_xd import (
    _general_scalar_control,
    _has_nonzero_floating_magnitude,
    _log_positive_with_gradual_underflow,
    _multiply_by_log_scale_with_gradual_underflow,
    posterior_components_general,
)
from .identity_xd import (
    EStep,
    Params,
    _nonnegative_conversion_is_valid,
    marginalized_posterior,
)


Array = jax.Array


def _failed_rows(e_step: EStep) -> Array:
    """Reduce pair status without confusing an empty batch with success."""

    return jnp.any(e_step.failed_pairs, axis=-1)


def _sentinel_scores(e_step: EStep) -> Array:
    """Return per-row scores with the contracted convenience sentinel."""

    return jnp.where(
        _failed_rows(e_step),
        jnp.asarray(jnp.nan, dtype=e_step.score_samples.dtype),
        e_step.score_samples,
    )


def _sentinel_probabilities(e_step: EStep) -> Array:
    """Return probabilities with every failed row replaced by NaNs."""

    return jnp.where(
        _failed_rows(e_step)[..., None],
        jnp.asarray(jnp.nan, dtype=e_step.responsibilities.dtype),
        e_step.responsibilities,
    )


def _sentinel_posterior(e_step: EStep) -> tuple[Array, Array]:
    """Marginalize component moments and mark every failed row."""

    failed = _failed_rows(e_step)
    mean, covariance = marginalized_posterior(e_step)
    nan = jnp.asarray(jnp.nan, dtype=mean.dtype)
    return (
        jnp.where(failed[..., None], nan, mean),
        jnp.where(failed[..., None, None], nan, covariance),
    )


def _canonical_reduction_weight(
    sample_weight: Array | None,
    *,
    batch_shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Prepare an exact-shape selected-dtype reduction weight.

    This is deliberately a numerical-leaf check: shape and numeric dtype are
    static errors, while value-domain failures stay device resident and poison
    the scalar reduction.  The eager general boundary remains responsible for
    actionable public validation and source-dtype precision diagnostics.
    """

    if sample_weight is None:
        return (
            jnp.ones(batch_shape, dtype=dtype),
            jnp.asarray(True),
        )

    try:
        original = jnp.asarray(sample_weight)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "sample_weight must be a real numeric array with shape "
            f"{batch_shape}"
        ) from error
    if original.shape != batch_shape:
        raise ValueError(
            "sample_weight must have exact batch shape "
            f"{batch_shape}; received {original.shape}"
        )
    is_real_numeric = jnp.issubdtype(
        original.dtype, jnp.integer
    ) or jnp.issubdtype(original.dtype, jnp.floating)
    if not is_real_numeric:
        raise TypeError(
            "sample_weight must have a real numeric dtype; received "
            f"{original.dtype}"
        )

    converted = jnp.asarray(original, dtype=dtype)
    is_valid = jnp.all(
        _nonnegative_conversion_is_valid(original, converted)
    )
    return converted, is_valid


def _positive_weight(weight: Array) -> Array:
    """Identify finite selected-dtype positive weights, including subnormals."""

    return (
        jnp.isfinite(weight)
        & (~jnp.signbit(weight))
        & _has_nonzero_floating_magnitude(weight)
    )


@jax.custom_jvp
def _value_from_bits_with_gradient(value_bits: Array, carrier: Array) -> Array:
    """Return exact IEEE value bits while differentiating through ``carrier``."""

    if carrier.dtype == jnp.dtype(jnp.float64):
        return jax.lax.bitcast_convert_type(value_bits, jnp.float64)
    if carrier.dtype == jnp.dtype(jnp.float32):
        return jax.lax.bitcast_convert_type(value_bits, jnp.float32)
    return carrier


@_value_from_bits_with_gradient.defjvp
def _value_from_bits_with_gradient_jvp(primals, tangents):
    value_bits, carrier = primals
    _, carrier_tangent = tangents
    return _value_from_bits_with_gradient(value_bits, carrier), carrier_tangent


def _weighted_total(
    scores: Array,
    weight: Array,
    *,
    included: Array,
) -> Array:
    """Reduce weighted scores without premature underflow or overflow.

    A positive failed row keeps its NaN score and therefore poisons the result;
    an excluded or exact-zero row contributes exact zero.  Finite participating
    terms use native products first, an exact shared-weight centering second,
    and a gradual power-of-two representation only as a final fallback.  This
    permits finite cancellation even when forming an individual
    ``weight * score`` would overflow, while retaining adjacent-weight low bits
    and representable subnormal results.
    """

    positive = _positive_weight(weight) & included
    diagnostic_scores = jax.lax.stop_gradient(scores)
    diagnostic_native_terms = _native_weighted_terms(
        diagnostic_scores, weight, active=positive
    )
    diagnostic_native_total = jnp.sum(diagnostic_native_terms)
    diagnostic_centered_total, centered_is_finite = _centered_weighted_total(
        diagnostic_scores,
        weight,
        active=positive,
    )
    diagnostic_scaled_total, diagnostic_reference_exponent = (
        _power_two_scaled_sum(
            diagnostic_scores,
            weight,
            active=positive,
        )
    )
    diagnostic_stable_bits = _ldexp_value_bits(
        diagnostic_scaled_total,
        diagnostic_reference_exponent,
    )
    if scores.dtype == jnp.dtype(jnp.float64):
        stable_magnitude_mask = jnp.asarray(
            0x7FFFFFFFFFFFFFFF, dtype=jnp.uint64
        )
    else:
        stable_magnitude_mask = jnp.asarray(0x7FFFFFFF, dtype=jnp.uint32)
    stable_is_zero = (
        diagnostic_stable_bits & stable_magnitude_mask
    ) == 0
    native_is_nonzero = _has_nonzero_floating_magnitude(
        diagnostic_native_total
    )
    native_is_usable = (
        jnp.all(jnp.isfinite(diagnostic_native_terms))
        & jnp.isfinite(diagnostic_native_total)
        & (native_is_nonzero | stable_is_zero)
    )
    centered_is_nonzero = _has_nonzero_floating_magnitude(
        diagnostic_centered_total
    )
    centered_is_usable = centered_is_finite & (
        centered_is_nonzero | stable_is_zero
    )

    def native_branch(_: None) -> Array:
        return jnp.sum(_native_weighted_terms(scores, weight, active=positive))

    def centered_branch(_: None) -> Array:
        return _centered_weighted_total(
            scores,
            weight,
            active=positive,
        )[0]

    def power_branch(_: None) -> Array:
        scaled_total, reference_exponent = _power_two_scaled_sum(
            scores,
            weight,
            active=positive,
        )
        return _ldexp_gradual(scaled_total, reference_exponent)

    def fallback_branch(_: None) -> Array:
        return jax.lax.cond(
            centered_is_usable,
            centered_branch,
            power_branch,
            operand=None,
        )

    gradient_carrier = jax.lax.cond(
        native_is_usable,
        native_branch,
        fallback_branch,
        operand=None,
    )
    if scores.dtype == jnp.dtype(jnp.float64):
        unsigned_dtype = jnp.uint64
    else:
        unsigned_dtype = jnp.uint32
    native_bits = jax.lax.bitcast_convert_type(
        diagnostic_native_total, unsigned_dtype
    )
    centered_bits = jax.lax.bitcast_convert_type(
        diagnostic_centered_total, unsigned_dtype
    )
    fallback_bits = jax.lax.select(
        centered_is_usable,
        centered_bits,
        diagnostic_stable_bits,
    )
    selected_bits = jax.lax.select(
        native_is_usable,
        native_bits,
        fallback_bits,
    )
    return _value_from_bits_with_gradient(selected_bits, gradient_carrier)


def _native_weighted_terms(
    values: Array,
    weight: Array,
    *,
    active: Array,
) -> Array:
    """Form exact native products, repairing only gradual underflow."""

    safe_value = jnp.where(active, values, jnp.zeros_like(values))
    safe_weight = jnp.where(active, weight, jnp.zeros_like(weight))
    direct_terms = safe_value * safe_weight
    log_weight = _log_positive_with_gradual_underflow(weight)
    safe_log_weight = jnp.where(
        active,
        log_weight,
        jnp.zeros_like(log_weight),
    )
    gradual_terms = _multiply_by_log_scale_with_gradual_underflow(
        safe_value,
        safe_log_weight,
    )
    value_is_nonzero = _has_nonzero_floating_magnitude(safe_value)
    direct_is_nonzero = _has_nonzero_floating_magnitude(direct_terms)
    direct_underflowed = (
        active
        & jnp.isfinite(safe_value)
        & value_is_nonzero
        & (~direct_is_nonzero)
    )
    return jnp.where(direct_underflowed, gradual_terms, direct_terms)


def _centered_weighted_total(
    values: Array,
    weight: Array,
    *,
    active: Array,
) -> tuple[Array, Array]:
    """Factor a shared exact weight before forming overflowing products."""

    if values.size == 0:
        return jnp.asarray(0.0, dtype=values.dtype), jnp.asarray(True)

    has_active = jnp.any(active)
    safe_value = jnp.where(active, values, jnp.zeros_like(values))
    safe_weight = jnp.where(active, weight, jnp.zeros_like(weight))
    anchor = jnp.min(jnp.where(active, weight, jnp.inf))
    safe_anchor = jnp.where(has_active, anchor, jnp.asarray(0.0, weight.dtype))
    value_sum = jnp.sum(safe_value)
    base_term = _native_weighted_terms(
        value_sum,
        safe_anchor,
        active=has_active,
    )
    residual_weight = jnp.where(
        active,
        safe_weight - safe_anchor,
        jnp.zeros_like(weight),
    )
    residual_active = active & _positive_weight(residual_weight)
    residual_terms = _native_weighted_terms(
        safe_value,
        residual_weight,
        active=residual_active,
    )
    centered_total = base_term + jnp.sum(residual_terms)
    centered_is_finite = (
        jnp.isfinite(value_sum)
        & jnp.isfinite(base_term)
        & jnp.all(jnp.isfinite(residual_terms))
        & jnp.isfinite(centered_total)
    )
    return centered_total, centered_is_finite


def _frexp_gradual(value: Array) -> tuple[Array, Array]:
    """Decompose floats while preserving IEEE subnormal significands."""

    native_mantissa, native_exponent = jnp.frexp(value)
    if value.dtype == jnp.dtype(jnp.float64):
        bits = jax.lax.bitcast_convert_type(value, jnp.uint64)
        fraction = bits & jnp.asarray(0x000FFFFFFFFFFFFF, dtype=jnp.uint64)
        exponent_bits = (
            bits >> jnp.asarray(52, dtype=jnp.uint64)
        ) & jnp.asarray(0x7FF, dtype=jnp.uint64)
        is_subnormal = (exponent_bits == 0) & (fraction != 0)
        safe_fraction = jnp.where(
            is_subnormal,
            fraction,
            jnp.asarray(1, dtype=jnp.uint64),
        )
        fraction_mantissa, fraction_exponent = jnp.frexp(
            safe_fraction.astype(jnp.float64)
        )
        subnormal_exponent = fraction_exponent - jnp.asarray(
            1074, dtype=fraction_exponent.dtype
        )
    elif value.dtype == jnp.dtype(jnp.float32):
        bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
        fraction = bits & jnp.asarray(0x007FFFFF, dtype=jnp.uint32)
        exponent_bits = (
            bits >> jnp.asarray(23, dtype=jnp.uint32)
        ) & jnp.asarray(0xFF, dtype=jnp.uint32)
        is_subnormal = (exponent_bits == 0) & (fraction != 0)
        safe_fraction = jnp.where(
            is_subnormal,
            fraction,
            jnp.asarray(1, dtype=jnp.uint32),
        )
        fraction_mantissa, fraction_exponent = jnp.frexp(
            safe_fraction.astype(jnp.float32)
        )
        subnormal_exponent = fraction_exponent - jnp.asarray(
            149, dtype=fraction_exponent.dtype
        )
    else:
        return native_mantissa, native_exponent

    subnormal_mantissa = jnp.where(
        jnp.signbit(value),
        -fraction_mantissa,
        fraction_mantissa,
    )
    return (
        jnp.where(is_subnormal, subnormal_mantissa, native_mantissa),
        jnp.where(is_subnormal, subnormal_exponent, native_exponent),
    )


def _ldexp_value_bits(mantissa: Array, exponent: Array) -> Array:
    """Return exact IEEE bits for ``mantissa * 2**exponent``."""

    native = jnp.ldexp(mantissa, exponent)
    normalized_mantissa, mantissa_exponent = _frexp_gradual(mantissa)
    combined_exponent = mantissa_exponent + exponent
    mantissa_is_nonzero = _has_nonzero_floating_magnitude(mantissa)
    if mantissa.dtype == jnp.dtype(jnp.float64):
        use_gradual = (
            jnp.isfinite(mantissa)
            & mantissa_is_nonzero
            & (
                combined_exponent
                <= jnp.asarray(-1022, dtype=combined_exponent.dtype)
            )
        )
        unit_exponent = combined_exponent + jnp.asarray(
            1074, dtype=combined_exponent.dtype
        )
        units = jnp.asarray(
            jnp.rint(
                jnp.ldexp(jnp.abs(normalized_mantissa), unit_exponent)
            ),
            dtype=jnp.uint64,
        )
        sign = jnp.where(
            jnp.signbit(normalized_mantissa),
            jnp.asarray(0x8000000000000000, dtype=jnp.uint64),
            jnp.asarray(0, dtype=jnp.uint64),
        )
        gradual_bits = units | sign
        native_bits = jax.lax.bitcast_convert_type(native, jnp.uint64)
    else:
        use_gradual = (
            jnp.isfinite(mantissa)
            & mantissa_is_nonzero
            & (
                combined_exponent
                <= jnp.asarray(-126, dtype=combined_exponent.dtype)
            )
        )
        unit_exponent = combined_exponent + jnp.asarray(
            149, dtype=combined_exponent.dtype
        )
        units = jnp.asarray(
            jnp.rint(
                jnp.ldexp(jnp.abs(normalized_mantissa), unit_exponent)
            ),
            dtype=jnp.uint32,
        )
        sign = jnp.where(
            jnp.signbit(normalized_mantissa),
            jnp.asarray(0x80000000, dtype=jnp.uint32),
            jnp.asarray(0, dtype=jnp.uint32),
        )
        gradual_bits = units | sign
        native_bits = jax.lax.bitcast_convert_type(native, jnp.uint32)
    return jax.lax.select(use_gradual, gradual_bits, native_bits)


def _ldexp_gradual(mantissa: Array, exponent: Array) -> Array:
    """Scale by a power of two without flushing a representable subnormal."""

    bits = _ldexp_value_bits(mantissa, exponent)
    if mantissa.dtype == jnp.dtype(jnp.float64):
        return jax.lax.bitcast_convert_type(bits, jnp.float64)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _power_two_scaled_weights(
    weight: Array,
    *,
    active: Array,
) -> tuple[Array, Array]:
    """Represent positive weights as ``scaled * 2**reference`` exactly."""

    if weight.size == 0:
        return (
            jnp.zeros_like(weight),
            jnp.asarray(0, dtype=jnp.int32),
        )

    safe_weight = jnp.where(active, weight, jnp.zeros_like(weight))
    mantissa, exponent = _frexp_gradual(safe_weight)
    minimum_exponent = jnp.asarray(jnp.iinfo(exponent.dtype).min, exponent.dtype)
    reference_exponent = jnp.max(
        jnp.where(active, exponent, minimum_exponent)
    )
    has_active = jnp.any(active)
    safe_reference = jnp.where(has_active, reference_exponent, 0)
    relative_exponent = exponent - safe_reference
    scaled_weight = jnp.where(
        active,
        _ldexp_gradual(mantissa, relative_exponent),
        jnp.zeros_like(weight),
    )
    return scaled_weight, safe_reference


def _power_two_scaled_sum(
    values: Array,
    weight: Array,
    *,
    active: Array,
) -> tuple[Array, Array]:
    """Represent a signed weighted sum without logarithmic rounding."""

    if values.size == 0:
        return (
            jnp.asarray(0.0, dtype=values.dtype),
            jnp.asarray(0, dtype=jnp.int32),
        )

    finite_active = active & jnp.isfinite(values) & jnp.isfinite(weight)
    safe_value = jnp.where(finite_active, values, jnp.zeros_like(values))
    safe_weight = jnp.where(finite_active, weight, jnp.zeros_like(weight))
    value_mantissa, value_exponent = _frexp_gradual(safe_value)
    weight_mantissa, weight_exponent = _frexp_gradual(safe_weight)
    nonzero_active = (
        finite_active
        & _has_nonzero_floating_magnitude(safe_value)
        & _has_nonzero_floating_magnitude(safe_weight)
    )
    term_exponent = value_exponent + weight_exponent
    minimum_exponent = jnp.asarray(
        jnp.iinfo(term_exponent.dtype).min,
        term_exponent.dtype,
    )
    reference_exponent = jnp.max(
        jnp.where(nonzero_active, term_exponent, minimum_exponent)
    )
    has_nonzero_term = jnp.any(nonzero_active)
    safe_reference = jnp.where(has_nonzero_term, reference_exponent, 0)
    relative_exponent = term_exponent - safe_reference
    scaled_terms = jnp.where(
        nonzero_active,
        _ldexp_gradual(
            value_mantissa * weight_mantissa,
            relative_exponent,
        ),
        jnp.zeros_like(values),
    )
    scaled_total = jnp.sum(scaled_terms)
    term_was_dropped = nonzero_active & (
        ~_has_nonzero_floating_magnitude(scaled_terms)
    )
    unresolved_scale_separation = jnp.any(term_was_dropped) & (
        ~_has_nonzero_floating_magnitude(scaled_total)
    )
    invalid_active = jnp.any(active & (~finite_active))
    return (
        jnp.where(
            invalid_active | unresolved_scale_separation,
            jnp.asarray(jnp.nan, dtype=values.dtype),
            scaled_total,
        ),
        safe_reference,
    )


def _weighted_mean(
    scores: Array,
    weight: Array,
    *,
    included: Array,
) -> Array:
    """Return the informative weighted mean without dividing raw reductions."""

    if scores.size == 0:
        return jnp.asarray(jnp.nan, dtype=scores.dtype)

    positive = _positive_weight(weight) & included
    has_positive_weight = jnp.any(positive)

    def active_branch(_: None) -> Array:
        scaled_weight, denominator_exponent = _power_two_scaled_weights(
            weight, active=positive
        )
        scaled_denominator = jnp.sum(scaled_weight)
        diagnostic_numerator, diagnostic_is_finite = (
            _centered_weighted_total(
                jax.lax.stop_gradient(scores),
                scaled_weight,
                active=positive,
            )
        )
        diagnostic_normalized = diagnostic_numerator / scaled_denominator
        numerator_is_nonzero = _has_nonzero_floating_magnitude(
            diagnostic_numerator
        )
        normalized_is_nonzero = _has_nonzero_floating_magnitude(
            diagnostic_normalized
        )
        centered_is_usable = (
            diagnostic_is_finite
            & jnp.isfinite(scaled_denominator)
            & _positive_weight(scaled_denominator)
            & jnp.isfinite(diagnostic_normalized)
            & ((~numerator_is_nonzero) | normalized_is_nonzero)
        )

        def centered_branch(_: None) -> Array:
            numerator = _centered_weighted_total(
                scores,
                scaled_weight,
                active=positive,
            )[0]
            return numerator / scaled_denominator

        def power_branch(_: None) -> Array:
            numerator, numerator_exponent = _power_two_scaled_sum(
                scores,
                weight,
                active=positive,
            )
            return _ldexp_gradual(
                numerator / scaled_denominator,
                numerator_exponent - denominator_exponent,
            )

        return jax.lax.cond(
            centered_is_usable,
            centered_branch,
            power_branch,
            operand=None,
        )

    return jax.lax.cond(
        has_positive_weight,
        active_branch,
        lambda _: jnp.asarray(jnp.nan, dtype=scores.dtype),
        operand=None,
    )


def _informative_weight_status(
    weight: Array,
    *,
    included: Array,
) -> tuple[Array, Array]:
    """Return positive-mass presence and selected-dtype representability."""

    if weight.size == 0:
        false = jnp.asarray(False)
        return false, false

    positive = _positive_weight(weight) & included
    has_positive_weight = jnp.any(positive)
    native_mass = jnp.sum(
        jnp.where(positive, weight, jnp.zeros_like(weight))
    )
    native_mass_is_positive = _positive_weight(native_mass)
    native_mass_is_zero = jnp.isfinite(native_mass) & (
        ~_has_nonzero_floating_magnitude(native_mass)
    )
    scaled_weight, reference_exponent = _power_two_scaled_weights(
        weight, active=positive
    )
    recovered_mass = _ldexp_gradual(
        jnp.sum(scaled_weight),
        reference_exponent,
    )
    recovered_mass_is_representable = (
        jnp.isfinite(recovered_mass) & _positive_weight(recovered_mass)
    )
    mass_is_representable = has_positive_weight & jnp.where(
        native_mass_is_positive,
        jnp.asarray(True),
        jnp.where(
            native_mass_is_zero,
            recovered_mass_is_representable,
            jnp.asarray(False),
        ),
    )
    return has_positive_weight, mass_is_representable


def _reduction_result(
    scores: Array,
    weight: Array,
    *,
    included: Array,
    weight_is_valid: Array,
    jitter_is_valid: Array,
    empty_observed_space: bool,
) -> tuple[Array, Array]:
    """Return raw and normalized reductions with global-status precedence."""

    total = _weighted_total(scores, weight, included=included)
    has_positive_weight, weight_mass_is_representable = (
        _informative_weight_status(weight, included=included)
    )
    nan = jnp.asarray(jnp.nan, dtype=scores.dtype)
    total_is_valid = (~has_positive_weight) | (
        weight_mass_is_representable & jnp.isfinite(total)
    )
    total = jnp.where(total_is_valid, total, nan)
    if empty_observed_space:
        normalized = jnp.asarray(0.0, dtype=scores.dtype)
    else:
        normalized = _weighted_mean(scores, weight, included=included)
        normalized = jax.lax.cond(
            weight_mass_is_representable & jnp.isfinite(normalized),
            lambda _: normalized,
            lambda _: nan,
            operand=None,
        )

    invalid_global = (~weight_is_valid) | (~jitter_is_valid)
    return (
        jnp.where(invalid_global, nan, total),
        jnp.where(invalid_global, nan, normalized),
    )


def _fixed_e_step(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array,
) -> tuple[EStep, Array]:
    """Evaluate one canonical fixed-``M`` E-step and its global control."""

    e_step = posterior_components_general(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    _, jitter_is_valid = _general_scalar_control(
        factor_jitter,
        dtype=e_step.score_samples.dtype,
        name="factor_jitter",
    )
    return e_step, jitter_is_valid


def _fixed_reductions(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    *,
    sample_weight: Array | None,
    factor_jitter: float | Array,
) -> tuple[Array, Array]:
    """Evaluate fixed-``M`` weighted total and informative mean."""

    e_step, jitter_is_valid = _fixed_e_step(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    scores = _sentinel_scores(e_step)
    weight, weight_is_valid = _canonical_reduction_weight(
        sample_weight,
        batch_shape=scores.shape,
        dtype=scores.dtype,
    )
    observed_dimension = observations.shape[-1]
    included = jnp.full(
        scores.shape,
        observed_dimension > 0,
        dtype=bool,
    )
    return _reduction_result(
        scores,
        weight,
        included=included,
        weight_is_valid=weight_is_valid,
        jitter_is_valid=jitter_is_valid,
        empty_observed_space=observed_dimension == 0,
    )


def score_samples_general(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return fixed-``M`` observed scores, with NaN for failed rows."""

    e_step = posterior_components_general(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    return _sentinel_scores(e_step)


def log_likelihood_general(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    *,
    sample_weight: Array | None = None,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return the weighted fixed-``M`` observed log likelihood."""

    total, _ = _fixed_reductions(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        sample_weight=sample_weight,
        factor_jitter=factor_jitter,
    )
    return total


def score_general(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    *,
    sample_weight: Array | None = None,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return the weighted fixed-``M`` informative mean score."""

    _, normalized = _fixed_reductions(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        sample_weight=sample_weight,
        factor_jitter=factor_jitter,
    )
    return normalized


def predict_proba_general(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return fixed-``M`` component probabilities or failed-row NaNs."""

    e_step = posterior_components_general(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    return _sentinel_probabilities(e_step)


def predict_general(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return the lowest-index maximum component or ``-1`` on failure."""

    probabilities = predict_proba_general(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    labels = jnp.argmax(probabilities, axis=-1)
    valid = jnp.all(jnp.isfinite(probabilities), axis=-1)
    return jnp.where(valid, labels, jnp.asarray(-1, dtype=labels.dtype))


def posterior_general(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> tuple[Array, Array]:
    """Return marginalized fixed-``M`` moments or failed-row NaNs."""

    e_step = posterior_components_general(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    return _sentinel_posterior(e_step)


def posterior_mean_general(
    params: Params,
    observations: Array,
    projection_matrices: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return only the marginalized fixed-``M`` latent mean."""

    return posterior_general(
        params,
        observations,
        projection_matrices,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )[0]


def _restore_group_values(
    grouped: GroupedGeneralInputs,
    values: Sequence[Array],
) -> Array:
    """Restore group-leading arrays without applying new validation policy."""

    concatenated = jnp.concatenate(tuple(values), axis=0)
    restoration = jnp.asarray(grouped.restoration_indices, dtype=jnp.int32)
    return concatenated[restoration]


def _grouped_weights_and_inclusion(
    grouped: GroupedGeneralInputs,
) -> tuple[Array, Array, bool]:
    """Restore stored weights and mark rows belonging to informative groups."""

    weights = _restore_group_values(
        grouped,
        tuple(group.sample_weight for group in grouped.groups),
    )
    included = _restore_group_values(
        grouped,
        tuple(
            jnp.full(
                (len(group.original_indices),),
                group.observations.shape[-1] > 0,
                dtype=bool,
            )
            for group in grouped.groups
        ),
    )
    has_informative_group = any(
        group.observations.shape[-1] > 0 for group in grouped.groups
    )
    return weights, included, has_informative_group


def _grouped_e_step(
    grouped: GroupedGeneralInputs,
    *,
    factor_jitter: float | Array,
) -> tuple[GroupedPosteriorResult, Array]:
    """Evaluate restored grouped inference and the global jitter status."""

    result = posterior_components_grouped(
        grouped,
        factor_jitter=factor_jitter,
    )
    _, jitter_is_valid = _general_scalar_control(
        factor_jitter,
        dtype=result.e_step.score_samples.dtype,
        name="factor_jitter",
    )
    return result, jitter_is_valid


def _grouped_reductions(
    grouped: GroupedGeneralInputs,
    *,
    factor_jitter: float | Array,
) -> tuple[Array, Array]:
    """Reduce grouped scores using stored weights and informative rows only."""

    detailed, jitter_is_valid = _grouped_e_step(
        grouped,
        factor_jitter=factor_jitter,
    )
    scores = _sentinel_scores(detailed.e_step)
    weights, included, has_informative_group = (
        _grouped_weights_and_inclusion(grouped)
    )
    weight_is_valid = jnp.all(jnp.isfinite(weights) & (weights >= 0.0))
    return _reduction_result(
        scores,
        weights,
        included=included,
        weight_is_valid=weight_is_valid,
        jitter_is_valid=jitter_is_valid,
        empty_observed_space=not has_informative_group,
    )


def score_samples_grouped(
    grouped: GroupedGeneralInputs,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return restored per-row grouped scores with failure sentinels."""

    detailed = posterior_components_grouped(
        grouped,
        factor_jitter=factor_jitter,
    )
    return _sentinel_scores(detailed.e_step)


def log_likelihood_grouped(
    grouped: GroupedGeneralInputs,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return the stored-weight grouped observed log likelihood."""

    total, _ = _grouped_reductions(
        grouped,
        factor_jitter=factor_jitter,
    )
    return total


def score_grouped(
    grouped: GroupedGeneralInputs,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return the stored-weight grouped informative mean score."""

    _, normalized = _grouped_reductions(
        grouped,
        factor_jitter=factor_jitter,
    )
    return normalized


def predict_proba_grouped(
    grouped: GroupedGeneralInputs,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return restored grouped probabilities or failed-row NaNs."""

    detailed = posterior_components_grouped(
        grouped,
        factor_jitter=factor_jitter,
    )
    return _sentinel_probabilities(detailed.e_step)


def predict_grouped(
    grouped: GroupedGeneralInputs,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return restored grouped labels or ``-1`` for failed rows."""

    probabilities = predict_proba_grouped(
        grouped,
        factor_jitter=factor_jitter,
    )
    labels = jnp.argmax(probabilities, axis=-1)
    valid = jnp.all(jnp.isfinite(probabilities), axis=-1)
    return jnp.where(valid, labels, jnp.asarray(-1, dtype=labels.dtype))


def posterior_grouped(
    grouped: GroupedGeneralInputs,
    *,
    factor_jitter: float | Array = 0.0,
) -> tuple[Array, Array]:
    """Return restored marginalized grouped moments or failed-row NaNs."""

    detailed = posterior_components_grouped(
        grouped,
        factor_jitter=factor_jitter,
    )
    return _sentinel_posterior(detailed.e_step)


def posterior_mean_grouped(
    grouped: GroupedGeneralInputs,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return only the restored marginalized grouped latent mean."""

    return posterior_grouped(
        grouped,
        factor_jitter=factor_jitter,
    )[0]


__all__ = [
    "log_likelihood_general",
    "log_likelihood_grouped",
    "posterior_general",
    "posterior_grouped",
    "posterior_mean_general",
    "posterior_mean_grouped",
    "predict_general",
    "predict_grouped",
    "predict_proba_general",
    "predict_proba_grouped",
    "score_general",
    "score_grouped",
    "score_samples_general",
    "score_samples_grouped",
]
