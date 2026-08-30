"""Temporary eager validation boundary for identity-projection XD inputs.

The numerical kernels intentionally assume canonical inputs. Functions here are
host-controlled, may synchronize device arrays for validation, and return JAX
arrays ready for those kernels. This module is not a final public API.
"""

from __future__ import annotations

import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .identity_xd import Params


Array = jax.Array


class ValidationError(ValueError):
    """An identity-XD public-boundary input is outside the contract."""


class PrecisionError(ValidationError):
    """The requested computation precision is unavailable or unsupported."""


class _ControlValueError(ValidationError):
    """A scalar control has an invalid value after static type/shape checks."""


class ValidatedIdentityInputs(NamedTuple):
    """Canonical parameter, observation, and full-noise arrays."""

    parameters: Params
    observations: Array
    measurement_covariances: Array


class PreparedControls(NamedTuple):
    """Eagerly validated scalar controls in the selected JAX dtype."""

    factor_jitter: Array
    covariance_ridge: Array


class PreparedConvergenceControls(NamedTuple):
    """Eagerly validated stopping controls in the selected JAX dtype."""

    tol: Array
    decrease_tol: Array


def _computation_dtype(dtype: object) -> tuple[np.dtype, jnp.dtype]:
    try:
        requested = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise PrecisionError(
            "dtype must explicitly request float32 or float64"
        ) from error
    if requested not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise PrecisionError(
            "dtype must explicitly request supported float32 or float64; "
            f"received {requested}"
        )
    if requested == np.dtype(np.float64) and not jax.config.x64_enabled:
        raise PrecisionError(
            "float64 was requested, but JAX x64 support is disabled; enable "
            "jax_enable_x64 before creating canonical inputs"
        )
    return requested, jnp.dtype(requested)


def _host_array(value: object, *, field: str) -> np.ndarray:
    try:
        return np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{field} must be a real numeric array") from error


def _source_dtype_is_real_numeric(array: np.ndarray) -> bool:
    try:
        return bool(
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        )
    except TypeError:
        return False


def _convert_array(
    value: object,
    *,
    field: str,
    requested_numpy_dtype: np.dtype,
    requested_jax_dtype: jnp.dtype,
    require_floating_source: bool,
    require_nonnegative_source: bool = False,
) -> Array:
    source = _host_array(value, field=field)
    if np.issubdtype(source.dtype, np.bool_):
        raise ValidationError(f"{field} must not be boolean")
    if np.issubdtype(source.dtype, np.complexfloating):
        raise ValidationError(f"{field} must not be complex")
    if not _source_dtype_is_real_numeric(source):
        raise ValidationError(f"{field} must be real numeric values")
    if require_floating_source and not np.issubdtype(
        source.dtype, np.floating
    ):
        raise ValidationError(f"{field} must use a floating input dtype")
    if require_nonnegative_source and np.any(source < 0):
        raise ValidationError(f"{field} must be nonnegative")

    with np.errstate(over="ignore", invalid="ignore"):
        converted = source.astype(requested_numpy_dtype, copy=False)
    if not np.all(np.isfinite(converted)):
        raise ValidationError(
            f"{field} must remain finite after conversion to "
            f"{requested_numpy_dtype.name}"
        )
    result = jnp.asarray(converted, dtype=requested_jax_dtype)
    if result.dtype != requested_jax_dtype:
        raise PrecisionError(
            f"requested {requested_numpy_dtype.name}, but JAX returned "
            f"{result.dtype}"
        )
    return result


def _shape(value: object, *, field: str) -> tuple[int, ...]:
    return tuple(_host_array(value, field=field).shape)


def _raise_shape(
    field: str, received: tuple[int, ...], expected: tuple[int, ...] | str
) -> None:
    raise ValidationError(
        f"{field}: received {received}; expected {expected}"
    )


def _parameter_dimensions(
    params: Params,
    observations_shape: tuple[int, ...],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    try:
        weights_shape = _shape(params.weights, field="weights")
        means_shape = _shape(params.means, field="means")
        covariances_shape = _shape(
            params.covariances, field="parameter covariances"
        )
    except AttributeError as error:
        raise ValidationError(
            "parameters must provide weights, means, and covariances"
        ) from error

    if len(weights_shape) == 1:
        n_components = weights_shape[0]
    elif len(means_shape) == 2:
        n_components = means_shape[0]
    elif len(covariances_shape) == 3:
        n_components = covariances_shape[0]
    else:
        n_components = 0

    if (
        len(covariances_shape) == 3
        and covariances_shape[-2] == covariances_shape[-1]
    ):
        dimension = covariances_shape[-1]
    elif len(means_shape) == 2:
        dimension = means_shape[-1]
    elif observations_shape:
        dimension = observations_shape[-1]
    else:
        dimension = 0
    return (
        n_components,
        dimension,
        weights_shape,
        means_shape,
        covariances_shape,
    )


def _validate_static_shapes(
    params: Params,
    observations: object,
    measurement_covariances: object,
    *,
    fitting: bool,
) -> tuple[int, int]:
    observations_shape = _shape(observations, field="observations")
    noise_shape = _shape(
        measurement_covariances, field="measurement covariances"
    )
    (
        n_components,
        dimension,
        weights_shape,
        means_shape,
        covariances_shape,
    ) = _parameter_dimensions(params, observations_shape)

    if fitting:
        if len(observations_shape) != 2:
            _raise_shape("observations", observations_shape, "(N, D)")
        if dimension < 1:
            _raise_shape("observations", observations_shape, "D >= 1")
        if observations_shape[-1] != dimension:
            expected_n = (
                noise_shape[0]
                if len(noise_shape) == 3
                and noise_shape[-2:] == (dimension, dimension)
                else observations_shape[0]
            )
            _raise_shape(
                "observations",
                observations_shape,
                (expected_n, dimension),
            )
        n_samples = observations_shape[0]
        if n_samples < 1:
            _raise_shape("observations", observations_shape, "N >= 1")
        expected_noise_shape = (n_samples, dimension, dimension)
        if noise_shape != expected_noise_shape:
            hint = (
                "; use shared_full_noise for an explicitly shared covariance"
                if len(noise_shape) == 2
                else ""
            )
            raise ValidationError(
                "measurement covariances: received "
                f"{noise_shape}; expected {expected_noise_shape}{hint}"
            )
    else:
        if len(observations_shape) < 1:
            _raise_shape(
                "observations", observations_shape, f"B + ({dimension},)"
            )
        if dimension < 1:
            _raise_shape("observations", observations_shape, "D >= 1")
        if observations_shape[-1] != dimension:
            _raise_shape(
                "observations",
                observations_shape,
                observations_shape[:-1] + (dimension,),
            )
        batch_shape = observations_shape[:-1]
        if any(size == 0 for size in batch_shape):
            raise ValidationError(
                f"observations: received {observations_shape}; expected a "
                "nonempty inference batch"
            )
        expected_noise_shape = batch_shape + (dimension, dimension)
        if noise_shape != expected_noise_shape:
            _raise_shape(
                "measurement covariances", noise_shape, expected_noise_shape
            )

    expected_weights_shape = (n_components,)
    expected_means_shape = (n_components, dimension)
    expected_covariances_shape = (n_components, dimension, dimension)
    if weights_shape != expected_weights_shape:
        _raise_shape("weights", weights_shape, expected_weights_shape)
    if means_shape != expected_means_shape:
        _raise_shape("means", means_shape, expected_means_shape)
    if covariances_shape != expected_covariances_shape:
        _raise_shape(
            "covariances",
            covariances_shape,
            expected_covariances_shape,
        )
    if n_components < 1:
        _raise_shape("weights", weights_shape, "K >= 1")
    return n_components, dimension


def _tolerance_profile(dtype: np.dtype) -> tuple[float, float, float]:
    if dtype == np.dtype(np.float64):
        return 5e-13, 2e-13, 2e-11
    return 2e-5, 2e-6, 5e-5


def _canonical_covariances(
    covariances: Array,
    *,
    field: str,
    requested_numpy_dtype: np.dtype,
    positive_definite: bool,
) -> Array:
    host = np.asarray(jax.device_get(covariances))
    flat = host.reshape((-1, host.shape[-2], host.shape[-1]))
    _, symmetry_tolerance, psd_tolerance = _tolerance_profile(
        requested_numpy_dtype
    )
    canonical = np.empty_like(flat)
    for index, matrix in enumerate(flat):
        # Scale before every metric operation. Even host float64 subtraction
        # and norms can overflow for finite selected-dtype values near max.
        metric_matrix = matrix.astype(np.float64, copy=False)
        entry_scale = max(
            1.0, float(np.max(np.abs(metric_matrix), initial=0.0))
        )
        scaled_matrix = metric_matrix / entry_scale
        scaled_spectral_norm = float(
            np.linalg.norm(scaled_matrix, ord=2)
        )
        scaled_spectral_scale = max(
            1.0 / entry_scale, scaled_spectral_norm
        )
        scaled_symmetry_norm = float(
            np.linalg.norm(
                scaled_matrix - scaled_matrix.T, ord=np.inf
            )
        )
        symmetry_residual = (
            scaled_symmetry_norm / scaled_spectral_scale
        )
        metrics = (
            entry_scale,
            scaled_spectral_norm,
            scaled_spectral_scale,
            scaled_symmetry_norm,
            symmetry_residual,
        )
        if not np.all(np.isfinite(metrics)) or (
            symmetry_residual > symmetry_tolerance
        ):
            raise ValidationError(
                f"{field} must be symmetric within the "
                f"{requested_numpy_dtype.name} tolerance; matrix {index} "
                f"has scaled residual {symmetry_residual}"
            )
        # Halve before adding so two valid near-dtype-limit entries cannot
        # overflow merely because the boundary repairs roundoff asymmetry.
        symmetric = 0.5 * matrix + 0.5 * matrix.T
        if not np.all(np.isfinite(symmetric)):
            raise ValidationError(
                f"{field} must remain finite after symmetrization; "
                f"matrix {index} produced a nonfinite value"
            )
        scaled_symmetric = (
            symmetric.astype(np.float64, copy=False) / entry_scale
        )
        if not np.all(np.isfinite(scaled_symmetric)):
            raise ValidationError(
                f"{field} produced nonfinite scaled covariance metrics; "
                f"matrix {index}"
            )
        if positive_definite:
            try:
                host_factor = np.linalg.cholesky(scaled_symmetric)
            except np.linalg.LinAlgError as error:
                raise ValidationError(
                    f"{field} must be positive definite; matrix {index} "
                    "failed Cholesky factorization"
                ) from error
            if not np.all(np.isfinite(host_factor)) or not np.all(
                np.diag(host_factor) > 0.0
            ):
                raise ValidationError(
                    f"{field} must be positive definite; matrix {index} "
                    "has an invalid host Cholesky factor"
                )

            # The kernels execute in the selected JAX dtype. The boundary has
            # already repaired symmetry safely, so disable JAX's additional
            # `(A + A.T) / 2` pass, which itself can overflow near dtype max.
            selected_factor = jax.lax.linalg.cholesky(
                jnp.asarray(symmetric, dtype=covariances.dtype),
                symmetrize_input=False,
            )
            selected_host_factor = np.asarray(
                jax.device_get(selected_factor)
            )
            if not np.all(np.isfinite(selected_host_factor)) or not np.all(
                np.diag(selected_host_factor) > 0.0
            ):
                raise ValidationError(
                    f"{field} must be positive definite in the selected "
                    f"{requested_numpy_dtype.name} JAX computation; matrix "
                    f"{index} failed Cholesky factorization"
                )
        else:
            minimum_eigenvalue = float(
                np.linalg.eigvalsh(scaled_symmetric)[0]
            )
            if not np.isfinite(minimum_eigenvalue) or (
                minimum_eigenvalue
                < -psd_tolerance * scaled_spectral_scale
            ):
                raise ValidationError(
                    f"{field} must be positive semidefinite; matrix {index} "
                    f"has minimum eigenvalue {minimum_eigenvalue}"
                )
        canonical[index] = symmetric
    reshaped = canonical.reshape(host.shape)
    return jnp.asarray(reshaped, dtype=covariances.dtype)


def _validate_weights(weights: Array, *, dtype: np.dtype) -> None:
    host = np.asarray(jax.device_get(weights))
    if not np.all(np.isfinite(host)):
        raise ValidationError("weights must be finite")
    if not np.all(host > 0.0):
        raise ValidationError("weights must be strictly positive")
    weight_tolerance, _, _ = _tolerance_profile(dtype)
    total = float(np.sum(host, dtype=host.dtype))
    if abs(total - 1.0) > weight_tolerance:
        raise ValidationError(
            "weights must sum to one within the selected dtype tolerance; "
            f"received sum {total}"
        )


def _canonicalize_inputs(
    params: Params,
    observations: object,
    measurement_covariances: object,
    *,
    dtype: object,
    fitting: bool,
) -> ValidatedIdentityInputs:
    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    _validate_static_shapes(
        params,
        observations,
        measurement_covariances,
        fitting=fitting,
    )

    weights = _convert_array(
        params.weights,
        field="weights",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=True,
    )
    means = _convert_array(
        params.means,
        field="means",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=True,
    )
    parameter_covariances = _convert_array(
        params.covariances,
        field="parameter covariances",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=True,
    )
    canonical_observations = _convert_array(
        observations,
        field="observations",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=False,
    )
    canonical_noise = _convert_array(
        measurement_covariances,
        field="measurement covariances",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=True,
    )

    _validate_weights(weights, dtype=requested_numpy_dtype)
    parameter_covariances = _canonical_covariances(
        parameter_covariances,
        field="parameter covariances",
        requested_numpy_dtype=requested_numpy_dtype,
        positive_definite=True,
    )
    canonical_noise = _canonical_covariances(
        canonical_noise,
        field="measurement covariances",
        requested_numpy_dtype=requested_numpy_dtype,
        positive_definite=False,
    )
    return ValidatedIdentityInputs(
        parameters=Params(weights, means, parameter_covariances),
        observations=canonical_observations,
        measurement_covariances=canonical_noise,
    )


def canonicalize_fit_inputs(
    params: Params,
    observations: object,
    measurement_covariances: object,
    *,
    dtype: object,
) -> ValidatedIdentityInputs:
    """Validate canonical fitting shapes ``(N,D)`` and ``(N,D,D)``."""

    return _canonicalize_inputs(
        params,
        observations,
        measurement_covariances,
        dtype=dtype,
        fitting=True,
    )


def canonicalize_inference_inputs(
    params: Params,
    observations: object,
    measurement_covariances: object,
    *,
    dtype: object,
) -> ValidatedIdentityInputs:
    """Validate single or batched inference inputs without adding axes."""

    return _canonicalize_inputs(
        params,
        observations,
        measurement_covariances,
        dtype=dtype,
        fitting=False,
    )


def _static_control_source(value: object, *, field: str) -> np.ndarray:
    """Validate one control's host-visible type and shape only.

    Host fit preparation validates this static domain for *both* controls
    before translating either control's value-domain failure into a device
    rollback status.  That preserves an actionable type/shape error in one
    control even when the other control is negative, nonfinite, or underflows.
    """

    source = _host_array(value, field=field)
    if source.ndim != 0:
        raise ValueError(
            f"{field} must be a rank-zero scalar; received shape "
            f"{source.shape}"
        )
    if np.issubdtype(source.dtype, np.bool_):
        raise TypeError(f"{field} must not be boolean")
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError(f"{field} must not be complex")
    if not _source_dtype_is_real_numeric(source):
        raise TypeError(f"{field} must be a real numeric scalar")
    return source


def _prepared_control(
    source: np.ndarray,
    *,
    field: str,
    requested_numpy_dtype: np.dtype,
    requested_jax_dtype: jnp.dtype,
) -> Array:
    """Validate one statically checked scalar without losing its host value."""

    if not bool(np.isfinite(source)):
        raise _ControlValueError(f"{field} must be finite")
    if bool(source < 0):
        raise _ControlValueError(f"{field} must be nonnegative")

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        converted = source.astype(requested_numpy_dtype, copy=False)
    if not bool(np.isfinite(converted)):
        raise _ControlValueError(
            f"{field} must remain finite after conversion to "
            f"{requested_numpy_dtype.name}"
        )
    if bool(source != 0) and bool(converted == 0):
        raise _ControlValueError(
            f"{field} is nonzero but becomes zero in "
            f"{requested_numpy_dtype.name} due to underflow"
        )

    result = jnp.asarray(converted, dtype=requested_jax_dtype)
    if result.shape != () or result.dtype != requested_jax_dtype:
        raise PrecisionError(
            f"{field} could not be represented as a rank-zero "
            f"{requested_numpy_dtype.name} JAX scalar"
        )
    return result


def validate_controls(
    *,
    factor_jitter: object = 0.0,
    covariance_ridge: object = 0.0,
    dtype: object,
) -> PreparedControls:
    """Eagerly prepare fit controls while their original values are visible.

    Unlike a raw compiled kernel, this host boundary can distinguish an
    intentional selected-dtype zero from a nonzero Python value lost during
    conversion. Invalid value-domain controls raise ``ValidationError``;
    boolean, complex, nonnumeric, and nonscalar controls retain their static
    type/shape error semantics.
    """

    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    # Establish static-error precedence across the pair before either
    # value-domain check can raise.  The host fit wrapper catches only the
    # private value error and converts it to an explicit rollback status.
    factor_source = _static_control_source(
        factor_jitter, field="factor_jitter"
    )
    ridge_source = _static_control_source(
        covariance_ridge, field="covariance_ridge"
    )
    factor = _prepared_control(
        factor_source,
        field="factor_jitter",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    ridge = _prepared_control(
        ridge_source,
        field="covariance_ridge",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    return PreparedControls(factor, ridge)


def validate_convergence_controls(
    *,
    tol: object = 1e-6,
    decrease_tol: object = 1e-10,
    dtype: object,
) -> PreparedConvergenceControls:
    """Prepare stopping tolerances without hiding conversion loss.

    Both controls complete static type/shape validation before either value is
    converted.  A finite source must remain finite in the selected dtype, and
    an intentional positive tolerance must not silently become exact zero.
    Unlike jitter/ridge value failures, invalid stopping tolerances are public
    validation errors rather than numerical-failure result sentinels.
    """

    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    tolerance_source = _static_control_source(tol, field="tol")
    decrease_source = _static_control_source(
        decrease_tol, field="decrease_tol"
    )
    tolerance = _prepared_control(
        tolerance_source,
        field="tol",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    decrease_tolerance = _prepared_control(
        decrease_source,
        field="decrease_tol",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    return PreparedConvergenceControls(tolerance, decrease_tolerance)


def _adapter_values(
    values: object,
    *,
    field: str,
    dtype: object,
    require_floating_source: bool,
    require_nonnegative_source: bool = False,
) -> tuple[Array, np.dtype, jnp.dtype]:
    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    converted = _convert_array(
        values,
        field=field,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=require_floating_source,
        require_nonnegative_source=require_nonnegative_source,
    )
    return converted, requested_numpy_dtype, requested_jax_dtype


def _positive_count(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a positive integer, not boolean")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValidationError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValidationError(f"{name} must be >= 1; received {result}")
    return int(result)


def isotropic_noise(
    variances: object,
    *,
    dimension: int,
    dtype: object,
) -> Array:
    """Construct ``variance * I`` from a scalar or explicit batch shape."""

    dimension_value = _positive_count(dimension, name="dimension")
    converted, _, requested_jax_dtype = _adapter_values(
        variances,
        field="isotropic variances",
        dtype=dtype,
        require_floating_source=False,
        require_nonnegative_source=True,
    )
    if converted.ndim == 2 and converted.shape[-1] == 1:
        _raise_shape(
            "isotropic variances",
            tuple(converted.shape),
            "(N,) or a non-ambiguous multi-axis batch shape",
        )
    if bool(jnp.any(converted < 0.0)):
        raise ValidationError("isotropic variances must be nonnegative")
    identity = jnp.eye(dimension_value, dtype=requested_jax_dtype)
    return converted[..., None, None] * identity


def _operation_specific_isotropic_noise(
    variances: object,
    *,
    dimension: int,
    dtype: object,
    fitting: bool,
) -> Array:
    """Construct isotropic noise with explicit fit/inference shape semantics."""

    dimension_value = _positive_count(dimension, name="dimension")
    field = (
        "fit isotropic variances"
        if fitting
        else "inference isotropic variances"
    )
    source = _host_array(variances, field=field)
    if fitting and (source.ndim != 1 or source.shape[0] < 1):
        _raise_shape(
            field,
            tuple(source.shape),
            "(N,) with N >= 1",
        )

    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    if np.issubdtype(source.dtype, np.bool_):
        raise ValidationError(f"{field} must not be boolean")
    if np.issubdtype(source.dtype, np.complexfloating):
        raise ValidationError(f"{field} must not be complex")
    if not _source_dtype_is_real_numeric(source):
        raise ValidationError(f"{field} must contain real numeric values")
    if not np.all(np.isfinite(source)):
        raise ValidationError(f"{field} must be finite")
    if np.any(source < 0):
        raise ValidationError(f"{field} must be nonnegative")

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        converted = source.astype(requested_numpy_dtype, copy=False)
    if not np.all(np.isfinite(converted)):
        raise ValidationError(
            f"{field} must remain finite after conversion to "
            f"{requested_numpy_dtype.name}"
        )
    if np.any((source != 0) & (converted == 0)):
        raise PrecisionError(
            f"{field} contains a nonzero variance that underflows to zero "
            f"in {requested_numpy_dtype.name}"
        )

    # Construct on the eager host boundary. XLA multiplication may flush a
    # representable selected-dtype subnormal even when transfer preserves its
    # bits; a host-built diagonal retains that value without weakening the
    # explicit host/device boundary of this module.
    host_identity = np.eye(
        dimension_value, dtype=requested_numpy_dtype
    )
    host_full = converted[..., None, None] * host_identity
    result = jnp.asarray(host_full, dtype=requested_jax_dtype)
    if result.dtype != requested_jax_dtype:
        raise PrecisionError(
            f"{field} could not be represented in the requested "
            f"{requested_numpy_dtype.name} computation"
        )
    return result


def fit_isotropic_noise(
    variances: object,
    *,
    dimension: int,
    dtype: object,
) -> Array:
    """Construct fitting noise from exact nonempty ``(N,)`` variances."""

    return _operation_specific_isotropic_noise(
        variances,
        dimension=dimension,
        dtype=dtype,
        fitting=True,
    )


def inference_isotropic_noise(
    variances: object,
    *,
    dimension: int,
    dtype: object,
) -> Array:
    """Construct inference noise, treating the full input shape as batch ``B``."""

    return _operation_specific_isotropic_noise(
        variances,
        dimension=dimension,
        dtype=dtype,
        fitting=False,
    )


def diagonal_noise(variances: object, *, dtype: object) -> Array:
    """Construct full matrices from explicit diagonal variances."""

    converted, _, requested_jax_dtype = _adapter_values(
        variances,
        field="diagonal variances",
        dtype=dtype,
        require_floating_source=False,
        require_nonnegative_source=True,
    )
    if converted.ndim < 1 or converted.shape[-1] < 1:
        _raise_shape(
            "diagonal variances",
            tuple(converted.shape),
            "B + (D,) with D >= 1",
        )
    if bool(jnp.any(converted < 0.0)):
        raise ValidationError("diagonal variances must be nonnegative")
    identity = jnp.eye(converted.shape[-1], dtype=requested_jax_dtype)
    return converted[..., :, None] * identity


def full_noise(covariances: object, *, dtype: object) -> Array:
    """Validate an explicit full covariance without broadcasting it."""

    converted, requested_numpy_dtype, _ = _adapter_values(
        covariances,
        field="full covariances",
        dtype=dtype,
        require_floating_source=True,
    )
    if (
        converted.ndim < 2
        or converted.shape[-1] < 1
        or converted.shape[-2] != converted.shape[-1]
    ):
        _raise_shape(
            "full covariances",
            tuple(converted.shape),
            "B + (D, D) with D >= 1",
        )
    return _canonical_covariances(
        converted,
        field="full covariances",
        requested_numpy_dtype=requested_numpy_dtype,
        positive_definite=False,
    )


def _batch_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        shape = (value,)
    else:
        try:
            shape = tuple(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValidationError(
                "batch_shape must be a tuple of nonnegative integers"
            ) from error
    for size in shape:
        if isinstance(size, (bool, np.bool_)):
            raise ValidationError(
                "batch_shape must contain nonnegative integers, not boolean"
            )
        try:
            integer_size = operator.index(size)
        except TypeError as error:
            raise ValidationError(
                "batch_shape must contain nonnegative integers"
            ) from error
        if integer_size < 0:
            raise ValidationError(
                "batch_shape must contain nonnegative integers"
            )
    return tuple(int(size) for size in shape)


def shared_full_noise(
    covariance: object,
    *,
    batch_shape: object,
    dtype: object,
) -> Array:
    """Validate one full covariance and explicitly broadcast it over a batch."""

    converted, requested_numpy_dtype, _ = _adapter_values(
        covariance,
        field="shared full covariance",
        dtype=dtype,
        require_floating_source=False,
    )
    if (
        converted.ndim != 2
        or converted.shape[-1] < 1
        or converted.shape[-2] != converted.shape[-1]
    ):
        _raise_shape(
            "shared full covariance",
            tuple(converted.shape),
            "(D, D) with D >= 1",
        )
    canonical = _canonical_covariances(
        converted,
        field="shared full covariance",
        requested_numpy_dtype=requested_numpy_dtype,
        positive_definite=False,
    )
    shape = _batch_shape(batch_shape)
    return jnp.broadcast_to(
        canonical, shape + tuple(canonical.shape)
    )


def validate_sample_initialization(
    *,
    n_samples: int,
    n_components: int,
    replace: bool = False,
) -> None:
    """Validate sample-based initializer counts without allocating samples."""

    sample_count = _positive_count(n_samples, name="n_samples")
    component_count = _positive_count(n_components, name="n_components")
    if not isinstance(replace, (bool, np.bool_)):
        raise ValidationError("replace must be boolean")
    if not bool(replace) and component_count > sample_count:
        raise ValidationError(
            f"n_components={component_count} exceeds n_samples={sample_count} "
            "when replace=False"
        )


__all__ = [
    "PreparedControls",
    "PrecisionError",
    "ValidatedIdentityInputs",
    "ValidationError",
    "canonicalize_fit_inputs",
    "canonicalize_inference_inputs",
    "diagonal_noise",
    "fit_isotropic_noise",
    "full_noise",
    "inference_isotropic_noise",
    "isotropic_noise",
    "shared_full_noise",
    "validate_controls",
    "validate_sample_initialization",
]
