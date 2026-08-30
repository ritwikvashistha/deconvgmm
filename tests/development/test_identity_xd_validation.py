"""Phase 1 public-boundary gates for identity-projection XD inputs.

These tests target a temporary eager ``development.validation`` layer.  The
pure numerical kernels remain free to assume canonical inputs; every rejection
in this file must happen before those kernels are entered.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import development.validation as validation_module
from development.identity_xd import Params, posterior_components
from development.validation import (
    PrecisionError,
    ValidatedIdentityInputs,
    ValidationError,
    canonicalize_fit_inputs,
    canonicalize_inference_inputs,
    diagonal_noise,
    full_noise,
    isotropic_noise,
    shared_full_noise,
    validate_sample_initialization,
)


DTYPES = (
    pytest.param(jnp.float64, 5e-10, 5e-12, id="float64"),
    pytest.param(jnp.float32, 1e-4, 1e-5, id="float32"),
)

BOUNDARY_PROFILES = (
    pytest.param(jnp.float64, 5e-13, 2e-13, 2e-11, id="float64"),
    pytest.param(jnp.float32, 2e-5, 2e-6, 5e-5, id="float32"),
)


def _valid_problem(dtype=jnp.float64, *, n_samples: int = 7, dimension: int = 2):
    n_components = 3
    weights = jnp.asarray([0.2, 0.35, 0.45], dtype=dtype)
    means = jnp.asarray(
        [[-1.0, 0.3], [0.2, -0.5], [1.1, 0.8]], dtype=dtype
    )
    covariances = jnp.asarray(
        [
            [[0.8, 0.1], [0.1, 0.6]],
            [[0.7, -0.08], [-0.08, 0.9]],
            [[0.5, 0.04], [0.04, 0.75]],
        ],
        dtype=dtype,
    )
    if dimension != 2:
        means = jnp.pad(means, ((0, 0), (0, dimension - 2)))
        covariances = jnp.stack(
            [jnp.eye(dimension, dtype=dtype) * scale for scale in (0.8, 0.7, 0.6)]
        )
    observations = jnp.reshape(
        jnp.linspace(-1.2, 1.4, n_samples * dimension, dtype=dtype),
        (n_samples, dimension),
    )
    noise = jnp.broadcast_to(
        jnp.eye(dimension, dtype=dtype) * jnp.asarray(0.1, dtype=dtype),
        (n_samples, dimension, dimension),
    )
    return Params(weights, means, covariances), observations, noise


def _assert_validated_container(value, *, dtype) -> None:
    assert isinstance(value, ValidatedIdentityInputs)
    assert value._fields == (
        "parameters",
        "observations",
        "measurement_covariances",
    )
    for array in (
        value.parameters.weights,
        value.parameters.means,
        value.parameters.covariances,
        value.observations,
        value.measurement_covariances,
    ):
        assert isinstance(array, jax.Array)
        assert array.dtype == dtype


def _assert_message(error: pytest.ExceptionInfo[BaseException], *parts: str) -> None:
    message = str(error.value).lower()
    for part in parts:
        assert part.lower() in message, message


def _assert_shape_error(call, *, received, expected) -> None:
    with pytest.raises(ValidationError) as error:
        call()
    _assert_message(error, f"received {received}", f"expected {expected}")


def _control_preparation_api():
    assert hasattr(validation_module, "PreparedControls")
    assert hasattr(validation_module, "validate_controls")
    return validation_module.PreparedControls, validation_module.validate_controls


def _operation_specific_isotropic_noise_api():
    """Return the explicit fit/inference adapters required by the contract.

    The generic adapter cannot tell whether ``(N, 1)`` is a forbidden fitting
    shape or the legitimate inference batch shape ``B=(N, 1)``.  Operation
    context, rather than shape guessing, must make that distinction.
    """

    assert hasattr(validation_module, "fit_isotropic_noise"), (
        "identity validation must expose an explicit fit_isotropic_noise "
        "entry point"
    )
    assert hasattr(validation_module, "inference_isotropic_noise"), (
        "identity validation must expose an explicit "
        "inference_isotropic_noise entry point"
    )
    return (
        validation_module.fit_isotropic_noise,
        validation_module.inference_isotropic_noise,
    )


def _adjacent_weight_totals(dtype, threshold: float) -> tuple[np.generic, np.generic]:
    numpy_dtype = np.dtype(dtype)
    one = np.asarray(1.0, dtype=numpy_dtype)[()]
    infinity = np.asarray(np.inf, dtype=numpy_dtype)[()]
    candidate = one
    inside = one
    while float(candidate - one) <= threshold:
        inside = candidate
        candidate = np.nextafter(candidate, infinity, dtype=numpy_dtype)
    return inside, candidate


def test_validation_exception_hierarchy_is_actionable_and_specific():
    assert issubclass(ValidationError, ValueError)
    assert issubclass(PrecisionError, ValidationError)


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
def test_valid_control_preparation_returns_exact_rank_zero_jax_dtype(
    dtype, _rtol, _atol
):
    prepared_type, validate_controls = _control_preparation_api()
    prepared = validate_controls(
        factor_jitter=1e-6,
        covariance_ridge=2e-4,
        dtype=dtype,
    )

    assert isinstance(prepared, prepared_type)
    assert prepared._fields == ("factor_jitter", "covariance_ridge")
    for value in prepared:
        assert isinstance(value, jax.Array)
        assert value.shape == ()
        assert value.dtype == dtype
    np.testing.assert_allclose(np.asarray(prepared.factor_jitter), 1e-6)
    np.testing.assert_allclose(np.asarray(prepared.covariance_ridge), 2e-4)


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
def test_xd_ip_shape_001_inference_preserves_single_and_batch_shapes(
    dtype, _rtol, _atol
):
    """XD-IP-SHAPE-001: no artificial length-one inference batch axis."""

    parameters, batch_observations, batch_noise = _valid_problem(
        dtype, n_samples=5
    )
    single = canonicalize_inference_inputs(
        parameters,
        batch_observations[0],
        batch_noise[0],
        dtype=dtype,
    )
    batch = canonicalize_inference_inputs(
        parameters,
        batch_observations,
        batch_noise,
        dtype=dtype,
    )

    _assert_validated_container(single, dtype=dtype)
    _assert_validated_container(batch, dtype=dtype)
    assert single.observations.shape == (2,)
    assert single.measurement_covariances.shape == (2, 2)
    assert batch.observations.shape == (5, 2)
    assert batch.measurement_covariances.shape == (5, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(single.observations), np.asarray(batch.observations[0])
    )
    np.testing.assert_array_equal(
        np.asarray(single.measurement_covariances),
        np.asarray(batch.measurement_covariances[0]),
    )


def test_xd_ip_shape_002_valid_fit_shapes_are_canonical_jax_arrays():
    parameters, observations, noise = _valid_problem(jnp.float64)
    validated = canonicalize_fit_inputs(
        parameters, observations, noise, dtype=jnp.float64
    )

    _assert_validated_container(validated, dtype=jnp.float64)
    assert validated.parameters.weights.shape == (3,)
    assert validated.parameters.means.shape == (3, 2)
    assert validated.parameters.covariances.shape == (3, 2, 2)
    assert validated.observations.shape == (7, 2)
    assert validated.measurement_covariances.shape == (7, 2, 2)


def test_xd_ip_shape_001_zero_sized_inference_batch_is_rejected():
    parameters, _, _ = _valid_problem(jnp.float64)
    observations = jnp.empty((0, 2), dtype=jnp.float64)
    noise = jnp.empty((0, 2, 2), dtype=jnp.float64)

    with pytest.raises(ValidationError) as error:
        canonicalize_inference_inputs(
            parameters, observations, noise, dtype=jnp.float64
        )
    _assert_message(error, "received (0, 2)", "nonempty")


@pytest.mark.parametrize(
    "field,received,expected",
    [
        pytest.param("observations", "(2, 7)", "(7, 2)", id="observations"),
        pytest.param(
            "measurement_covariances",
            "(2, 7, 2)",
            "(7, 2, 2)",
            id="measurement-covariances",
        ),
        pytest.param("weights", "(3, 1)", "(3,)", id="weights"),
        pytest.param("means", "(2, 3)", "(3, 2)", id="means"),
        pytest.param(
            "covariances", "(2, 3, 2)", "(3, 2, 2)", id="covariances"
        ),
    ],
)
def test_xd_ip_shape_002_transposed_fit_inputs_name_received_and_expected_shapes(
    field, received, expected
):
    parameters, observations, noise = _valid_problem(jnp.float64)
    if field == "observations":
        observations = observations.T
    elif field == "measurement_covariances":
        noise = jnp.transpose(noise, (1, 0, 2))
    elif field == "weights":
        parameters = parameters._replace(weights=parameters.weights[:, None])
    elif field == "means":
        parameters = parameters._replace(means=parameters.means.T)
    else:
        parameters = parameters._replace(
            covariances=jnp.transpose(parameters.covariances, (1, 0, 2))
        )

    _assert_shape_error(
        lambda: canonicalize_fit_inputs(
            parameters, observations, noise, dtype=jnp.float64
        ),
        received=received,
        expected=expected,
    )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_xd_ip_val_001_nonfinite_observations_are_rejected(invalid_value):
    parameters, observations, noise = _valid_problem(jnp.float64)
    observations = observations.at[0, 0].set(invalid_value)

    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters, observations, noise, dtype=jnp.float64
        )
    _assert_message(error, "observations", "finite")


@pytest.mark.parametrize(
    "weights,reason",
    [
        pytest.param([0.0, 0.55, 0.45], "strictly positive", id="zero"),
        pytest.param([-0.1, 0.55, 0.55], "strictly positive", id="negative"),
        pytest.param([0.2, 0.35, 0.46], "sum", id="sum-error"),
        pytest.param([0.2, np.nan, 0.8], "finite", id="nonfinite"),
    ],
)
def test_xd_ip_val_001_invalid_mixture_weights_are_rejected(weights, reason):
    parameters, observations, noise = _valid_problem(jnp.float64)
    parameters = parameters._replace(weights=jnp.asarray(weights, dtype=jnp.float64))

    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters, observations, noise, dtype=jnp.float64
        )
    _assert_message(error, "weights", reason)


@pytest.mark.parametrize(
    "dtype,weights",
    [
        pytest.param(
            jnp.float64, [0.2, 0.35, 0.4500000000002], id="float64"
        ),
        pytest.param(jnp.float32, [0.2, 0.35, 0.45001], id="float32"),
    ],
)
def test_xd_ip_val_001_weight_sum_within_tolerance_is_preserved_not_rescaled(
    dtype, weights
):
    parameters, observations, noise = _valid_problem(dtype)
    supplied = jnp.asarray(weights, dtype=dtype)
    parameters = parameters._replace(weights=supplied)

    validated = canonicalize_fit_inputs(
        parameters, observations, noise, dtype=dtype
    )

    np.testing.assert_array_equal(
        np.asarray(validated.parameters.weights), np.asarray(supplied)
    )


@pytest.mark.parametrize(
    "dtype,weight_tolerance,_symmetry_tolerance,_psd_tolerance",
    BOUNDARY_PROFILES,
)
def test_xd_ip_val_001_weight_sum_acceptance_uses_exact_selected_dtype_boundary(
    dtype, weight_tolerance, _symmetry_tolerance, _psd_tolerance
):
    inside_total, outside_total = _adjacent_weight_totals(
        dtype, weight_tolerance
    )
    assert float(inside_total - 1.0) <= weight_tolerance
    assert float(outside_total - 1.0) > weight_tolerance

    def problem(total):
        weights = np.asarray([0.5, total - 0.5], dtype=np.dtype(dtype))
        parameters = Params(
            weights=jnp.asarray(weights),
            means=jnp.zeros((2, 2), dtype=dtype),
            covariances=jnp.asarray(
                [np.eye(2), np.eye(2)], dtype=dtype
            ),
        )
        observations = jnp.zeros((1, 2), dtype=dtype)
        noise = jnp.zeros((1, 2, 2), dtype=dtype)
        return weights, parameters, observations, noise

    inside_weights, parameters, observations, noise = problem(inside_total)
    accepted = canonicalize_fit_inputs(
        parameters, observations, noise, dtype=dtype
    )
    np.testing.assert_array_equal(
        np.asarray(accepted.parameters.weights), inside_weights
    )

    _, parameters, observations, noise = problem(outside_total)
    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters, observations, noise, dtype=dtype
        )
    _assert_message(error, "weights", "sum", "tolerance")


@pytest.mark.parametrize(
    "dtype,_weight_tolerance,symmetry_tolerance,_psd_tolerance",
    BOUNDARY_PROFILES,
)
def test_xd_ip_val_001_symmetry_acceptance_uses_adjacent_dtype_values(
    dtype, _weight_tolerance, symmetry_tolerance, _psd_tolerance
):
    numpy_dtype = np.dtype(dtype)
    inside = np.nextafter(
        np.asarray(symmetry_tolerance, dtype=numpy_dtype),
        np.asarray(0.0, dtype=numpy_dtype),
    )
    outside = np.nextafter(
        np.asarray(symmetry_tolerance, dtype=numpy_dtype),
        np.asarray(np.inf, dtype=numpy_dtype),
    )

    def problem(asymmetry):
        covariance = np.asarray(
            [[0.5, asymmetry], [0.0, 0.5]], dtype=numpy_dtype
        )
        parameters = Params(
            weights=jnp.asarray([1.0], dtype=dtype),
            means=jnp.zeros((1, 2), dtype=dtype),
            covariances=jnp.asarray(covariance[None, :, :]),
        )
        return parameters

    observations = jnp.zeros((1, 2), dtype=dtype)
    noise = jnp.zeros((1, 2, 2), dtype=dtype)
    accepted = canonicalize_fit_inputs(
        problem(inside), observations, noise, dtype=dtype
    )
    returned = np.asarray(accepted.parameters.covariances[0])
    np.testing.assert_array_equal(returned, returned.T)

    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            problem(outside), observations, noise, dtype=dtype
        )
    _assert_message(error, "parameter covariances", "symmetric")


@pytest.mark.parametrize(
    "dtype,_weight_tolerance,_symmetry_tolerance,psd_tolerance",
    BOUNDARY_PROFILES,
)
def test_xd_ip_val_001_psd_acceptance_uses_adjacent_dtype_values(
    dtype, _weight_tolerance, _symmetry_tolerance, psd_tolerance
):
    numpy_dtype = np.dtype(dtype)
    inside_magnitude = np.nextafter(
        np.asarray(psd_tolerance, dtype=numpy_dtype),
        np.asarray(0.0, dtype=numpy_dtype),
    )
    outside_magnitude = np.nextafter(
        np.asarray(psd_tolerance, dtype=numpy_dtype),
        np.asarray(np.inf, dtype=numpy_dtype),
    )
    parameters = Params(
        weights=jnp.asarray([1.0], dtype=dtype),
        means=jnp.zeros((1, 2), dtype=dtype),
        covariances=jnp.asarray([np.eye(2)], dtype=dtype),
    )
    observations = jnp.zeros((1, 2), dtype=dtype)

    inside_noise = jnp.asarray(
        [[[0.5, 0.0], [0.0, -inside_magnitude]]], dtype=dtype
    )
    accepted = canonicalize_fit_inputs(
        parameters, observations, inside_noise, dtype=dtype
    )
    np.testing.assert_array_equal(
        np.asarray(accepted.measurement_covariances),
        np.asarray(inside_noise),
    )

    outside_noise = jnp.asarray(
        [[[0.5, 0.0], [0.0, -outside_magnitude]]], dtype=dtype
    )
    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters, observations, outside_noise, dtype=dtype
        )
    _assert_message(error, "measurement covariances", "positive semidefinite")


@pytest.mark.parametrize(
    "target,reason",
    [
        pytest.param("asymmetric-v", "symmetric", id="asymmetric-v"),
        pytest.param("non-pd-v", "positive definite", id="non-pd-v"),
        pytest.param("asymmetric-s", "symmetric", id="asymmetric-s"),
        pytest.param("negative-s", "positive semidefinite", id="negative-s"),
    ],
)
def test_xd_ip_val_001_invalid_covariance_domains_are_rejected(target, reason):
    parameters, observations, noise = _valid_problem(jnp.float64)
    if target == "asymmetric-v":
        covariances = parameters.covariances.at[0, 0, 1].set(0.3)
        parameters = parameters._replace(covariances=covariances)
        field = "parameter covariances"
    elif target == "non-pd-v":
        covariances = parameters.covariances.at[0].set(
            jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
        )
        parameters = parameters._replace(covariances=covariances)
        field = "parameter covariances"
    elif target == "asymmetric-s":
        noise = noise.at[0, 0, 1].set(0.2)
        field = "measurement covariances"
    else:
        noise = noise.at[0].set(jnp.asarray([[0.1, 0.0], [0.0, -0.1]]))
        field = "measurement covariances"

    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters, observations, noise, dtype=jnp.float64
        )
    _assert_message(error, field, reason)


def test_xd_ip_val_001_rounding_scale_asymmetry_is_symmetrized_not_rejected():
    parameters, observations, noise = _valid_problem(jnp.float64)
    nearly_symmetric_v = parameters.covariances.at[0, 0, 1].add(1e-14)
    nearly_symmetric_s = noise.at[0, 0, 1].add(1e-14)
    parameters = parameters._replace(covariances=nearly_symmetric_v)

    validated = canonicalize_fit_inputs(
        parameters,
        observations,
        nearly_symmetric_s,
        dtype=jnp.float64,
    )

    np.testing.assert_array_equal(
        np.asarray(validated.parameters.covariances),
        np.swapaxes(np.asarray(validated.parameters.covariances), -1, -2),
    )
    np.testing.assert_array_equal(
        np.asarray(validated.measurement_covariances),
        np.swapaxes(np.asarray(validated.measurement_covariances), -1, -2),
    )


def test_xd_ip_val_001_rounding_scale_psd_residual_is_not_eigenvalue_clipped():
    parameters, observations, noise = _valid_problem(jnp.float64)
    rounding_psd = noise.at[0].set(
        jnp.asarray([[0.1, 0.0], [0.0, -1e-12]], dtype=jnp.float64)
    )

    validated = canonicalize_fit_inputs(
        parameters, observations, rounding_psd, dtype=jnp.float64
    )

    np.testing.assert_array_equal(
        np.asarray(validated.measurement_covariances[0]),
        np.asarray(rounding_psd[0]),
    )


@pytest.mark.parametrize(
    "input_kind",
    [
        pytest.param("bool-observations", id="bool-observations"),
        pytest.param("complex-observations", id="complex-observations"),
        pytest.param("integer-weights", id="integer-weights"),
        pytest.param("integer-parameters", id="integer-parameters"),
        pytest.param("integer-parameter-covariances", id="integer-v"),
        pytest.param("integer-measurement-covariances", id="integer-s"),
        pytest.param("complex-covariances", id="complex-covariances"),
    ],
)
def test_xd_ip_val_001_invalid_input_dtypes_are_rejected_before_casting(input_kind):
    parameters, observations, noise = _valid_problem(jnp.float64)
    if input_kind == "bool-observations":
        observations = observations.astype(jnp.bool_)
        reason = "boolean"
    elif input_kind == "complex-observations":
        observations = observations.astype(jnp.complex128)
        reason = "complex"
    elif input_kind == "integer-weights":
        parameters = parameters._replace(
            weights=jnp.asarray([1, 1, 1], dtype=jnp.int32)
        )
        reason = "floating"
    elif input_kind == "integer-parameters":
        parameters = parameters._replace(means=parameters.means.astype(jnp.int32))
        reason = "floating"
    elif input_kind == "integer-parameter-covariances":
        parameters = parameters._replace(
            covariances=parameters.covariances.astype(jnp.int32)
        )
        reason = "floating"
    elif input_kind == "integer-measurement-covariances":
        noise = noise.astype(jnp.int32)
        reason = "floating"
    else:
        parameters = parameters._replace(
            covariances=parameters.covariances.astype(jnp.complex128)
        )
        reason = "complex"

    with pytest.raises((TypeError, ValidationError)) as error:
        canonicalize_fit_inputs(
            parameters, observations, noise, dtype=jnp.float64
        )
    _assert_message(error, reason)


def test_xd_ip_val_001_values_must_remain_finite_after_selected_dtype_conversion():
    parameters, observations, noise = _valid_problem(jnp.float64)
    large_but_float64_finite = np.asarray(observations, dtype=np.float64).copy()
    large_but_float64_finite[0, 0] = 1e40

    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters,
            large_but_float64_finite,
            noise,
            dtype=jnp.float32,
        )
    _assert_message(error, "observations", "finite", "float32")


@pytest.mark.parametrize(
    "field,value_kind",
    [
        pytest.param("means", "nan", id="means-nan"),
        pytest.param("means", "inf", id="means-inf"),
        pytest.param("means", "boolean", id="means-boolean"),
        pytest.param("means", "complex", id="means-complex"),
        pytest.param("measurement covariances", "nan", id="noise-nan"),
        pytest.param("measurement covariances", "inf", id="noise-inf"),
        pytest.param("measurement covariances", "boolean", id="noise-boolean"),
        pytest.param("measurement covariances", "complex", id="noise-complex"),
    ],
)
def test_xd_ip_val_001_parameter_and_noise_fields_reject_invalid_values_and_dtypes(
    field, value_kind
):
    parameters, observations, noise = _valid_problem(jnp.float64)
    if field == "means":
        value = parameters.means
    else:
        value = noise

    if value_kind == "nan":
        value = value.at[(0,) * value.ndim].set(np.nan)
        reason = "finite"
    elif value_kind == "inf":
        value = value.at[(0,) * value.ndim].set(np.inf)
        reason = "finite"
    elif value_kind == "boolean":
        value = value.astype(jnp.bool_)
        reason = "boolean"
    else:
        value = value.astype(jnp.complex128) + 1.0j
        reason = "complex"

    if field == "means":
        parameters = parameters._replace(means=value)
    else:
        noise = value

    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters, observations, noise, dtype=jnp.float64
        )
    _assert_message(error, field, reason)


@pytest.mark.parametrize(
    "observations,received,expected",
    [
        pytest.param(np.empty((0, 2)), "(0, 2)", "N >= 1", id="empty"),
        pytest.param(np.empty((7, 0)), "(7, 0)", "D >= 1", id="zero-dimension"),
    ],
)
def test_xd_ip_val_001_empty_or_zero_dimension_fit_is_rejected(
    observations, received, expected
):
    parameters, _, noise = _valid_problem(jnp.float64)
    if observations.shape[0] == 0:
        noise = jnp.empty((0, 2, 2), dtype=jnp.float64)
    else:
        parameters = Params(
            weights=jnp.asarray([0.2, 0.35, 0.45], dtype=jnp.float64),
            means=jnp.empty((3, 0), dtype=jnp.float64),
            covariances=jnp.empty((3, 0, 0), dtype=jnp.float64),
        )
        noise = jnp.empty((7, 0, 0), dtype=jnp.float64)

    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters,
            jnp.asarray(observations),
            noise,
            dtype=jnp.float64,
        )
    _assert_message(error, f"received {received}", expected)


def test_xd_ip_val_001_sample_initializer_rejects_k_greater_than_n_without_replacement():
    with pytest.raises(ValidationError) as error:
        validate_sample_initialization(
            n_samples=3, n_components=4, replace=False
        )
    _assert_message(error, "n_components", "4", "n_samples", "3", "replace=false")

    validate_sample_initialization(n_samples=3, n_components=4, replace=True)


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
def test_xd_ip_noise_001_explicit_noise_adapters_construct_only_intended_entries(
    dtype, _rtol, _atol
):
    isotropic_variances = jnp.asarray([0.0, 0.1, 0.5, 2.0], dtype=dtype)
    isotropic = isotropic_noise(
        isotropic_variances, dimension=3, dtype=dtype
    )
    expected_isotropic = np.asarray(isotropic_variances)[:, None, None] * np.eye(
        3, dtype=np.asarray(isotropic_variances).dtype
    )[None, :, :]
    assert isinstance(isotropic, jax.Array)
    np.testing.assert_array_equal(np.asarray(isotropic), expected_isotropic)

    diagonal_variances = jnp.asarray(
        [[0.1, 0.2, 0.3], [0.4, 0.7, 0.5]], dtype=dtype
    )
    diagonal = diagonal_noise(diagonal_variances, dtype=dtype)
    expected_diagonal = np.stack(
        [np.diag(row) for row in np.asarray(diagonal_variances)]
    )
    assert isinstance(diagonal, jax.Array)
    np.testing.assert_array_equal(np.asarray(diagonal), expected_diagonal)

    correlated = jnp.asarray(
        [
            [[0.4, 0.1, 0.0], [0.1, 0.5, 0.03], [0.0, 0.03, 0.3]],
            [[0.8, -0.05, 0.01], [-0.05, 0.6, 0.08], [0.01, 0.08, 0.7]],
        ],
        dtype=dtype,
    )
    full = full_noise(correlated, dtype=dtype)
    assert isinstance(full, jax.Array)
    np.testing.assert_array_equal(np.asarray(full), np.asarray(correlated))

    single_full = full_noise(correlated[0], dtype=dtype)
    assert single_full.shape == (3, 3)
    np.testing.assert_array_equal(
        np.asarray(single_full), np.asarray(correlated[0])
    )

    integer_isotropic = isotropic_noise([0, 2], dimension=3, dtype=dtype)
    integer_diagonal = diagonal_noise([[1, 2, 3]], dtype=dtype)
    integer_shared = shared_full_noise(
        np.eye(3, dtype=np.int32), batch_shape=(2,), dtype=dtype
    )
    for constructed in (integer_isotropic, integer_diagonal, integer_shared):
        assert isinstance(constructed, jax.Array)
        assert constructed.dtype == dtype

    with pytest.raises((TypeError, ValidationError)) as error:
        full_noise(np.eye(3, dtype=np.int32), dtype=dtype)
    _assert_message(error, "full", "floating")


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
def test_xd_ip_noise_001_invalid_variances_and_ambiguous_isotropic_shape_fail(
    dtype, _rtol, _atol
):
    with pytest.raises(ValidationError) as error:
        isotropic_noise(jnp.ones((4, 1), dtype=dtype), dimension=3, dtype=dtype)
    _assert_message(error, "received (4, 1)", "expected (N,)")

    with pytest.raises(ValidationError) as error:
        isotropic_noise(jnp.asarray([0.1, -0.2], dtype=dtype), dimension=3, dtype=dtype)
    _assert_message(error, "isotropic", "nonnegative")

    with pytest.raises(ValidationError) as error:
        diagonal_noise(jnp.asarray([[0.1, -0.2, 0.3]], dtype=dtype), dtype=dtype)
    _assert_message(error, "diagonal", "nonnegative")


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
def test_xd_ip_noise_001_isotropic_supports_multi_axis_batch_shape(
    dtype, _rtol, _atol
):
    variances = jnp.asarray(
        [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=dtype
    )
    actual = isotropic_noise(variances, dimension=2, dtype=dtype)
    expected = np.asarray(variances)[..., None, None] * np.eye(
        2, dtype=np.asarray(variances).dtype
    )

    assert isinstance(actual, jax.Array)
    assert actual.shape == (2, 3, 2, 2)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
def test_xd_ip_noise_001_fit_isotropic_entry_point_requires_nonempty_rank_one_values(
    dtype, _rtol, _atol
):
    fit_isotropic_noise, _ = _operation_specific_isotropic_noise_api()
    variances = np.asarray([0.0, 0.1, 0.5, 2.0], dtype=np.dtype(dtype))

    actual = fit_isotropic_noise(variances, dimension=3, dtype=dtype)
    expected = variances[:, None, None] * np.eye(3, dtype=np.dtype(dtype))

    assert isinstance(actual, jax.Array)
    assert actual.shape == (4, 3, 3)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(np.asarray(actual), expected)
    off_diagonal = np.asarray(actual).copy()
    diagonal = np.arange(3)
    off_diagonal[:, diagonal, diagonal] = 0.0
    np.testing.assert_array_equal(off_diagonal, np.zeros_like(off_diagonal))


@pytest.mark.parametrize(
    "values,received",
    [
        pytest.param(np.asarray(0.2), "()", id="scalar"),
        pytest.param(np.empty((0,)), "(0,)", id="empty"),
        pytest.param(np.ones((4, 1)), "(4, 1)", id="forbidden-n-by-one"),
        pytest.param(np.ones((2, 3)), "(2, 3)", id="multi-axis"),
    ],
)
def test_xd_ip_noise_001_fit_isotropic_entry_point_rejects_non_fit_batch_shapes(
    values, received
):
    fit_isotropic_noise, _ = _operation_specific_isotropic_noise_api()

    with pytest.raises(ValidationError) as error:
        fit_isotropic_noise(values, dimension=2, dtype=jnp.float64)

    _assert_message(
        error,
        "fit isotropic",
        f"received {received}",
        "(N,)",
        "N >= 1",
    )


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
@pytest.mark.parametrize(
    "batch_shape",
    [
        pytest.param((), id="single-observation"),
        pytest.param((3,), id="one-batch-axis"),
        pytest.param((2, 1), id="formerly-ambiguous-two-by-one"),
        pytest.param((2, 1, 3), id="three-batch-axes"),
    ],
)
def test_xd_ip_noise_001_inference_isotropic_entry_point_accepts_exact_batch_shape(
    dtype, _rtol, _atol, batch_shape
):
    _, inference_isotropic_noise = _operation_specific_isotropic_noise_api()
    size = int(np.prod(batch_shape, dtype=np.int64)) if batch_shape else 1
    variances = np.linspace(0.0, 0.5, size, dtype=np.dtype(dtype)).reshape(
        batch_shape
    )

    actual = inference_isotropic_noise(
        variances, dimension=2, dtype=dtype
    )
    expected = variances[..., None, None] * np.eye(2, dtype=np.dtype(dtype))

    assert isinstance(actual, jax.Array)
    assert actual.shape == batch_shape + (2, 2)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(np.asarray(actual), expected)
    off_diagonal = np.asarray(actual).copy()
    diagonal = np.arange(2)
    off_diagonal[..., diagonal, diagonal] = 0.0
    np.testing.assert_array_equal(off_diagonal, np.zeros_like(off_diagonal))


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
def test_xd_ip_noise_001_two_by_one_isotropic_batch_round_trips_through_inference(
    dtype, _rtol, _atol
):
    _, inference_isotropic_noise = _operation_specific_isotropic_noise_api()
    parameters, _, _ = _valid_problem(dtype)
    observations = jnp.asarray(
        [[[-0.7, 0.2]], [[0.6, -0.1]]], dtype=dtype
    )
    variances = np.asarray([[0.1], [0.3]], dtype=np.dtype(dtype))
    noise = inference_isotropic_noise(
        variances, dimension=2, dtype=dtype
    )

    validated = canonicalize_inference_inputs(
        parameters, observations, noise, dtype=dtype
    )

    assert validated.observations.shape == (2, 1, 2)
    assert validated.measurement_covariances.shape == (2, 1, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(validated.measurement_covariances),
        variances[..., None, None] * np.eye(2, dtype=np.dtype(dtype)),
    )


def test_xd_ip_noise_001_inference_only_batch_output_cannot_broadcast_into_fit():
    _, inference_isotropic_noise = _operation_specific_isotropic_noise_api()
    parameters, observations, _ = _valid_problem(
        jnp.float64, n_samples=4
    )
    inference_batch = inference_isotropic_noise(
        np.ones((4, 1), dtype=np.float64),
        dimension=2,
        dtype=jnp.float64,
    )
    assert inference_batch.shape == (4, 1, 2, 2)

    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters, observations, inference_batch, dtype=jnp.float64
        )

    _assert_message(
        error,
        "measurement covariances",
        "received (4, 1, 2, 2)",
        "expected (4, 2, 2)",
    )


@pytest.mark.parametrize(
    "value,reason",
    [
        pytest.param(np.nan, "finite", id="nan"),
        pytest.param(np.inf, "finite", id="positive-infinity"),
        pytest.param(-np.inf, "finite", id="negative-infinity"),
        pytest.param(np.bool_(True), "boolean", id="boolean"),
        pytest.param(np.complex128(0.1 + 0.2j), "complex", id="complex"),
        pytest.param("0.1", "real numeric", id="nonnumeric"),
    ],
)
@pytest.mark.parametrize("operation", ["fit", "inference"])
def test_xd_ip_noise_001_operation_specific_isotropic_values_fail_before_cast(
    value, reason, operation
):
    fit_isotropic_noise, inference_isotropic_noise = (
        _operation_specific_isotropic_noise_api()
    )
    if operation == "fit":
        call = lambda: fit_isotropic_noise(
            np.asarray([value]), dimension=2, dtype=jnp.float32
        )
    else:
        call = lambda: inference_isotropic_noise(
            np.asarray(value), dimension=2, dtype=jnp.float32
        )

    with pytest.raises(ValidationError) as error:
        call()

    _assert_message(error, "isotropic", reason)


@pytest.mark.parametrize("operation", ["fit", "inference"])
@pytest.mark.parametrize(
    "value,reason",
    [
        pytest.param(-1e-50, "nonnegative", id="negative-before-underflow"),
        pytest.param(1e-50, "underflow", id="positive-underflow-to-zero"),
    ],
)
def test_xd_ip_noise_001_operation_specific_isotropic_preserves_source_domain(
    operation, value, reason
):
    source = np.asarray(value, dtype=np.float64)
    assert source != 0.0
    assert source.astype(np.float32) == 0.0
    fit_isotropic_noise, inference_isotropic_noise = (
        _operation_specific_isotropic_noise_api()
    )
    if operation == "fit":
        call = lambda: fit_isotropic_noise(
            source.reshape(1), dimension=2, dtype=jnp.float32
        )
    else:
        call = lambda: inference_isotropic_noise(
            source, dimension=2, dtype=jnp.float32
        )

    with pytest.raises(ValidationError) as error:
        call()

    expected_parts = ["isotropic", reason]
    if value > 0:
        expected_parts.append("float32")
    _assert_message(error, *expected_parts)


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
@pytest.mark.parametrize("operation", ["fit", "inference"])
def test_xd_ip_noise_001_operation_specific_isotropic_preserves_selected_subnormal(
    dtype, _rtol, _atol, operation
):
    numpy_dtype = np.dtype(dtype)
    tiny = np.nextafter(
        np.asarray(0.0, dtype=numpy_dtype),
        np.asarray(1.0, dtype=numpy_dtype),
        dtype=numpy_dtype,
    )
    assert tiny != 0.0
    fit_isotropic_noise, inference_isotropic_noise = (
        _operation_specific_isotropic_noise_api()
    )
    if operation == "fit":
        actual = fit_isotropic_noise(
            np.asarray([tiny], dtype=numpy_dtype),
            dimension=2,
            dtype=dtype,
        )[0]
    else:
        actual = inference_isotropic_noise(
            np.asarray(tiny, dtype=numpy_dtype),
            dimension=2,
            dtype=dtype,
        )

    returned = np.asarray(actual)
    assert returned[0, 0] != 0.0
    assert returned[1, 1] != 0.0
    np.testing.assert_array_equal(
        np.diag(returned), np.asarray([tiny, tiny], dtype=numpy_dtype)
    )
    assert returned[0, 1] == 0.0
    assert returned[1, 0] == 0.0


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
@pytest.mark.parametrize("operation", ["fit", "inference"])
def test_xd_ip_noise_001_operation_specific_isotropic_accepts_integer_variances(
    dtype, _rtol, _atol, operation
):
    fit_isotropic_noise, inference_isotropic_noise = (
        _operation_specific_isotropic_noise_api()
    )
    if operation == "fit":
        actual = fit_isotropic_noise(
            np.asarray([0, 2], dtype=np.int32),
            dimension=2,
            dtype=dtype,
        )
        expected_shape = (2, 2, 2)
    else:
        actual = inference_isotropic_noise(
            np.asarray(2, dtype=np.int32),
            dimension=2,
            dtype=dtype,
        )
        expected_shape = (2, 2)

    assert actual.shape == expected_shape
    assert actual.dtype == dtype
    assert np.issubdtype(np.asarray(actual).dtype, np.floating)


@pytest.mark.parametrize("representation", ["isotropic", "diagonal"])
def test_xd_ip_noise_001_negative_variance_is_checked_before_float32_cast(
    representation,
):
    negative_float64 = np.asarray([-1e-50], dtype=np.float64)
    assert negative_float64[0] < 0.0
    assert negative_float64.astype(np.float32)[0] == 0.0

    with pytest.raises(ValidationError) as error:
        if representation == "isotropic":
            isotropic_noise(
                negative_float64, dimension=2, dtype=jnp.float32
            )
        else:
            diagonal_noise(negative_float64, dtype=jnp.float32)
    _assert_message(error, representation, "nonnegative")


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPES)
def test_xd_ip_noise_001_shared_full_noise_requires_explicit_adapter(
    dtype, _rtol, _atol
):
    parameters, observations, _ = _valid_problem(
        dtype, n_samples=3, dimension=3
    )
    shared = jnp.asarray(
        [[0.4, 0.05, 0.0], [0.05, 0.3, 0.02], [0.0, 0.02, 0.5]],
        dtype=dtype,
    )

    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters, observations, shared, dtype=dtype
        )
    _assert_message(error, "received (3, 3)", "expected (3, 3, 3)", "shared_full_noise")

    expanded = shared_full_noise(shared, batch_shape=(3,), dtype=dtype)
    assert isinstance(expanded, jax.Array)
    assert expanded.shape == (3, 3, 3)
    np.testing.assert_array_equal(
        np.asarray(expanded), np.broadcast_to(np.asarray(shared), (3, 3, 3))
    )
    validated = canonicalize_fit_inputs(
        parameters, observations, expanded, dtype=dtype
    )
    np.testing.assert_array_equal(
        np.asarray(validated.measurement_covariances), np.asarray(expanded)
    )


def _large_float32_covariance() -> np.ndarray:
    covariance = np.eye(2, dtype=np.float32) * np.float32(2e38)
    assert np.all(np.isfinite(covariance))
    assert np.all(np.isfinite(np.linalg.cholesky(covariance)))
    return covariance


def test_xd_ip_val_001_large_finite_float32_parameter_covariance_stays_finite():
    large_covariance = _large_float32_covariance()
    parameters = Params(
        weights=jnp.asarray([1.0], dtype=jnp.float32),
        means=jnp.zeros((1, 2), dtype=jnp.float32),
        covariances=jnp.asarray(large_covariance[None, :, :]),
    )
    observations = jnp.asarray([[0.1, -0.2]], dtype=jnp.float32)
    noise = jnp.zeros((1, 2, 2), dtype=jnp.float32)

    validated = canonicalize_fit_inputs(
        parameters, observations, noise, dtype=jnp.float32
    )
    returned = np.asarray(validated.parameters.covariances[0])

    assert np.all(np.isfinite(returned))
    assert np.all(np.isfinite(np.linalg.cholesky(returned)))
    np.testing.assert_array_equal(returned, large_covariance)


@pytest.mark.parametrize("execution", ["eager", "jit"])
def test_xd_ip_val_001_large_float32_covariance_survives_eager_and_jit_posterior(
    execution,
):
    """VAL-001: canonical near-max V must not overflow in the E-step."""

    large_covariance = _large_float32_covariance()
    parameters = Params(
        weights=jnp.asarray([1.0], dtype=jnp.float32),
        means=jnp.zeros((1, 2), dtype=jnp.float32),
        covariances=jnp.asarray(large_covariance[None, :, :]),
    )
    observations = jnp.asarray([[0.1, -0.2]], dtype=jnp.float32)
    noise = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    validated = canonicalize_fit_inputs(
        parameters, observations, noise, dtype=jnp.float32
    )

    operation = posterior_components
    if execution == "jit":
        operation = jax.jit(posterior_components)
    result = operation(
        validated.parameters,
        validated.observations,
        validated.measurement_covariances,
    )

    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), result)
    assert not bool(np.asarray(result.numerical_failure))
    assert not np.any(np.asarray(result.failed_pairs))
    for value in (
        result.component_log_density,
        result.component_log_joint,
        result.score_samples,
        result.responsibilities,
        result.conditional_mean,
        result.conditional_covariance,
    ):
        assert np.all(np.isfinite(np.asarray(value)))
    np.testing.assert_array_equal(
        np.asarray(result.responsibilities), np.asarray([[1.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(result.conditional_mean),
        np.asarray([[[0.1, -0.2]]], dtype=np.float32),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.conditional_covariance),
        np.zeros((1, 1, 2, 2), dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )


@pytest.mark.parametrize("representation", ["full", "shared"])
def test_xd_ip_noise_001_large_finite_float32_measurement_covariance_stays_finite(
    representation,
):
    large_covariance = _large_float32_covariance()
    representable_total = large_covariance + np.eye(2, dtype=np.float32)
    assert np.all(np.isfinite(representable_total))
    assert np.all(np.isfinite(np.linalg.cholesky(representable_total)))
    if representation == "full":
        constructed = full_noise(
            np.stack([large_covariance, large_covariance]),
            dtype=jnp.float32,
        )
    else:
        constructed = shared_full_noise(
            large_covariance, batch_shape=(2,), dtype=jnp.float32
        )

    parameters = Params(
        weights=jnp.asarray([1.0], dtype=jnp.float32),
        means=jnp.zeros((1, 2), dtype=jnp.float32),
        covariances=jnp.asarray([np.eye(2, dtype=np.float32)]),
    )
    observations = jnp.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=jnp.float32)
    validated = canonicalize_fit_inputs(
        parameters, observations, constructed, dtype=jnp.float32
    )
    returned = np.asarray(validated.measurement_covariances)
    total = returned + np.asarray(parameters.covariances[0])[None, :, :]

    assert np.all(np.isfinite(np.asarray(constructed)))
    assert np.all(np.isfinite(returned))
    assert np.all(np.isfinite(total))
    assert np.all(np.isfinite(np.linalg.cholesky(total)))
    np.testing.assert_array_equal(
        returned, np.broadcast_to(large_covariance, (2, 2, 2))
    )


@pytest.mark.parametrize("target", ["parameter", "full-noise"])
def test_xd_ip_val_001_near_max_asymmetry_is_rejected_without_overflow_bypass(
    target,
):
    grossly_asymmetric = np.asarray(
        [[1.6e308, 1.6e308], [-1.6e308, 1.6e308]], dtype=np.float64
    )
    assert np.all(np.isfinite(grossly_asymmetric))
    parameters = Params(
        weights=jnp.asarray([1.0], dtype=jnp.float64),
        means=jnp.zeros((1, 2), dtype=jnp.float64),
        covariances=jnp.asarray([np.eye(2, dtype=np.float64)]),
    )
    observations = jnp.zeros((1, 2), dtype=jnp.float64)
    noise = jnp.zeros((1, 2, 2), dtype=jnp.float64)
    expected_field = "measurement covariances"
    if target == "parameter":
        parameters = parameters._replace(
            covariances=jnp.asarray(grossly_asymmetric[None, :, :])
        )
        expected_field = "parameter covariances"
    else:
        noise = jnp.asarray(grossly_asymmetric[None, :, :])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError) as error:
            canonicalize_fit_inputs(
                parameters, observations, noise, dtype=jnp.float64
            )

    _assert_message(error, expected_field, "symmetric")
    unsafe_warnings = [
        warning
        for warning in caught
        if "overflow" in str(warning.message).lower()
        or "invalid" in str(warning.message).lower()
    ]
    assert not unsafe_warnings


def test_xd_ip_val_001_parameter_pd_check_matches_selected_jax_float32_cholesky():
    covariance = np.asarray(
        [
            [2.0119961e16, 1.6779924e9],
            [1.6779924e9, 1.3994356e2],
        ],
        dtype=np.float32,
    )
    assert np.all(np.isfinite(np.linalg.cholesky(covariance)))
    jax_factor = np.asarray(jnp.linalg.cholesky(jnp.asarray(covariance)))
    assert not np.all(np.isfinite(jax_factor))

    parameters = Params(
        weights=jnp.asarray([1.0], dtype=jnp.float32),
        means=jnp.zeros((1, 2), dtype=jnp.float32),
        covariances=jnp.asarray(covariance[None, :, :]),
    )
    observations = jnp.zeros((1, 2), dtype=jnp.float32)
    noise = jnp.zeros((1, 2, 2), dtype=jnp.float32)

    with pytest.raises(ValidationError) as error:
        canonicalize_fit_inputs(
            parameters, observations, noise, dtype=jnp.float32
        )
    _assert_message(error, "parameter covariances", "positive definite")


@pytest.mark.parametrize("dtype,rtol,atol", DTYPES)
def test_xd_ip_dtype_001_integer_observation_produces_floating_noninteger_posterior(
    dtype, rtol, atol
):
    parameters = Params(
        weights=jnp.asarray([1.0], dtype=dtype),
        means=jnp.asarray([[0.0, 0.0]], dtype=dtype),
        covariances=jnp.asarray([np.eye(2)], dtype=dtype),
    )
    observations = jnp.asarray([0, 1], dtype=jnp.int32)
    noise = jnp.asarray([[0.5, 0.0], [0.0, 0.5]], dtype=dtype)

    validated = canonicalize_inference_inputs(
        parameters, observations, noise, dtype=dtype
    )
    posterior = posterior_components(
        validated.parameters,
        validated.observations,
        validated.measurement_covariances,
    )

    assert validated.observations.dtype == dtype
    assert posterior.conditional_mean.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(posterior.conditional_mean[0]),
        [0.0, 2.0 / 3.0],
        rtol=rtol,
        atol=atol,
    )
    assert not np.issubdtype(np.asarray(posterior.conditional_mean).dtype, np.integer)


@pytest.mark.parametrize(
    "requested_dtype",
    [
        pytest.param(jnp.float16, id="float16"),
        pytest.param(jnp.bfloat16, id="bfloat16"),
        pytest.param(jnp.bool_, id="boolean"),
        pytest.param(jnp.int32, id="integer"),
    ],
)
def test_xd_ip_dtype_002_unsupported_requested_dtype_raises_precision_error(
    requested_dtype,
):
    parameters, observations, noise = _valid_problem(jnp.float32)

    with pytest.raises(PrecisionError) as error:
        canonicalize_fit_inputs(
            parameters,
            observations,
            noise,
            dtype=requested_dtype,
        )
    _assert_message(error, "dtype", "float32", "float64")


def test_xd_ip_dtype_002_float64_request_fails_when_jax_x64_is_disabled():
    project_root = Path(__file__).resolve().parents[2]
    code = """
import sys
import jax
import jax.numpy as jnp
from development.identity_xd import Params
from development.validation import PrecisionError, canonicalize_inference_inputs

assert not jax.config.x64_enabled
params = Params(
    jnp.asarray([1.0], dtype=jnp.float32),
    jnp.asarray([[0.0]], dtype=jnp.float32),
    jnp.asarray([[[1.0]]], dtype=jnp.float32),
)
try:
    canonicalize_inference_inputs(
        params,
        jnp.asarray([0.0], dtype=jnp.float32),
        jnp.asarray([[0.1]], dtype=jnp.float32),
        dtype=jnp.float64,
    )
except PrecisionError as error:
    message = str(error).lower()
    if "float64" in message and "x64" in message:
        raise SystemExit(0)
    print(message)
    raise SystemExit(2)
raise SystemExit(3)
"""
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_host_controls_preserve_tiny_python_values_when_jax_x64_is_disabled():
    """Host validation retains information unavailable to a raw jitted kernel."""

    project_root = Path(__file__).resolve().parents[2]
    code = """
import numpy as np
import jax
import jax.numpy as jnp
from development.fit_control import FitStatus, fit_converged, fit_fixed_steps
from development.identity_xd import Params
from development.validation import (
    PreparedControls,
    ValidationError,
    validate_controls,
)

assert not jax.config.x64_enabled

prepared = validate_controls(
    factor_jitter=0.125,
    covariance_ridge=0.25,
    dtype=jnp.float32,
)
assert isinstance(prepared, PreparedControls)
assert prepared._fields == ("factor_jitter", "covariance_ridge")
assert prepared.factor_jitter.shape == ()
assert prepared.covariance_ridge.shape == ()
assert prepared.factor_jitter.dtype == jnp.float32
assert prepared.covariance_ridge.dtype == jnp.float32

def expect_control_error(field, value, *, negative):
    try:
        validate_controls(dtype=jnp.float32, **{field: value})
    except ValidationError as error:
        message = str(error).lower()
        assert field in message
        if negative:
            assert "nonnegative" in message
        else:
            assert "float32" in message
            assert "zero" in message or "underflow" in message
        return
    raise AssertionError(f"{field}={value!r} was accepted")

params = Params(
    jnp.asarray([1.0], dtype=jnp.float32),
    jnp.asarray([[0.0]], dtype=jnp.float32),
    jnp.asarray([[[1.0]]], dtype=jnp.float32),
)
x = jnp.asarray([[0.1]], dtype=jnp.float32)
noise = jnp.asarray([[[0.1]]], dtype=jnp.float32)

for field in ("factor_jitter", "covariance_ridge"):
    for value in (-1e-50, 1e-50):
        expect_control_error(field, value, negative=value < 0.0)
        for mode in ("converged", "fixed"):
            if mode == "converged":
                result = fit_converged(
                    params, x, noise, max_iter=0, **{field: value}
                )
            else:
                result = fit_fixed_steps(
                    params, x, noise, n_steps=0, **{field: value}
                )
            assert bool(np.asarray(result.numerical_failure))
            assert not bool(np.asarray(result.collapsed))
            assert not np.any(np.asarray(result.collapsed_components))
            assert int(np.asarray(result.status)) == int(FitStatus.NUMERICAL_FAILURE)
            assert int(np.asarray(result.n_iter)) == 0
            assert np.asarray(result.history).shape == (1,)
            for actual, expected in zip(result.parameters, params):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

# Deliberately no assertion for a raw already-jitted kernel: with x64 disabled,
# JAX may canonicalize a dynamic Python scalar to float32 zero before kernel code.
"""
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
