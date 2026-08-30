"""Red tests for the temporary eager general-XD input boundary.

These tests intentionally target the not-yet-created
``development.general_validation`` module.  The fixed-``M`` numerical leaf in
``development.general_xd`` continues to accept canonical arrays; this eager
layer is responsible for explicit projection/noise modes, selected-dtype
validation, deterministic boolean-mask grouping, and public
``no_informative_weight`` failures.
"""

from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError, fields, is_dataclass
import os
from pathlib import Path
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import development
from development.identity_xd import Params
from development.validation import PrecisionError, ValidationError


DTYPES = (
    pytest.param(jnp.float64, id="float64"),
    pytest.param(jnp.float32, id="float32"),
)


@pytest.fixture
def general_validation():
    """Import inside the test so the complete red inventory still collects."""

    return importlib.import_module("development.general_validation")


def _params(dtype=jnp.float64, *, dimension: int = 3) -> Params:
    means = np.asarray(
        [[-0.8, 0.2, 0.5], [1.1, -0.4, 0.3]], dtype=np.float64
    )[:, :dimension]
    base_covariances = np.asarray(
        [
            [[0.9, 0.08, -0.03], [0.08, 0.7, 0.04], [-0.03, 0.04, 0.6]],
            [[0.6, -0.05, 0.02], [-0.05, 1.0, 0.06], [0.02, 0.06, 0.8]],
        ],
        dtype=np.float64,
    )[:, :dimension, :dimension]
    return Params(
        weights=jnp.asarray([0.4, 0.6], dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(base_covariances, dtype=dtype),
    )


def _canonical_problem(
    dtype=jnp.float64,
    *,
    batch_shape: tuple[int, ...] = (5,),
    observed_dimension: int = 2,
    latent_dimension: int = 3,
):
    params = _params(dtype, dimension=latent_dimension)
    size = int(np.prod(batch_shape, dtype=np.int64))
    x = jnp.reshape(
        jnp.linspace(
            -1.25,
            1.4,
            size * observed_dimension,
            dtype=dtype,
        ),
        batch_shape + (observed_dimension,),
    )
    row_projection = jnp.asarray(
        [
            [1.0, 0.2, -0.1],
            [-0.3, 0.8, 0.4],
            [0.15, -0.25, 0.9],
        ],
        dtype=dtype,
    )[:observed_dimension, :latent_dimension]
    projection = jnp.broadcast_to(
        row_projection,
        batch_shape + (observed_dimension, latent_dimension),
    )
    row_noise = jnp.eye(observed_dimension, dtype=dtype) * jnp.asarray(
        0.2, dtype=dtype
    )
    noise = jnp.broadcast_to(
        row_noise,
        batch_shape + (observed_dimension, observed_dimension),
    )
    return params, x, projection, noise


def _mask_fixture(dtype=jnp.float64):
    n_samples = 9
    potential_dimension = 4
    latent_dimension = 3
    params = _params(dtype, dimension=latent_dimension)
    x_full = jnp.asarray(
        np.arange(n_samples * potential_dimension, dtype=np.float64).reshape(
            n_samples, potential_dimension
        )
        / 7.0
        - 1.0,
        dtype=dtype,
    )
    observed_mask = np.asarray(
        [
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    )
    projection_full = jnp.asarray(
        np.arange(
            n_samples * potential_dimension * latent_dimension,
            dtype=np.float64,
        ).reshape(n_samples, potential_dimension, latent_dimension)
        / 23.0
        - 0.8,
        dtype=dtype,
    )
    covariance_rows = []
    for sample in range(n_samples):
        diagonal = np.asarray([0.3, 0.5, 0.7, 0.9]) + 0.03 * sample
        covariance_rows.append(
            np.diag(diagonal) + 0.01 * np.ones((potential_dimension,) * 2)
        )
    noise_full = jnp.asarray(covariance_rows, dtype=dtype)
    sample_weight = jnp.asarray(
        [1.0, 0.5, 2.0, 7.0, 1.5, 0.25, 3.0, 11.0, 0.75],
        dtype=dtype,
    )
    return (
        params,
        x_full,
        observed_mask,
        projection_full,
        noise_full,
        sample_weight,
    )


def _assert_message(error: pytest.ExceptionInfo[BaseException], *parts: str) -> None:
    message = str(error.value).lower()
    for part in parts:
        assert part.lower() in message, message


def _canonical_inference(api, params, x, projection, noise, *, dtype):
    return api.canonicalize_general_inference_inputs(
        params,
        x,
        projection=api.PerItemProjection(projection),
        noise=api.PerItemFullNoise(noise),
        dtype=dtype,
    )


def _canonical_fit(
    api,
    params,
    x,
    projection,
    noise,
    *,
    dtype,
    sample_weight=None,
):
    return api.canonicalize_general_fit_inputs(
        params,
        x,
        projection=api.PerItemProjection(projection),
        noise=api.PerItemFullNoise(noise),
        sample_weight=sample_weight,
        dtype=dtype,
    )


def _group_masked(
    api,
    params,
    x,
    mask,
    projection,
    noise,
    sample_weight=None,
    *,
    dtype,
    fitting=False,
    factor_jitter=0.0,
    covariance_ridge=0.0,
):
    operation = (
        api.group_masked_general_fit_inputs
        if fitting
        else api.group_masked_general_inputs
    )
    keywords = {}
    if fitting:
        keywords.update(
            factor_jitter=factor_jitter,
            covariance_ridge=covariance_ridge,
        )
    return operation(
        params,
        x,
        mask,
        projection=api.PerItemProjection(projection),
        noise=api.PerItemFullNoise(noise),
        sample_weight=sample_weight,
        dtype=dtype,
        **keywords,
    )


def test_general_validation_schema_and_exception_hierarchy(general_validation):
    required_module_names = {
        "ValidationError",
        "PrecisionError",
        "NoInformativeWeightError",
        "ValidatedGeneralInputs",
        "ValidatedGeneralFitInputs",
        "GeneralMaskGroup",
        "GroupedGeneralInputs",
        "GroupedGeneralFitInputs",
        "PerItemProjection",
        "SharedProjection",
        "IdentityProjection",
        "PerItemIsotropicNoise",
        "PerItemDiagonalNoise",
        "PerItemFullNoise",
        "SharedIsotropicNoise",
        "SharedDiagonalNoise",
        "SharedFullNoise",
        "canonicalize_general_inference_inputs",
        "canonicalize_general_fit_inputs",
        "group_masked_general_inputs",
        "group_masked_general_fit_inputs",
        "restore_grouped_rows",
    }
    assert required_module_names <= set(general_validation.__all__)
    assert general_validation.ValidationError is ValidationError
    assert general_validation.PrecisionError is PrecisionError
    assert issubclass(
        general_validation.NoInformativeWeightError, ValidationError
    )

    expected_validated_fields = {
        "ValidatedGeneralInputs": (
            "parameters",
            "observations",
            "projection_matrices",
            "measurement_covariances",
        ),
        "ValidatedGeneralFitInputs": (
            "parameters",
            "observations",
            "projection_matrices",
            "measurement_covariances",
            "sample_weight",
            "informative_weight",
        ),
    }
    for name, expected in expected_validated_fields.items():
        value = getattr(general_validation, name)
        assert value._fields == expected

    expected_group_fields = {
        "GeneralMaskGroup": (
            "group_index",
            "mask",
            "original_indices",
            "coordinate_indices",
            "observations",
            "projection_matrices",
            "measurement_covariances",
            "sample_weight",
        ),
        "GroupedGeneralInputs": (
            "parameters",
            "groups",
            "grouped_indices",
            "restoration_indices",
            "n_samples",
            "potential_observed_dimension",
            "latent_dimension",
            "informative_weight",
        ),
        "GroupedGeneralFitInputs": (
            "grouped",
            "informative_weight",
            "controls",
        ),
    }
    for name, expected in expected_group_fields.items():
        value = getattr(general_validation, name)
        assert is_dataclass(value)
        assert tuple(field.name for field in fields(value)) == expected
        assert "__dict__" not in value.__slots__

    tag_fields = {
        "PerItemProjection": ("values",),
        "SharedProjection": ("matrix",),
        "IdentityProjection": ("dimension",),
        "PerItemIsotropicNoise": ("variances",),
        "PerItemDiagonalNoise": ("variances",),
        "PerItemFullNoise": ("covariances",),
        "SharedIsotropicNoise": ("variance",),
        "SharedDiagonalNoise": ("variances",),
        "SharedFullNoise": ("covariance",),
    }
    for name, expected in tag_fields.items():
        value = getattr(general_validation, name)
        assert is_dataclass(value)
        assert tuple(field.name for field in fields(value)) == expected
        assert "__dict__" not in value.__slots__
        instance = value(*([3] * len(expected)))
        with pytest.raises(FrozenInstanceError):
            setattr(instance, expected[0], 4)

    for name in required_module_names:
        assert getattr(development, name) is getattr(general_validation, name)


@pytest.mark.parametrize("dtype", DTYPES)
def test_xd_gen_shape_001_canonical_single_batch_and_multi_axis_shapes(
    general_validation, dtype
):
    params, x, projection, noise = _canonical_problem(dtype)
    single = _canonical_inference(
        general_validation, params, x[0], projection[0], noise[0], dtype=dtype
    )
    batch = _canonical_inference(
        general_validation, params, x, projection, noise, dtype=dtype
    )
    multi_params, multi_x, multi_projection, multi_noise = _canonical_problem(
        dtype, batch_shape=(2, 1)
    )
    multi = _canonical_inference(
        general_validation,
        multi_params,
        multi_x,
        multi_projection,
        multi_noise,
        dtype=dtype,
    )

    assert isinstance(single, general_validation.ValidatedGeneralInputs)
    assert single.observations.shape == (2,)
    assert single.projection_matrices.shape == (2, 3)
    assert single.measurement_covariances.shape == (2, 2)
    assert batch.observations.shape == (5, 2)
    assert batch.projection_matrices.shape == (5, 2, 3)
    assert batch.measurement_covariances.shape == (5, 2, 2)
    assert multi.observations.shape == (2, 1, 2)
    assert multi.projection_matrices.shape == (2, 1, 2, 3)
    assert multi.measurement_covariances.shape == (2, 1, 2, 2)
    for result in (single, batch, multi):
        for array in (
            *result.parameters,
            result.observations,
            result.projection_matrices,
            result.measurement_covariances,
        ):
            assert isinstance(array, jax.Array)
            assert array.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_xd_gen_shape_001_m_zero_inference_is_canonical(
    general_validation, dtype
):
    params = _params(dtype, dimension=3)
    x = np.empty((4, 0), dtype=np.int32)
    projection = np.empty((4, 0, 3), dtype=np.dtype(dtype))
    noise = np.empty((4, 0, 0), dtype=np.dtype(dtype))

    validated = _canonical_inference(
        general_validation, params, x, projection, noise, dtype=dtype
    )

    assert validated.observations.shape == (4, 0)
    assert validated.projection_matrices.shape == (4, 0, 3)
    assert validated.measurement_covariances.shape == (4, 0, 0)
    assert validated.observations.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "batch_shape",
    (
        pytest.param((0,), id="zero-leading-axis"),
        pytest.param((2, 0), id="zero-inner-axis"),
    ),
)
@pytest.mark.parametrize(
    "observed_dimension",
    (
        pytest.param(2, id="m-positive"),
        pytest.param(0, id="m-zero"),
    ),
)
def test_xd_gen_shape_001_zero_sized_inference_batch_axes_are_canonical(
    general_validation, dtype, batch_shape, observed_dimension
):
    params, x, projection, noise = _canonical_problem(
        dtype,
        batch_shape=batch_shape,
        observed_dimension=observed_dimension,
        latent_dimension=3,
    )

    validated = _canonical_inference(
        general_validation,
        params,
        x,
        projection,
        noise,
        dtype=dtype,
    )

    assert validated.observations.shape == batch_shape + (observed_dimension,)
    assert validated.projection_matrices.shape == batch_shape + (
        observed_dimension,
        3,
    )
    assert validated.measurement_covariances.shape == batch_shape + (
        observed_dimension,
        observed_dimension,
    )
    for array in (
        validated.observations,
        validated.projection_matrices,
        validated.measurement_covariances,
    ):
        assert array.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_xd_gen_proj_001_shared_identity_and_repeated_per_item_agree_exactly(
    general_validation, dtype
):
    shared_source = np.asarray(
        [[1.0, 0.2, -0.1], [-0.3, 0.8, 0.4]], dtype=np.dtype(dtype)
    )
    params, x, per_item_projection, noise = _canonical_problem(
        dtype, batch_shape=(2, 1)
    )
    shared = general_validation.canonicalize_general_inference_inputs(
        params,
        x,
        projection=general_validation.SharedProjection(shared_source),
        noise=general_validation.PerItemFullNoise(noise),
        dtype=dtype,
    )
    per_item = general_validation.canonicalize_general_inference_inputs(
        params,
        x,
        projection=general_validation.PerItemProjection(per_item_projection),
        noise=general_validation.PerItemFullNoise(noise),
        dtype=dtype,
    )
    np.testing.assert_array_equal(
        np.asarray(shared.projection_matrices),
        np.broadcast_to(shared_source, (2, 1, 2, 3)),
    )
    np.testing.assert_array_equal(
        np.asarray(shared.projection_matrices),
        np.asarray(per_item.projection_matrices),
    )

    identity_params, identity_x, _, identity_noise = _canonical_problem(
        dtype,
        batch_shape=(2, 1),
        observed_dimension=3,
        latent_dimension=3,
    )
    identity = general_validation.canonicalize_general_inference_inputs(
        identity_params,
        identity_x,
        projection=general_validation.IdentityProjection(3),
        noise=general_validation.PerItemFullNoise(identity_noise),
        dtype=dtype,
    )
    np.testing.assert_array_equal(
        np.asarray(identity.projection_matrices),
        np.broadcast_to(np.eye(3, dtype=np.dtype(dtype)), (2, 1, 3, 3)),
    )
    assert shared.projection_matrices.dtype == dtype
    assert identity.projection_matrices.dtype == dtype

    for invalid_dimension in (True, 0, -1, 1.5):
        with pytest.raises((TypeError, ValueError)) as error:
            general_validation.IdentityProjection(invalid_dimension)
        _assert_message(error, "dimension")

    # Tags retain sources without inspecting array shape/dtype prematurely.
    source = np.asarray([[1, 0], [0, 1]], dtype=np.int32)
    tag = general_validation.SharedProjection(source)
    assert tag.matrix is source
    with pytest.raises(ValidationError):
        general_validation.canonicalize_general_inference_inputs(
            params,
            x,
            projection=tag,
            noise=general_validation.PerItemFullNoise(noise),
            dtype=dtype,
        )


@pytest.mark.parametrize("target", ["rank-two", "singleton"])
def test_xd_gen_proj_001_raw_rank_two_and_singleton_projection_do_not_broadcast(
    general_validation, target
):
    params, x, projection, noise = _canonical_problem(jnp.float64)
    supplied = (
        projection[0]
        if target == "rank-two"
        else general_validation.PerItemProjection(projection[:1])
    )

    with pytest.raises(ValidationError) as error:
        general_validation.canonicalize_general_fit_inputs(
            params,
            x,
            projection=supplied,
            noise=general_validation.PerItemFullNoise(noise),
            dtype=jnp.float64,
        )

    _assert_message(error, "projection", "received", "expected", "(5, 2, 3)")


@pytest.mark.parametrize("dtype", DTYPES)
def test_xd_gen_noise_001_all_explicit_noise_modes_construct_expected_full_arrays(
    general_validation, dtype
):
    numpy_dtype = np.dtype(dtype)
    params, x, projection, _ = _canonical_problem(
        dtype,
        batch_shape=(2, 1),
        observed_dimension=3,
        latent_dimension=3,
    )
    per_isotropic_source = np.asarray([[0.1], [0.4]], dtype=numpy_dtype)
    per_diagonal_source = np.asarray(
        [[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]], dtype=numpy_dtype
    )
    per_full_source = np.stack(
        [
            np.stack([np.diag(row) for row in item])
            for item in per_diagonal_source
        ]
    )

    shared_full_source = np.asarray(
        [[0.4, 0.05, 0.0], [0.05, 0.3, 0.02], [0.0, 0.02, 0.5]],
        dtype=numpy_dtype,
    )
    specifications = (
        general_validation.PerItemIsotropicNoise(per_isotropic_source),
        general_validation.PerItemDiagonalNoise(per_diagonal_source),
        general_validation.PerItemFullNoise(per_full_source),
        general_validation.SharedIsotropicNoise(
            np.asarray(0.25, dtype=numpy_dtype)
        ),
        general_validation.SharedDiagonalNoise(
            np.asarray([0.1, 0.2, 0.3], dtype=numpy_dtype)
        ),
        general_validation.SharedFullNoise(shared_full_source),
    )
    canonical = [
        general_validation.canonicalize_general_inference_inputs(
            params,
            x,
            projection=general_validation.PerItemProjection(projection),
            noise=specification,
            dtype=dtype,
        ).measurement_covariances
        for specification in specifications
    ]
    (
        per_isotropic,
        per_diagonal,
        per_full,
        shared_isotropic,
        shared_diagonal,
        shared_full,
    ) = canonical

    expected_isotropic = per_isotropic_source[..., None, None] * np.eye(
        3, dtype=numpy_dtype
    )
    np.testing.assert_array_equal(np.asarray(per_isotropic), expected_isotropic)
    np.testing.assert_array_equal(np.asarray(per_diagonal), per_full_source)
    np.testing.assert_array_equal(np.asarray(per_full), per_full_source)
    np.testing.assert_array_equal(
        np.asarray(shared_isotropic),
        np.broadcast_to(0.25 * np.eye(3, dtype=numpy_dtype), (2, 1, 3, 3)),
    )
    np.testing.assert_array_equal(
        np.asarray(shared_diagonal),
        np.broadcast_to(np.diag([0.1, 0.2, 0.3]).astype(numpy_dtype), (2, 1, 3, 3)),
    )
    np.testing.assert_array_equal(
        np.asarray(shared_full)[0, 0],
        shared_full_source,
    )
    for value in (
        per_isotropic,
        per_diagonal,
        per_full,
        shared_isotropic,
        shared_diagonal,
        shared_full,
    ):
        assert value.shape == (2, 1, 3, 3)
        assert value.dtype == dtype

    integer_specifications = (
        general_validation.PerItemIsotropicNoise(
            np.ones((2, 1), dtype=np.int32)
        ),
        general_validation.PerItemDiagonalNoise(
            np.ones((2, 1, 3), dtype=np.int32)
        ),
        general_validation.PerItemFullNoise(
            np.broadcast_to(np.eye(3, dtype=np.int32), (2, 1, 3, 3))
        ),
        general_validation.SharedIsotropicNoise(np.asarray(1, dtype=np.int32)),
        general_validation.SharedDiagonalNoise(np.ones(3, dtype=np.int32)),
        general_validation.SharedFullNoise(np.eye(3, dtype=np.int32)),
    )
    for specification in integer_specifications:
        with pytest.raises(ValidationError) as error:
            general_validation.canonicalize_general_inference_inputs(
                params,
                x,
                projection=general_validation.PerItemProjection(projection),
                noise=specification,
                dtype=dtype,
            )
        _assert_message(error, "floating")

    negative_specifications = (
        general_validation.PerItemIsotropicNoise(
            np.asarray([[0.1], [-0.2]], dtype=numpy_dtype)
        ),
        general_validation.PerItemDiagonalNoise(
            np.asarray(
                [[[0.1, 0.2, 0.3]], [[0.4, -0.5, 0.6]]],
                dtype=numpy_dtype,
            )
        ),
        general_validation.SharedIsotropicNoise(
            np.asarray(-0.2, dtype=numpy_dtype)
        ),
        general_validation.SharedDiagonalNoise(
            np.asarray([0.1, -0.2, 0.3], dtype=numpy_dtype)
        ),
    )
    for specification in negative_specifications:
        with pytest.raises(ValidationError) as error:
            general_validation.canonicalize_general_inference_inputs(
                params,
                x,
                projection=general_validation.PerItemProjection(projection),
                noise=specification,
                dtype=dtype,
            )
        _assert_message(error, "nonnegative")


@pytest.mark.parametrize("target", ["rank-two", "singleton"])
def test_xd_gen_noise_001_raw_shared_and_singleton_full_noise_do_not_broadcast(
    general_validation, target
):
    params, x, projection, noise = _canonical_problem(jnp.float64)
    supplied = (
        noise[0]
        if target == "rank-two"
        else general_validation.PerItemFullNoise(noise[:1])
    )

    with pytest.raises(ValidationError) as error:
        general_validation.canonicalize_general_fit_inputs(
            params,
            x,
            projection=general_validation.PerItemProjection(projection),
            noise=supplied,
            dtype=jnp.float64,
        )

    _assert_message(
        error, "measurement", "received", "expected", "(5, 2, 2)"
    )


@pytest.mark.parametrize(
    "target,reason",
    (
        pytest.param("projection-shape", "expected", id="projection-shape"),
        pytest.param("projection-nan", "finite", id="projection-nan"),
        pytest.param("projection-bool", "boolean", id="projection-bool"),
        pytest.param("projection-integer", "floating", id="projection-integer"),
        pytest.param("noise-inf", "finite", id="noise-inf"),
        pytest.param("noise-complex", "complex", id="noise-complex"),
        pytest.param("noise-integer", "floating", id="noise-integer"),
        pytest.param("noise-asymmetric", "symmetric", id="noise-asymmetric"),
        pytest.param("noise-indefinite", "positive semidefinite", id="noise-indefinite"),
        pytest.param("parameter-integer", "floating", id="parameter-integer"),
        pytest.param("parameter-zero-weight", "strictly positive", id="zero-weight"),
        pytest.param("parameter-weight-sum", "sum", id="weight-sum"),
        pytest.param("parameter-non-pd", "positive definite", id="parameter-non-pd"),
        pytest.param("observation-nan", "finite", id="observation-nan"),
        pytest.param("observation-bool", "boolean", id="observation-bool"),
    ),
)
def test_xd_gen_val_001_projection_noise_parameter_and_observation_domains_fail_actionably(
    general_validation, target, reason
):
    params, x, projection, noise = _canonical_problem(jnp.float64)
    field = target.split("-")[0]
    if target == "projection-shape":
        projection = projection[:, :, :2]
    elif target == "projection-nan":
        projection = projection.at[0, 0, 0].set(np.nan)
    elif target == "projection-bool":
        projection = projection.astype(jnp.bool_)
    elif target == "projection-integer":
        projection = projection.astype(jnp.int32)
    elif target == "noise-inf":
        noise = noise.at[0, 0, 0].set(np.inf)
        field = "measurement"
    elif target == "noise-complex":
        noise = noise.astype(jnp.complex128) + 1.0j
        field = "measurement"
    elif target == "noise-integer":
        noise = noise.astype(jnp.int32)
        field = "measurement"
    elif target == "noise-asymmetric":
        noise = noise.at[0, 0, 1].set(0.1)
        field = "measurement"
    elif target == "noise-indefinite":
        noise = noise.at[0].set(jnp.diag(jnp.asarray([0.2, -0.1])))
        field = "measurement"
    elif target == "parameter-integer":
        params = params._replace(means=params.means.astype(jnp.int32))
        field = "means"
    elif target == "parameter-zero-weight":
        params = params._replace(
            weights=jnp.asarray([0.0, 1.0], dtype=jnp.float64)
        )
        field = "weights"
    elif target == "parameter-weight-sum":
        params = params._replace(
            weights=jnp.asarray([0.4, 0.7], dtype=jnp.float64)
        )
        field = "weights"
    elif target == "parameter-non-pd":
        params = params._replace(
            covariances=params.covariances.at[0, -1, -1].set(0.0)
        )
        field = "parameter"
    elif target == "observation-nan":
        x = x.at[0, 0].set(np.nan)
    else:
        x = x.astype(jnp.bool_)

    with pytest.raises(ValidationError) as error:
        _canonical_fit(
            general_validation,
            params,
            x,
            projection,
            noise,
            dtype=jnp.float64,
        )
    _assert_message(error, field, reason)


@pytest.mark.parametrize(
    "sample_weight,exception,reason",
    (
        pytest.param(1.0, ValidationError, "shape", id="scalar"),
        pytest.param(np.ones(1), ValidationError, "shape", id="singleton"),
        pytest.param(np.ones((5, 1)), ValidationError, "shape", id="column"),
        pytest.param(
            np.asarray([1, -1, 1, 1, 1]),
            ValidationError,
            "nonnegative",
            id="negative",
        ),
        pytest.param(
            np.asarray([1.0, np.nan, 1.0, 1.0, 1.0]),
            ValidationError,
            "finite",
            id="nan",
        ),
        pytest.param(
            np.asarray([1e-50, 1.0, 1.0, 1.0, 1.0]),
            PrecisionError,
            "underflow",
            id="positive-underflow",
        ),
        pytest.param(
            np.asarray([-1e-50, 1.0, 1.0, 1.0, 1.0]),
            ValidationError,
            "nonnegative",
            id="negative-underflow",
        ),
        pytest.param(
            np.asarray([1e40, 1.0, 1.0, 1.0, 1.0]),
            PrecisionError,
            "finite",
            id="conversion-overflow",
        ),
        pytest.param(
            np.asarray([3e38, 3e38, 0.0, 0.0, 0.0]),
            PrecisionError,
            "informative",
            id="total-overflow",
        ),
    ),
)
def test_xd_gen_val_001_sample_weight_exact_shape_domain_and_selected_dtype_guards(
    general_validation, sample_weight, exception, reason
):
    params, x, projection, noise = _canonical_problem(jnp.float32)
    with pytest.raises(exception) as error:
        _canonical_fit(
            general_validation,
            params,
            x,
            projection,
            noise,
            dtype=jnp.float32,
            sample_weight=sample_weight,
        )
    _assert_message(error, "sample_weight", reason)

    accepted_none = _canonical_fit(
        general_validation,
        params,
        x,
        projection,
        noise,
        sample_weight=None,
        dtype=jnp.float32,
    )
    accepted_integer = _canonical_fit(
        general_validation,
        params,
        x,
        projection,
        noise,
        sample_weight=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        dtype=jnp.float32,
    )
    accepted_integer_observations = _canonical_fit(
        general_validation,
        params,
        np.asarray(x).astype(np.int32),
        projection,
        noise,
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(accepted_none.sample_weight, np.ones(5))
    np.testing.assert_array_equal(
        accepted_integer.sample_weight, np.asarray([0, 1, 2, 3, 4])
    )
    assert accepted_none.sample_weight.dtype == jnp.float32
    assert accepted_integer.sample_weight.dtype == jnp.float32
    assert accepted_integer_observations.observations.dtype == jnp.float32


def test_xd_gen_val_001_float64_request_fails_with_x64_disabled(
    general_validation,
):
    del general_validation
    project_root = Path(__file__).resolve().parents[2]
    code = """
import importlib
import jax
import jax.numpy as jnp
from development.identity_xd import Params

assert not jax.config.x64_enabled
api = importlib.import_module("development.general_validation")
params = Params(
    jnp.asarray([1.0], dtype=jnp.float32),
    jnp.asarray([[0.0]], dtype=jnp.float32),
    jnp.asarray([[[1.0]]], dtype=jnp.float32),
)
try:
    api.canonicalize_general_inference_inputs(
        params,
        jnp.asarray([0.0], dtype=jnp.float32),
        projection=api.PerItemProjection(
            jnp.asarray([[1.0]], dtype=jnp.float32)
        ),
        noise=api.PerItemFullNoise(
            jnp.asarray([[0.1]], dtype=jnp.float32)
        ),
        dtype=jnp.float64,
    )
except api.PrecisionError as error:
    message = str(error).lower()
    assert "float64" in message and "x64" in message
    raise SystemExit(0)
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


@pytest.mark.parametrize("case", ["m-zero", "zero-weight"])
def test_xd_gen_m0_001_fixed_m_fit_requires_informative_weight(
    general_validation, case
):
    params = _params(jnp.float64)
    if case == "m-zero":
        x = np.empty((3, 0), dtype=np.float64)
        projection = np.empty((3, 0, 3), dtype=np.float64)
        noise = np.empty((3, 0, 0), dtype=np.float64)
        sample_weight = np.asarray([1.0, 0.0, 2.0])
    else:
        params, x, projection, noise = _canonical_problem(jnp.float64)
        sample_weight = np.zeros(5, dtype=np.float64)

    with pytest.raises(general_validation.NoInformativeWeightError) as error:
        _canonical_fit(
            general_validation,
            params,
            x,
            projection,
            noise,
            dtype=jnp.float64,
            sample_weight=sample_weight,
        )
    _assert_message(error, "no_informative_weight")


@pytest.mark.parametrize("dtype", DTYPES)
def test_xd_gen_missing_001_groups_literal_matrix_fixture_deterministically(
    general_validation, dtype
):
    fixture = _mask_fixture(dtype)
    grouped = _group_masked(
        general_validation, *fixture, dtype=dtype
    )
    params, x_full, mask, projection_full, noise_full, sample_weight = fixture
    del params

    assert isinstance(grouped, general_validation.GroupedGeneralInputs)
    assert grouped.n_samples == 9
    assert grouped.potential_observed_dimension == 4
    expected_masks = sorted({tuple(row) for row in mask})
    assert [group.mask for group in grouped.groups] == expected_masks
    expected_grouped_indices = tuple(
        index
        for group_mask in expected_masks
        for index in np.flatnonzero(np.all(mask == group_mask, axis=1))
    )
    expected_restoration = tuple(np.argsort(expected_grouped_indices))
    assert grouped.grouped_indices == expected_grouped_indices
    assert grouped.restoration_indices == expected_restoration
    assert grouped.latent_dimension == 3
    assert grouped.informative_weight.dtype == dtype
    assert float(np.asarray(grouped.informative_weight)) == pytest.approx(
        float(np.sum(np.asarray(sample_weight)[np.any(mask, axis=1)]))
    )

    for group_index, (group, group_mask) in enumerate(
        zip(grouped.groups, expected_masks, strict=True)
    ):
        row_indices = np.flatnonzero(np.all(mask == group_mask, axis=1))
        coordinates = np.flatnonzero(group_mask)
        assert group.group_index == group_index
        assert group.original_indices == tuple(row_indices)
        assert group.coordinate_indices == tuple(coordinates)
        np.testing.assert_array_equal(
            group.observations,
            np.asarray(x_full)[np.ix_(row_indices, coordinates)],
        )
        np.testing.assert_array_equal(
            group.projection_matrices,
            np.asarray(projection_full)[row_indices][:, coordinates, :],
        )
        expected_noise = np.asarray(noise_full)[row_indices]
        expected_noise = expected_noise[:, coordinates, :][:, :, coordinates]
        np.testing.assert_array_equal(group.measurement_covariances, expected_noise)
        np.testing.assert_array_equal(
            group.sample_weight, np.asarray(sample_weight)[row_indices]
        )


def test_xd_gen_missing_001_grouping_accepts_explicit_shared_projection_and_noise_tags(
    general_validation,
):
    params, x_full, mask, _, _, sample_weight = _mask_fixture(jnp.float64)
    shared_projection_source = np.asarray(
        [[1.0, 0.1, -0.2], [0.2, 0.9, 0.3], [-0.1, 0.4, 0.8], [0.5, -0.2, 0.1]]
    )
    shared_noise_source = np.asarray(
        [
            [0.5, 0.02, 0.0, 0.01],
            [0.02, 0.6, 0.03, 0.0],
            [0.0, 0.03, 0.7, -0.02],
            [0.01, 0.0, -0.02, 0.8],
        ]
    )
    grouped = general_validation.group_masked_general_inputs(
        params,
        x_full,
        mask,
        projection=general_validation.SharedProjection(shared_projection_source),
        noise=general_validation.SharedFullNoise(shared_noise_source),
        sample_weight=sample_weight,
        dtype=jnp.float64,
    )

    for group in grouped.groups:
        coordinates = np.asarray(group.coordinate_indices, dtype=np.intp)
        np.testing.assert_array_equal(
            group.projection_matrices,
            np.broadcast_to(
                shared_projection_source[coordinates],
                group.projection_matrices.shape,
            ),
        )
        expected_noise = shared_noise_source[np.ix_(coordinates, coordinates)]
        np.testing.assert_array_equal(
            group.measurement_covariances,
            np.broadcast_to(expected_noise, group.measurement_covariances.shape),
        )

    for shared_noise in (
        general_validation.SharedIsotropicNoise(np.asarray(0.4)),
        general_validation.SharedDiagonalNoise(
            np.asarray([0.4, 0.5, 0.6, 0.7])
        ),
    ):
        alternate = general_validation.group_masked_general_inputs(
            params,
            x_full,
            mask,
            projection=general_validation.SharedProjection(
                shared_projection_source
            ),
            noise=shared_noise,
            sample_weight=sample_weight,
            dtype=jnp.float64,
        )
        assert [group.mask for group in alternate.groups] == [
            group.mask for group in grouped.groups
        ]


def test_xd_gen_missing_001_grouping_accepts_identity_when_p_equals_d(
    general_validation,
):
    params = _params(jnp.float64, dimension=3)
    x = np.asarray([[0.2, -0.1, 0.5], [0.4, 0.7, -0.3]])
    mask = np.asarray([[True, False, True], [True, True, True]])
    grouped = general_validation.group_masked_general_inputs(
        params,
        x,
        mask,
        projection=general_validation.IdentityProjection(3),
        noise=general_validation.SharedIsotropicNoise(np.asarray(0.2)),
        dtype=jnp.float64,
    )

    for group in grouped.groups:
        coordinates = np.asarray(group.coordinate_indices, dtype=np.intp)
        expected = np.eye(3)[coordinates]
        np.testing.assert_array_equal(
            group.projection_matrices,
            np.broadcast_to(expected, group.projection_matrices.shape),
        )


@pytest.mark.parametrize(
    "projection_mode,noise_mode,reason",
    (
        pytest.param("raw", "full", "projection", id="raw-projection"),
        pytest.param("per-item", "raw", "noise", id="raw-noise"),
        pytest.param(
            "per-item", "per-isotropic", "per-item full", id="per-isotropic"
        ),
        pytest.param(
            "per-item", "per-diagonal", "per-item full", id="per-diagonal"
        ),
        pytest.param("identity", "full", "P == D", id="identity-dimension"),
    ),
)
def test_xd_gen_missing_001_rejects_uncontracted_or_raw_group_modes(
    general_validation, projection_mode, noise_mode, reason
):
    params, x, mask, projection, noise, sample_weight = _mask_fixture(jnp.float64)
    projection_spec = general_validation.PerItemProjection(projection)
    if projection_mode == "raw":
        projection_spec = projection
    elif projection_mode == "identity":
        projection_spec = general_validation.IdentityProjection(3)

    noise_spec = general_validation.PerItemFullNoise(noise)
    if noise_mode == "raw":
        noise_spec = noise
    elif noise_mode == "per-isotropic":
        noise_spec = general_validation.PerItemIsotropicNoise(
            np.ones(9, dtype=np.float64)
        )
    elif noise_mode == "per-diagonal":
        noise_spec = general_validation.PerItemDiagonalNoise(
            np.ones((9, 4), dtype=np.float64)
        )

    with pytest.raises(ValidationError) as error:
        general_validation.group_masked_general_inputs(
            params,
            x,
            mask,
            projection=projection_spec,
            noise=noise_spec,
            sample_weight=sample_weight,
            dtype=jnp.float64,
        )
    _assert_message(error, reason)


@pytest.mark.parametrize(
    "target",
    (
        "mask-dtype",
        "mask-shape",
        "observations",
        "projection",
        "noise",
        "sample-weight",
    ),
)
def test_xd_gen_missing_001_rejects_nonboolean_or_mismatched_mask_and_shapes(
    general_validation, target
):
    params, x, mask, projection, noise, sample_weight = _mask_fixture(jnp.float64)
    if target == "mask-dtype":
        mask = mask.astype(np.int8)
    elif target == "mask-shape":
        mask = mask[:-1]
    elif target == "observations":
        x = x[:, :-1]
    elif target == "projection":
        projection = projection[:, :-1]
    elif target == "noise":
        noise = noise[:, :-1, :]
    else:
        sample_weight = sample_weight[:-1]

    with pytest.raises(ValidationError) as error:
        _group_masked(
            general_validation,
            params,
            x,
            mask,
            projection,
            noise,
            sample_weight,
            dtype=jnp.float64,
        )
    _assert_message(error, "mask" if target.startswith("mask") else target.split("-")[0])


@pytest.mark.parametrize("target", ["observations", "projection", "noise", "sample-weight"])
def test_xd_gen_missing_001_rejects_nonfinite_values_even_when_masked_out(
    general_validation, target
):
    params, x, mask, projection, noise, sample_weight = _mask_fixture(jnp.float64)
    # Row 1 masks out source coordinate 1.  Values remain invalid even there.
    if target == "observations":
        x = x.at[1, 1].set(np.nan)
    elif target == "projection":
        projection = projection.at[1, 1, 0].set(np.inf)
    elif target == "noise":
        noise = noise.at[1, 1, 1].set(np.nan)
    else:
        sample_weight = sample_weight.at[1].set(np.inf)

    with pytest.raises(ValidationError) as error:
        _group_masked(
            general_validation,
            params,
            x,
            mask,
            projection,
            noise,
            sample_weight,
            dtype=jnp.float64,
        )
    _assert_message(error, "finite")


@pytest.mark.parametrize("defect", ["indefinite", "asymmetric"])
def test_xd_gen_missing_001_validates_full_covariance_before_slicing(
    general_validation, defect
):
    params = _params(jnp.float64)
    x = np.asarray([[0.2, 9.0]])
    mask = np.asarray([[True, False]])
    projection = np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    if defect == "indefinite":
        noise = np.asarray([[[0.5, 0.0], [0.0, -1.0]]])
        reason = "positive semidefinite"
    else:
        noise = np.asarray([[[0.5, 0.0], [0.2, 0.5]]])
        reason = "symmetric"

    with pytest.raises(ValidationError) as error:
        _group_masked(
            general_validation,
            params,
            x,
            mask,
            projection,
            noise,
            dtype=jnp.float64,
        )
    _assert_message(error, "measurement", reason)


@pytest.mark.parametrize(
    "dtype,full_scale,psd_tolerance",
    (
        pytest.param(jnp.float64, 1e20, 2e-11, id="float64"),
        pytest.param(jnp.float32, 1e10, 5e-5, id="float32"),
    ),
)
@pytest.mark.parametrize("noise_mode", ["per-item", "shared"])
def test_xd_gen_missing_001_revalidates_selected_covariance_blocks_at_group_scale(
    general_validation, dtype, full_scale, psd_tolerance, noise_mode
):
    """A globally tiny PSD residual can be material after coordinate selection."""

    numpy_dtype = np.dtype(dtype)
    params = _params(dtype)
    observations = np.zeros((1, 2), dtype=numpy_dtype)
    observed_mask = np.asarray([[False, True]], dtype=bool)
    projection = np.zeros((1, 2, 3), dtype=numpy_dtype)
    full_covariance = np.asarray(
        [[[full_scale, 0.0], [0.0, -1.0]]], dtype=numpy_dtype
    )

    # The full matrix satisfies the contract's scale-relative residual, while
    # its selected 1 x 1 principal block is materially indefinite at its own
    # scale.  Full-before-slice validation must therefore be followed by group-
    # scale validation of the canonical block.
    full_norm = float(np.linalg.norm(full_covariance[0], ord=2))
    assert -1.0 >= -psd_tolerance * max(1.0, full_norm)
    assert -1.0 < -psd_tolerance

    noise = (
        general_validation.PerItemFullNoise(full_covariance)
        if noise_mode == "per-item"
        else general_validation.SharedFullNoise(full_covariance[0])
    )
    with pytest.raises(ValidationError) as error:
        general_validation.group_masked_general_inputs(
            params,
            observations,
            observed_mask,
            projection=general_validation.PerItemProjection(projection),
            noise=noise,
            dtype=dtype,
        )

    _assert_message(error, "measurement", "positive semidefinite")
    message = str(error.value).lower()
    assert any(word in message for word in ("group", "principal", "selected"))


@pytest.mark.parametrize(
    "dtype,full_scale,symmetry_tolerance",
    (
        pytest.param(jnp.float64, 1e20, 2e-13, id="float64"),
        pytest.param(jnp.float32, 1e10, 2e-6, id="float32"),
    ),
)
@pytest.mark.parametrize("noise_mode", ["per-item", "shared"])
def test_xd_gen_missing_001_validates_selected_blocks_before_group_scale_symmetrization(
    general_validation, dtype, full_scale, symmetry_tolerance, noise_mode
):
    """Global scaling must not hide material asymmetry in a selected block."""

    numpy_dtype = np.dtype(dtype)
    params = _params(dtype)
    observations = np.zeros((1, 3), dtype=numpy_dtype)
    observed_mask = np.asarray([[False, True, True]], dtype=bool)
    projection = np.zeros((1, 3, 3), dtype=numpy_dtype)
    full_covariance = np.asarray(
        [
            [
                [full_scale, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=numpy_dtype,
    )

    full_matrix = full_covariance[0]
    selected_block = full_matrix[1:, 1:]
    full_residual = float(
        np.linalg.norm(full_matrix - full_matrix.T, ord=np.inf)
        / max(1.0, float(np.linalg.norm(full_matrix, ord=2)))
    )
    selected_residual = float(
        np.linalg.norm(selected_block - selected_block.T, ord=np.inf)
        / max(1.0, float(np.linalg.norm(selected_block, ord=2)))
    )
    assert full_residual <= symmetry_tolerance
    assert selected_residual > symmetry_tolerance

    noise = (
        general_validation.PerItemFullNoise(full_covariance)
        if noise_mode == "per-item"
        else general_validation.SharedFullNoise(full_covariance[0])
    )
    with pytest.raises(ValidationError) as error:
        general_validation.group_masked_general_inputs(
            params,
            observations,
            observed_mask,
            projection=general_validation.PerItemProjection(projection),
            noise=noise,
            dtype=dtype,
        )

    _assert_message(error, "measurement", "symmetric")
    message = str(error.value).lower()
    assert any(word in message for word in ("group", "principal", "selected"))


def test_xd_gen_hugenoise_001_large_finite_noise_is_not_missingness(
    general_validation,
):
    params = _params(jnp.float64)
    x = np.asarray([[0.2, -0.4], [0.7, 0.1]])
    mask = np.ones((2, 2), dtype=bool)
    projection = np.broadcast_to(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (2, 2, 3)
    )
    noise = np.broadcast_to(np.eye(2) * 1e200, (2, 2, 2))

    grouped = _group_masked(
        general_validation,
        params,
        x,
        mask,
        projection,
        noise,
        dtype=jnp.float64,
    )

    assert len(grouped.groups) == 1
    assert grouped.groups[0].mask == (True, True)
    assert grouped.groups[0].observations.shape == (2, 2)
    np.testing.assert_array_equal(grouped.groups[0].measurement_covariances, noise)


@pytest.mark.parametrize(
    "bad_field",
    [None, "mask", "projection", "noise"],
    ids=["valid", "bad-mask", "bad-projection", "bad-noise"],
)
def test_xd_gen_m0_001_grouped_inference_accepts_all_empty_but_grouped_fit_rejects(
    general_validation, bad_field
):
    params = _params(jnp.float64)
    x = np.empty((3, 0), dtype=np.float64)
    mask = np.empty((3, 0), dtype=bool)
    projection = np.empty((3, 0, 3), dtype=np.float64)
    noise = np.empty((3, 0, 0), dtype=np.float64)
    if bad_field == "mask":
        mask = np.empty((2, 0), dtype=bool)
    elif bad_field == "projection":
        projection = np.empty((3, 0, 2), dtype=np.float64)
    elif bad_field == "noise":
        noise = np.empty((3, 0, 1), dtype=np.float64)

    if bad_field is not None:
        with pytest.raises(ValidationError):
            _group_masked(
                general_validation,
                params,
                x,
                mask,
                projection,
                noise,
                dtype=jnp.float64,
            )
        return

    per_item_empty = general_validation.canonicalize_general_inference_inputs(
        params,
        x,
        projection=general_validation.PerItemProjection(projection),
        noise=general_validation.PerItemFullNoise(noise),
        dtype=jnp.float64,
    )
    shared_empty = general_validation.canonicalize_general_inference_inputs(
        params,
        x,
        projection=general_validation.PerItemProjection(projection),
        noise=general_validation.SharedFullNoise(
            np.empty((0, 0), dtype=np.float64)
        ),
        dtype=jnp.float64,
    )
    np.testing.assert_array_equal(per_item_empty.measurement_covariances, noise)
    np.testing.assert_array_equal(shared_empty.measurement_covariances, noise)
    grouped = _group_masked(
        general_validation,
        params,
        x,
        mask,
        projection,
        noise,
        dtype=jnp.float64,
    )
    assert len(grouped.groups) == 1
    group = grouped.groups[0]
    assert group.mask == ()
    assert group.observations.shape == (3, 0)
    assert group.projection_matrices.shape == (3, 0, 3)
    assert group.measurement_covariances.shape == (3, 0, 0)
    assert group.original_indices == (0, 1, 2)
    assert group.coordinate_indices == ()
    np.testing.assert_array_equal(group.sample_weight, np.ones(3))
    assert grouped.grouped_indices == (0, 1, 2)
    assert grouped.restoration_indices == (0, 1, 2)
    assert float(np.asarray(grouped.informative_weight)) == 0.0

    with pytest.raises(general_validation.NoInformativeWeightError) as error:
        general_validation.group_masked_general_fit_inputs(
            params,
            x,
            mask,
            projection=general_validation.PerItemProjection(projection),
            noise=general_validation.PerItemFullNoise(noise),
            dtype=jnp.float64,
        )
    _assert_message(error, "no_informative_weight")


def test_xd_gen_m0_001_grouped_fit_excludes_empty_rows_from_informative_weight(
    general_validation,
):
    params = _params(jnp.float64)
    x = np.asarray([[3.0, 4.0], [0.2, -0.1], [9.0, 10.0]])
    mask = np.asarray([[False, False], [True, False], [False, False]])
    projection = np.broadcast_to(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (3, 2, 3)
    )
    noise = np.broadcast_to(np.eye(2) * 0.2, (3, 2, 2))
    sample_weight = np.asarray([1e200, 2.5, 1e200])

    fit = general_validation.group_masked_general_fit_inputs(
        params,
        x,
        mask,
        projection=general_validation.PerItemProjection(projection),
        noise=general_validation.PerItemFullNoise(noise),
        sample_weight=sample_weight,
        factor_jitter=1e-6,
        covariance_ridge=2e-4,
        dtype=jnp.float64,
    )
    assert isinstance(fit, general_validation.GroupedGeneralFitInputs)
    assert float(np.asarray(fit.informative_weight)) == 2.5
    assert float(np.asarray(fit.grouped.informative_weight)) == 2.5
    assert float(np.asarray(fit.controls.factor_jitter)) == pytest.approx(1e-6)
    assert float(np.asarray(fit.controls.covariance_ridge)) == pytest.approx(2e-4)
    assert [group.mask for group in fit.grouped.groups] == [
        (False, False),
        (True, False),
    ]

    zero_informative = sample_weight.copy()
    zero_informative[1] = 0.0
    with pytest.raises(general_validation.NoInformativeWeightError) as error:
        general_validation.group_masked_general_fit_inputs(
            params,
            x,
            mask,
            projection=general_validation.PerItemProjection(projection),
            noise=general_validation.PerItemFullNoise(noise),
            sample_weight=zero_informative,
            dtype=jnp.float64,
        )
    _assert_message(error, "no_informative_weight")


def test_xd_gen_m0_001_control_errors_precede_no_informative_weight(
    general_validation,
):
    params = _params(jnp.float64)
    x = np.empty((3, 0), dtype=np.float64)
    mask = np.empty((3, 0), dtype=bool)
    projection = np.empty((3, 0, 3), dtype=np.float64)
    noise = np.empty((3, 0, 0), dtype=np.float64)
    common = dict(
        projection=general_validation.PerItemProjection(projection),
        noise=general_validation.PerItemFullNoise(noise),
        dtype=jnp.float64,
    )

    # Static validity of both controls is established before the negative
    # value-domain error in the other control and before the all-M=0 check.
    with pytest.raises(ValueError) as static_error:
        general_validation.group_masked_general_fit_inputs(
            params,
            x,
            mask,
            factor_jitter=-1.0,
            covariance_ridge=np.asarray([0.0]),
            **common,
        )
    assert not isinstance(
        static_error.value, general_validation.NoInformativeWeightError
    )
    _assert_message(static_error, "covariance_ridge", "rank-zero")

    with pytest.raises(ValidationError) as value_error:
        general_validation.group_masked_general_fit_inputs(
            params,
            x,
            mask,
            factor_jitter=-1.0,
            covariance_ridge=0.0,
            **common,
        )
    assert not isinstance(
        value_error.value, general_validation.NoInformativeWeightError
    )
    _assert_message(value_error, "factor_jitter", "nonnegative")

    with pytest.raises(general_validation.NoInformativeWeightError):
        general_validation.group_masked_general_fit_inputs(
            params,
            x,
            mask,
            factor_jitter=0.0,
            covariance_ridge=0.0,
            **common,
        )


def test_xd_gen_missing_001_restore_rows_supports_scalar_and_trailing_array_results(
    general_validation,
):
    grouped = _group_masked(
        general_validation, *_mask_fixture(jnp.float64), dtype=jnp.float64
    )
    scalar_values = [
        jnp.asarray(group.original_indices, dtype=jnp.float64) + 0.25
        for group in grouped.groups
    ]
    trailing_values = [
        jnp.stack(
            [
                jnp.asarray(group.original_indices, dtype=jnp.float64),
                -jnp.asarray(group.original_indices, dtype=jnp.float64),
            ],
            axis=-1,
        )
        for group in grouped.groups
    ]

    restored_scalar = general_validation.restore_grouped_rows(
        grouped, scalar_values, field="score_samples"
    )
    restored_trailing = general_validation.restore_grouped_rows(
        grouped, trailing_values, field="posterior_mean"
    )
    np.testing.assert_array_equal(restored_scalar, np.arange(9) + 0.25)
    np.testing.assert_array_equal(
        restored_trailing,
        np.stack([np.arange(9), -np.arange(9)], axis=-1),
    )


@pytest.mark.parametrize("defect", ["group-count", "leading-axis", "trailing-shape"])
def test_xd_gen_missing_001_restore_rows_rejects_group_count_leading_axis_or_trailing_shape_mismatch(
    general_validation, defect
):
    grouped = _group_masked(
        general_validation, *_mask_fixture(jnp.float64), dtype=jnp.float64
    )
    values = [
        jnp.zeros((len(group.original_indices), 2), dtype=jnp.float64)
        for group in grouped.groups
    ]
    if defect == "group-count":
        values = values[:-1]
    elif defect == "leading-axis":
        values[0] = jnp.zeros((values[0].shape[0] + 1, 2), dtype=jnp.float64)
    else:
        values[-1] = jnp.zeros((values[-1].shape[0], 3), dtype=jnp.float64)

    with pytest.raises(ValidationError) as error:
        general_validation.restore_grouped_rows(
            grouped, values, field="posterior_mean"
        )
    _assert_message(error, "posterior_mean", "group")
