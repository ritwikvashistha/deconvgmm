"""Tests-first contract for temporary general observed-space sampling.

The intended development-only surface has two deliberately separate layers::

    canonicalize_general_sampling_inputs(
        params, n, *, projection, noise, dtype
    )
    sample_observed_general(
        params, key, n, projection_matrices, measurement_covariances
    )
    sample_observed_general_from_specs(
        params, key, n, *, projection, noise, dtype
    )

The first and third functions are eager tagged boundaries.  The middle function
is the pure canonical JAX leaf.  General observed sampling does not introduce a
second latent sampler: :func:`development.inference.sample_latent` remains the
one canonical latent-mixture operation.

These tests intentionally precede ``development.general_sampling``.  They do
not claim public capability or grouped/ragged sampling support.
"""

from __future__ import annotations

import importlib
import inspect

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import development
from development.identity_xd import Params
from development.validation import ValidationError


DTYPES = (
    pytest.param(jnp.float64, id="float64"),
    pytest.param(jnp.float32, id="float32"),
)


@pytest.fixture(scope="module")
def general_sampling():
    """Import lazily so the complete tests-first inventory still collects."""

    return importlib.import_module("development.general_sampling")


def _params(dtype, *, one_component: bool = False) -> Params:
    if one_component:
        return Params(
            weights=jnp.asarray([1.0], dtype=dtype),
            means=jnp.asarray([[0.2, -0.4, 0.7]], dtype=dtype),
            covariances=jnp.asarray(
                [
                    [
                        [0.9, 0.12, -0.04],
                        [0.12, 0.7, 0.08],
                        [-0.04, 0.08, 0.6],
                    ]
                ],
                dtype=dtype,
            ),
        )
    return Params(
        weights=jnp.asarray([0.35, 0.65], dtype=dtype),
        means=jnp.asarray(
            [[-0.9, 0.2, 0.5], [1.2, -0.4, 0.1]], dtype=dtype
        ),
        covariances=jnp.asarray(
            [
                [
                    [0.8, 0.10, -0.04],
                    [0.10, 0.6, 0.05],
                    [-0.04, 0.05, 0.7],
                ],
                [
                    [0.5, -0.06, 0.03],
                    [-0.06, 0.9, 0.07],
                    [0.03, 0.07, 0.65],
                ],
            ],
            dtype=dtype,
        ),
    )


def _row_projection(dtype, observed_dimension: int = 2) -> np.ndarray:
    return np.asarray(
        [[1.0, 0.2, -0.1], [-0.3, 0.8, 0.4]], dtype=np.dtype(dtype)
    )[:observed_dimension]


def _row_noise(dtype, observed_dimension: int = 2) -> np.ndarray:
    return np.asarray(
        [[0.30, 0.08], [0.08, 0.20]], dtype=np.dtype(dtype)
    )[:observed_dimension, :observed_dimension]


def _canonical_arrays(
    dtype,
    *,
    n_samples: int = 5,
    observed_dimension: int = 2,
) -> tuple[jax.Array, jax.Array]:
    projection = jnp.asarray(
        np.broadcast_to(
            _row_projection(dtype, observed_dimension),
            (n_samples, observed_dimension, 3),
        ),
        dtype=dtype,
    )
    noise = jnp.asarray(
        np.broadcast_to(
            _row_noise(dtype, observed_dimension),
            (n_samples, observed_dimension, observed_dimension),
        ),
        dtype=dtype,
    )
    return projection, noise


def _tagged_specs(
    api,
    dtype,
    *,
    n_samples: int,
    observed_dimension: int,
    shared_projection: bool,
    shared_noise: bool,
):
    row_projection = _row_projection(dtype, observed_dimension)
    row_noise = _row_noise(dtype, observed_dimension)
    projection_values = np.broadcast_to(
        row_projection, (n_samples, observed_dimension, 3)
    )
    noise_values = np.broadcast_to(
        row_noise, (n_samples, observed_dimension, observed_dimension)
    )
    projection = (
        api.SharedProjection(row_projection)
        if shared_projection
        else api.PerItemProjection(projection_values)
    )
    noise = (
        api.SharedFullNoise(row_noise)
        if shared_noise
        else api.PerItemFullNoise(noise_values)
    )
    return projection, noise, projection_values, noise_values


def _assert_signature(function, positional_names, keyword_only_names) -> None:
    parameters = inspect.signature(function).parameters
    expected_names = tuple(positional_names) + tuple(keyword_only_names)
    assert tuple(parameters) == expected_names
    for name in positional_names:
        parameter = parameters[name]
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameter.default is inspect.Parameter.empty
    for name in keyword_only_names:
        parameter = parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


def _synchronized_numpy(array: jax.Array) -> np.ndarray:
    array.block_until_ready()
    return np.asarray(array)


def _analytic_latent_moments(params: Params) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(params.weights, dtype=np.float64)
    means = np.asarray(params.means, dtype=np.float64)
    covariances = np.asarray(params.covariances, dtype=np.float64)
    mean = np.sum(weights[:, None] * means, axis=0)
    centered = means - mean
    covariance = np.sum(
        weights[:, None, None]
        * (covariances + centered[:, :, None] * centered[:, None, :]),
        axis=0,
    )
    return mean, 0.5 * (covariance + covariance.T)


def _assert_empirical_moments(
    samples: np.ndarray,
    expected_mean: np.ndarray,
    expected_covariance: np.ndarray,
    *,
    dtype_atol: float,
    covariance_relative_bound: float,
) -> None:
    n_samples = samples.shape[0]
    empirical_mean = np.mean(samples, axis=0, dtype=np.float64)
    empirical_covariance = np.cov(
        np.asarray(samples, dtype=np.float64), rowvar=False, ddof=0
    )
    mean_bound = (
        6.0 * np.sqrt(np.diag(expected_covariance) / n_samples) + dtype_atol
    )
    assert np.all(np.abs(empirical_mean - expected_mean) <= mean_bound)
    relative_covariance_error = np.linalg.norm(
        empirical_covariance - expected_covariance, ord="fro"
    ) / np.linalg.norm(expected_covariance, ord="fro")
    assert relative_covariance_error <= covariance_relative_bound


def test_general_sampling_schema_signatures_and_exports(general_sampling):
    """The canonical leaf and eager tagged boundary have exact roles."""

    required = {
        "ValidatedGeneralSamplingInputs",
        "canonicalize_general_sampling_inputs",
        "sample_observed_general",
        "sample_observed_general_from_specs",
    }
    assert required <= set(general_sampling.__all__)
    assert general_sampling.ValidatedGeneralSamplingInputs._fields == (
        "parameters",
        "projection_matrices",
        "measurement_covariances",
        "n_samples",
        "observed_dimension",
    )
    for name in required:
        assert getattr(development, name) is getattr(general_sampling, name)

    _assert_signature(
        general_sampling.canonicalize_general_sampling_inputs,
        ("params", "n"),
        ("projection", "noise", "dtype"),
    )
    _assert_signature(
        general_sampling.sample_observed_general,
        (
            "params",
            "key",
            "n",
            "projection_matrices",
            "measurement_covariances",
        ),
        (),
    )
    _assert_signature(
        general_sampling.sample_observed_general_from_specs,
        ("params", "key", "n"),
        ("projection", "noise", "dtype"),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shared_projection,shared_noise",
    (
        pytest.param(True, True, id="shared-r-shared-s"),
        pytest.param(True, False, id="shared-r-per-item-s"),
        pytest.param(False, True, id="per-item-r-shared-s"),
        pytest.param(False, False, id="per-item-r-per-item-s"),
    ),
)
def test_xd_gen_sample_002_all_projection_noise_sharing_combinations_canonicalize(
    general_sampling, dtype, shared_projection, shared_noise
):
    """XD-GEN-SAMPLE-002: all four explicit sharing combinations are valid."""

    n_samples = 5
    params = _params(dtype)
    projection, noise, expected_projection, expected_noise = _tagged_specs(
        general_sampling,
        dtype,
        n_samples=n_samples,
        observed_dimension=2,
        shared_projection=shared_projection,
        shared_noise=shared_noise,
    )
    validated = general_sampling.canonicalize_general_sampling_inputs(
        params,
        n_samples,
        projection=projection,
        noise=noise,
        dtype=dtype,
    )

    assert isinstance(
        validated, general_sampling.ValidatedGeneralSamplingInputs
    )
    assert validated.n_samples == n_samples
    assert validated.observed_dimension == 2
    assert validated.projection_matrices.shape == (n_samples, 2, 3)
    assert validated.measurement_covariances.shape == (n_samples, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(validated.projection_matrices), expected_projection
    )
    np.testing.assert_array_equal(
        np.asarray(validated.measurement_covariances), expected_noise
    )
    for array in (
        *validated.parameters,
        validated.projection_matrices,
        validated.measurement_covariances,
    ):
        assert isinstance(array, jax.Array)
        assert array.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_xd_gen_sample_002_all_explicit_noise_tags_construct_full_covariances(
    general_sampling, dtype
):
    """Every Section 4 noise tag has an exact sampling-boundary meaning."""

    numpy_dtype = np.dtype(dtype)
    n_samples = 3
    projection = general_sampling.SharedProjection(_row_projection(dtype))
    per_isotropic = np.asarray([0.10, 0.20, 0.30], dtype=numpy_dtype)
    per_diagonal = np.asarray(
        [[0.10, 0.20], [0.20, 0.30], [0.30, 0.40]], dtype=numpy_dtype
    )
    per_full = np.stack([np.diag(row) for row in per_diagonal])
    shared_full = np.asarray(
        [[0.30, 0.08], [0.08, 0.20]], dtype=numpy_dtype
    )
    cases = (
        (
            general_sampling.PerItemIsotropicNoise(per_isotropic),
            per_isotropic[:, None, None] * np.eye(2, dtype=numpy_dtype),
        ),
        (general_sampling.PerItemDiagonalNoise(per_diagonal), per_full),
        (general_sampling.PerItemFullNoise(per_full), per_full),
        (
            general_sampling.SharedIsotropicNoise(
                np.asarray(0.25, dtype=numpy_dtype)
            ),
            np.broadcast_to(
                0.25 * np.eye(2, dtype=numpy_dtype), (n_samples, 2, 2)
            ),
        ),
        (
            general_sampling.SharedDiagonalNoise(
                np.asarray([0.15, 0.35], dtype=numpy_dtype)
            ),
            np.broadcast_to(
                np.diag([0.15, 0.35]).astype(numpy_dtype),
                (n_samples, 2, 2),
            ),
        ),
        (
            general_sampling.SharedFullNoise(shared_full),
            np.broadcast_to(shared_full, (n_samples, 2, 2)),
        ),
    )

    for noise, expected in cases:
        validated = general_sampling.canonicalize_general_sampling_inputs(
            _params(dtype),
            n_samples,
            projection=projection,
            noise=noise,
            dtype=dtype,
        )
        assert validated.measurement_covariances.shape == (n_samples, 2, 2)
        assert validated.measurement_covariances.dtype == dtype
        np.testing.assert_array_equal(
            np.asarray(validated.measurement_covariances), expected
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_xd_gen_sample_002_explicit_identity_projection_is_canonical(
    general_sampling, dtype
):
    """Identity sampling is explicit and only valid when ``M == D``."""

    n_samples = 4
    validated = general_sampling.canonicalize_general_sampling_inputs(
        _params(dtype),
        n_samples,
        projection=general_sampling.IdentityProjection(3),
        noise=general_sampling.SharedIsotropicNoise(
            np.asarray(0.2, dtype=np.dtype(dtype))
        ),
        dtype=dtype,
    )
    np.testing.assert_array_equal(
        np.asarray(validated.projection_matrices),
        np.broadcast_to(
            np.eye(3, dtype=np.dtype(dtype)), (n_samples, 3, 3)
        ),
    )
    draws = general_sampling.sample_observed_general_from_specs(
        _params(dtype),
        jax.random.key(20260825),
        n_samples,
        projection=general_sampling.IdentityProjection(3),
        noise=general_sampling.SharedIsotropicNoise(
            np.asarray(0.2, dtype=np.dtype(dtype))
        ),
        dtype=dtype,
    )
    assert draws.shape == (n_samples, 3)
    assert draws.dtype == dtype


@pytest.mark.parametrize(
    "defect,reason",
    (
        pytest.param("raw-projection", "projection", id="raw-projection"),
        pytest.param("raw-noise", "noise", id="raw-noise"),
        pytest.param("singleton-projection", "projection", id="singleton-r"),
        pytest.param("singleton-noise", "noise", id="singleton-s"),
        pytest.param("projection-n", "projection", id="wrong-r-leading-axis"),
        pytest.param("noise-n", "noise", id="wrong-s-leading-axis"),
        pytest.param("projection-d", "projection", id="wrong-latent-axis"),
        pytest.param("noise-m", "noise", id="wrong-observed-axis"),
    ),
)
def test_xd_gen_prng_001_eager_tags_never_infer_broadcast_or_leading_axes(
    general_sampling, defect, reason
):
    """XD-GEN-PRNG-001: per-item axes equal authoritative ``n`` exactly."""

    dtype = jnp.float64
    n_samples = 3
    projection_values, noise_values = _canonical_arrays(
        dtype, n_samples=n_samples
    )
    projection = general_sampling.PerItemProjection(projection_values)
    noise = general_sampling.PerItemFullNoise(noise_values)
    if defect == "raw-projection":
        projection = projection_values
    elif defect == "raw-noise":
        noise = noise_values
    elif defect == "singleton-projection":
        projection = general_sampling.PerItemProjection(projection_values[:1])
    elif defect == "singleton-noise":
        noise = general_sampling.PerItemFullNoise(noise_values[:1])
    elif defect == "projection-n":
        projection = general_sampling.PerItemProjection(
            jnp.broadcast_to(projection_values[0], (n_samples + 1, 2, 3))
        )
    elif defect == "noise-n":
        noise = general_sampling.PerItemFullNoise(
            jnp.broadcast_to(noise_values[0], (n_samples + 1, 2, 2))
        )
    elif defect == "projection-d":
        projection = general_sampling.PerItemProjection(
            jnp.zeros((n_samples, 2, 2), dtype=dtype)
        )
    else:
        noise = general_sampling.PerItemFullNoise(
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (n_samples, 3, 3))
        )

    with pytest.raises((TypeError, ValueError), match=reason):
        general_sampling.canonicalize_general_sampling_inputs(
            _params(dtype),
            n_samples,
            projection=projection,
            noise=noise,
            dtype=dtype,
        )


@pytest.mark.parametrize(
    "defect,reason",
    (
        pytest.param("shared-projection", "projection", id="shared-r"),
        pytest.param("singleton-projection", "projection", id="singleton-r"),
        pytest.param("projection-n", "projection", id="wrong-r-n"),
        pytest.param("projection-d", "projection", id="wrong-r-d"),
        pytest.param("shared-noise", "covariance|noise", id="shared-s"),
        pytest.param("singleton-noise", "covariance|noise", id="singleton-s"),
        pytest.param("noise-n", "covariance|noise", id="wrong-s-n"),
        pytest.param("noise-m", "covariance|noise", id="wrong-s-m"),
    ),
)
def test_xd_gen_prng_001_canonical_leaf_requires_exact_per_item_shapes(
    general_sampling, defect, reason
):
    """The pure leaf performs no singleton or shared-array broadcasting."""

    dtype = jnp.float64
    n_samples = 3
    projection, noise = _canonical_arrays(dtype, n_samples=n_samples)
    if defect == "shared-projection":
        projection = projection[0]
    elif defect == "singleton-projection":
        projection = projection[:1]
    elif defect == "projection-n":
        projection = jnp.broadcast_to(projection[0], (n_samples + 1, 2, 3))
    elif defect == "projection-d":
        projection = jnp.zeros((n_samples, 2, 2), dtype=dtype)
    elif defect == "shared-noise":
        noise = noise[0]
    elif defect == "singleton-noise":
        noise = noise[:1]
    elif defect == "noise-n":
        noise = jnp.broadcast_to(noise[0], (n_samples + 1, 2, 2))
    else:
        noise = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (n_samples, 3, 3))

    with pytest.raises((TypeError, ValueError), match=reason):
        general_sampling.sample_observed_general(
            _params(dtype),
            jax.random.key(20260825),
            n_samples,
            projection,
            noise,
        )


@pytest.mark.parametrize(
    "invalid_n",
    (
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="python-float"),
        pytest.param(np.float64(2.0), id="numpy-float"),
        pytest.param(True, id="python-bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param(jnp.asarray(True), id="jax-bool"),
        pytest.param("2", id="string"),
        pytest.param(jnp.asarray([2]), id="rank-positive"),
    ),
)
def test_xd_gen_prng_001_every_sampling_layer_rejects_invalid_n(
    general_sampling, invalid_n
):
    """XD-GEN-PRNG-001: ``n`` is static, integral, nonboolean, and nonnegative."""

    dtype = jnp.float64
    intended_n = 2
    projection_values, noise_values = _canonical_arrays(
        dtype, n_samples=intended_n
    )
    projection = general_sampling.PerItemProjection(projection_values)
    noise = general_sampling.PerItemFullNoise(noise_values)
    message = "n|sample|count|integer|nonnegative|static"

    with pytest.raises((TypeError, ValueError), match=message):
        general_sampling.canonicalize_general_sampling_inputs(
            _params(dtype),
            invalid_n,
            projection=projection,
            noise=noise,
            dtype=dtype,
        )
    with pytest.raises((TypeError, ValueError), match=message):
        general_sampling.sample_observed_general(
            _params(dtype),
            jax.random.key(20260825),
            invalid_n,
            projection_values,
            noise_values,
        )
    with pytest.raises((TypeError, ValueError), match=message):
        general_sampling.sample_observed_general_from_specs(
            _params(dtype),
            jax.random.key(20260825),
            invalid_n,
            projection=projection,
            noise=noise,
            dtype=dtype,
        )


@pytest.mark.parametrize(
    "n_value",
    (
        pytest.param(np.int64(2), id="numpy-integer"),
        pytest.param(jnp.asarray(2, dtype=jnp.int32), id="concrete-jax-scalar"),
    ),
)
def test_xd_gen_prng_001_concrete_scalar_integer_counts_are_static(
    general_sampling, n_value
):
    """Concrete index-like scalar counts retain the identity sampler policy."""

    dtype = jnp.float64
    projection_values, noise_values = _canonical_arrays(dtype, n_samples=2)
    validated = general_sampling.canonicalize_general_sampling_inputs(
        _params(dtype),
        n_value,
        projection=general_sampling.PerItemProjection(projection_values),
        noise=general_sampling.PerItemFullNoise(noise_values),
        dtype=dtype,
    )
    draws = general_sampling.sample_observed_general(
        validated.parameters,
        jax.random.key(20260825),
        n_value,
        validated.projection_matrices,
        validated.measurement_covariances,
    )
    assert validated.n_samples == 2
    assert draws.shape == (2, 2)


def test_xd_gen_prng_001_traced_dynamic_n_fails_actionably(general_sampling):
    """A dynamic traced count cannot determine random output shape."""

    dtype = jnp.float64
    projection, noise = _canonical_arrays(dtype, n_samples=2)
    params = _params(dtype)
    projection_spec = general_sampling.PerItemProjection(projection)
    noise_spec = general_sampling.PerItemFullNoise(noise)
    message = "n|static|integer|concrete|tracer"

    compiled_core = jax.jit(
        lambda count: general_sampling.sample_observed_general(
            params,
            jax.random.key(20260825),
            count,
            projection,
            noise,
        )
    )
    with pytest.raises(TypeError, match=message):
        compiled_core(jnp.asarray(2, dtype=jnp.int32))

    compiled_boundary = jax.jit(
        lambda count: general_sampling.sample_observed_general_from_specs(
            params,
            jax.random.key(20260825),
            count,
            projection=projection_spec,
            noise=noise_spec,
            dtype=dtype,
        )
    )
    with pytest.raises(TypeError, match=message):
        compiled_boundary(jnp.asarray(2, dtype=jnp.int32))


def test_xd_gen_prng_001_random_functions_require_key_by_signature(
    general_sampling,
):
    """No general random API can manufacture a hidden default key."""

    params = _params(jnp.float64)
    projection, noise = _canonical_arrays(jnp.float64, n_samples=1)
    with pytest.raises(TypeError):
        general_sampling.sample_observed_general(
            params,
            n=1,
            projection_matrices=projection,
            measurement_covariances=noise,
        )
    with pytest.raises(TypeError):
        general_sampling.sample_observed_general_from_specs(
            params,
            n=1,
            projection=general_sampling.PerItemProjection(projection),
            noise=general_sampling.PerItemFullNoise(noise),
            dtype=jnp.float64,
        )


@pytest.mark.parametrize(
    "n_samples,observed_dimension",
    (
        pytest.param(3, 2, id="ordinary"),
        pytest.param(0, 2, id="n-zero"),
        pytest.param(3, 0, id="m-zero"),
        pytest.param(0, 0, id="n-and-m-zero"),
    ),
)
def test_xd_gen_prng_001_invalid_keys_fail_before_zero_size_return(
    general_sampling, n_samples, observed_dimension
):
    """Typed/legacy key validation is not bypassed by ``n=0`` or ``M=0``."""

    dtype = jnp.float64
    projection, noise = _canonical_arrays(
        dtype,
        n_samples=n_samples,
        observed_dimension=observed_dimension,
    )
    projection_spec = general_sampling.PerItemProjection(projection)
    noise_spec = general_sampling.PerItemFullNoise(noise)
    invalid_keys = (
        None,
        1,
        jnp.asarray(1, dtype=jnp.uint32),
        jnp.zeros((3,), dtype=jnp.uint32),
        jnp.zeros((2, 2), dtype=jnp.uint32),
        jax.random.split(jax.random.key(9), 2),
    )
    for invalid_key in invalid_keys:
        with pytest.raises((TypeError, ValueError)):
            general_sampling.sample_observed_general(
                _params(dtype),
                invalid_key,
                n_samples,
                projection,
                noise,
            )
        with pytest.raises((TypeError, ValueError)):
            general_sampling.sample_observed_general_from_specs(
                _params(dtype),
                invalid_key,
                n_samples,
                projection=projection_spec,
                noise=noise_spec,
                dtype=dtype,
            )


@pytest.mark.parametrize("key_kind", ["typed", "legacy"])
def test_xd_gen_prng_001_typed_and_legacy_key_reuse_and_split_semantics(
    general_sampling, key_kind
):
    """Same explicit key repeats exactly; a split key changes a nonempty draw."""

    dtype = jnp.float64
    n_samples = 32
    params = _params(dtype)
    projection, noise = _canonical_arrays(dtype, n_samples=n_samples)
    key = (
        jax.random.key(20260825)
        if key_kind == "typed"
        else jax.random.PRNGKey(20260825)
    )
    split_key = jax.random.split(key, 2)[1]

    first = _synchronized_numpy(
        general_sampling.sample_observed_general(
            params, key, n_samples, projection, noise
        )
    )
    repeated = _synchronized_numpy(
        general_sampling.sample_observed_general(
            params, key, n_samples, projection, noise
        )
    )
    independent = _synchronized_numpy(
        general_sampling.sample_observed_general(
            params, split_key, n_samples, projection, noise
        )
    )
    wrapped = _synchronized_numpy(
        general_sampling.sample_observed_general_from_specs(
            params,
            key,
            n_samples,
            projection=general_sampling.PerItemProjection(projection),
            noise=general_sampling.PerItemFullNoise(noise),
            dtype=dtype,
        )
    )

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first, wrapped)
    assert np.any(first != independent)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "n_samples,observed_dimension",
    (
        pytest.param(0, 2, id="n-zero"),
        pytest.param(4, 0, id="m-zero"),
        pytest.param(0, 0, id="n-and-m-zero"),
    ),
)
def test_xd_gen_sample_002_zero_size_shapes_dtypes_and_all_sharing_modes(
    general_sampling, dtype, n_samples, observed_dimension
):
    """XD-GEN-SAMPLE-002: zero-sized results retain exact ``(n,M)`` shape."""

    params = _params(dtype)
    key = jax.random.key(20260825)
    for shared_projection, shared_noise in (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ):
        projection, noise, _, _ = _tagged_specs(
            general_sampling,
            dtype,
            n_samples=n_samples,
            observed_dimension=observed_dimension,
            shared_projection=shared_projection,
            shared_noise=shared_noise,
        )
        validated = general_sampling.canonicalize_general_sampling_inputs(
            params,
            n_samples,
            projection=projection,
            noise=noise,
            dtype=dtype,
        )
        canonical = general_sampling.sample_observed_general(
            validated.parameters,
            key,
            n_samples,
            validated.projection_matrices,
            validated.measurement_covariances,
        )
        wrapped = general_sampling.sample_observed_general_from_specs(
            params,
            key,
            n_samples,
            projection=projection,
            noise=noise,
            dtype=dtype,
        )

        assert validated.projection_matrices.shape == (
            n_samples,
            observed_dimension,
            3,
        )
        assert validated.measurement_covariances.shape == (
            n_samples,
            observed_dimension,
            observed_dimension,
        )
        assert canonical.shape == (n_samples, observed_dimension)
        assert wrapped.shape == (n_samples, observed_dimension)
        assert canonical.dtype == dtype
        assert wrapped.dtype == dtype
        np.testing.assert_array_equal(canonical, wrapped)


@pytest.mark.parametrize("zero_case", ["n-zero", "m-zero"])
def test_xd_gen_sample_002_zero_size_calls_still_validate_static_specs(
    general_sampling, zero_case
):
    """An empty result cannot excuse a malformed projection or noise tag."""

    dtype = jnp.float64
    n_samples = 0 if zero_case == "n-zero" else 3
    observed_dimension = 2 if zero_case == "n-zero" else 0
    row_projection = _row_projection(dtype, observed_dimension)
    bad_projection = general_sampling.PerItemProjection(
        np.broadcast_to(
            row_projection,
            (n_samples + 1, observed_dimension, 3),
        )
    )
    valid_noise = general_sampling.SharedFullNoise(
        _row_noise(dtype, observed_dimension)
    )
    with pytest.raises((TypeError, ValueError), match="projection"):
        general_sampling.sample_observed_general_from_specs(
            _params(dtype),
            jax.random.key(20260825),
            n_samples,
            projection=bad_projection,
            noise=valid_noise,
            dtype=dtype,
        )

    with pytest.raises((TypeError, ValueError), match="noise|nonnegative"):
        general_sampling.sample_observed_general_from_specs(
            _params(dtype),
            jax.random.key(20260825),
            n_samples,
            projection=general_sampling.SharedProjection(row_projection),
            noise=general_sampling.SharedIsotropicNoise(
                np.asarray(-0.1, dtype=np.float64)
            ),
            dtype=dtype,
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "noise_matrix",
    (
        pytest.param([[0.0, 0.0], [0.0, 0.0]], id="zero"),
        pytest.param([[0.2, 0.2], [0.2, 0.2]], id="singular"),
    ),
)
def test_xd_gen_sample_002_zero_and_singular_psd_noise_are_finite(
    general_sampling, dtype, noise_matrix
):
    """The deterministic symmetric square root accepts the whole PSD cone."""

    n_samples = 16
    params = _params(dtype)
    projection, _ = _canonical_arrays(dtype, n_samples=n_samples)
    row_noise = jnp.asarray(noise_matrix, dtype=dtype)
    noise = jnp.broadcast_to(row_noise, (n_samples, 2, 2))
    key = jax.random.key(20260825)

    canonical = general_sampling.sample_observed_general(
        params, key, n_samples, projection, noise
    )
    eager = general_sampling.sample_observed_general_from_specs(
        params,
        key,
        n_samples,
        projection=general_sampling.PerItemProjection(projection),
        noise=general_sampling.PerItemFullNoise(noise),
        dtype=dtype,
    )
    actual = _synchronized_numpy(canonical)
    assert canonical.shape == (n_samples, 2)
    assert canonical.dtype == dtype
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, _synchronized_numpy(eager))


@pytest.mark.parametrize("dtype", DTYPES)
def test_xd_gen_sample_002_raw_indefinite_core_is_nan_and_eager_rejects(
    general_sampling, dtype
):
    """Canonical status is observable; actionable PSD rejection stays eager."""

    n_samples = 2
    params = _params(dtype)
    projection, _ = _canonical_arrays(dtype, n_samples=n_samples)
    noise = jnp.asarray(
        [
            [[0.2, 0.04], [0.04, 0.3]],
            [[1.0, 0.0], [0.0, -0.1]],
        ],
        dtype=dtype,
    )
    key = jax.random.key(20260825)

    raw = _synchronized_numpy(
        general_sampling.sample_observed_general(
            params, key, n_samples, projection, noise
        )
    )
    assert np.all(np.isfinite(raw[0]))
    assert np.all(np.isnan(raw[1]))

    with pytest.raises(ValidationError, match="positive semidefinite|noise"):
        general_sampling.canonicalize_general_sampling_inputs(
            params,
            n_samples,
            projection=general_sampling.PerItemProjection(projection),
            noise=general_sampling.PerItemFullNoise(noise),
            dtype=dtype,
        )
    with pytest.raises(ValidationError, match="positive semidefinite|noise"):
        general_sampling.sample_observed_general_from_specs(
            params,
            key,
            n_samples,
            projection=general_sampling.PerItemProjection(projection),
            noise=general_sampling.PerItemFullNoise(noise),
            dtype=dtype,
        )


def test_general_observed_sampling_reuses_the_canonical_latent_sampler(
    general_sampling, monkeypatch
):
    """General projection/noise composition must not duplicate latent draws."""

    dtype = jnp.float64
    params = _params(dtype)
    n_samples = 3
    projection = jnp.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [0.0, 0.5, 0.0]],
        ],
        dtype=dtype,
    )
    noise = jnp.zeros((n_samples, 2, 2), dtype=dtype)
    latent = jnp.asarray(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.25, -2.0, 1.5]],
        dtype=dtype,
    )
    calls = []

    def fake_sample_latent(parameters, key, n):
        calls.append((parameters, key, n))
        return latent

    monkeypatch.setattr(general_sampling, "sample_latent", fake_sample_latent)
    actual = general_sampling.sample_observed_general(
        params,
        jax.random.key(20260825),
        n_samples,
        projection,
        noise,
    )
    expected = jnp.einsum("nmd,nd->nm", projection, latent)

    assert len(calls) == 1
    assert calls[0][2] == n_samples
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("dtype", DTYPES)
def test_general_observed_sampling_is_callback_free_jittable_and_reuses_trace(
    general_sampling, dtype
):
    """The canonical random leaf stays device-resident with static ``n``."""

    n_samples = 7
    params = _params(dtype)
    projection, noise = _canonical_arrays(dtype, n_samples=n_samples)
    key = jax.random.key(20260825)

    def operation(parameters, random_key, r, s):
        return general_sampling.sample_observed_general(
            parameters, random_key, n_samples, r, s
        )

    jaxpr_text = str(jax.make_jaxpr(operation)(params, key, projection, noise))
    assert "callback" not in jaxpr_text.lower()

    eager = operation(params, key, projection, noise)
    compiled = jax.jit(operation)
    first = compiled(params, key, projection, noise)
    first.block_until_ready()
    np.testing.assert_allclose(
        np.asarray(first),
        np.asarray(eager),
        rtol=2e-6 if dtype == jnp.float32 else 2e-13,
        atol=2e-6 if dtype == jnp.float32 else 2e-13,
    )

    trace_count = 0

    def counted(parameters, random_key, r, s):
        nonlocal trace_count
        trace_count += 1
        return operation(parameters, random_key, r, s)

    counted_compiled = jax.jit(counted)
    counted_compiled(params, key, projection, noise).block_until_ready()
    counted_compiled(
        params,
        jax.random.split(key, 2)[1],
        projection + jnp.asarray(0.01, dtype=dtype),
        noise,
    ).block_until_ready()
    assert trace_count == 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.slow
def test_xd_gen_sample_001_shared_projection_noise_analytic_moments(
    general_sampling, dtype
):
    """XD-GEN-SAMPLE-001: one-component observed moments match analysis."""

    n_samples = 150_000
    params = _params(dtype, one_component=True)
    projection = _row_projection(dtype)
    noise = np.asarray(
        [[0.25, 0.07], [0.07, 0.18]], dtype=np.dtype(dtype)
    )
    draws = general_sampling.sample_observed_general_from_specs(
        params,
        jax.random.key(20260825),
        n_samples,
        projection=general_sampling.SharedProjection(projection),
        noise=general_sampling.SharedFullNoise(noise),
        dtype=dtype,
    )
    samples = _synchronized_numpy(draws)
    latent_mean, latent_covariance = _analytic_latent_moments(params)
    expected_mean = projection.astype(np.float64) @ latent_mean
    expected_covariance = (
        projection.astype(np.float64)
        @ latent_covariance
        @ projection.astype(np.float64).T
        + noise.astype(np.float64)
    )

    assert draws.shape == (n_samples, 2)
    assert draws.dtype == dtype
    assert np.all(np.isfinite(samples))
    _assert_empirical_moments(
        samples,
        expected_mean,
        expected_covariance,
        dtype_atol=2e-3 if dtype == jnp.float64 else 3e-3,
        covariance_relative_bound=0.025 if dtype == jnp.float64 else 0.035,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "mode",
    (
        pytest.param("shared-shared-isotropic", id="shared-r-shared-isotropic"),
        pytest.param("shared-per-diagonal", id="shared-r-per-diagonal"),
        pytest.param("per-shared-correlated", id="per-r-shared-full"),
        pytest.param("per-per-zero", id="per-r-per-full-zero"),
        pytest.param("per-per-singular", id="per-r-per-full-singular"),
    ),
)
@pytest.mark.slow
def test_xd_gen_sample_002_mixture_moments_across_modes(
    general_sampling, dtype, mode
):
    """XD-GEN-SAMPLE-002: homogeneous explicit modes meet mixture moments."""

    n_samples = 100_000
    params = _params(dtype)
    numpy_dtype = np.dtype(dtype)
    row_projection = _row_projection(dtype)
    per_projection = np.broadcast_to(row_projection, (n_samples, 2, 3))

    if mode == "shared-shared-isotropic":
        projection = general_sampling.SharedProjection(row_projection)
        noise = general_sampling.SharedIsotropicNoise(
            np.asarray(0.12, dtype=numpy_dtype)
        )
        row_noise = np.eye(2, dtype=numpy_dtype) * 0.12
    elif mode == "shared-per-diagonal":
        projection = general_sampling.SharedProjection(row_projection)
        diagonal = np.asarray([0.10, 0.25], dtype=numpy_dtype)
        noise = general_sampling.PerItemDiagonalNoise(
            np.broadcast_to(diagonal, (n_samples, 2))
        )
        row_noise = np.diag(diagonal)
    elif mode == "per-shared-correlated":
        projection = general_sampling.PerItemProjection(per_projection)
        row_noise = np.asarray(
            [[0.30, 0.08], [0.08, 0.20]], dtype=numpy_dtype
        )
        noise = general_sampling.SharedFullNoise(row_noise)
    elif mode == "per-per-zero":
        projection = general_sampling.PerItemProjection(per_projection)
        row_noise = np.zeros((2, 2), dtype=numpy_dtype)
        noise = general_sampling.PerItemFullNoise(
            np.broadcast_to(row_noise, (n_samples, 2, 2))
        )
    else:
        projection = general_sampling.PerItemProjection(per_projection)
        row_noise = np.asarray(
            [[0.20, 0.20], [0.20, 0.20]], dtype=numpy_dtype
        )
        noise = general_sampling.PerItemFullNoise(
            np.broadcast_to(row_noise, (n_samples, 2, 2))
        )

    draws = general_sampling.sample_observed_general_from_specs(
        params,
        jax.random.key(20260825),
        n_samples,
        projection=projection,
        noise=noise,
        dtype=dtype,
    )
    samples = _synchronized_numpy(draws)
    latent_mean, latent_covariance = _analytic_latent_moments(params)
    projection64 = row_projection.astype(np.float64)
    expected_mean = projection64 @ latent_mean
    expected_covariance = (
        projection64 @ latent_covariance @ projection64.T
        + row_noise.astype(np.float64)
    )

    assert draws.shape == (n_samples, 2)
    assert draws.dtype == dtype
    assert np.all(np.isfinite(samples))
    _assert_empirical_moments(
        samples,
        expected_mean,
        expected_covariance,
        dtype_atol=2e-3 if dtype == jnp.float64 else 3e-3,
        covariance_relative_bound=0.03 if dtype == jnp.float64 else 0.04,
    )
