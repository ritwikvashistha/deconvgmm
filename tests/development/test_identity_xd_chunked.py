"""Red tests for bounded-memory identity-XD sufficient-statistic updates.

The temporary chunked operation is deliberately specified through independent
NumPy and existing unchunked-kernel comparisons before it is implemented.  Its
result exposes only reduced statistics: no observation/component posterior
matrix is part of the return schema.
"""

from __future__ import annotations

from collections.abc import Iterable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jax_core

from development.chunked import (
    ChunkedEMStepResult,
    ChunkedSufficientStatistics,
    chunked_em_step,
)
from development.identity_xd import Params, em_step, posterior_components
from tests.reference.identity_xd import identity_em_step as reference_em_step


DTYPE_CASES = (
    pytest.param(jnp.float64, 5e-10, 5e-12, 5e-10, id="float64"),
    pytest.param(jnp.float32, 2e-4, 2e-5, 5e-3, id="float32"),
)


def _ordinary_problem(dtype) -> tuple[Params, jax.Array, jax.Array]:
    observations = jnp.asarray(
        [
            [-1.5, 0.2],
            [-0.9, -0.4],
            [-0.2, 0.7],
            [0.4, -0.5],
            [1.1, 0.8],
            [1.8, 0.4],
            [0.7, 1.5],
        ],
        dtype=dtype,
    )
    measurement_covariances = jnp.asarray(
        [
            [[0.12, 0.02], [0.02, 0.08]],
            [[0.09, -0.01], [-0.01, 0.16]],
            [[0.20, 0.04], [0.04, 0.14]],
            [[0.07, 0.00], [0.00, 0.11]],
            [[0.15, -0.03], [-0.03, 0.18]],
            [[0.10, 0.01], [0.01, 0.06]],
            [[0.18, 0.05], [0.05, 0.22]],
        ],
        dtype=dtype,
    )
    parameters = Params(
        weights=jnp.asarray([0.25, 0.45, 0.30], dtype=dtype),
        means=jnp.asarray(
            [[-1.0, 0.1], [0.3, -0.2], [1.2, 1.0]], dtype=dtype
        ),
        covariances=jnp.asarray(
            [
                [[0.8, 0.12], [0.12, 0.6]],
                [[0.7, -0.08], [-0.08, 0.9]],
                [[0.5, 0.09], [0.09, 0.7]],
            ],
            dtype=dtype,
        ),
    )
    return parameters, observations, measurement_covariances


def _assert_allclose(actual, expected, *, rtol: float, atol: float) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol
    )


def _assert_exact_rollback(result: ChunkedEMStepResult, initial: Params) -> None:
    for actual, expected in zip(result.parameters, initial, strict=True):
        actual_array = np.asarray(actual)
        assert np.all(np.isfinite(actual_array))
        np.testing.assert_array_equal(actual_array, np.asarray(expected))


def _centered_numerator(
    responsibilities: np.ndarray,
    conditional_mean: np.ndarray,
    conditional_covariance: np.ndarray,
    component_means: np.ndarray,
) -> np.ndarray:
    """Independently form the contract's stable centered covariance numerator."""

    centered = conditional_mean - component_means[None, :, :]
    return np.einsum(
        "nk,nkde->kde",
        responsibilities,
        conditional_covariance
        + centered[:, :, :, None] * centered[:, :, None, :],
    )


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 11])
@pytest.mark.parametrize("dtype,rtol,atol,log_atol", DTYPE_CASES)
def test_chunked_em_step_matches_unchunked_and_independent_oracle_for_all_partitions(
    dtype, rtol, atol, log_atol, chunk_size
):
    """Chunks 1, non-divisor, N, and >N preserve one exact XD update."""

    parameters, observations, noise = _ordinary_problem(dtype)
    factor_jitter = 0.025
    covariance_ridge = 0.015
    unchunked = em_step(
        parameters,
        observations,
        noise,
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
    )
    reference_parameters, reference_e_step, reference_statistics = (
        reference_em_step(
            np.asarray(observations, dtype=np.float64),
            np.asarray(noise, dtype=np.float64),
            np.asarray(parameters.weights, dtype=np.float64),
            np.asarray(parameters.means, dtype=np.float64),
            np.asarray(parameters.covariances, dtype=np.float64),
            factor_jitter=factor_jitter,
            covariance_ridge=covariance_ridge,
        )
    )
    expected_centered_numerator = _centered_numerator(
        reference_e_step.responsibilities,
        reference_e_step.conditional_mean,
        reference_e_step.conditional_covariance,
        reference_parameters.means,
    )
    expected_log_likelihood = np.sum(reference_e_step.score_samples)
    expected_objective = np.mean(reference_e_step.score_samples)

    result = chunked_em_step(
        parameters,
        observations,
        noise,
        chunk_size=chunk_size,
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
    )

    assert isinstance(result, ChunkedEMStepResult)
    assert isinstance(result.statistics, ChunkedSufficientStatistics)
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    np.testing.assert_array_equal(
        np.asarray(result.collapsed_components), np.zeros(3, dtype=bool)
    )
    assert int(np.asarray(result.actual_count)) == observations.shape[0]
    assert int(np.asarray(result.padded_count)) == (
        (observations.shape[0] + chunk_size - 1) // chunk_size
    ) * chunk_size

    _assert_allclose(
        result.log_likelihood,
        expected_log_likelihood,
        rtol=rtol,
        atol=log_atol,
    )
    _assert_allclose(
        result.objective, expected_objective, rtol=rtol, atol=log_atol
    )
    _assert_allclose(
        result.log_likelihood,
        np.sum(np.asarray(unchunked.e_step.score_samples)),
        rtol=rtol,
        atol=log_atol,
    )
    _assert_allclose(
        result.objective,
        np.mean(np.asarray(unchunked.e_step.score_samples)),
        rtol=rtol,
        atol=log_atol,
    )
    # The mean objective is defined from the actual N rows, never allocated
    # padding. This equality also fixes which of the two scalar diagnostics is
    # primary rather than permitting two separately rounded accumulations.
    np.testing.assert_array_equal(
        np.asarray(result.objective),
        np.asarray(result.log_likelihood / result.actual_count),
    )

    _assert_allclose(
        result.statistics.mass,
        reference_statistics.mass,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(
        result.statistics.mass,
        unchunked.statistics.mass,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(
        result.statistics.means,
        reference_parameters.means,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(
        result.statistics.means,
        unchunked.parameters.means,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(
        result.statistics.centered_covariance_numerator,
        expected_centered_numerator,
        rtol=rtol,
        atol=atol,
    )

    for field in ("weights", "means", "covariances"):
        _assert_allclose(
            getattr(result.parameters, field),
            getattr(reference_parameters, field),
            rtol=rtol,
            atol=atol,
        )
        _assert_allclose(
            getattr(result.parameters, field),
            getattr(unchunked.parameters, field),
            rtol=rtol,
            atol=atol,
        )

    dimension = observations.shape[-1]
    reconstructed_covariances = (
        result.statistics.centered_covariance_numerator
        / result.statistics.mass[:, None, None]
        + jnp.asarray(covariance_ridge, dtype=dtype)
        * jnp.eye(dimension, dtype=dtype)[None, :, :]
    )
    _assert_allclose(
        reconstructed_covariances,
        result.parameters.covariances,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype,rtol,atol,log_atol", DTYPE_CASES)
def test_chunk_padding_is_excluded_from_likelihood_mass_and_update(
    dtype, rtol, atol, log_atol
):
    """Logical tail slots contribute neither likelihood nor statistics."""

    actual_count = 5
    chunk_size = 8
    parameters = Params(
        weights=jnp.ones((1,), dtype=dtype),
        means=jnp.zeros((1, 1), dtype=dtype),
        covariances=jnp.ones((1, 1, 1), dtype=dtype),
    )
    observations = jnp.zeros((actual_count, 1), dtype=dtype)
    noise = jnp.full((actual_count, 1, 1), 0.2, dtype=dtype)
    unchunked = em_step(parameters, observations, noise)

    result = chunked_em_step(
        parameters, observations, noise, chunk_size=chunk_size
    )

    assert int(np.asarray(result.actual_count)) == actual_count
    # ``padded_count`` is the total logical scan-slot count, not an allocated
    # padded-input shape or the number of dummy rows.
    assert int(np.asarray(result.padded_count)) == chunk_size
    np.testing.assert_array_equal(
        np.asarray(result.statistics.mass),
        np.asarray(
            [float(actual_count)],
            dtype=np.asarray(result.statistics.mass).dtype,
        ),
    )
    expected_log_likelihood = np.sum(
        np.asarray(unchunked.e_step.score_samples)
    )
    _assert_allclose(
        result.log_likelihood,
        expected_log_likelihood,
        rtol=rtol,
        atol=log_atol,
    )
    _assert_allclose(
        result.objective,
        expected_log_likelihood / actual_count,
        rtol=rtol,
        atol=log_atol,
    )
    for field in ("weights", "means", "covariances"):
        _assert_allclose(
            getattr(result.parameters, field),
            getattr(unchunked.parameters, field),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "dtype,tiny",
    [
        pytest.param(jnp.float64, 1.0e-200, id="float64"),
        pytest.param(jnp.float32, 1.0e-30, id="float32"),
    ],
)
def test_tiny_component_mass_preserves_chan_cross_chunk_covariance(dtype, tiny):
    """The Chan cross weight must not form a tiny-times-tiny product first."""

    parameters = Params(
        weights=jnp.asarray([1.0 - tiny, tiny], dtype=dtype),
        means=jnp.zeros((2, 1), dtype=dtype),
        covariances=jnp.ones((2, 1, 1), dtype=dtype),
    )
    observations = jnp.asarray([[-10.0], [-10.0], [10.0], [10.0]], dtype=dtype)
    noise = jnp.ones((4, 1, 1), dtype=dtype)

    unchunked = em_step(parameters, observations, noise)
    result = chunked_em_step(
        parameters, observations, noise, chunk_size=2
    )

    assert not bool(np.asarray(unchunked.numerical_failure))
    assert not bool(np.asarray(unchunked.collapsed))
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert float(np.asarray(result.statistics.mass[1])) > 0.0
    # Each component has conditional variance 1/2 and conditional means -5,+5,
    # so its updated variance is exactly 1/2 + 25.  For the tiny component,
    # the naive ``mass_a * mass_b / (mass_a + mass_b)`` cross weight underflows
    # before division in both selected dtypes.
    np.testing.assert_allclose(
        np.asarray(unchunked.parameters.covariances[:, 0, 0]),
        np.asarray([25.5, 25.5]),
        rtol=2e-5 if dtype == jnp.float32 else 5e-12,
        atol=2e-5 if dtype == jnp.float32 else 5e-12,
    )
    np.testing.assert_allclose(
        np.asarray(result.parameters.covariances),
        np.asarray(unchunked.parameters.covariances),
        rtol=2e-5 if dtype == jnp.float32 else 5e-12,
        atol=2e-5 if dtype == jnp.float32 else 5e-12,
    )


@pytest.mark.parametrize("chunk_size", [2, 3])
def test_padded_zero_responsibility_cannot_overflow_centered_products(chunk_size):
    """Dummy rows are masked before a large centered displacement is squared."""

    dtype = jnp.float32
    parameters = Params(
        weights=jnp.ones((1,), dtype=dtype),
        means=jnp.asarray([[1.9e19]], dtype=dtype),
        covariances=jnp.asarray([[[3.0e38]]], dtype=dtype),
    )
    observations = jnp.zeros((1, 1), dtype=dtype)
    noise = jnp.ones((1, 1, 1), dtype=dtype)

    unpadded = chunked_em_step(
        parameters, observations, noise, chunk_size=1
    )
    result = chunked_em_step(
        parameters, observations, noise, chunk_size=chunk_size
    )

    assert not bool(np.asarray(unpadded.numerical_failure))
    assert not bool(np.asarray(unpadded.collapsed))
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert np.all(
        np.isfinite(
            np.asarray(result.statistics.centered_covariance_numerator)
        )
    )
    for actual, expected in zip(
        result.parameters, unpadded.parameters, strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_within_chunk_centering_uses_anchored_displacements_under_scan(compiled):
    """Compiled scan arithmetic must not recenter by subtracting two offsets."""

    dtype = jnp.float32
    parameters = Params(
        weights=jnp.ones((1,), dtype=dtype),
        means=jnp.asarray([[1.0e10]], dtype=dtype),
        covariances=jnp.asarray([[[8.0e19]]], dtype=dtype),
    )
    observations = jnp.zeros((1, 1), dtype=dtype)
    noise = jnp.ones((1, 1, 1), dtype=dtype)
    unchunked = em_step(parameters, observations, noise)

    def operation(p, x, errors):
        return chunked_em_step(p, x, errors, chunk_size=1)

    result = (jax.jit(operation) if compiled else operation)(
        parameters, observations, noise
    )

    assert not bool(np.asarray(unchunked.numerical_failure))
    assert not bool(np.asarray(unchunked.collapsed))
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    np.testing.assert_allclose(
        np.asarray(result.parameters.means),
        np.asarray(unchunked.parameters.means),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.parameters.covariances),
        np.asarray(unchunked.parameters.covariances),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_nonfinite_aggregate_likelihood_is_numerical_failure(chunk_size):
    """Finite row scores whose sum overflows must roll back as failure."""

    dtype = jnp.float32
    parameters = Params(
        weights=jnp.ones((1,), dtype=dtype),
        means=jnp.zeros((1, 1), dtype=dtype),
        covariances=jnp.ones((1, 1, 1), dtype=dtype),
    )
    observations = jnp.full((5, 1), 1.8e19, dtype=dtype)
    noise = jnp.ones((5, 1, 1), dtype=dtype)
    e_step = posterior_components(parameters, observations, noise)

    assert not bool(np.asarray(e_step.numerical_failure))
    assert np.all(np.isfinite(np.asarray(e_step.score_samples)))
    result = chunked_em_step(
        parameters, observations, noise, chunk_size=chunk_size
    )

    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not np.any(np.asarray(result.collapsed_components))
    assert not np.isfinite(np.asarray(result.log_likelihood))
    assert not np.isfinite(np.asarray(result.objective))
    _assert_exact_rollback(result, parameters)


def _collapse_problem(dtype) -> tuple[Params, jax.Array, jax.Array]:
    parameters = Params(
        weights=jnp.asarray([0.5, 0.5], dtype=dtype),
        means=jnp.asarray(
            [[0.0, 0.0], [1_000_000.0, 1_000_000.0]], dtype=dtype
        ),
        covariances=jnp.asarray([np.eye(2), np.eye(2)], dtype=dtype),
    )
    observations = jnp.asarray(
        [
            [-1.0, -0.5],
            [-0.8, 0.4],
            [-0.3, -0.7],
            [0.0, 0.2],
            [0.2, 0.9],
            [0.6, -0.4],
            [0.9, 0.5],
            [1.2, -0.1],
        ],
        dtype=dtype,
    )
    noise = jnp.broadcast_to(
        jnp.eye(2, dtype=dtype) * jnp.asarray(0.1, dtype=dtype),
        (8, 2, 2),
    )
    return parameters, observations, noise


@pytest.mark.parametrize("chunk_size", [3, 11])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_chunked_collapse_reports_component_and_rolls_back_whole_state(
    dtype, chunk_size
):
    parameters, observations, noise = _collapse_problem(dtype)
    unchunked = em_step(parameters, observations, noise)

    result = chunked_em_step(
        parameters, observations, noise, chunk_size=chunk_size
    )

    assert bool(np.asarray(unchunked.collapsed))
    assert bool(np.asarray(result.collapsed))
    assert not bool(np.asarray(result.numerical_failure))
    np.testing.assert_array_equal(
        np.asarray(result.collapsed_components), np.asarray([False, True])
    )
    assert float(np.asarray(result.statistics.mass[1])) == 0.0
    assert int(np.asarray(result.actual_count)) == observations.shape[0]
    assert int(np.asarray(result.padded_count)) == (
        (observations.shape[0] + chunk_size - 1) // chunk_size
    ) * chunk_size
    _assert_exact_rollback(result, parameters)


@pytest.mark.parametrize("chunk_size", [1, 3, 5])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_chunked_factor_failure_is_numerical_not_collapse_and_rolls_back(
    dtype, chunk_size
):
    parameters = Params(
        weights=jnp.asarray([0.5, 0.5], dtype=dtype),
        means=jnp.asarray([[0.0, 0.0], [0.8, -0.4]], dtype=dtype),
        covariances=jnp.asarray(
            [0.5 * np.eye(2), 2.0 * np.eye(2)], dtype=dtype
        ),
    )
    observations = jnp.asarray(
        [[0.1, -0.2], [0.4, 0.3], [-0.5, 0.1]], dtype=dtype
    )
    noise = jnp.asarray(
        [0.1 * np.eye(2), -1.0 * np.eye(2), 0.2 * np.eye(2)],
        dtype=dtype,
    )
    unchunked = em_step(parameters, observations, noise)

    result = chunked_em_step(
        parameters, observations, noise, chunk_size=chunk_size
    )

    assert bool(np.asarray(unchunked.numerical_failure))
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not np.any(np.asarray(result.collapsed_components))
    _assert_exact_rollback(result, parameters)
    np.testing.assert_allclose(
        np.asarray(result.log_likelihood),
        np.sum(np.asarray(unchunked.e_step.score_samples)),
        rtol=2e-4 if dtype == jnp.float32 else 5e-10,
        atol=5e-3 if dtype == jnp.float32 else 5e-10,
    )


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_singular_m_step_proposal_is_collapse_not_pair_failure(dtype, compiled):
    """One noiseless point has a valid E-step but a singular covariance update."""

    parameters = Params(
        weights=jnp.ones((1,), dtype=dtype),
        means=jnp.zeros((1, 1), dtype=dtype),
        covariances=jnp.ones((1, 1, 1), dtype=dtype),
    )
    observations = jnp.asarray([[0.25]], dtype=dtype)
    noise = jnp.zeros((1, 1, 1), dtype=dtype)

    def operation(p, x, errors):
        return chunked_em_step(p, x, errors, chunk_size=3)

    unchunked = em_step(parameters, observations, noise)
    result = (jax.jit(operation) if compiled else operation)(
        parameters, observations, noise
    )

    assert not bool(np.asarray(unchunked.numerical_failure))
    assert bool(np.asarray(unchunked.collapsed))
    assert not bool(np.asarray(result.numerical_failure))
    assert bool(np.asarray(result.collapsed))
    np.testing.assert_array_equal(
        np.asarray(result.collapsed_components), np.asarray([True])
    )
    np.testing.assert_array_equal(
        np.asarray(result.statistics.mass), np.asarray([1.0], dtype=np.dtype(dtype))
    )
    _assert_exact_rollback(result, parameters)


@pytest.mark.parametrize(
    "invalid_chunk_size",
    [
        pytest.param(True, id="bool"),
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="nonintegral"),
        pytest.param("3", id="string"),
        pytest.param(jnp.asarray([2]), id="rank-one"),
    ],
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit-static"])
def test_chunk_size_must_be_a_positive_static_index_integer(
    invalid_chunk_size, compiled
):
    parameters, observations, noise = _ordinary_problem(jnp.float64)

    def operation(p, x, errors):
        return chunked_em_step(
            p, x, errors, chunk_size=invalid_chunk_size
        )

    run = jax.jit(operation) if compiled else operation
    with pytest.raises((TypeError, ValueError), match="chunk_size"):
        run(parameters, observations, noise)


def test_chunk_size_is_required_and_keyword_only():
    parameters, observations, noise = _ordinary_problem(jnp.float64)

    with pytest.raises(TypeError):
        chunked_em_step(parameters, observations, noise)
    with pytest.raises(TypeError):
        chunked_em_step(parameters, observations, noise, 3)


def _shape_problem(dtype) -> tuple[Params, jax.Array, jax.Array]:
    n_samples, n_components, dimension = 13, 2, 3
    parameters = Params(
        weights=jnp.asarray([0.4, 0.6], dtype=dtype),
        means=jnp.asarray(
            [[-0.5, 0.2, 0.4], [0.8, -0.3, 0.7]], dtype=dtype
        ),
        covariances=jnp.asarray(
            [
                [[0.8, 0.1, 0.0], [0.1, 0.7, 0.05], [0.0, 0.05, 0.6]],
                [[0.6, -0.04, 0.03], [-0.04, 0.9, 0.1], [0.03, 0.1, 0.75]],
            ],
            dtype=dtype,
        ),
    )
    observations = jnp.reshape(
        jnp.linspace(-1.2, 1.5, n_samples * dimension, dtype=dtype),
        (n_samples, dimension),
    )
    noise = jnp.broadcast_to(
        jnp.eye(dimension, dtype=dtype) * jnp.asarray(0.12, dtype=dtype),
        (n_samples, dimension, dimension),
    )
    assert parameters.weights.shape == (n_components,)
    return parameters, observations, noise


def _jaxpr_shapes(value: object) -> set[tuple[int, ...]]:
    """Collect array shapes recursively from main and nested JAX programs."""

    shapes: set[tuple[int, ...]] = set()
    visited: set[int] = set()

    def add_variables(variables: Iterable[object]) -> None:
        for variable in variables:
            aval = getattr(variable, "aval", None)
            shape = getattr(aval, "shape", None)
            if shape is not None:
                shapes.add(tuple(shape))

    def visit(item: object) -> None:
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(item, jax_core.ClosedJaxpr):
            visit(item.jaxpr)
            return
        if isinstance(item, jax_core.Jaxpr):
            add_variables(item.constvars)
            add_variables(item.invars)
            add_variables(item.outvars)
            for equation in item.eqns:
                add_variables(equation.invars)
                add_variables(equation.outvars)
                visit(equation.params)
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)

    visit(value)
    return shapes


def _jaxpr_equations(value: object) -> list[object]:
    """Collect equations recursively from main and nested JAX programs."""

    equations: list[object] = []
    visited: set[int] = set()

    def visit(item: object) -> None:
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(item, jax_core.ClosedJaxpr):
            visit(item.jaxpr)
            return
        if isinstance(item, jax_core.Jaxpr):
            for equation in item.eqns:
                equations.append(equation)
                visit(equation.params)
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)

    visit(value)
    return equations


def _variable_shape(variable: object) -> tuple[int, ...] | None:
    """Return one jaxpr variable's static array shape when it has one."""

    aval = getattr(variable, "aval", None)
    shape = getattr(aval, "shape", None)
    return None if shape is None else tuple(shape)


@pytest.mark.parametrize(
    "n_samples,chunk_size",
    [
        pytest.param(13, 4, id="nondivisible"),
        pytest.param(3, 7, id="chunk-larger-than-input"),
    ],
)
def test_chunk_scan_reads_original_inputs_without_full_padded_staging(
    n_samples, chunk_size
):
    """The scan indexes original inputs and never receives a padded stack."""

    parameters, observations, noise = _shape_problem(jnp.float64)
    observations = observations[:n_samples]
    noise = noise[:n_samples]
    n_components = parameters.weights.shape[0]
    dimension = observations.shape[-1]
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    padded_count = n_chunks * chunk_size

    def operation(p, x, errors):
        return chunked_em_step(p, x, errors, chunk_size=chunk_size)

    closed_jaxpr = jax.make_jaxpr(operation)(parameters, observations, noise)
    scan_equations = [
        equation
        for equation in _jaxpr_equations(closed_jaxpr)
        if equation.primitive.name == "scan"
    ]
    assert len(scan_equations) == 1
    scan_equation = scan_equations[0]
    assert scan_equation.params["length"] == n_chunks

    n_constants = scan_equation.params["num_consts"]
    n_carry = scan_equation.params["num_carry"]
    constant_shapes = {
        _variable_shape(variable)
        for variable in scan_equation.invars[:n_constants]
    }
    sequence_shapes = [
        _variable_shape(variable)
        for variable in scan_equation.invars[n_constants + n_carry :]
    ]
    body_shapes = _jaxpr_shapes(scan_equation.params["jaxpr"])
    all_program_shapes = _jaxpr_shapes(closed_jaxpr)

    violations: list[str] = []
    for original_shape in (
        (n_samples, dimension),
        (n_samples, dimension, dimension),
    ):
        if original_shape not in constant_shapes:
            violations.append(
                f"scan constants do not include original input {original_shape}"
            )
    if sequence_shapes != [(n_chunks,)]:
        violations.append(
            "scan sequence must contain only chunk indices with shape "
            f"{(n_chunks,)}; received {sequence_shapes}"
        )

    required_chunk_shapes = {
        (chunk_size, dimension),
        (chunk_size, dimension, dimension),
        (chunk_size, n_components, dimension, dimension),
    }
    missing_chunk_shapes = required_chunk_shapes - body_shapes
    if missing_chunk_shapes:
        violations.append(
            "scan body is missing per-chunk work arrays "
            f"{sorted(missing_chunk_shapes)}"
        )

    # When P and C differ, array shapes alone distinguish a global padded
    # staging buffer from the permitted original-N inputs and C-sized gathers.
    # For C>N, the scan-constant and sequence checks above provide the needed
    # distinction even though P==C.
    if padded_count > chunk_size:
        forbidden_staging_shapes = {
            (padded_count,),
            (padded_count, dimension),
            (padded_count, dimension, dimension),
            (n_chunks, chunk_size),
            (n_chunks, chunk_size, dimension),
            (n_chunks, chunk_size, dimension, dimension),
        }
        present_staging_shapes = forbidden_staging_shapes & all_program_shapes
        if present_staging_shapes:
            violations.append(
                "program contains full-P input staging arrays "
                f"{sorted(present_staging_shapes)}"
            )

    assert not violations, "; ".join(violations)


@pytest.mark.parametrize(
    "n_samples,chunk_size",
    [
        pytest.param(13, 4, id="nondivisible"),
        pytest.param(3, 7, id="chunk-larger-than-input"),
    ],
)
@pytest.mark.parametrize("dtype,rtol,atol,log_atol", DTYPE_CASES)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_unpadded_chunk_execution_preserves_eager_jit_and_unchunked_results(
    compiled, dtype, rtol, atol, log_atol, n_samples, chunk_size
):
    """Per-chunk input gathering preserves every reduced numerical endpoint."""

    parameters, observations, noise = _shape_problem(dtype)
    observations = observations[:n_samples]
    noise = noise[:n_samples]
    unchunked = em_step(parameters, observations, noise)

    def operation(p, x, errors):
        return chunked_em_step(p, x, errors, chunk_size=chunk_size)

    eager = operation(parameters, observations, noise)
    result = (jax.jit(operation) if compiled else operation)(
        parameters, observations, noise
    )
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), result)

    for actual, expected in zip(
        jax.tree_util.tree_leaves(result),
        jax.tree_util.tree_leaves(eager),
        strict=True,
    ):
        _assert_allclose(actual, expected, rtol=rtol, atol=log_atol)

    assert int(np.asarray(result.actual_count)) == n_samples
    assert int(np.asarray(result.padded_count)) == (
        (n_samples + chunk_size - 1) // chunk_size
    ) * chunk_size
    np.testing.assert_array_equal(
        np.asarray(result.numerical_failure),
        np.asarray(unchunked.numerical_failure),
    )
    np.testing.assert_array_equal(
        np.asarray(result.collapsed), np.asarray(unchunked.collapsed)
    )
    np.testing.assert_array_equal(
        np.asarray(result.collapsed_components),
        np.asarray(unchunked.collapsed_components),
    )

    _assert_allclose(
        result.log_likelihood,
        jnp.sum(unchunked.e_step.score_samples),
        rtol=rtol,
        atol=log_atol,
    )
    _assert_allclose(
        result.objective,
        jnp.mean(unchunked.e_step.score_samples),
        rtol=rtol,
        atol=log_atol,
    )
    _assert_allclose(
        result.statistics.mass,
        unchunked.statistics.mass,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(
        result.statistics.means,
        unchunked.parameters.means,
        rtol=rtol,
        atol=atol,
    )
    expected_centered_numerator = _centered_numerator(
        np.asarray(unchunked.e_step.responsibilities),
        np.asarray(unchunked.e_step.conditional_mean),
        np.asarray(unchunked.e_step.conditional_covariance),
        np.asarray(unchunked.parameters.means),
    )
    _assert_allclose(
        result.statistics.centered_covariance_numerator,
        expected_centered_numerator,
        rtol=rtol,
        atol=atol,
    )
    for field in ("weights", "means", "covariances"):
        _assert_allclose(
            getattr(result.parameters, field),
            getattr(unchunked.parameters, field),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("dtype,rtol,atol,_log_atol", DTYPE_CASES)
def test_chunked_kernel_is_jittable_with_static_size_and_exposes_only_reductions(
    dtype, rtol, atol, _log_atol
):
    """The program is bounded by C, and result leaves contain no N-axis state."""

    parameters, observations, noise = _shape_problem(dtype)
    chunk_size = 4
    n_samples, dimension = observations.shape
    n_components = parameters.weights.shape[0]
    padded_count = 16

    def operation(p, x, errors):
        return chunked_em_step(p, x, errors, chunk_size=chunk_size)

    eager = operation(parameters, observations, noise)
    compiled = jax.jit(operation)(parameters, observations, noise)
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), compiled)
    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager),
        jax.tree_util.tree_leaves(compiled),
        strict=True,
    ):
        _assert_allclose(compiled_leaf, eager_leaf, rtol=rtol, atol=atol)

    allowed_result_shapes = {
        (),
        (n_components,),
        (n_components, dimension),
        (n_components, dimension, dimension),
    }
    assert all(
        tuple(np.shape(leaf)) in allowed_result_shapes
        for leaf in jax.tree_util.tree_leaves(compiled)
    )

    closed_jaxpr = jax.make_jaxpr(operation)(parameters, observations, noise)
    all_program_shapes = _jaxpr_shapes(closed_jaxpr)
    # A correct implementation may materialize C x K x D x D inside one scan
    # body. It must not construct posterior-covariance arrays for all actual or
    # padded rows at once.
    assert (chunk_size, n_components, dimension, dimension) in all_program_shapes
    assert (n_samples, n_components, dimension, dimension) not in all_program_shapes
    assert (padded_count, n_components, dimension, dimension) not in all_program_shapes


def _high_offset_problem(dtype) -> tuple[Params, jax.Array, jax.Array]:
    offset = 1.0e9 if dtype == jnp.float64 else 1.0e4
    deviations = np.asarray(
        [
            [-0.60, -0.20],
            [-0.45, 0.35],
            [-0.20, -0.55],
            [-0.05, 0.10],
            [0.10, 0.50],
            [0.25, -0.30],
            [0.40, 0.25],
            [0.55, -0.10],
            [0.00, -0.05],
        ]
    )
    observations = jnp.asarray(offset + deviations, dtype=dtype)
    parameters = Params(
        weights=jnp.ones((1,), dtype=dtype),
        means=jnp.asarray([[offset + 0.2, offset - 0.1]], dtype=dtype),
        covariances=jnp.asarray(
            [[[1.0, 0.1], [0.1, 0.8]]], dtype=dtype
        ),
    )
    noise = jnp.zeros((observations.shape[0], 2, 2), dtype=dtype)
    return parameters, observations, noise


@pytest.mark.parametrize("chunk_size", [1, 4])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_chunked_centered_accumulation_survives_high_offset_low_variance_data(
    dtype, chunk_size
):
    """A raw E[zz']-E[z]E[z]' accumulator would catastrophically cancel."""

    parameters, observations, noise = _high_offset_problem(dtype)
    unchunked = em_step(parameters, observations, noise)
    reference_parameters, reference_e_step, _ = reference_em_step(
        np.asarray(observations, dtype=np.float64),
        np.asarray(noise, dtype=np.float64),
        np.asarray(parameters.weights, dtype=np.float64),
        np.asarray(parameters.means, dtype=np.float64),
        np.asarray(parameters.covariances, dtype=np.float64),
    )
    expected_numerator = _centered_numerator(
        reference_e_step.responsibilities,
        reference_e_step.conditional_mean,
        reference_e_step.conditional_covariance,
        reference_parameters.means,
    )

    # Demonstrate that this fixture discriminates the required centered method
    # from the tempting raw-second-moment subtraction in the selected dtype.
    raw_mean = jnp.mean(observations, axis=0)
    raw_covariance = (
        jnp.mean(
            observations[:, :, None] * observations[:, None, :], axis=0
        )
        - raw_mean[:, None] * raw_mean[None, :]
    )
    stable_covariance = np.asarray(reference_parameters.covariances[0])
    assert np.linalg.norm(
        np.asarray(raw_covariance, dtype=np.float64) - stable_covariance,
        ord="fro",
    ) > 0.05

    result = chunked_em_step(
        parameters, observations, noise, chunk_size=chunk_size
    )

    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    covariance_atol = 5e-7 if dtype == jnp.float64 else 5e-4
    numerator_atol = 5e-6 if dtype == jnp.float64 else 5e-3
    mean_atol = 5e-7 if dtype == jnp.float64 else 3e-3
    np.testing.assert_allclose(
        np.asarray(result.statistics.means),
        reference_parameters.means,
        rtol=0.0,
        atol=mean_atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.statistics.centered_covariance_numerator),
        expected_numerator,
        rtol=2e-5 if dtype == jnp.float32 else 5e-9,
        atol=numerator_atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.parameters.covariances),
        reference_parameters.covariances,
        rtol=2e-5 if dtype == jnp.float32 else 5e-9,
        atol=covariance_atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.parameters.covariances),
        np.asarray(unchunked.parameters.covariances),
        rtol=2e-5 if dtype == jnp.float32 else 5e-9,
        atol=covariance_atol,
    )
