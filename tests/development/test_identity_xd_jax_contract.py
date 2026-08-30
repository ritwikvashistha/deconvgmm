"""Callback, retrace, and gradient evidence for canonical JAX operations."""

from __future__ import annotations

from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import fit_fixed_steps_kernel, mean_log_likelihood
from development.identity_xd import Params, posterior_components
from tests.fixtures.identity_likelihood import likelihood_fixture


def _params(dtype, weights, means, covariances) -> Params:
    return Params(
        weights=jnp.asarray(weights, dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(covariances, dtype=dtype),
    )


def _ordinary_problem(dtype):
    observations, noise, weights, means, covariances = likelihood_fixture(
        np.float64
    )
    return (
        _params(dtype, weights, means, covariances),
        jnp.asarray(observations, dtype=dtype),
        jnp.asarray(noise, dtype=dtype),
    )


def _block_tree(value) -> None:
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), value)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_xd_ip_jit_001_canonical_operations_have_no_callback_and_do_not_retrace(
    dtype,
):
    """XD-IP-JIT-001: same-shape repeat calls reuse callback-free traces."""

    params, observations, noise = _ordinary_problem(dtype)

    def posterior_operation(parameters, x, errors):
        return posterior_components(parameters, x, errors)

    def likelihood_operation(parameters, x, errors):
        return mean_log_likelihood(parameters, x, errors)

    def one_step_operation(parameters, x, errors):
        return fit_fixed_steps_kernel(
            parameters, x, errors, n_steps=1
        )

    for operation in (
        posterior_operation,
        likelihood_operation,
        one_step_operation,
    ):
        jaxpr_text = str(jax.make_jaxpr(operation)(params, observations, noise))
        assert "callback" not in jaxpr_text.lower()

        trace_count = 0

        def counted(parameters, x, errors):
            nonlocal trace_count
            trace_count += 1
            return operation(parameters, x, errors)

        compiled = jax.jit(counted)
        first = compiled(params, observations, noise)
        _block_tree(first)
        second = compiled(params, observations + dtype(0.01), noise)
        _block_tree(second)
        assert trace_count == 1


def _gradient_problem():
    observations = jnp.asarray(
        [[-0.9, 0.2], [-0.1, -0.6], [0.7, 0.4], [1.3, 1.0]],
        dtype=jnp.float64,
    )
    noise = jnp.asarray(
        [
            [[0.12, 0.02], [0.02, 0.08]],
            [[0.09, -0.01], [-0.01, 0.15]],
            [[0.16, 0.03], [0.03, 0.11]],
            [[0.10, 0.00], [0.00, 0.18]],
        ],
        dtype=jnp.float64,
    )
    params = _params(
        jnp.float64,
        [0.42, 0.58],
        [[-0.55, -0.10], [0.85, 0.60]],
        [
            [[0.74, 0.11], [0.11, 0.63]],
            [[0.68, -0.07], [-0.07, 0.82]],
        ],
    )
    return params, observations, noise


def _central_difference(
    function: Callable[[jax.Array], jax.Array],
    value: jax.Array,
    *,
    step: float,
) -> np.ndarray:
    base = np.asarray(value, dtype=np.float64)
    gradient = np.empty_like(base)
    for index in np.ndindex(base.shape):
        positive = base.copy()
        negative = base.copy()
        positive[index] += step
        negative[index] -= step
        upper = float(np.asarray(function(jnp.asarray(positive))))
        lower = float(np.asarray(function(jnp.asarray(negative))))
        gradient[index] = (upper - lower) / (2.0 * step)
    return gradient


def test_xd_ip_grad_001_likelihood_and_one_step_gradients_match_central_difference():
    """XD-IP-GRAD-001: advertised f64 gradients match ``h=1e-5`` differences."""

    params, observations, noise = _gradient_problem()

    def likelihood_from_observations(x):
        return jnp.sum(posterior_components(params, x, noise).score_samples)

    def likelihood_from_means(means):
        candidate = Params(params.weights, means, params.covariances)
        return jnp.sum(
            posterior_components(candidate, observations, noise).score_samples
        )

    def post_update_from_observations(x):
        result = fit_fixed_steps_kernel(params, x, noise, n_steps=1)
        return jnp.sum(
            posterior_components(result.parameters, x, noise).score_samples
        )

    def post_update_from_means(means):
        initial = Params(params.weights, means, params.covariances)
        result = fit_fixed_steps_kernel(initial, observations, noise, n_steps=1)
        return jnp.sum(
            posterior_components(
                result.parameters, observations, noise
            ).score_samples
        )

    cases = (
        (likelihood_from_observations, observations),
        (likelihood_from_means, params.means),
        (post_update_from_observations, observations),
        (post_update_from_means, params.means),
    )
    for function, argument in cases:
        automatic = np.asarray(jax.grad(function)(argument))
        numeric = _central_difference(function, argument, step=1e-5)
        assert np.all(np.isfinite(automatic))
        np.testing.assert_allclose(
            automatic, numeric, rtol=2e-5, atol=2e-6
        )

