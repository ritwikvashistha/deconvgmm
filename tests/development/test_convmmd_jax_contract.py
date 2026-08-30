"""JAX-contract evidence for the convMMD kernels (contract xdgmm-jax.convmmd).

Covers jit callback-freedom and retrace invariance, float32/float64 correctness,
autodiff-vs-finite-difference gradients, vmap over the observation batch,
explicit-PRNG semantics for the Monte-Carlo loss, its statistical convergence to
the analytic value, device residency, and honest failure/status behavior (a
degenerate input surfaces as a documented status or a visible NaN, never as a
finite success).
"""

from __future__ import annotations

from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.convmmd import (
    ConvMMDParams,
    ConvMMDUnconstrained,
    convmmd_loss_analytic,
    convmmd_loss_mc,
    denoise,
    posterior_components,
    to_canonical,
)
from development.convmmd_fit import (
    ConvMMDFitStatus,
    fit_analytic,
    fit_analytic_state,
    fit_mc_state,
)


def _problem(dtype=jnp.float64):
    observations = jnp.asarray(
        [[-0.9, 0.2], [-0.1, -0.6], [0.7, 0.4], [1.3, 1.0], [0.2, -0.2]],
        dtype=dtype,
    )
    noise = jnp.asarray(
        [
            [[0.12, 0.02], [0.02, 0.08]],
            [[0.09, -0.01], [-0.01, 0.15]],
            [[0.16, 0.03], [0.03, 0.11]],
            [[0.10, 0.00], [0.00, 0.18]],
            [[0.13, 0.01], [0.01, 0.09]],
        ],
        dtype=dtype,
    )
    params = ConvMMDParams(
        weights=jnp.asarray([0.42, 0.58], dtype=dtype),
        means=jnp.asarray([[-0.55, -0.10], [0.85, 0.60]], dtype=dtype),
        covariances=jnp.asarray(
            [[[0.74, 0.11], [0.11, 0.63]], [[0.68, -0.07], [-0.07, 0.82]]],
            dtype=dtype,
        ),
    )
    bandwidths = jnp.asarray([0.5, 1.0, 2.0], dtype=dtype)
    return params, observations, noise, bandwidths


def _unconstrained(dtype=jnp.float64):
    return ConvMMDUnconstrained(
        alphas=jnp.asarray([0.1, -0.2], dtype=dtype),
        means=jnp.asarray([[-0.5, -0.1], [0.8, 0.6]], dtype=dtype),
        unconstrained_L=jnp.asarray(
            [[[0.3, 0.0], [0.2, 0.1]], [[0.25, 0.0], [-0.1, 0.2]]], dtype=dtype
        ),
    )


def _block(value):
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), value)


# --------------------------------------------------------------------------- #
# jit: callback-free and retrace-invariant
# --------------------------------------------------------------------------- #


def test_operations_are_callback_free_and_do_not_retrace():
    params, observations, noise, bandwidths = _problem()

    def analytic(p, x, s):
        return convmmd_loss_analytic(p, x, s, bandwidths)

    def denoiser(p, x, s):
        return denoise(p, x, s)

    for operation in (analytic, denoiser):
        jaxpr_text = str(jax.make_jaxpr(operation)(params, observations, noise))
        assert "callback" not in jaxpr_text.lower()

        trace_count = 0

        def counted(p, x, s):
            nonlocal trace_count
            trace_count += 1
            return operation(p, x, s)

        compiled = jax.jit(counted)
        first = compiled(params, observations, noise)
        _block(first)
        second = compiled(params, observations + 0.01, noise)
        _block(second)
        assert trace_count == 1


# --------------------------------------------------------------------------- #
# dtype correctness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_dtype_is_preserved(dtype):
    params, observations, noise, bandwidths = _problem(dtype)
    loss = convmmd_loss_analytic(params, observations, noise, bandwidths)
    posterior_mean = denoise(params, observations, noise)
    assert loss.dtype == dtype
    assert posterior_mean.dtype == dtype
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(posterior_mean))


# --------------------------------------------------------------------------- #
# gradients: autodiff matches central differences (f64)
# --------------------------------------------------------------------------- #


def _central_difference(
    function: Callable[[jax.Array], jax.Array], value: jax.Array, *, step: float
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


def test_analytic_loss_gradient_matches_central_difference():
    _, observations, noise, bandwidths = _problem()
    base = _unconstrained()

    def loss_from_means(means):
        candidate = ConvMMDUnconstrained(base.alphas, means, base.unconstrained_L)
        return convmmd_loss_analytic(
            to_canonical(candidate), observations, noise, bandwidths
        )

    def loss_from_alphas(alphas):
        candidate = ConvMMDUnconstrained(alphas, base.means, base.unconstrained_L)
        return convmmd_loss_analytic(
            to_canonical(candidate), observations, noise, bandwidths
        )

    for function, argument in (
        (loss_from_means, base.means),
        (loss_from_alphas, base.alphas),
    ):
        automatic = np.asarray(jax.grad(function)(argument))
        numeric = _central_difference(function, argument, step=1e-5)
        assert np.all(np.isfinite(automatic))
        np.testing.assert_allclose(automatic, numeric, rtol=2e-5, atol=2e-6)


# --------------------------------------------------------------------------- #
# vmap over the observation batch
# --------------------------------------------------------------------------- #


def test_denoise_vmaps_over_observations():
    params, observations, noise, _ = _problem()
    batched = denoise(params, observations, noise)

    def single(x, s):
        return denoise(params, x[None, :], s[None, :, :])[0]

    mapped = jax.vmap(single)(observations, noise)
    np.testing.assert_allclose(np.asarray(mapped), np.asarray(batched), rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# explicit-PRNG semantics for the Monte-Carlo loss
# --------------------------------------------------------------------------- #


def test_mc_prng_semantics():
    params, observations, noise, bandwidths = _problem()
    key = jax.random.PRNGKey(0)

    reuse_a = convmmd_loss_mc(params, observations, noise, bandwidths, key, 128)
    reuse_b = convmmd_loss_mc(params, observations, noise, bandwidths, key, 128)
    assert float(reuse_a) == float(reuse_b)  # identical key -> identical draw

    left, right = jax.random.split(key)
    split_a = convmmd_loss_mc(params, observations, noise, bandwidths, left, 128)
    split_b = convmmd_loss_mc(params, observations, noise, bandwidths, right, 128)
    assert float(split_a) != float(split_b)  # split keys -> independent draws

    with pytest.raises(TypeError):
        convmmd_loss_mc(params, observations, noise, bandwidths)  # missing key


def test_mc_converges_to_analytic():
    params, observations, noise, bandwidths = _problem()
    analytic = float(convmmd_loss_analytic(params, observations, noise, bandwidths))

    def error_at(num_samples: int) -> tuple[float, float]:
        keys = jax.random.split(jax.random.PRNGKey(11), 8)
        estimates = [
            float(convmmd_loss_mc(params, observations, noise, bandwidths, k, num_samples))
            for k in keys
        ]
        mean = float(np.mean(estimates))
        standard_error = float(np.std(estimates) / np.sqrt(len(estimates)))
        return abs(mean - analytic), standard_error

    _, coarse_se = error_at(256)
    fine_error, fine_se = error_at(8192)
    # The estimator spread (standard error) shrinks reliably with M...
    assert fine_se < coarse_se
    # ...and the fine estimate agrees with the analytic oracle within noise.
    assert fine_error < 4.0 * fine_se + 1e-9


# --------------------------------------------------------------------------- #
# device residency
# --------------------------------------------------------------------------- #


def test_outputs_reside_on_the_default_device():
    params, observations, noise, bandwidths = _problem()
    loss = convmmd_loss_analytic(params, observations, noise, bandwidths)
    posterior_mean = denoise(params, observations, noise)
    default = jax.devices()[0]
    assert default in loss.devices()
    assert default in posterior_mean.devices()


# --------------------------------------------------------------------------- #
# failure / status behavior
# --------------------------------------------------------------------------- #


def test_non_pd_covariance_surfaces_as_visible_nan_not_finite_success():
    params, observations, noise, bandwidths = _problem()
    # A clearly indefinite "covariance" (negative diagonal): the differentiable
    # core does not silently repair it; the loss is NaN, never a finite value
    # that could pass as success.
    broken = ConvMMDParams(
        weights=params.weights,
        means=params.means,
        covariances=params.covariances.at[0].set(-jnp.eye(2)),
    )
    loss = convmmd_loss_analytic(broken, observations, noise, bandwidths)
    assert jnp.isnan(loss)


def test_valid_fit_reports_non_failure_and_reduces_loss():
    _, observations, noise, bandwidths = _problem()
    result = fit_analytic(_unconstrained(), observations, noise, bandwidths, n_steps=200)
    assert not bool(result.numerical_failure)
    assert int(result.status) in (
        int(ConvMMDFitStatus.CONVERGED),
        int(ConvMMDFitStatus.MAX_ITER),
    )
    assert float(result.loss) <= float(result.history[0])
    assert jnp.all(jnp.isfinite(result.parameters.covariances))
    assert result.metadata.contract_id == "xdgmm-jax.convmmd"


def test_divergent_fit_rolls_back_to_finite_state_with_failure_status():
    _, observations, noise, bandwidths = _problem()
    # A valid start but an enormous learning rate drives the iterate to
    # non-finite; the fit must report NUMERICAL_FAILURE and keep a finite
    # best-so-far parameter set.
    result = fit_analytic(
        _unconstrained(), observations, noise, bandwidths,
        n_steps=50, learning_rate=1e30, clip_norm=1e30,
    )
    assert int(result.status) == int(ConvMMDFitStatus.NUMERICAL_FAILURE)
    assert bool(result.numerical_failure)
    assert jnp.isfinite(result.loss)
    assert jnp.all(jnp.isfinite(result.parameters.covariances))


def test_analytic_reported_loss_is_exact_at_returned_params():
    """The analytic fit's reported loss recomputes exactly at its returned
    parameters -- no biased ``min`` over noisy per-step estimates."""

    _, observations, noise, bandwidths = _problem()
    result = fit_analytic(_unconstrained(), observations, noise, bandwidths, n_steps=150)
    recomputed = convmmd_loss_analytic(
        result.parameters, observations, noise, bandwidths
    )
    np.testing.assert_allclose(
        float(result.loss), float(recomputed), rtol=1e-12, atol=1e-12
    )


def test_n_steps_one_does_not_spuriously_converge():
    """A single-step fit must not claim convergence via an out-of-bounds
    two-point change."""

    _, observations, noise, bandwidths = _problem()
    result = fit_analytic(_unconstrained(), observations, noise, bandwidths, n_steps=1)
    assert not bool(result.converged)
    assert int(result.status) == int(ConvMMDFitStatus.MAX_ITER)


def test_mc_fit_never_reports_spurious_convergence():
    """The stochastic MC path uses different keys per step, so the two-point
    convergence test is meaningless; it must never report CONVERGED -- even when
    the parameters are frozen at an optimum (learning_rate=0)."""

    from development.convmmd_fit import fit_mc

    _, observations, noise, bandwidths = _problem()
    for learning_rate in (1e-2, 0.0):
        result = fit_mc(
            _unconstrained(), observations, noise, bandwidths, jax.random.PRNGKey(5),
            num_samples=128, n_steps=60, learning_rate=learning_rate,
        )
        assert not bool(result.converged)
        assert int(result.status) in (
            int(ConvMMDFitStatus.MAX_ITER),
            int(ConvMMDFitStatus.NUMERICAL_FAILURE),
        )


def test_mc_reported_loss_is_an_honest_estimate_not_a_biased_min():
    """The MC fit's reported loss is a single estimate at the returned params,
    close to the exact analytic loss there -- not an optimistically-biased
    minimum over noisy draws."""

    from development.convmmd_fit import fit_mc

    _, observations, noise, bandwidths = _problem()
    result = fit_mc(
        _unconstrained(), observations, noise, bandwidths, jax.random.PRNGKey(9),
        num_samples=256, n_steps=120,
    )
    analytic_at_returned = float(
        convmmd_loss_analytic(result.parameters, observations, noise, bandwidths)
    )
    assert np.isfinite(float(result.loss))
    # An unbiased single estimate sits within a small band of the exact value;
    # a biased min-over-draws would be conspicuously below it.
    assert abs(float(result.loss) - analytic_at_returned) < 0.05


def test_fit_state_kernels_are_jitkable():
    _, observations, noise, bandwidths = _problem()
    analytic = jax.jit(
        lambda i, x, s, b: fit_analytic_state(i, x, s, b, n_steps=25)
    )(_unconstrained(), observations, noise, bandwidths)
    _block(analytic)
    assert jnp.isfinite(analytic.loss)

    mc = jax.jit(
        lambda i, x, s, b, k: fit_mc_state(i, x, s, b, k, num_samples=64, n_steps=25)
    )(_unconstrained(), observations, noise, bandwidths, jax.random.PRNGKey(2))
    _block(mc)
    assert jnp.isfinite(mc.loss)
