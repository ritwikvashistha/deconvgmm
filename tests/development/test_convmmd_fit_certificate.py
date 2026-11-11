"""Evidence for the convMMD fit status certificate (B1) and the constant schedule (B2).

``converged`` used to be a two-point loss-change test, which reported convergence at
points far from stationary (recovered finding B1). It is now the certificate
``max|grad| <= grad_tol`` at the returned unconstrained parameters, on the
deterministic path only. ``schedule="constant"`` runs the same AdamW at a fixed
learning rate (finding B2); the shipped default stays the cosine schedule.
"""

from __future__ import annotations

import functools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.convmmd import (
    ConvMMDUnconstrained,
    convmmd_loss_analytic,
    median_bandwidths,
    to_canonical,
)
from development.convmmd_fit import (
    ConvMMDFitStatus,
    fit_analytic,
    fit_analytic_state,
    fit_mc,
)


def _two_component_problem(n=200, seed=7):
    rng = np.random.default_rng(seed)
    means = np.array([[-1.5, 0.0], [1.5, 0.5]])
    covs = np.array([[[0.30, 0.05], [0.05, 0.20]], [[0.25, -0.04], [-0.04, 0.35]]])
    labels = rng.choice(2, size=n, p=[0.45, 0.55])
    latent = means[labels] + np.einsum(
        "nij,nj->ni", np.linalg.cholesky(covs)[labels], rng.standard_normal((n, 2))
    )
    variances = rng.uniform(0.05, 0.2, size=(n, 2))
    x = latent + np.sqrt(variances) * rng.standard_normal((n, 2))
    noise = np.zeros((n, 2, 2))
    noise[:, np.arange(2), np.arange(2)] = variances
    x, noise = jnp.asarray(x), jnp.asarray(noise)
    return x, noise, median_bandwidths(x)


def _start():
    return ConvMMDUnconstrained(
        alphas=jnp.asarray([0.2, -0.1]),
        means=jnp.asarray([[-1.0, 0.3], [1.0, 0.2]]),
        unconstrained_L=jnp.asarray(
            [[[-0.4, 0.0], [0.1, -0.6]], [[-0.5, 0.0], [-0.05, -0.3]]]
        ),
    )


def _independent_max_abs_grad(unc, x, noise, bandwidths):
    grads = jax.grad(
        lambda u: convmmd_loss_analytic(to_canonical(u), x, noise, bandwidths)
    )(unc)
    return max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree_util.tree_leaves(grads))


_constant_fit = functools.partial(
    fit_analytic, learning_rate=3.0e-2, weight_decay=0.0, schedule="constant"
)


def _certified_fit(start, x, noise, bandwidths, **kwargs):
    return _constant_fit(
        start, x, noise, bandwidths, n_steps=3000, stop_at_certificate=True, **kwargs
    )


def test_old_two_point_rule_would_have_certified_a_non_stationary_point():
    """A frozen optimizer makes the loss change vanish anywhere; the two-point
    rule then said CONVERGED. The certificate looks at the gradient instead."""

    x, noise, bandwidths = _two_component_problem()
    result = fit_analytic(
        _start(), x, noise, bandwidths, n_steps=5, learning_rate=1.0e-12
    )
    history = np.asarray(result.history)
    old_rule = abs(history[-1] - history[-2]) / (abs(history[-1]) + 1e-12) < 1.0e-6
    assert old_rule, "the planted point no longer fools the two-point rule"
    assert float(result.max_abs_grad) > 1.0e-3
    assert not bool(result.converged)
    assert int(result.status) == int(ConvMMDFitStatus.MAX_ITER)


def test_max_abs_grad_is_the_gradient_at_the_returned_parameters():
    x, noise, bandwidths = _two_component_problem()
    result = fit_analytic(_start(), x, noise, bandwidths, n_steps=40)
    expected = _independent_max_abs_grad(result.unconstrained, x, noise, bandwidths)
    np.testing.assert_allclose(float(result.max_abs_grad), expected, rtol=1e-12, atol=0)
    assert float(result.grad_tol) == 1.0e-6


def test_converged_flips_exactly_at_grad_tol():
    """Same fixed-length trajectory, three thresholds: certified iff
    max|grad| <= grad_tol at the returned point."""

    x, noise, bandwidths = _two_component_problem()
    stopped = _certified_fit(_start(), x, noise, bandwidths)
    assert bool(stopped.converged), f"no certificate in 3000 steps: {float(stopped.max_abs_grad):.3e}"
    steps = int(stopped.n_iter)
    assert 0 < steps < 3000

    # A constant schedule does not depend on n_steps, so a plain fixed-length fit
    # of exactly that many updates ends at the same certified point.
    result = _constant_fit(_start(), x, noise, bandwidths, n_steps=steps)
    for got, want in zip(
        jax.tree_util.tree_leaves(result.unconstrained),
        jax.tree_util.tree_leaves(stopped.unconstrained),
    ):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12, atol=1e-13)
    achieved = float(result.max_abs_grad)
    assert achieved <= 1.0e-6
    assert bool(result.converged)
    assert int(result.status) == int(ConvMMDFitStatus.CONVERGED)
    np.testing.assert_allclose(
        achieved,
        _independent_max_abs_grad(result.unconstrained, x, noise, bandwidths),
        rtol=1e-10,
        atol=0,
    )

    strict = _constant_fit(_start(), x, noise, bandwidths, n_steps=steps, grad_tol=0.5 * achieved)
    assert not bool(strict.converged)
    assert int(strict.status) == int(ConvMMDFitStatus.MAX_ITER)
    # Without stopping, grad_tol only reads the returned point; it never changes
    # the trajectory.
    np.testing.assert_array_equal(np.asarray(strict.history), np.asarray(result.history))
    assert float(strict.max_abs_grad) == achieved

    loose = _constant_fit(_start(), x, noise, bandwidths, n_steps=steps, grad_tol=2.0 * achieved)
    assert bool(loose.converged)


def test_stop_at_certificate_freezes_at_the_first_certified_iterate():
    x, noise, bandwidths = _two_component_problem()
    stopped = _certified_fit(_start(), x, noise, bandwidths)
    k = int(stopped.n_iter)
    history = np.asarray(stopped.history)
    assert history.shape == (3000,)
    # Every slot from the stop on repeats the certified iterate's loss ...
    np.testing.assert_array_equal(history[k:], np.full(3000 - k, history[k]))
    np.testing.assert_allclose(history[k], float(stopped.loss), rtol=1e-13, atol=1e-15)
    # ... the path up to it is the unstopped path ...
    plain = _constant_fit(_start(), x, noise, bandwidths, n_steps=k + 25)
    np.testing.assert_allclose(history[: k + 1], np.asarray(plain.history)[: k + 1], rtol=1e-12, atol=1e-14)
    # ... and no earlier iterate was certified.
    before = _constant_fit(_start(), x, noise, bandwidths, n_steps=k - 1)
    assert float(before.max_abs_grad) > 1.0e-6
    assert not bool(before.converged)


def test_stop_at_certificate_is_off_by_default():
    x, noise, bandwidths = _two_component_problem(n=40)
    result = fit_analytic(_start(), x, noise, bandwidths, n_steps=7)
    assert int(result.n_iter) == 7


def test_certificate_ignores_the_retired_tol_control():
    x, noise, bandwidths = _two_component_problem()
    a = fit_analytic(_start(), x, noise, bandwidths, n_steps=20, tol=1.0)
    b = fit_analytic(_start(), x, noise, bandwidths, n_steps=20, tol=1.0e-30)
    assert bool(a.converged) == bool(b.converged) == (float(a.max_abs_grad) <= 1.0e-6)


def test_zero_step_fit_certifies_only_a_stationary_start():
    x, noise, bandwidths = _two_component_problem()
    rough = fit_analytic(_start(), x, noise, bandwidths, n_steps=0)
    assert not bool(rough.converged)
    polished = _certified_fit(_start(), x, noise, bandwidths)
    again = fit_analytic(polished.unconstrained, x, noise, bandwidths, n_steps=0)
    assert bool(again.converged)
    assert float(again.max_abs_grad) == float(polished.max_abs_grad)


def test_failed_fit_never_certifies_and_reports_gradient_at_rollback_point():
    x, noise, bandwidths = _two_component_problem()
    result = fit_analytic(
        _start(), x, noise, bandwidths, n_steps=50, learning_rate=1e30, clip_norm=1e30
    )
    assert int(result.status) == int(ConvMMDFitStatus.NUMERICAL_FAILURE)
    assert not bool(result.converged)
    assert np.isfinite(float(result.max_abs_grad))


def test_mc_path_reports_a_gradient_estimate_but_never_certifies():
    x, noise, bandwidths = _two_component_problem(n=60)
    polished = _certified_fit(_start(), x, noise, bandwidths)
    # Parameters frozen at a certified analytic optimum: the MC estimate of the
    # gradient is reported, but the stochastic path never claims convergence.
    result = fit_mc(
        polished.unconstrained, x, noise, bandwidths, jax.random.PRNGKey(3),
        num_samples=64, n_steps=5, learning_rate=0.0,
    )
    assert np.isfinite(float(result.max_abs_grad))
    assert not bool(result.converged)
    assert int(result.status) == int(ConvMMDFitStatus.MAX_ITER)


# ---------------------------------------------------------------------------
# Constant learning-rate schedule (B2)
# ---------------------------------------------------------------------------


def _hand_rolled_adamw(unc, x, noise, bandwidths, *, steps, lr, wd, clip):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    objective = lambda u: convmmd_loss_analytic(to_canonical(u), x, noise, bandwidths)
    leaves, treedef = jax.tree_util.tree_flatten(unc)
    m = [jnp.zeros_like(leaf) for leaf in leaves]
    v = [jnp.zeros_like(leaf) for leaf in leaves]
    for t in range(1, steps + 1):
        grads = jax.tree_util.tree_leaves(jax.grad(objective)(jax.tree_util.tree_unflatten(treedef, leaves)))
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in grads))
        grads = [g * jnp.minimum(1.0, clip / (norm + 1e-12)) for g in grads]
        m = [beta1 * a + (1 - beta1) * g for a, g in zip(m, grads)]
        v = [beta2 * b + (1 - beta2) * g * g for b, g in zip(v, grads)]
        leaves = [
            p - lr * ((a / (1 - beta1**t)) / (jnp.sqrt(b / (1 - beta2**t)) + eps) + wd * p)
            for p, a, b in zip(leaves, m, v)
        ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_constant_schedule_is_fixed_rate_adamw(weight_decay):
    x, noise, bandwidths = _two_component_problem(n=80)
    result = fit_analytic(
        _start(), x, noise, bandwidths, n_steps=4, learning_rate=3e-2,
        weight_decay=weight_decay, schedule="constant",
    )
    expected = _hand_rolled_adamw(
        _start(), x, noise, bandwidths, steps=4, lr=3e-2, wd=weight_decay, clip=1.0
    )
    for got, want in zip(
        jax.tree_util.tree_leaves(result.unconstrained), jax.tree_util.tree_leaves(expected)
    ):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12, atol=1e-14)


def test_default_schedule_is_still_cosine():
    x, noise, bandwidths = _two_component_problem(n=80)
    default = fit_analytic(_start(), x, noise, bandwidths, n_steps=6)
    cosine = fit_analytic(_start(), x, noise, bandwidths, n_steps=6, schedule="cosine")
    constant = fit_analytic(_start(), x, noise, bandwidths, n_steps=6, schedule="constant")
    np.testing.assert_array_equal(np.asarray(default.history), np.asarray(cosine.history))
    assert not np.array_equal(
        np.asarray(default.unconstrained.means), np.asarray(constant.unconstrained.means)
    )
    assert float(default.learning_rate) == 1.0e-2
    assert float(default.weight_decay) == 1.0e-1


def test_unknown_schedule_is_rejected():
    x, noise, bandwidths = _two_component_problem(n=20)
    with pytest.raises(ValueError, match="schedule"):
        fit_analytic(_start(), x, noise, bandwidths, n_steps=2, schedule="linear")


def test_constant_schedule_fit_is_jittable_with_static_schedule():
    x, noise, bandwidths = _two_component_problem(n=40)
    jitted = jax.jit(
        fit_analytic_state, static_argnames=("n_steps", "schedule")
    )
    a = jitted(_start(), x, noise, bandwidths, n_steps=10, schedule="constant",
               learning_rate=3e-2, weight_decay=0.0)
    b = fit_analytic_state(_start(), x, noise, bandwidths, n_steps=10, schedule="constant",
                           learning_rate=3e-2, weight_decay=0.0)
    np.testing.assert_allclose(np.asarray(a.history), np.asarray(b.history), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(float(a.max_abs_grad), float(b.max_abs_grad), rtol=1e-10)
