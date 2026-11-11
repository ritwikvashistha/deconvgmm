"""Evidence for phase 2 of the J1 convMMD recipe (``development.convmmd_recipe``).

Covers saddle-aware stopping (the resumable Adam chunk, the identifiable basis,
the exact and Lanczos curvature checks, the escape) and the HU/Kkz witness-repair
chain (the witness statistic, the birth move, per-row losses and the paired z),
then the whole recipe against J1 fits recovered from
``results/convmmd_g/j_20261025/j1/``: a clean fit where no move is attempted and an
outlier fit where J1 kept an H-triggered move. The J1 outlier generator and greedy
seeding are ported below (data only) so that dataset and its start are rebuilt
draw for draw.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import development.convmmd_recipe as recipe
from development.convmmd import (
    ConvMMDParams,
    convmmd_loss_analytic,
    expected_rbf_kernel,
    median_bandwidths,
    to_canonical,
)
from development.convmmd_fit import ConvMMDFitStatus, fit_analytic_state
from development.convmmd_recipe import (
    _adam_chunk,
    birth_at,
    effective,
    fit_convmmd_recipe,
    fit_saddle_aware,
    floor_value,
    floored_loss,
    identifiable_basis,
    paired_z,
    row_losses,
    smallest_curvature,
    to_floor_unc,
    witness_candidates,
    witness_stats,
)
from tests.development.test_convmmd_recipe import (
    J1_SEED_BASE,
    _J1_REFERENCE,
    _j1_clean_dataset,
    _random_cov,
    _recovered,
    _small_problem,
    _truth_d,
)


def _problem():
    truth, x, noise, bandwidths = _small_problem()
    floor = floor_value(noise)
    start = to_floor_unc(ConvMMDParams(*(jnp.asarray(a) for a in truth)), floor)
    return truth, x, noise, bandwidths, floor, start


# ---------------------------------------------------------------------------
# Resumable Adam chunk and the chunked driver
# ---------------------------------------------------------------------------


def _constant_fit(start, x, noise, bandwidths, floor, n_steps):
    return fit_analytic_state(
        start, x, noise, bandwidths, n_steps=n_steps, learning_rate=3e-2,
        weight_decay=0.0, schedule="constant", covariance_floor=floor,
    )


def test_adam_chunks_resume_to_one_constant_rate_fit():
    _, x, noise, bandwidths, floor, start = _problem()
    reference = _constant_fit(start, x, noise, bandwidths, floor, 60)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, start)
    args = (x, noise, bandwidths, floor, jnp.asarray(3e-2), jnp.asarray(0.0), jnp.asarray(1.0))
    state = (start, zeros, zeros, jnp.asarray(0, jnp.int32))
    histories = []
    for n in (25, 35):
        unc, m, v, t, failed, loss, gmax, history, applied = _adam_chunk(*state, *args, n_steps=n)
        assert not bool(failed) and int(applied) == n
        state = (unc, m, v, t)
        histories.append(history)
    assert int(state[3]) == 60
    np.testing.assert_allclose(np.concatenate(histories), np.asarray(reference.history), rtol=1e-12, atol=1e-14)
    for got, want in zip(state[0], reference.unconstrained):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(float(loss), float(reference.loss), rtol=1e-12)
    np.testing.assert_allclose(float(gmax), float(reference.max_abs_grad), rtol=1e-8, atol=1e-14)


def test_chunked_fit_without_a_certificate_is_the_plain_constant_fit():
    """grad_tol = 0 never certifies, so no curvature check runs: chunks of 30, 30
    and 20 are one 80-step constant-rate fit, honestly MAX_ITER."""

    _, x, noise, bandwidths, floor, start = _problem()
    fit = fit_saddle_aware(start, x, noise, bandwidths, floor, max_steps=80, chunk=30, grad_tol=0.0)
    reference = _constant_fit(start, x, noise, bandwidths, floor, 80)
    assert int(fit.n_iter) == 80 and int(fit.curvature_checks) == 0
    assert int(fit.status) == int(ConvMMDFitStatus.MAX_ITER) and not bool(fit.converged)
    assert np.isnan(float(fit.min_eigenvalue))
    np.testing.assert_allclose(np.asarray(fit.history), np.asarray(reference.history), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(float(fit.loss), float(reference.loss), rtol=1e-12)
    for got, want in zip(fit.parameters, reference.parameters):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-10, atol=1e-12)


def test_chunked_fit_stops_at_a_certified_minimum_on_a_chunk_boundary():
    _, x, noise, bandwidths, floor, start = _problem()
    fit = fit_saddle_aware(start, x, noise, bandwidths, floor)
    assert bool(fit.converged) and int(fit.status) == int(ConvMMDFitStatus.CONVERGED)
    assert int(fit.n_iter) % 200 == 0 and int(fit.n_iter) < 3000
    assert float(fit.max_abs_grad) <= 1e-6
    assert int(fit.curvature_checks) == 1 and int(fit.escapes) == 0 and not bool(fit.saddle)
    assert float(fit.min_eigenvalue) >= -1e-10
    np.testing.assert_allclose(
        float(fit.loss), float(floored_loss(to_canonical(fit.unconstrained), x, noise, bandwidths, floor)), rtol=1e-13
    )
    # One chunk fewer was not yet certified.
    shorter = fit_saddle_aware(start, x, noise, bandwidths, floor, max_steps=int(fit.n_iter) - 200)
    assert not bool(shorter.converged)


def test_chunked_fit_rolls_back_a_step_that_fails_inside_a_chunk():
    """A ridge of minus (lambda_min - 1e-3) leaves the floored covariance nearly
    singular; one step at learning rate 1 makes it indefinite, so step 1 scores
    non-finite. The fit stops and returns iterate 0, the last finite one."""

    truth, x, noise, bandwidths = _small_problem()
    params = ConvMMDParams(*(jnp.asarray(a) for a in truth))
    floor = -(float(jnp.min(jnp.linalg.eigvalsh(params.covariances))) - 1e-3)
    start = to_floor_unc(params, 0.0)
    fit = fit_saddle_aware(
        start, x, noise, bandwidths, floor, max_steps=100, chunk=100, learning_rate=1.0, grad_tol=0.0
    )
    assert bool(fit.numerical_failure) and int(fit.status) == int(ConvMMDFitStatus.NUMERICAL_FAILURE)
    assert not bool(fit.converged)
    history = np.asarray(fit.history)
    assert np.isfinite(history[0]) and not np.isfinite(history[1])
    assert int(fit.n_iter) == 1  # the update that produced the failing iterate
    assert float(fit.loss) == history[0]
    for got, want in zip(fit.unconstrained, start):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_chunked_fit_never_updates_a_non_finite_start():
    _, x, noise, bandwidths, floor, start = _problem()
    poisoned = start._replace(means=start.means.at[0, 0].set(jnp.nan))
    fit = fit_saddle_aware(poisoned, x, noise, bandwidths, floor, max_steps=50)
    assert bool(fit.numerical_failure) and int(fit.status) == int(ConvMMDFitStatus.NUMERICAL_FAILURE)
    assert not bool(fit.converged) and int(fit.n_iter) == 0


# ---------------------------------------------------------------------------
# Identifiable basis and curvature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k,d", [(1, 2), (3, 2), (3, 5), (2, 7)])
def test_identifiable_basis_is_orthonormal_and_spans_the_used_parameters(k, d):
    q = np.asarray(identifiable_basis(k, d))
    p = k + k * d + k * d * d
    assert q.shape == (p, (k - 1) + k * d + k * d * (d + 1) // 2)
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-14)
    shift = np.zeros(p)
    shift[:k] = 1.0
    np.testing.assert_allclose(shift @ q, 0.0, atol=1e-14)
    upper = np.zeros((k, d, d), bool)
    upper[:, np.triu_indices(d, 1)[0], np.triu_indices(d, 1)[1]] = True
    unused = np.concatenate([np.zeros(k + k * d, bool), upper.ravel()])
    np.testing.assert_array_equal(q[unused], 0.0)


def test_exact_curvature_is_the_smallest_eigenvalue_of_the_projected_hessian():
    _, x, noise, bandwidths, floor, start = _problem()
    flat, unravel = ravel_pytree(start)
    hessian = np.asarray(jax.hessian(
        lambda v: floored_loss(to_canonical(unravel(v)), x, noise, bandwidths, floor)
    )(flat))
    q = np.asarray(identifiable_basis(3, 2))
    values = np.linalg.eigvalsh(q.T @ hessian @ q)
    curvature = smallest_curvature(start, x, noise, bandwidths, floor)
    assert curvature.mode == "exact" and curvature.n_hvps == q.shape[1]
    assert float(curvature.min_eigenvalue) == pytest.approx(values[0], rel=1e-9, abs=1e-12)
    vector = np.asarray(curvature.vector)
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, rtol=1e-12)
    np.testing.assert_allclose(vector @ hessian @ vector, values[0], rtol=1e-8, atol=1e-12)
    # The dropped directions do not move the loss.
    shift = np.zeros_like(flat)
    shift[:3] = 1.0
    np.testing.assert_allclose(hessian @ shift, 0.0, atol=1e-10)


def _seven_dimensional_point():
    """K = 3, D = 7: P = 2 + 21 + 84 = 107 > 80, so the default check is Lanczos."""

    rng = np.random.default_rng(3)
    truth = _truth_d(rng, 3, 7)
    labels = rng.choice(3, size=40, p=truth[0])
    latent = truth[1][labels] + np.einsum(
        "nij,nj->ni", np.linalg.cholesky(truth[2])[labels], rng.standard_normal((40, 7))
    )
    variances = rng.uniform(0.05, 0.2, size=(40, 7))
    x = jnp.asarray(latent + np.sqrt(variances) * rng.standard_normal((40, 7)))
    noise = jnp.asarray(variances[:, :, None] * np.eye(7))
    floor = floor_value(noise)
    point = to_floor_unc(ConvMMDParams(*(jnp.asarray(a) for a in truth)), floor)
    return point, x, noise, median_bandwidths(x), floor


def test_lanczos_agrees_with_the_exact_check_above_the_cut():
    point, x, noise, bandwidths, floor = _seven_dimensional_point()
    exact = smallest_curvature(point, x, noise, bandwidths, floor, exact_limit=200)
    assert exact.mode == "exact" and exact.n_hvps == 107
    full = smallest_curvature(point, x, noise, bandwidths, floor, lanczos_steps=107)
    assert full.mode == "lanczos"
    assert float(full.min_eigenvalue) == pytest.approx(float(exact.min_eigenvalue), rel=1e-7, abs=1e-12)
    default = smallest_curvature(point, x, noise, bandwidths, floor)
    assert default.mode == "lanczos" and default.n_hvps == 80
    # A Ritz value never undershoots the smallest eigenvalue; 80 steps get its
    # sign and size.
    assert float(default.min_eigenvalue) >= float(exact.min_eigenvalue) - 1e-12
    assert np.sign(float(default.min_eigenvalue)) == np.sign(float(exact.min_eigenvalue))
    assert float(default.min_eigenvalue) == pytest.approx(float(exact.min_eigenvalue), rel=1e-2)


def _two_cluster_saddle():
    """A certified K = 1 fit to two separated clusters, duplicated: the K = 2
    mixture with both halves on the merged component is a stationary point that
    splitting the halves lowers (a planted saddle)."""

    rng = np.random.default_rng(5)
    centres = np.array([[-2.0, 0.0], [2.0, 0.0]])
    latent = centres[rng.integers(2, size=300)] + 0.4 * rng.standard_normal((300, 2))
    variances = rng.uniform(0.05, 0.2, size=(300, 2))
    x = jnp.asarray(latent + np.sqrt(variances) * rng.standard_normal((300, 2)))
    noise = jnp.asarray(variances[:, :, None] * np.eye(2))
    bandwidths, floor = median_bandwidths(x), floor_value(noise)
    one = ConvMMDParams(jnp.ones(1), jnp.mean(x, axis=0)[None], jnp.cov(x.T)[None])
    single = fit_saddle_aware(to_floor_unc(one, floor), x, noise, bandwidths, floor)
    assert bool(single.converged) and not bool(single.saddle)
    merged = single.unconstrained
    duplicated = merged._replace(
        alphas=jnp.zeros(2),
        means=jnp.concatenate([merged.means, merged.means]),
        unconstrained_L=jnp.concatenate([merged.unconstrained_L, merged.unconstrained_L]),
    )
    return duplicated, x, noise, bandwidths, floor, centres


def test_a_planted_saddle_is_flagged_and_escaped():
    start, x, noise, bandwidths, floor, centres = _two_cluster_saddle()
    curvature = smallest_curvature(start, x, noise, bandwidths, floor)
    assert float(curvature.min_eigenvalue) < -1e-10
    at_saddle = float(floored_loss(to_canonical(start), x, noise, bandwidths, floor))

    stuck = fit_saddle_aware(start, x, noise, bandwidths, floor, max_escapes=0)
    assert bool(stuck.saddle) and int(stuck.escapes) == 0 and int(stuck.n_iter) == 0
    # The saddle is a flag: the point is still gradient-certified.
    assert bool(stuck.converged) and int(stuck.status) == int(ConvMMDFitStatus.CONVERGED)
    assert float(stuck.min_eigenvalue) == pytest.approx(float(curvature.min_eigenvalue), rel=1e-12)

    fit = fit_saddle_aware(start, x, noise, bandwidths, floor)
    assert int(fit.escapes) >= 1 and not bool(fit.saddle)
    assert bool(fit.converged) and float(fit.min_eigenvalue) >= -1e-10
    assert float(fit.loss) < at_saddle - 1e-4
    means = np.sort(np.asarray(fit.parameters.means)[:, 0])
    np.testing.assert_allclose(means, centres[:, 0], atol=0.3)


# ---------------------------------------------------------------------------
# Witness statistic, birth move, per-row losses, paired z
# ---------------------------------------------------------------------------


def _reference_witness(params, x, noise, bandwidths, candidates):
    """h_i(z) from the package kernel, then J1's raw / leave-one-out statistics."""

    weights, means, covariances = (np.asarray(a) for a in params)
    x, noise = np.asarray(x), np.asarray(noise)
    n = len(x)
    h = np.zeros((n, len(candidates)))
    for gamma in np.asarray(bandwidths):
        for c, row in enumerate(candidates):
            z = x[row]
            h[:, c] += np.asarray(expected_rbf_kernel(jnp.asarray(x - z), jnp.asarray(noise), gamma))
            for k in range(len(weights)):
                omega = covariances[k][None] + 2.0 * noise
                h[:, c] -= weights[k] * np.asarray(
                    expected_rbf_kernel(jnp.asarray(np.broadcast_to(means[k] - z, x.shape)), jnp.asarray(omega), gamma)
                )
    h /= len(bandwidths)
    own = h[np.asarray(candidates), np.arange(len(candidates))]
    raw = h.sum(0) / n
    loo = (h.sum(0) - own) / (n - 1)
    var = (((h * h).sum(0) - own**2) - (n - 1) * loo**2) / (n - 2)
    se = np.sqrt(np.maximum(var, 1e-300) / (n - 1))
    j_raw, j_loo = int(np.argmax(raw)), int(np.argmax(loo))
    return {"raw_peak": raw[j_raw], "raw_row": candidates[j_raw], "loo_peak": loo[j_loo],
            "loo_row": candidates[j_loo], "se": se[j_loo], "t": loo[j_loo] / se[j_loo],
            "t_max": np.max(loo / se)}


@pytest.mark.parametrize("full_noise", [False, True])
def test_witness_stats_match_the_package_kernel(full_noise, monkeypatch):
    truth, x, noise, bandwidths = _small_problem(n=40)
    if full_noise:
        rng = np.random.default_rng(8)
        noise = jnp.asarray(np.stack([_random_cov(rng, 2, 0.05, 0.2) for _ in range(40)]))
    params = ConvMMDParams(*(jnp.asarray(a) for a in truth))
    candidates = np.array([0, 3, 7, 11, 19, 23, 31])
    want = _reference_witness(params, x, noise, bandwidths, candidates)
    for entries in (10**9, 40 * 3):  # one block; blocks of 3 with a padded last block
        monkeypatch.setattr(recipe, "_WITNESS_BLOCK_ENTRIES", entries)
        got = witness_stats(params, x, noise, bandwidths, jnp.asarray(candidates))
        for name, value in want.items():
            assert float(getattr(got, name)) == pytest.approx(float(value), rel=1e-10, abs=1e-15), name


@pytest.mark.parametrize("cell", sorted(_J1_REFERENCE))
def test_witness_stats_reproduce_j1_at_its_recorded_fits(cell):
    reference = _J1_REFERENCE[cell]
    _, x, noise = _j1_clean_dataset(*cell)
    x, noise = jnp.asarray(x), jnp.asarray(noise)
    params = ConvMMDParams(*(jnp.asarray(reference[key]) for key in ("weights", "means", "covariances")))
    got = witness_stats(params, x, noise, median_bandwidths(x), witness_candidates(None, 500))
    want = _J1_WITNESS[cell]
    for name, value in want.items():
        assert float(getattr(got, name)) == pytest.approx(value, rel=1e-9, abs=1e-15), name


# Recorded chain[0]["stats"] of j1_clean_d2_n500_r{dataset}_kplus1_floor.
_J1_WITNESS = {
    (2, 500, 2): {"raw_peak": 0.0008346307839766385, "raw_row": 12, "loo_peak": 0.0004816431429586023,
                  "loo_row": 12, "se": 0.0018170650428773574, "t": 0.265066539498174,
                  "t_max": 0.28384920465527685, "t_max_row": 454},
    (2, 500, 0): {"raw_peak": 0.0011094946686098167, "raw_row": 157, "loo_peak": 0.0007329509781243527,
                  "loo_row": 157, "se": 0.0028332125085536127, "t": 0.2586996125110758,
                  "t_max": 0.2586996125110758, "t_max_row": 157},
}


def test_witness_candidates_subsample_only_above_the_limit():
    np.testing.assert_array_equal(np.asarray(witness_candidates(jax.random.PRNGKey(0), 50, size=50)), np.arange(50))
    rows = np.asarray(witness_candidates(jax.random.PRNGKey(0), 120, size=50))
    assert len(np.unique(rows)) == 50 and np.all(np.diff(rows) > 0) and rows.max() < 120


def test_birth_at_follows_the_reference_move():
    truth, x, noise, _ = _small_problem(n=1500)
    params = ConvMMDParams(*(jnp.asarray(a) for a in truth))
    row, floor = 17, 0.02
    got = birth_at(params, x, noise, row, floor)

    w, mu, cov = (np.array(a, float) for a in truth)
    xs, s = np.asarray(x), np.asarray(noise)
    j = int(np.argmin(w))
    near = np.argsort(((xs - xs[row]) ** 2).sum(axis=1))[: max(10, len(xs) // 100)]
    assert len(near) == 15
    mu[j] = xs[near].mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(xs[near].T) - s[near].mean(axis=0))
    cov[j] = (vectors * np.maximum(values, floor)) @ vectors.T
    new = max(w[j], 0.05)
    others = np.arange(3) != j
    w[others] *= (1 - new) / w[others].sum()
    w[j] = new
    for got_a, want_a in zip(got, (w, mu, cov)):
        np.testing.assert_allclose(np.asarray(got_a), want_a, rtol=1e-12, atol=1e-14)


def test_row_losses_differences_average_to_the_loss_difference():
    truth, x, noise, bandwidths = _small_problem()
    a = ConvMMDParams(*(jnp.asarray(t) for t in truth))
    b = a._replace(means=a.means + 0.1)
    rows_a = row_losses(a, x, noise, bandwidths, chunk=50)  # 50, 50 and a padded 20
    np.testing.assert_allclose(np.asarray(rows_a), np.asarray(row_losses(a, x, noise, bandwidths)), rtol=1e-14)
    drop = float(jnp.mean(row_losses(b, x, noise, bandwidths) - rows_a))
    want = float(convmmd_loss_analytic(b, x, noise, bandwidths) - convmmd_loss_analytic(a, x, noise, bandwidths))
    assert drop == pytest.approx(want, rel=1e-10)


def test_paired_z_is_positive_when_the_loss_falls():
    rng = np.random.default_rng(0)
    changes = -0.01 + 0.002 * rng.standard_normal(400)
    want = -np.mean(changes) / (np.std(changes, ddof=1) / np.sqrt(400))
    assert float(paired_z(jnp.asarray(changes))) == pytest.approx(want, rel=1e-12)
    assert float(paired_z(-jnp.asarray(changes))) == pytest.approx(-want, rel=1e-12)
    assert float(paired_z(jnp.zeros(10))) == 0.0
    assert float(paired_z(jnp.full(10, -1.0))) == np.inf


# ---------------------------------------------------------------------------
# The chain and the recipe
# ---------------------------------------------------------------------------


def test_recipe_without_witness_moves_is_the_base_fit():
    _, x, noise, _ = _small_problem()
    result = fit_convmmd_recipe(x, noise, 3, key=jax.random.PRNGKey(0), witness_moves=0)
    assert result.repairs == ()
    assert int(result.total_steps) == int(result.base.n_iter) == int(result.n_iter)
    np.testing.assert_array_equal(np.asarray(result.history), np.asarray(result.base.history))
    assert float(result.loss) == float(result.base.loss)


def test_a_forced_health_trigger_attempts_a_move_and_keeps_it_only_under_kkz():
    """A threshold no mixture passes fires H on every fit: each move is attempted,
    and the chain keeps a refit only if Kkz accepts it."""

    _, x, noise, _ = _small_problem()
    result = fit_convmmd_recipe(x, noise, 3, key=jax.random.PRNGKey(0), health_collapsed=0.999)
    assert len(result.repairs) >= 1
    current_loss = float(result.base.loss)
    for move in result.repairs:
        assert bool(move.fired_H) and bool(move.attempted) and move.fit is not None
        assert float(move.loss_drop) == pytest.approx(current_loss - float(move.fit.loss), abs=1e-15)
        kkz = (bool(move.accepted) and float(move.kappa_after) >= float(move.kappa_before) and float(move.z) >= 3.0)
        assert bool(move.kept) == kkz
        assert bool(move.accepted) == (float(move.loss_drop) > 1e-9)
        if not bool(move.kept):
            break
        current_loss = float(move.fit.loss)
    kept = [m for m in result.repairs if bool(m.kept)]
    returned = kept[-1].fit if kept else result.base
    assert float(result.loss) == float(returned.loss)
    assert int(result.total_steps) == int(result.base.n_iter) + sum(int(m.fit.n_iter) for m in result.repairs)


@pytest.mark.parametrize("cell", sorted(_J1_REFERENCE))
def test_recipe_attempts_no_move_on_the_healthy_j1_fits(cell):
    truth, x, noise = _j1_clean_dataset(*cell)
    result = fit_convmmd_recipe(jnp.asarray(x), jnp.asarray(noise), 3, key=jax.random.PRNGKey(cell[2]))
    (move,) = result.repairs
    assert not bool(move.attempted) and move.fit is None
    assert float(move.witness.raw_peak) < 5e-3 and float(move.witness.t) < 4.0 and not bool(move.fired_H)
    assert float(result.loss) == float(result.base.loss)
    assert int(result.n_iter) % 200 == 0 and float(result.min_eigenvalue) > 0.0
    assert _recovered(result.parameters, truth)[0]


# --- J1 outlier fit where the chain kept an H-triggered move -----------------

_OUTLIERS_INDEX = 2  # 1 + ("clean", "outliers", ...).index("outliers")


def _outlier_points(rng, truth, count):
    _, means, covs = truth
    smax = float(np.sqrt(max(np.linalg.eigvalsh(c).max() for c in covs)))
    out = []
    while len(out) < count:
        direction = rng.standard_normal(means.shape[1])
        point = means[rng.integers(len(means))] + rng.uniform(8, 15) * smax * direction / np.linalg.norm(direction)
        if np.min(np.linalg.norm(means - point, axis=1)) >= 8 * smax:
            out.append(point)
    return np.asarray(out).reshape(count, means.shape[1])


def _j1_outlier_dataset(dim, n, dataset, k=3):
    """``convmmd_j1_guard.make_dataset`` for the outliers situation (5% outliers)."""

    truth = _truth_d(np.random.default_rng([J1_SEED_BASE, dim, dataset]), k, dim)
    rng = np.random.default_rng([J1_SEED_BASE, _OUTLIERS_INDEX, dim, n, dataset, 1])
    weights, means, covs = truth
    labels = rng.choice(k, size=n, p=weights)
    latent = means[labels] + np.einsum("nij,nj->ni", np.linalg.cholesky(covs)[labels], rng.standard_normal((n, dim)))
    variances = rng.uniform(0.05, 0.2, size=(n, dim))
    e = np.sqrt(variances) * rng.standard_normal((n, dim))
    x = latent + e
    rows = rng.choice(n, size=int(round(0.05 * n)), replace=False)
    x[rows] = _outlier_points(rng, truth, len(rows)) + e[rows]
    noise = np.zeros((n, dim, dim))
    noise[:, np.arange(dim), np.arange(dim)] = variances
    return truth, x, noise


def _j1_greedy(rng, x, k, trials=10):
    """``convmmd_exp_i2_init.greedy_kmeanspp`` (numpy), draw for draw."""

    def total(d2, i):
        return float(np.minimum(d2, ((x - x[i]) ** 2).sum(axis=1)).sum())

    d2 = np.full(len(x), np.inf)
    centres = []
    for j in range(k):
        cand = rng.integers(len(x), size=trials) if j == 0 else rng.choice(len(x), size=trials, p=d2 / d2.sum())
        best = min(cand, key=lambda i: total(d2, i))
        centres.append(x[best])
        d2 = np.minimum(d2, ((x - x[best]) ** 2).sum(axis=1))
    return np.asarray(centres)


@pytest.mark.slow
def test_recipe_reproduces_j1s_kept_repair_on_an_outlier_fit(monkeypatch):
    """j1_outliers_d5_n500_r10_kplus1_floor: from J1's own start the base fit misses
    the truth (b = 19.8, kappa = 0.021); H fires, the birth move at row 322 lowers
    the loss by 7.1e-4 with paired z = 5.6 and kappa 0.181, Kkz keeps it, and the
    refit recovers the truth."""

    truth, x, noise = _j1_outlier_dataset(5, 500, 10)
    centres = _j1_greedy(np.random.default_rng([J1_SEED_BASE, _OUTLIERS_INDEX, 5, 500, 10, 21]), x, 4)
    monkeypatch.setattr(recipe, "greedy_kmeanspp", lambda key, x_, k, trials=10: jnp.asarray(centres))
    result = fit_convmmd_recipe(jnp.asarray(x), jnp.asarray(noise), 3)

    assert float(result.covariance_floor) == pytest.approx(0.003747621812106809, rel=1e-13)
    base = result.base
    assert int(base.n_iter) == 200 and not _recovered(base.parameters, truth)[0]
    assert float(base.loss) == pytest.approx(-0.5162381088694032, abs=1e-12)
    assert float(base.min_eigenvalue) == pytest.approx(5.711997456756129e-08, rel=1e-4)

    first = result.repairs[0]
    assert bool(first.fired_H) and not bool(first.fired_F) and not bool(first.fired_Z)
    assert int(first.witness.raw_row) == 322
    assert float(first.witness.raw_peak) == pytest.approx(0.0031699691796650232, rel=1e-9)
    assert float(first.health.b) == pytest.approx(19.796261104274496, rel=1e-9)
    assert float(first.loss_drop) == pytest.approx(0.0007091632000523207, rel=1e-8)
    assert float(first.z) == pytest.approx(5.595574836383374, rel=1e-8)
    assert float(first.kappa_after) == pytest.approx(0.18055750414509936, rel=1e-8)
    assert bool(first.accepted) and bool(first.kept)
    (second,) = result.repairs[1:]
    assert not bool(second.attempted)

    assert float(result.loss) == pytest.approx(-0.5169472720694556, abs=1e-12)
    assert int(result.n_iter) == 200 and int(result.total_steps) == 400
    assert bool(result.converged) and not bool(result.collapsed)
    assert _recovered(result.parameters, truth)[0]
