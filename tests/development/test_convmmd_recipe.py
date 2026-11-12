"""Evidence for phase 1 of the J1 convMMD recipe (``development.convmmd_recipe``).

Covers the covariance floor as a loss ridge (``Sigma_k + f I`` inside the
objective), the collapse/health guard, the K + 1 greedy zero-noise-EM start, the
bandwidth subsample, and an end-to-end check against fits recovered from the J1
campaign (``results/convmmd_g/j_20261025/j1/j1_clean_*_kplus1_floor.json``, whose
witness chain attempted no move, so the full recipe is the base fit there).
Saddle-aware stopping and witness repair are covered in
``test_convmmd_recipe_phase2``. The J1 generator is ported below (data only) so
those datasets are rebuilt draw for draw; the reference numbers are embedded
because the recovered artifacts are not part of the repository.
"""

from __future__ import annotations

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.convmmd import (
    ConvMMDParams,
    ConvMMDUnconstrained,
    convmmd_loss_analytic,
    median_bandwidths,
    to_canonical,
)
from development.convmmd_fit import ConvMMDFitStatus, fit_analytic_state
from development.convmmd_recipe import (
    BANDWIDTH_STREAM,
    effective,
    fit_convmmd_recipe,
    floor_cov,
    floor_value,
    floored_loss,
    gmm_start,
    greedy_kmeanspp,
    health,
    kplus1_start,
    subsample_bandwidths,
    to_floor_unc,
    to_unconstrained,
    _nearest_centre_moments,
)
from development.fit_control import FitStatus


# ---------------------------------------------------------------------------
# J1 data generator (port of convmmd_g_common.truth_d / make_data and
# convmmd_exp_i1_init.random_cov; clean situation, Gaussian noise U[0.05, 0.2])
# ---------------------------------------------------------------------------

J1_SEED_BASE = 2026102601
J1_CLEAN_INDEX = 1  # 1 + SITUATIONS.index("clean")


def _random_cov(rng, d, lo, hi):
    rotation, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return (rotation * rng.uniform(lo, hi, size=d)) @ rotation.T


def _truth_d(rng, k, dim):
    a = 3.0 * np.sqrt(2.0 / dim) * max(1.0, (k / 3.0) ** (1.0 / dim))
    while True:
        means = rng.uniform(-a, a, size=(k, dim))
        gaps = np.linalg.norm(means[:, None] - means[None], axis=-1)[np.triu_indices(k, 1)]
        if k == 1 or gaps.min() >= 3.0:
            break
    covs = np.stack([_random_cov(rng, dim, 0.15, 0.5) for _ in range(k)])
    floor = min(0.15, 0.5 / k)
    while True:
        weights = rng.dirichlet(np.full(k, 4.0))
        if weights.min() >= floor:
            break
    return np.asarray(weights, float), means, covs


def _make_data(rng, truth, n):
    weights, means, covs = truth
    labels = rng.choice(len(weights), size=n, p=weights)
    latent = means[labels] + np.einsum(
        "nij,nj->ni", np.linalg.cholesky(covs)[labels], rng.standard_normal((n, means.shape[1]))
    )
    dim = latent.shape[1]
    variances = rng.uniform(0.05, 0.2, size=(n, dim))
    x = latent + np.sqrt(variances) * rng.standard_normal((n, dim))
    noise = np.zeros((n, dim, dim))
    noise[:, np.arange(dim), np.arange(dim)] = variances
    return x, noise


def _j1_clean_dataset(dim, n, dataset, k=3):
    truth = _truth_d(np.random.default_rng([J1_SEED_BASE, dim, dataset]), k, dim)
    rng = np.random.default_rng([J1_SEED_BASE, J1_CLEAN_INDEX, dim, n, dataset, 1])
    x, noise = _make_data(rng, truth, n)
    return truth, x, noise


def _recovered(params, truth):
    """J1's recovery rule (``convmmd_exp_i1_init.recovery``): best permutation,
    Mahalanobis mean error <= 1 and weight ratio within 50% for every component."""

    weights, means, _ = (np.asarray(a) for a in params)
    t_weights, t_means, t_covs = truth
    k = len(t_weights)
    best = None
    for perm in itertools.permutations(range(len(weights)), k):
        maha = [
            float(np.sqrt((means[perm[j]] - t_means[j]) @ np.linalg.solve(t_covs[j], means[perm[j]] - t_means[j])))
            for j in range(k)
        ]
        score = sum(e * e for e in maha)
        if best is None or score < best[0]:
            best = (score, perm, maha)
    _, perm, maha = best
    ratios = [weights[perm[j]] / t_weights[j] for j in range(k)]
    return all(m <= 1.0 for m in maha) and all(abs(r - 1.0) <= 0.5 for r in ratios), perm


def _match(params, reference):
    """Permutation of ``params`` components closest (by means) to ``reference``."""

    means = np.asarray(params.means)
    ref_means = np.asarray(reference[1])
    k = len(ref_means)
    return min(
        itertools.permutations(range(k)),
        key=lambda p: sum(np.sum((means[p[j]] - ref_means[j]) ** 2) for j in range(k)),
    )


def _small_problem(n=120, seed=11):
    rng = np.random.default_rng(seed)
    truth = _truth_d(rng, 3, 2)
    x, noise = _make_data(rng, truth, n)
    return truth, jnp.asarray(x), jnp.asarray(noise), median_bandwidths(jnp.asarray(x))


def _params(truth):
    return ConvMMDParams(*(jnp.asarray(a) for a in truth))


# ---------------------------------------------------------------------------
# Covariance floor as a loss ridge
# ---------------------------------------------------------------------------


def test_floor_value_is_three_percent_of_the_median_noise_variance():
    _, _, noise, _ = _small_problem()
    expected = 0.03 * np.median(np.diagonal(np.asarray(noise), axis1=1, axis2=2))
    assert float(floor_value(noise)) == pytest.approx(expected, rel=1e-15)
    assert float(floor_value(noise, floor_fraction=0.5)) == pytest.approx(
        expected / 0.03 * 0.5, rel=1e-15
    )


def test_effective_and_to_floor_unc_round_trip():
    truth, _, noise, _ = _small_problem()
    params = _params(truth)
    f = floor_value(noise)
    back = effective(to_canonical(to_floor_unc(params, f)), f)
    for got, want in zip(back, params):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0, atol=1e-12)
    eye = jnp.eye(2)
    np.testing.assert_array_equal(
        np.asarray(effective(params, f).covariances),
        np.asarray(params.covariances + f * eye),
    )


def test_to_unconstrained_inverts_to_canonical():
    rng = np.random.default_rng(4)
    unc = ConvMMDUnconstrained(
        alphas=jnp.asarray(rng.normal(size=3)),
        means=jnp.asarray(rng.normal(size=(3, 2))),
        unconstrained_L=jnp.asarray(np.tril(rng.normal(size=(3, 2, 2)))),
    )
    canonical = to_canonical(unc)
    again = to_canonical(to_unconstrained(canonical))
    for got, want in zip(again, canonical):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(to_unconstrained(canonical).unconstrained_L),
        np.asarray(unc.unconstrained_L),
        rtol=0,
        atol=1e-9,
    )


def test_floored_loss_is_the_plain_loss_on_ridged_covariances():
    truth, x, noise, bandwidths = _small_problem()
    params = _params(truth)
    f = floor_value(noise)
    ridged = ConvMMDParams(params.weights, params.means, params.covariances + f * jnp.eye(2))
    plain_on_ridged = convmmd_loss_analytic(ridged, x, noise, bandwidths)
    np.testing.assert_allclose(
        float(floored_loss(params, x, noise, bandwidths, f)),
        float(plain_on_ridged),
        rtol=1e-14,
        atol=1e-14,
    )
    assert float(floored_loss(params, x, noise, bandwidths, f)) != float(
        convmmd_loss_analytic(params, x, noise, bandwidths)
    )


def test_fit_with_floor_optimizes_and_reports_the_ridged_objective():
    truth, x, noise, bandwidths = _small_problem()
    f = floor_value(noise)
    init = to_floor_unc(_params(truth), f)
    state = fit_analytic_state(
        init, x, noise, bandwidths, n_steps=30, learning_rate=3e-2,
        weight_decay=0.0, schedule="constant", covariance_floor=f,
    )
    # Reported parameters are the floored mixture and the loss recomputes there.
    for got, want in zip(state.parameters, effective(to_canonical(state.unconstrained), f)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_allclose(
        float(state.loss),
        float(convmmd_loss_analytic(state.parameters, x, noise, bandwidths)),
        rtol=1e-13,
        atol=1e-14,
    )
    # The optimized objective (and the certificate's gradient) is the ridged one.
    grads = jax.grad(
        lambda u: floored_loss(to_canonical(u), x, noise, bandwidths, f)
    )(state.unconstrained)
    expected = max(float(jnp.max(jnp.abs(g))) for g in jax.tree_util.tree_leaves(grads))
    np.testing.assert_allclose(float(state.max_abs_grad), expected, rtol=1e-12)
    # The history's first entry is the floored loss at the start.
    np.testing.assert_allclose(
        float(state.history[0]),
        float(floored_loss(to_canonical(init), x, noise, bandwidths, f)),
        rtol=1e-13,
    )


def test_zero_floor_array_reproduces_the_unfloored_fit():
    truth, x, noise, bandwidths = _small_problem()
    init = to_unconstrained(_params(truth))
    plain = fit_analytic_state(init, x, noise, bandwidths, n_steps=10)
    zero = fit_analytic_state(
        init, x, noise, bandwidths, n_steps=10, covariance_floor=jnp.asarray(0.0)
    )
    np.testing.assert_allclose(np.asarray(zero.history), np.asarray(plain.history), rtol=1e-15)


# ---------------------------------------------------------------------------
# Health / collapse guard
# ---------------------------------------------------------------------------


def _reference_health(covariances):
    """``convmmd_h1_repair.health``, in numpy."""

    ev = [np.maximum(np.linalg.eigvalsh(np.asarray(c)), 1e-300) for c in covariances]
    spread = np.array([np.exp(np.mean(np.log(e))) for e in ev])
    return spread.max() / np.median(spread), min(e.min() / e.max() for e in ev)


# J1 reference fit j1_clean_d2_n500_r2_kplus1_floor (trigger none): its recorded
# floored covariances and health.
_J1_R2_COVS = [
    [[0.31464261988814596, -0.042085882736489486], [-0.042085882736489486, 0.2728723308995575]],
    [[0.23657786603842237, -0.07684688127309904], [-0.07684688127309904, 0.4097100752563988]],
    [[0.28906067109398553, 0.07836837818100906], [0.07836837818100906, 0.1678651711362323]],
]


def test_health_reproduces_the_recorded_j1_values():
    params = ConvMMDParams(jnp.full((3,), 1 / 3), jnp.zeros((3, 2)), jnp.asarray(_J1_R2_COVS))
    report = health(params)
    assert float(report.b) == pytest.approx(1.0404311734149156, rel=1e-12)
    assert float(report.kappa) == pytest.approx(0.3950784343581632, rel=1e-12)
    assert not bool(report.collapsed)


def test_health_flags_a_planted_degenerate_component_not_a_healthy_mixture():
    truth, _, _, _ = _small_problem()
    healthy = _params(truth)
    report = health(healthy)
    b, kappa = _reference_health(truth[2])
    assert float(report.b) == pytest.approx(b, rel=1e-12)
    assert float(report.kappa) == pytest.approx(kappa, rel=1e-12)
    assert not bool(report.collapsed)

    vectors = np.linalg.qr(np.random.default_rng(2).normal(size=(2, 2)))[0]
    needle = (vectors * np.array([1.0e-4, 0.3])) @ vectors.T  # kappa ~ 3e-4
    degenerate = healthy._replace(covariances=healthy.covariances.at[1].set(needle))
    report = health(degenerate)
    assert bool(report.ill_conditioned)
    assert bool(report.collapsed)
    assert float(report.kappa) < 0.03

    # Truth eigenvalues lie in [0.15, 0.5], so a 30x component has b >= 9.
    broad = healthy._replace(covariances=healthy.covariances.at[0].multiply(30.0))
    report = health(broad)
    assert bool(report.broad) and bool(report.collapsed)
    assert not bool(report.ill_conditioned)


# ---------------------------------------------------------------------------
# K + 1 greedy start
# ---------------------------------------------------------------------------


def _blobs(seed=0, n_per=60):
    rng = np.random.default_rng(seed)
    centres = np.array([[-6.0, 0.0], [0.0, 6.0], [6.0, 0.0]])
    x = np.concatenate([c + 0.3 * rng.standard_normal((n_per, 2)) for c in centres])
    return jnp.asarray(x), centres


def test_greedy_kmeanspp_picks_one_data_row_per_separated_cluster():
    x, centres = _blobs()
    chosen = greedy_kmeanspp(jax.random.PRNGKey(5), x, 3)
    assert chosen.shape == (3, 2)
    rows = np.asarray(x)
    for c in np.asarray(chosen):
        assert np.any(np.all(rows == c, axis=1)), "centres must be data rows"
    nearest = {int(np.argmin(np.sum((centres - c) ** 2, axis=1))) for c in np.asarray(chosen)}
    assert nearest == {0, 1, 2}
    again = greedy_kmeanspp(jax.random.PRNGKey(5), x, 3)
    np.testing.assert_array_equal(np.asarray(again), np.asarray(chosen))


def _reference_moments(x, centres, floor):
    """``convmmd_exp_i1_init.moments(..., deflate=False)``, in numpy."""

    x = np.asarray(x)
    d = x.shape[1]
    k = len(centres)
    labels = np.argmin(((x[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
    weights, means, covs = np.zeros(k), np.zeros((k, d)), np.zeros((k, d, d))
    for j in range(k):
        rows = labels == j
        if rows.sum() > d:
            means[j] = x[rows].mean(axis=0)
            c = np.atleast_2d(np.cov(x[rows].T))
            weights[j] = rows.sum()
        else:
            means[j] = centres[j]
            c = np.eye(d) * np.mean(np.var(x, axis=0))
            weights[j] = max(rows.sum(), 1)
        values, vectors = np.linalg.eigh(0.5 * (c + c.T))
        covs[j] = (vectors * np.maximum(values, floor)) @ vectors.T
    return weights / weights.sum(), means, covs


def test_nearest_centre_moments_follow_the_reference_rule():
    x, _ = _blobs()
    # The fourth centre is far away: it owns no rows and takes the fallback.
    centres = np.array([[-6.0, 0.2], [0.1, 6.0], [6.0, -0.1], [40.0, 40.0]])
    floor = 0.01 * float(np.mean(np.var(np.asarray(x), axis=0)))
    got = _nearest_centre_moments(x, jnp.asarray(centres), floor)
    for a, b in zip(got, _reference_moments(x, centres, floor)):
        np.testing.assert_allclose(np.asarray(a), b, rtol=1e-12, atol=1e-13)


def test_floor_cov_floors_every_eigenvalue():
    covs = jnp.asarray([[[1.0, 0.0], [0.0, -0.5]], [[0.2, 0.1], [0.1, 0.2]]])
    floored = floor_cov(covs, 0.15)
    values = np.linalg.eigvalsh(np.asarray(floored))
    assert np.all(values >= 0.15 - 1e-14)
    np.testing.assert_allclose(values[0], [0.15, 1.0], atol=1e-14)
    np.testing.assert_allclose(values[1], [0.15, 0.3], atol=1e-14)


def test_gmm_start_runs_zero_noise_em_then_deflates_and_floors():
    truth, x, noise, _ = _small_problem()
    centres = greedy_kmeanspp(jax.random.PRNGKey(1), x, 3)
    start, status = gmm_start(x, noise, centres)
    assert int(status) == int(FitStatus.FIXED_STEPS_COMPLETE)
    np.testing.assert_allclose(float(jnp.sum(start.weights)), 1.0, rtol=1e-14)
    floor = 0.01 * float(np.mean(np.var(np.asarray(x), axis=0)))
    assert np.all(np.linalg.eigvalsh(np.asarray(start.covariances)) >= floor - 1e-12)
    zero_steps, _ = gmm_start(x, noise, centres, em_steps=0)
    raw = _reference_moments(x, np.asarray(centres), floor)
    mean_noise = np.asarray(noise).mean(axis=0)
    expected = floor_cov(jnp.asarray(raw[2] - mean_noise), floor)
    np.testing.assert_allclose(np.asarray(zero_steps.covariances), np.asarray(expected), rtol=1e-12, atol=1e-13)


def test_kplus1_start_drops_the_smallest_weight_component():
    _, x, noise, _ = _small_problem()
    key = jax.random.PRNGKey(9)
    start, status = kplus1_start(key, x, noise, 3)
    full, full_status = gmm_start(x, noise, greedy_kmeanspp(key, x, 4))
    keep = np.arange(4) != int(np.argmin(np.asarray(full.weights)))
    weights = np.asarray(full.weights)[keep]
    np.testing.assert_allclose(np.asarray(start.weights), weights / weights.sum(), rtol=1e-14)
    np.testing.assert_array_equal(np.asarray(start.means), np.asarray(full.means)[keep])
    np.testing.assert_array_equal(np.asarray(start.covariances), np.asarray(full.covariances)[keep])
    assert int(status) == int(full_status)


# ---------------------------------------------------------------------------
# Recipe status plumbing (cheap)
# ---------------------------------------------------------------------------


def test_recipe_health_is_a_flag_that_never_changes_the_status():
    _, x, noise, _ = _small_problem()
    # The base fit alone (no witness chain, whose H trigger reads the flag).
    healthy = fit_convmmd_recipe(x, noise, 3, key=jax.random.PRNGKey(0), witness_moves=0)
    assert bool(healthy.converged)
    assert float(healthy.max_abs_grad) <= float(healthy.grad_tol)
    assert int(healthy.n_iter) < 3000 and int(healthy.n_iter) % 200 == 0  # a certified chunk boundary
    assert not bool(healthy.collapsed)
    assert int(healthy.status) == int(ConvMMDFitStatus.CONVERGED)

    # A threshold no mixture passes: the flag fires, while the status, the
    # certificate and the trajectory are untouched.
    flagged = fit_convmmd_recipe(
        x, noise, 3, key=jax.random.PRNGKey(0), health_collapsed=0.999, witness_moves=0
    )
    assert bool(flagged.collapsed) and bool(flagged.health.ill_conditioned)
    assert int(flagged.status) == int(ConvMMDFitStatus.CONVERGED)
    assert bool(flagged.converged)
    np.testing.assert_array_equal(np.asarray(flagged.history), np.asarray(healthy.history))


def test_recipe_flags_an_unfinished_degenerate_looking_fit():
    """Fifty steps in, this fit still has a needle-like component (kappa ~ 0.007):
    the flag reports it, and the status stays the honest MAX_ITER."""

    _, x, noise, _ = _small_problem()
    early = fit_convmmd_recipe(x, noise, 3, key=jax.random.PRNGKey(0), n_steps=50, witness_moves=0)
    assert not bool(early.converged)
    assert bool(early.collapsed) and float(early.health.kappa) < 0.03
    assert int(early.status) == int(ConvMMDFitStatus.MAX_ITER)


def test_component_collapsed_status_is_reserved():
    assert int(ConvMMDFitStatus.COMPONENT_COLLAPSED) == int(FitStatus.COMPONENT_COLLAPSED) == 5


def test_recipe_returns_the_floored_mixture_and_its_inputs():
    _, x, noise, _ = _small_problem()
    result = fit_convmmd_recipe(x, noise, 3, key=jax.random.PRNGKey(0), n_steps=20, witness_moves=0)
    f = floor_value(noise)
    assert float(result.covariance_floor) == pytest.approx(float(f), rel=1e-15)
    np.testing.assert_allclose(np.asarray(result.bandwidths), np.asarray(median_bandwidths(x)), rtol=1e-15)
    for got, want in zip(result.parameters, effective(to_canonical(result.unconstrained), f)):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        float(result.loss),
        float(convmmd_loss_analytic(result.parameters, x, noise, result.bandwidths)),
        rtol=1e-13,
        atol=1e-14,
    )
    # The optimizer starts at the K + 1 start, read through the floor.
    start, _ = kplus1_start(jax.random.PRNGKey(0), x, noise, 3)
    np.testing.assert_allclose(
        float(result.history[0]),
        float(convmmd_loss_analytic(start, x, noise, result.bandwidths)),
        rtol=1e-10,
    )
    assert result.metadata.contract_id == "xdgmm-jax.convmmd"


def test_subsample_bandwidths_use_every_row_up_to_the_limit():
    _, x, _, _ = _small_problem(n=120)
    for rows in (120, 2000):
        np.testing.assert_array_equal(
            np.asarray(subsample_bandwidths(jax.random.PRNGKey(5), x, rows=rows)),
            np.asarray(median_bandwidths(x)),
        )


def test_subsample_bandwidths_draw_distinct_rows_above_the_limit():
    _, x, _, _ = _small_problem(n=120)
    key = jax.random.PRNGKey(5)
    chosen = np.sort(np.asarray(jax.random.choice(key, 120, shape=(50,), replace=False)))
    assert len(np.unique(chosen)) == 50
    got = np.asarray(subsample_bandwidths(key, x, rows=50))
    np.testing.assert_array_equal(got, np.asarray(median_bandwidths(x[chosen])))
    assert not np.array_equal(got, np.asarray(median_bandwidths(x)))
    np.testing.assert_array_equal(got, np.asarray(subsample_bandwidths(key, x, rows=50)))


def test_recipe_default_bandwidths_subsample_with_a_folded_key():
    _, x, noise, _ = _small_problem(n=120)
    key = jax.random.PRNGKey(3)
    result = fit_convmmd_recipe(x, noise, 3, key=key, n_steps=0, bandwidth_rows=50, witness_moves=0)
    want = subsample_bandwidths(jax.random.fold_in(key, BANDWIDTH_STREAM), x, rows=50)
    np.testing.assert_array_equal(np.asarray(result.bandwidths), np.asarray(want))
    # The subsample key is separate from the start's: the start is unchanged.
    full = fit_convmmd_recipe(x, noise, 3, key=key, n_steps=0, witness_moves=0)
    for got, ref in zip(result.initial_parameters, full.initial_parameters):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


# ---------------------------------------------------------------------------
# End to end against the recovered J1 reference fits
# ---------------------------------------------------------------------------

# (dim, n, dataset) -> recorded floor_value, base loss (floored), and the
# recorded floored mixture of j1_clean_d{dim}_n{n}_r{dataset}_kplus1_floor.
_J1_REFERENCE = {
    (2, 500, 2): {
        "floor": 0.0038183521260068917,
        "loss": -0.5370138921305024,
        "weights": [0.5005891703441657, 0.25955071631731497, 0.2398601133385194],
        "means": [[2.095417447103977, 0.31599161136607934], [-1.2509496625381462, 0.9027334618843288],
                  [-1.4286209685848597, -3.0356014429764238]],
        "covariances": _J1_R2_COVS,
    },
    (2, 500, 0): {
        "floor": 0.00377727972283347,
        "loss": -0.5418274654762538,
        "weights": [0.19185655590224804, 0.28194501980585995, 0.526198424291892],
        "means": [[0.3526058390648743, -2.292959849788886], [-2.9246658303305835, -2.475420237321682],
                  [0.6000127096287035, 1.9894285240976122]],
        "covariances": [
            [[0.2563819884929843, -0.060243829781918654], [-0.060243829781918654, 0.10279201370523687]],
            [[0.2571532622096938, 0.11595623066809617], [0.11595623066809617, 0.21963859578878092]],
            [[0.328070327505112, 0.03493706862594865], [0.03493706862594865, 0.3215742886017253]],
        ],
    },
}


@pytest.mark.parametrize("cell", sorted(_J1_REFERENCE))
def test_recipe_recovers_clean_truth_and_matches_the_j1_reference_fit(cell):
    dim, n, dataset = cell
    reference = _J1_REFERENCE[cell]
    truth, x, noise = _j1_clean_dataset(dim, n, dataset)
    x, noise = jnp.asarray(x), jnp.asarray(noise)
    # The rebuilt dataset is the recorded one (its floor is a function of every
    # noise draw, which follow every latent draw in the stream).
    assert float(floor_value(noise)) == pytest.approx(reference["floor"], rel=1e-13)

    result = fit_convmmd_recipe(x, noise, 3, key=jax.random.PRNGKey(dataset))

    assert not bool(result.numerical_failure)
    assert bool(result.converged), f"max|grad| = {float(result.max_abs_grad):.3e}"
    assert float(result.max_abs_grad) <= 1.0e-6
    assert int(result.status) == int(ConvMMDFitStatus.CONVERGED)
    assert not bool(result.collapsed)
    assert int(result.initialization_status) == int(FitStatus.FIXED_STEPS_COMPLETE)

    ok, _ = _recovered(result.parameters, truth)
    assert ok, "the recipe did not recover the clean truth"

    # Same certified minimum as the recovered J1 fit (which used a numpy-seeded
    # start): at a certified point the loss gap is O(max|grad|^2 / lambda_min).
    # No witness move fires on these fits (see test_convmmd_recipe_phase2).
    np.testing.assert_allclose(float(result.loss), reference["loss"], rtol=0, atol=1e-8)
    perm = _match(result.parameters, (reference["weights"], reference["means"]))
    got = [np.asarray(a)[list(perm)] for a in result.parameters]
    # Observed: |dloss| ~ 1e-10, parameters within ~1e-4.
    np.testing.assert_allclose(got[0], reference["weights"], atol=1e-3)
    np.testing.assert_allclose(got[1], reference["means"], atol=2e-3)
    np.testing.assert_allclose(got[2], reference["covariances"], atol=2e-3)
    report = health(result.parameters)
    b, kappa = _reference_health(reference["covariances"])
    assert float(report.b) == pytest.approx(b, abs=5e-3)
    assert float(report.kappa) == pytest.approx(kappa, abs=5e-3)
