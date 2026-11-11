# SPDX-License-Identifier: MIT
# Provenance: convMMD is the maintainer's own method (Vashistha, Sarkar, Farahi,
# arXiv:2606.21907). The recipe is ported from the maintainer's own J1 benchmark
# harness (unpublished; the file:line references below point into it), not from
# astroML or Bovy XD code.
"""Pure-JAX convMMD fit recipe of J1.

The convMMD robustness campaign adopted the fit recipe ``kplus1+floor:HU Kkz``
(J1, see ``docs/convmmd-recipe-crosscheck.md``). This module lands it:

* **K + 1 greedy start** -- greedy k-means++ seeds ``K + 1`` centres, the
  nearest-centre moments are polished by 20 zero-noise XD EM steps (the
  package's :func:`development.fit_control.fit_fixed_steps_kernel`), the mean
  noise is subtracted and each covariance eigenvalue-floored, and the
  smallest-weight component is dropped (``convmmd_j1_guard.py:290-293``).
* **Covariance floor as a loss ridge** -- ``f = 0.03 x median(diag S_i)`` over
  every row and coordinate; the optimizer moves ``Sigma_k`` but the loss sees
  ``Sigma_k + f I`` (``convmmd_j1_guard.py:113-139``).
* **Saddle-aware constant-rate Adam** -- learning rate ``3e-2``, no weight
  decay, 200-step chunks (at most 3000 steps); ``converged`` iff ``max|grad| <=
  1e-6`` (findings B2, G2). At a certified chunk boundary the smallest Hessian
  eigenvalue on the identifiable parameters is checked (exact for ``P <= 80``,
  Lanczos ``m = 80`` above); a saddle is escaped by a line search along its
  eigenvector with Adam reset, at most 3 times (J1 section 5).
* **Witness repair, HU trigger, Kkz keep** -- while the health flag, the raw
  witness peak (``> 5e-3``) or its standardized leave-one-out peak (``t > 4``)
  fires, the lowest-weight component is moved to the peak row and the mixture
  refitted; a move is kept only if the loss falls, ``kappa`` does not and the
  paired z of the per-row loss changes is at least 3 (at most 2 moves; J1
  sections 3-4).
* **Collapse guard** -- the health check of ``convmmd_h1_repair.py:175-181``
  (``b`` = broadest geometric-mean eigenvalue over the median, ``kappa`` = worst
  component conditioning) is reported as the ``collapsed`` flag. It is a flag
  only and never changes ``status``: in the recovered J1 records it fired on
  39/40 hetero-broad and 31/40 hetero-thin fits that had recovered the truth,
  and J1 used it as a trigger to attempt a repair (H above), not as a verdict.

:func:`fit_convmmd_recipe` and its result types are re-exported by the public
``deconvgmm.convmmd`` facade (contract ``0.6.0-draft.1``, §18; every recipe row
of the capability matrix is Pending). It validates its inputs eagerly and runs
in float64 only. The remaining helpers are internal. The non-Gaussian noise
rules (G4/H3) are a separate workstream.
"""

from __future__ import annotations

import functools
import math
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .convmmd import (
    EPS_L,
    EPS_SIGMA,
    ConvMMDParams,
    ConvMMDUnconstrained,
    convmmd_loss_analytic,
    median_bandwidths,
    to_canonical,
)
from .convmmd_fit import (
    _ADAM_EPS,
    _BETA1,
    _BETA2,
    ConvMMDFitStatus,
    ConvMMDResultMetadata,
    _add_covariance_floor,
    _all_finite,
    _clip_by_global_norm,
    _max_abs,
    _tree_where,
    current_convmmd_metadata,
)
from .fit_control import fit_fixed_steps_kernel
from .identity_xd import Params
from .validation import (
    PrecisionError,
    ValidationError,
    _convert_array,
    _positive_count,
    _raise_shape,
    _shape,
    full_noise,
)


Array = jax.Array

# Recipe constants, quoted from the recovered J1 settings
# (results/convmmd_g/j_20261025/j1/*.json "settings").
FLOOR_FRACTION = 0.03  # ridge f = FLOOR_FRACTION x median noise variance
START_FLOOR_FRACTION = 0.01  # start eigenvalue floor = this x mean(var(x))
GREEDY_TRIALS = 10
GMM_STEPS = 20
LEARNING_RATE = 3.0e-2
WEIGHT_DECAY = 0.0
GRAD_TOL = 1.0e-6
MAX_STEPS = 3000
HEALTH_BROAD = 3.0  # b above this fires the health flag
HEALTH_COLLAPSED = 0.03  # kappa below this fires the health flag
BANDWIDTH_ROWS = 2000  # median heuristic on at most this many rows
BANDWIDTH_STREAM = 3  # fold_in stream of the subsample key (J1's stream 3)
# Saddle-aware stopping (settings "fit"; convmmd_l1_scaling.py "check").
CHUNK = 200  # Adam steps between certificate checks
EXACT_LIMIT = 80  # exact curvature check for P <= this, Lanczos above
LANCZOS_STEPS = 80
EXACT_TOLERANCE = 1.0e-10  # a minimum if lambda_min >= -this (exact check)
LANCZOS_TOLERANCE = 1.0e-6  # ... (Lanczos check)
ESCAPE_STEPS = (0.01, 0.03, 0.1, 0.3, 1.0)  # line search at +/- these along the eigenvector
MAX_ESCAPES = 3
LANCZOS_STREAM = 7  # fold_in stream of the Lanczos start vectors
# Witness repair (HU trigger, Kkz keep).
WITNESS_MOVES = 2
RAW_TRIGGER = 5.0e-3  # F fires when the raw witness peak exceeds this
Z_TRIGGER = 4.0  # Z fires when the leave-one-out peak is this many SEs above zero
Z_MIN = 3.0  # Kkz keeps a move only if its paired z is at least this
KEEP_DROP = 1.0e-9  # K0 accepts a move only if the loss falls by more than this
WITNESS_CANDIDATES = 2000  # candidate rows for the witness peak
WITNESS_STREAM = 99  # fold_in stream of the candidate subsample
BIRTH_MIN_WEIGHT = 0.05
ROW_CHUNK = 2000  # rows per per-row-loss evaluation
_WITNESS_BLOCK_ENTRIES = 25_000_000  # (N x candidates) entries per witness block


class ConvMMDHealth(NamedTuple):
    """Post-fit health of a mixture (``convmmd_h1_repair.py:175-181``).

    ``b`` is the largest per-component geometric-mean eigenvalue divided by the
    median one; ``kappa`` is ``min_k lambda_min / lambda_max``. ``collapsed`` is
    ``broad | ill_conditioned`` (J1's ``H`` trigger).
    """

    b: Array
    kappa: Array
    broad: Array
    ill_conditioned: Array
    collapsed: Array


class ConvMMDRecipeResult(NamedTuple):
    """Host result of :func:`fit_convmmd_recipe`.

    The leading fields describe the returned fit: the last witness refit kept,
    or the base fit if none was. ``parameters`` is the mixture the model sees
    (covariances ``Sigma_k + f I``); ``unconstrained`` are the optimizer's
    coordinates, whose ``to_canonical`` covariances are the un-floored
    ``Sigma_k``. ``loss`` and ``max_abs_grad`` are the floored objective and its
    gradient at the returned point; ``converged`` is the certificate
    ``max_abs_grad <= grad_tol``; ``history`` and ``n_iter`` are that fit's.
    ``status`` is its :class:`~development.convmmd_fit.ConvMMDFitStatus` value
    (``NUMERICAL_FAILURE``, ``CONVERGED`` or ``MAX_ITER``). ``collapsed`` and
    ``health`` (the post-fit health check) and ``saddle`` (certified at negative
    curvature) are flags; they never change ``status`` or ``converged``.
    ``base`` is the base fit, ``repairs`` the witness-chain moves examined in
    order, and ``total_steps`` the Adam updates over the base fit and every
    attempted refit.
    """

    parameters: ConvMMDParams
    unconstrained: ConvMMDUnconstrained
    loss: Array
    history: Array
    n_iter: Array
    status: Array
    converged: Array
    numerical_failure: Array
    max_abs_grad: Array
    grad_tol: Array
    collapsed: Array
    health: ConvMMDHealth
    covariance_floor: Array
    bandwidths: Array
    initial_parameters: ConvMMDParams
    initialization_status: Array  # FitStatus of the zero-noise EM warm-up
    metadata: ConvMMDResultMetadata
    saddle: Array
    escapes: Array
    min_eigenvalue: Array  # at the returned fit's last curvature check
    base: "ConvMMDSaddleFit"
    repairs: tuple
    total_steps: Array


# ---------------------------------------------------------------------------
# Bandwidths
# ---------------------------------------------------------------------------


def subsample_bandwidths(
    key: Array, observations: Array, *, rows: int = BANDWIDTH_ROWS
) -> Array:
    """Median-heuristic bandwidths on ``min(N, rows)`` rows (``convmmd_g_common.py:159-162``).

    With ``N <= rows`` this is :func:`~development.convmmd.median_bandwidths` on
    every row. Otherwise ``rows`` distinct rows are drawn with ``key``, which
    keeps the pairwise-distance array at ``rows^2 x D`` instead of ``N^2 x D``
    (about 4 GB at ``N = 10,000, D = 5``). On the recovered J1 datasets with
    ``N = 10,000`` the subsample median was within 2.5% of the full one.
    """

    x = jnp.asarray(observations)
    n = x.shape[0]
    if n <= rows:
        return median_bandwidths(x)
    chosen = jnp.sort(jax.random.choice(key, n, shape=(rows,), replace=False))
    return median_bandwidths(x[chosen])


# ---------------------------------------------------------------------------
# Covariance floor (J1 section 2)
# ---------------------------------------------------------------------------


def floor_value(
    measurement_covariances: Array, *, floor_fraction: float = FLOOR_FRACTION
) -> Array:
    """``f = floor_fraction x median`` of every row's noise-variance diagonal."""

    diagonals = jnp.diagonal(measurement_covariances, axis1=-2, axis2=-1)
    return floor_fraction * jnp.median(diagonals)


def effective(params: ConvMMDParams, floor) -> ConvMMDParams:
    """The fitted mixture as the model sees it: covariances ``Sigma_k + f I``."""

    return _add_covariance_floor(params, floor)


def floored_loss(
    params: ConvMMDParams,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    floor,
) -> Array:
    """The analytic convMMD loss evaluated on ``Sigma_k + f I``."""

    return convmmd_loss_analytic(
        effective(params, floor), observations, measurement_covariances, bandwidths
    )


def to_unconstrained(
    params: ConvMMDParams,
    *,
    eps_sigma: float = EPS_SIGMA,
    eps_l: float = EPS_L,
) -> ConvMMDUnconstrained:
    """Inverse of :func:`~development.convmmd.to_canonical`.

    Eigenvalues of ``Sigma_k - eps_sigma I`` are floored at ``(2 eps_l)^2`` so
    the Cholesky diagonal exceeds ``eps_l`` (as the recovered harness's
    ``to_unconstrained_np``); covariances already in the image of
    ``to_canonical`` round-trip to rounding error.
    """

    covariances = params.covariances
    dimension = covariances.shape[-1]
    eye = jnp.eye(dimension, dtype=covariances.dtype)
    target = covariances - eps_sigma * eye
    values, vectors = jnp.linalg.eigh(0.5 * (target + jnp.swapaxes(target, -1, -2)))
    values = jnp.maximum(values, (2.0 * eps_l) ** 2)
    target = jnp.einsum("kij,kj,klj->kil", vectors, values, vectors)
    factor = jnp.linalg.cholesky(target)
    diagonal = jnp.maximum(
        jnp.diagonal(factor, axis1=-2, axis2=-1) - eps_l, 1.0e-12
    )
    unconstrained_l = jnp.tril(factor, k=-1) + jnp.log(jnp.expm1(diagonal))[
        ..., None
    ] * eye
    return ConvMMDUnconstrained(
        alphas=jnp.log(params.weights),
        means=params.means,
        unconstrained_L=unconstrained_l,
    )


def to_floor_unc(params: ConvMMDParams, floor) -> ConvMMDUnconstrained:
    """Unconstrained parameters whose floored covariances equal ``params``'.

    ``effective(to_canonical(to_floor_unc(p, f)), f) == p`` whenever every
    eigenvalue of ``p.covariances`` exceeds ``f + EPS_SIGMA + (2 EPS_L)^2``.
    """

    return to_unconstrained(effective(params, -floor))


# ---------------------------------------------------------------------------
# Health / collapse guard (J1)
# ---------------------------------------------------------------------------


def health(
    params: ConvMMDParams,
    *,
    broad: float = HEALTH_BROAD,
    collapsed: float = HEALTH_COLLAPSED,
) -> ConvMMDHealth:
    """Broadness ``b`` and worst conditioning ``kappa`` of a mixture."""

    eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(params.covariances), 1.0e-300)
    spread = jnp.exp(jnp.mean(jnp.log(eigenvalues), axis=-1))
    b = jnp.max(spread) / jnp.median(spread)
    kappa = jnp.min(eigenvalues[:, 0] / eigenvalues[:, -1])
    is_broad = b > broad
    is_ill_conditioned = kappa < collapsed
    return ConvMMDHealth(
        b=b,
        kappa=kappa,
        broad=is_broad,
        ill_conditioned=is_ill_conditioned,
        collapsed=is_broad | is_ill_conditioned,
    )


# ---------------------------------------------------------------------------
# K + 1 greedy start (J1 section 1; EXP-I1/I2 starts)
# ---------------------------------------------------------------------------


def _squared_distances(x: Array, centres: Array) -> Array:
    difference = x[:, None, :] - centres[None, :, :]
    return jnp.sum(difference * difference, axis=-1)


def greedy_kmeanspp(
    key: Array, x: Array, k: int, trials: int = GREEDY_TRIALS
) -> Array:
    """Greedy k-means++ centres ``(k, D)`` (``convmmd_exp_i2_init.py:256-268``).

    Every centre, the first included, is the best of ``trials`` candidates by the
    reduction in total squared distance; the first candidates are uniform, the
    rest drawn by ``D^2`` (with replacement).
    """

    n = x.shape[0]
    d2 = jnp.full((n,), jnp.inf, dtype=x.dtype)
    centres = []
    for j in range(k):
        key, draw_key = jax.random.split(key)
        if j == 0:
            candidates = jax.random.randint(draw_key, (trials,), 0, n)
        else:
            candidates = jax.random.choice(
                draw_key, n, shape=(trials,), replace=True, p=d2 / jnp.sum(d2)
            )
        candidate_d2 = _squared_distances(x, x[candidates]).T  # (trials, N)
        totals = jnp.sum(jnp.minimum(d2[None, :], candidate_d2), axis=-1)
        best = candidates[jnp.argmin(totals)]
        centres.append(x[best])
        d2 = jnp.minimum(d2, jnp.sum((x - x[best]) ** 2, axis=-1))
    return jnp.stack(centres)


def floor_cov(covariances: Array, floor) -> Array:
    """Symmetrize and floor every eigenvalue at ``floor`` (``convmmd_exp_i1_init.py:204-206``)."""

    symmetric = 0.5 * (covariances + jnp.swapaxes(covariances, -1, -2))
    values, vectors = jnp.linalg.eigh(symmetric)
    return jnp.einsum(
        "...ij,...j,...lj->...il", vectors, jnp.maximum(values, floor), vectors
    )


def _nearest_centre_moments(x: Array, centres: Array, floor) -> ConvMMDParams:
    """Raw (not noise-deflated) cluster moments (``convmmd_exp_i1_init.py:209-226``).

    A cluster with more than ``D`` members gets its sample mean and covariance;
    otherwise its centre and ``mean(var(x)) I``. Weights are cluster sizes (at
    least one) normalized; every covariance is eigenvalue-floored.
    """

    n, dimension = x.shape
    k = centres.shape[0]
    labels = jnp.argmin(_squared_distances(x, centres), axis=1)
    members = (labels[None, :] == jnp.arange(k)[:, None]).astype(x.dtype)  # (K, N)
    counts = jnp.sum(members, axis=1)
    enough = counts > dimension
    sample_means = (members @ x) / jnp.maximum(counts, 1.0)[:, None]
    centred = x[None, :, :] - sample_means[:, None, :]  # (K, N, D)
    scatter = jnp.einsum("kn,kna,knb->kab", members, centred, centred)
    sample_covs = scatter / jnp.maximum(counts - 1.0, 1.0)[:, None, None]
    fallback_cov = jnp.mean(jnp.var(x, axis=0)) * jnp.eye(dimension, dtype=x.dtype)
    means = jnp.where(enough[:, None], sample_means, centres)
    covs = jnp.where(enough[:, None, None], sample_covs, fallback_cov[None])
    weights = jnp.where(enough, counts, jnp.maximum(counts, 1.0))
    return ConvMMDParams(
        weights=weights / jnp.sum(weights),
        means=means,
        covariances=floor_cov(covs, floor),
    )


def gmm_start(
    observations: Array,
    measurement_covariances: Array,
    centres: Array,
    *,
    em_steps: int = GMM_STEPS,
    start_floor=None,
) -> tuple[ConvMMDParams, Array]:
    """Zero-noise GMM start (``convmmd_exp_i2_init.py:290-294``).

    Nearest-centre moments (no deflation), ``em_steps`` XD EM steps with zero
    measurement noise, then the mean noise is subtracted from every covariance
    and each is eigenvalue-floored at ``start_floor`` (default ``0.01 x
    mean(var(x))``, distinct from the loss ridge). Returns the start and the
    EM warm-up's ``FitStatus`` value.
    """

    x = observations
    if start_floor is None:
        start_floor = START_FLOOR_FRACTION * jnp.mean(jnp.var(x, axis=0))
    raw = _nearest_centre_moments(x, centres, start_floor)
    em = fit_fixed_steps_kernel(
        Params(raw.weights, raw.means, raw.covariances),
        x,
        jnp.zeros_like(measurement_covariances),
        n_steps=em_steps,
    )
    mean_noise = jnp.mean(measurement_covariances, axis=0)
    start = ConvMMDParams(
        weights=em.parameters.weights,
        means=em.parameters.means,
        covariances=floor_cov(em.parameters.covariances - mean_noise, start_floor),
    )
    return start, em.status


def kplus1_start(
    key: Array,
    observations: Array,
    measurement_covariances: Array,
    k: int,
    *,
    trials: int = GREEDY_TRIALS,
    em_steps: int = GMM_STEPS,
    start_floor=None,
) -> tuple[ConvMMDParams, Array]:
    """K + 1 start (``convmmd_j1_guard.py:290-293``).

    Seed ``k + 1`` greedy k-means++ centres, run :func:`gmm_start`, drop the
    smallest-weight component and renormalize the rest.
    """

    centres = greedy_kmeanspp(key, observations, k + 1, trials)
    start, status = gmm_start(
        observations,
        measurement_covariances,
        centres,
        em_steps=em_steps,
        start_floor=start_floor,
    )
    drop = jnp.argmin(start.weights)
    keep = jnp.nonzero(jnp.arange(k + 1) != drop, size=k)[0]
    weights = start.weights[keep]
    return (
        ConvMMDParams(
            weights=weights / jnp.sum(weights),
            means=start.means[keep],
            covariances=start.covariances[keep],
        ),
        status,
    )


# ---------------------------------------------------------------------------
# Saddle-aware chunked fit (J1 section 5; convmmd_exp_d_xd.py:234-348)
# ---------------------------------------------------------------------------


class ConvMMDCurvature(NamedTuple):
    """Smallest Hessian eigenvalue of the floored loss on the identifiable parameters.

    ``vector`` is its unit eigenvector in flat unconstrained coordinates
    (``ravel_pytree(ConvMMDUnconstrained)`` order); ``mode`` is ``"exact"`` or
    ``"lanczos"``; ``n_hvps`` counts the Hessian-vector products used.
    """

    min_eigenvalue: Array
    vector: Array
    mode: str
    n_hvps: int


class ConvMMDSaddleFit(NamedTuple):
    """Host result of :func:`fit_saddle_aware`.

    ``parameters`` is the floored mixture; ``loss`` and ``max_abs_grad`` are
    recomputed at the returned point. ``status`` is a
    :class:`~development.convmmd_fit.ConvMMDFitStatus` value and ``converged``
    the certificate ``max_abs_grad <= grad_tol``. ``saddle`` marks a certified
    point whose smallest curvature stayed below the tolerance (no escapes left,
    or no line-search step lowered the loss); it is a flag and does not change
    ``status``. ``min_eigenvalue`` is from the last curvature check (NaN if
    none ran); ``history`` holds the pre-update loss of every Adam step;
    ``n_iter`` counts the updates applied.
    """

    parameters: ConvMMDParams
    unconstrained: ConvMMDUnconstrained
    loss: Array
    max_abs_grad: Array
    history: Array
    n_iter: Array
    status: Array
    converged: Array
    numerical_failure: Array
    saddle: Array
    escapes: Array
    min_eigenvalue: Array
    curvature_checks: Array
    grad_tol: Array


def identifiable_basis(k: int, dimension: int, *, dtype=jnp.float64) -> Array:
    """Orthonormal basis ``(p, P)`` of the parameters the loss depends on (``convmmd_exp_i2_init.py:114-127``).

    Columns live in flat unconstrained coordinates (``alphas``, then ``means``,
    then ``unconstrained_L``, each row-major). The basis drops the softmax's
    common shift of ``alphas`` (Helmert contrasts span the rest) and the unused
    strict upper triangle of ``unconstrained_L``, leaving ``P = (K - 1) + K D +
    K D (D + 1) / 2`` columns.
    """

    p = k + k * dimension + k * dimension * dimension
    helmert = []
    for j in range(1, k):
        norm = math.sqrt(j * (j + 1))
        helmert.append([1.0 / norm] * j + [-j / norm] + [0.0] * (k - j - 1))
    contrasts = jnp.zeros((p, k - 1), dtype=dtype)
    if k > 1:
        contrasts = contrasts.at[:k].set(jnp.asarray(helmert, dtype=dtype).T)
    used = list(range(k, k + k * dimension))
    offset = k + k * dimension
    for component in range(k):
        for row in range(dimension):
            for col in range(row + 1):
                used.append(offset + component * dimension * dimension + row * dimension + col)
    units = jnp.eye(p, dtype=dtype)[:, jnp.asarray(used)]
    return jnp.concatenate([contrasts, units], axis=1)


def _value_and_grad_fn(x, noise, bandwidths, floor):
    def objective(unc):
        return floored_loss(to_canonical(unc), x, noise, bandwidths, floor)

    return jax.value_and_grad(objective)


def _flat_objective(unc, x, noise, bandwidths, floor):
    flat, unravel = ravel_pytree(unc)

    def objective(vector):
        return floored_loss(to_canonical(unravel(vector)), x, noise, bandwidths, floor)

    return flat, unravel, objective


@jax.jit
def _value_and_max_grad(unc, x, noise, bandwidths, floor):
    loss, grads = _value_and_grad_fn(x, noise, bandwidths, floor)(unc)
    return loss, _max_abs(grads)


@functools.partial(jax.jit, static_argnames=("n_steps",))
def _adam_chunk(
    unc, moment1, moment2, step, x, noise, bandwidths, floor, learning_rate,
    weight_decay, clip_norm, *, n_steps
):
    """``n_steps`` constant-rate AdamW updates resumed from ``(moment1, moment2, step)``.

    The update is :func:`development.convmmd_fit._run`'s, step for step (same
    clip, moments, bias correction and honest rollback), so a fit split into
    chunks equals one ``schedule="constant"`` run of the same total length.
    Returns the returned iterate (the last finite one if a step failed), the
    moments and step count to resume from, the failure flag, the loss and
    ``max|grad|`` at the returned iterate, the per-step pre-update losses and the
    number of updates applied.
    """

    value_and_grad = _value_and_grad_fn(x, noise, bandwidths, floor)
    dtype = unc.means.dtype

    def body(carry, _):
        u, m, v, t, last_finite, failed, applied = carry
        loss, grads = value_and_grad(u)
        finite = jnp.isfinite(loss) & _all_finite(grads)
        live = finite & jnp.logical_not(failed)
        last_finite = _tree_where(live, u, last_finite)
        grads = _clip_by_global_norm(grads, clip_norm)
        t_next = t + 1
        m_new = jax.tree_util.tree_map(lambda a, g: _BETA1 * a + (1.0 - _BETA1) * g, m, grads)
        v_new = jax.tree_util.tree_map(lambda a, g: _BETA2 * a + (1.0 - _BETA2) * g * g, v, grads)
        bias1 = 1.0 - _BETA1 ** t_next.astype(dtype)
        bias2 = 1.0 - _BETA2 ** t_next.astype(dtype)

        def apply(param, a, b):
            m_hat = a / bias1
            v_hat = b / bias2
            return param - learning_rate * (
                m_hat / (jnp.sqrt(v_hat) + _ADAM_EPS) + weight_decay * param
            )

        u_new = jax.tree_util.tree_map(apply, u, m_new, v_new)
        carry = (
            _tree_where(live, u_new, u),
            _tree_where(live, m_new, m),
            _tree_where(live, v_new, v),
            jnp.where(live, t_next, t),
            last_finite,
            failed | jnp.logical_not(finite),
            applied + live.astype(applied.dtype),
        )
        return carry, loss

    carry0 = (unc, moment1, moment2, step, unc, jnp.asarray(False), jnp.asarray(0, jnp.int32))
    (u, m, v, t, last_finite, failed, applied), history = jax.lax.scan(
        body, carry0, None, length=n_steps
    )
    returned = _tree_where(failed, last_finite, u)
    loss, grads = value_and_grad(returned)
    return returned, m, v, t, failed, loss, _max_abs(grads), history, applied


@jax.jit
def _projected_hessian(unc, basis, x, noise, bandwidths, floor):
    flat, _, objective = _flat_objective(unc, x, noise, bandwidths, floor)
    gradient = jax.grad(objective)
    columns = jax.lax.map(lambda column: jax.jvp(gradient, (flat,), (column,))[1], basis.T)
    hessian = columns @ basis
    return 0.5 * (hessian + hessian.T)


@jax.jit
def _projected_hvp(unc, vector, basis, x, noise, bandwidths, floor):
    flat, _, objective = _flat_objective(unc, x, noise, bandwidths, floor)
    gradient = jax.grad(objective)
    return basis.T @ jax.jvp(gradient, (flat,), (basis @ vector,))[1]


def smallest_curvature(
    unconstrained: ConvMMDUnconstrained,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    floor,
    *,
    key: Array | None = None,
    exact_limit: int = EXACT_LIMIT,
    lanczos_steps: int = LANCZOS_STEPS,
) -> ConvMMDCurvature:
    """Smallest Hessian eigenvalue of the floored loss on :func:`identifiable_basis`.

    With ``P <= exact_limit`` the projected Hessian is formed from ``P``
    Hessian-vector products and diagonalized (``convmmd_exp_d_xd.py:271-277``).
    Above it, ``min(lanczos_steps, P)`` Lanczos iterations with full
    reorthogonalization from a ``key``-seeded start (``convmmd_g_common.py:431-471``);
    the smallest Ritz value never falls below the true eigenvalue, so an
    unconverged run can miss a saddle but never invent one.
    """

    x = jnp.asarray(observations)
    noise = jnp.asarray(measurement_covariances)
    bandwidths = jnp.asarray(bandwidths, dtype=x.dtype)
    floor = jnp.asarray(floor, dtype=x.dtype)
    k, dimension = unconstrained.means.shape
    basis = identifiable_basis(k, dimension, dtype=x.dtype)
    size = basis.shape[1]
    if size <= exact_limit:
        values, vectors = jnp.linalg.eigh(
            _projected_hessian(unconstrained, basis, x, noise, bandwidths, floor)
        )
        return ConvMMDCurvature(values[0], basis @ vectors[:, 0], "exact", size)

    if key is None:
        key = jax.random.PRNGKey(0)
    steps = min(lanczos_steps, size)
    start = jax.random.normal(key, (size,), dtype=x.dtype)
    lanczos = [start / jnp.linalg.norm(start)]
    alphas, betas = [], []
    for j in range(steps):
        w = _projected_hvp(unconstrained, lanczos[j], basis, x, noise, bandwidths, floor)
        alphas.append(jnp.dot(w, lanczos[j]))
        stacked = jnp.stack(lanczos)
        for _ in range(2):  # full reorthogonalization, twice
            w = w - stacked.T @ (stacked @ w)
        norm = jnp.linalg.norm(w)
        if j + 1 == steps or float(norm) < 1.0e-13:
            break
        betas.append(norm)
        lanczos.append(w / norm)
    tridiagonal = jnp.diag(jnp.stack(alphas))
    if betas:
        off = jnp.stack(betas)
        tridiagonal = tridiagonal + jnp.diag(off, 1) + jnp.diag(off, -1)
    values, vectors = jnp.linalg.eigh(tridiagonal)
    ritz = jnp.stack(lanczos).T @ vectors[:, 0]
    return ConvMMDCurvature(values[0], basis @ ritz, "lanczos", len(alphas))


@jax.jit
def _escape_line_search(unc, direction, steps, x, noise, bandwidths, floor):
    flat, unravel, objective = _flat_objective(unc, x, noise, bandwidths, floor)
    values = jax.vmap(lambda step: objective(flat + step * direction))(steps)
    values = jnp.where(jnp.isfinite(values), values, jnp.inf)
    best = jnp.argmin(values)
    return unravel(flat + steps[best] * direction), values[best]


def fit_saddle_aware(
    initial: ConvMMDUnconstrained,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    floor,
    *,
    key: Array | None = None,
    max_steps: int = MAX_STEPS,
    chunk: int = CHUNK,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    clip_norm: float = 1.0,
    grad_tol: float = GRAD_TOL,
    exact_limit: int = EXACT_LIMIT,
    lanczos_steps: int = LANCZOS_STEPS,
    exact_tolerance: float = EXACT_TOLERANCE,
    lanczos_tolerance: float = LANCZOS_TOLERANCE,
    escape_steps: tuple[float, ...] = ESCAPE_STEPS,
    max_escapes: int = MAX_ESCAPES,
) -> ConvMMDSaddleFit:
    """Host-chunked constant-rate Adam with saddle-aware stopping (``convmmd_exp_d_xd.py:291-348``).

    Adam runs in chunks of ``chunk`` steps (at most ``max_steps`` in all) on the
    floored loss. Before each chunk (and at the start), if ``max|grad| <=
    grad_tol`` the smallest curvature is checked (:func:`smallest_curvature`).
    At or above ``-exact_tolerance`` (``-lanczos_tolerance`` for a Lanczos
    check) the point is a minimum and the fit stops. Below it the fit tries an
    escape: the lowest loss over ``+/- escape_steps`` along the eigenvector; if
    that is below the current loss the fit moves there, resets Adam's moments and
    step count, and continues. After ``max_escapes`` escapes, or when no step
    lowers the loss, it stops at the saddle (``saddle=True``). A failed step
    rolls back to the last finite iterate and stops the fit
    (``NUMERICAL_FAILURE``). ``key`` seeds the Lanczos start vectors (folded with
    the check number).
    """

    x = jnp.asarray(observations)
    noise = jnp.asarray(measurement_covariances)
    dtype = x.dtype
    bandwidths = jnp.asarray(bandwidths, dtype=dtype)
    floor = jnp.asarray(floor, dtype=dtype)
    lr = jnp.asarray(learning_rate, dtype=dtype)
    wd = jnp.asarray(weight_decay, dtype=dtype)
    clip = jnp.asarray(clip_norm, dtype=dtype)
    if key is None:
        key = jax.random.PRNGKey(0)
    alphas = jnp.asarray([s * sign for s in escape_steps for sign in (1.0, -1.0)], dtype=dtype)
    args = (x, noise, bandwidths, floor)

    unc = initial
    zeros = jax.tree_util.tree_map(jnp.zeros_like, unc)
    moment1, moment2, step = zeros, zeros, jnp.asarray(0, jnp.int32)
    loss, max_grad = _value_and_max_grad(unc, *args)
    failed = not bool(jnp.isfinite(loss))
    steps = updates = escapes = checks = 0
    min_eigenvalue = jnp.asarray(jnp.nan, dtype=dtype)
    saddle = False
    histories = []
    while not failed:
        if float(max_grad) <= grad_tol:
            curvature = smallest_curvature(
                unc, *args, key=jax.random.fold_in(key, checks),
                exact_limit=exact_limit, lanczos_steps=lanczos_steps,
            )
            checks += 1
            min_eigenvalue = curvature.min_eigenvalue
            tolerance = exact_tolerance if curvature.mode == "exact" else lanczos_tolerance
            if float(min_eigenvalue) >= -tolerance:
                break
            if escapes >= max_escapes:
                saddle = True
                break
            moved, moved_loss = _escape_line_search(unc, curvature.vector, alphas, *args)
            if not float(moved_loss) < float(loss):
                saddle = True
                break
            unc, loss, max_grad = moved, moved_loss, jnp.inf
            moment1, moment2, step = zeros, zeros, jnp.asarray(0, jnp.int32)
            escapes += 1
        if steps >= max_steps:
            break
        n = min(chunk, max_steps - steps)
        unc, moment1, moment2, step, chunk_failed, loss, max_grad, history, applied = _adam_chunk(
            unc, moment1, moment2, step, *args, lr, wd, clip, n_steps=n
        )
        histories.append(history)
        steps += n
        updates += int(applied)
        failed = bool(chunk_failed)

    loss, max_grad = _value_and_max_grad(unc, *args)
    converged = (not failed) and float(max_grad) <= grad_tol
    if failed:
        status = ConvMMDFitStatus.NUMERICAL_FAILURE
    elif converged:
        status = ConvMMDFitStatus.CONVERGED
    else:
        status = ConvMMDFitStatus.MAX_ITER
    history = jnp.concatenate(histories) if histories else jnp.zeros((0,), dtype=dtype)
    return ConvMMDSaddleFit(
        parameters=effective(to_canonical(unc), floor),
        unconstrained=unc,
        loss=loss,
        max_abs_grad=max_grad,
        history=history,
        n_iter=jnp.asarray(updates),
        status=jnp.asarray(int(status)),
        converged=jnp.asarray(converged),
        numerical_failure=jnp.asarray(failed),
        saddle=jnp.asarray(saddle),
        escapes=jnp.asarray(escapes),
        min_eigenvalue=min_eigenvalue,
        curvature_checks=jnp.asarray(checks),
        grad_tol=jnp.asarray(grad_tol, dtype=dtype),
    )


# ---------------------------------------------------------------------------
# Witness repair (J1 sections 3-4; convmmd_g_common.py:276-361)
# ---------------------------------------------------------------------------


class ConvMMDWitness(NamedTuple):
    """Peak of the witness score over candidate rows (``convmmd_g_common.py:276-342``).

    ``raw_peak`` is the mean of ``h_i(z)`` over every row (the shipped
    statistic) at its best candidate ``raw_row``; ``loo_peak`` leaves the
    candidate's own row out, ``se`` is its standard error and ``t = loo_peak /
    se``; ``t_max`` is the largest ``loo / se`` over candidates. Rows are data
    row indices.
    """

    raw_peak: Array
    raw_row: Array
    own_term_at_raw_peak: Array
    loo_peak: Array
    loo_row: Array
    se: Array
    t: Array
    t_max: Array
    t_max_row: Array
    raw_at_loo_row: Array


class ConvMMDRepairMove(NamedTuple):
    """One step of the HU-triggered, Kkz-kept witness-repair chain.

    ``witness`` and ``health`` describe the mixture before the move; ``fired_F``
    (raw peak above the trigger), ``fired_Z`` (``t`` above the trigger) and
    ``fired_H`` (health flag) decide ``attempted``. For an attempted move,
    ``fit`` is the refit from the birth proposal, ``loss_drop`` its loss
    decrease, ``z`` the paired z of the per-row loss changes, ``accepted`` the
    K0 rule (drop above ``KEEP_DROP``) and ``kept`` the Kkz rule (accepted,
    ``kappa_after >= kappa_before`` and ``z >= Z_MIN``). Unattempted moves carry
    NaN and ``None`` there.
    """

    witness: ConvMMDWitness
    health: ConvMMDHealth
    fired_F: Array
    fired_Z: Array
    fired_H: Array
    attempted: Array
    fit: ConvMMDSaddleFit | None
    loss_drop: Array
    z: Array
    kappa_before: Array
    kappa_after: Array
    accepted: Array
    kept: Array


def witness_candidates(key: Array, n: int, *, size: int = WITNESS_CANDIDATES) -> Array:
    """Every row up to ``size``; above it ``size`` distinct ``key``-drawn rows, sorted (``convmmd_g_common.py:345-347``)."""

    if n <= size:
        return jnp.arange(n)
    return jnp.sort(jax.random.choice(key, n, shape=(size,), replace=False))


def _witness_terms(params, x, noise, bandwidths):
    """Per-row Gaussian terms of ``h_i(z)``, stacked over bandwidth and component.

    Each term is ``coef * |I + A_i / gamma^2|^{-1/2} exp(-(c_i - z)^T A_i^{-1}
    (c_i - z) / 2)``: the data term has ``A_i = S_i + gamma^2 I``, ``c_i = x_i``
    and ``coef = 1/G``; component ``k`` has ``A_i = Sigma_k + 2 S_i + gamma^2
    I``, ``c_i = mu_k`` and ``coef = -pi_k / G``. Returned as the exponent's
    constant, linear and (upper-triangle) quadratic coefficients in ``z``.
    """

    weights, means, covariances = params
    n, dimension = x.shape
    upper = jnp.triu_indices(dimension)
    eye = jnp.eye(dimension, dtype=x.dtype)
    count = bandwidths.shape[0]

    def term(a, centre, gamma_sq):
        inverse = jnp.linalg.inv(a)
        _, log_det = jnp.linalg.slogdet(a)
        linear = jnp.einsum("nij,nj->ni", inverse, centre)
        constant = -0.5 * (log_det - dimension * jnp.log(gamma_sq)) - 0.5 * jnp.sum(linear * centre, axis=-1)
        return constant, linear, inverse[:, upper[0], upper[1]]

    constants, linears, quadratics, coefficients = [], [], [], []
    for g in range(count):
        gamma_sq = bandwidths[g] * bandwidths[g]
        pieces = [(noise + gamma_sq * eye, x, 1.0 / count)]
        for k in range(weights.shape[0]):
            a = covariances[k][None] + 2.0 * noise + gamma_sq * eye
            pieces.append((a, jnp.broadcast_to(means[k], (n, dimension)), -weights[k] / count))
        for a, centre, coefficient in pieces:
            constant, linear, quadratic = term(a, centre, gamma_sq)
            constants.append(constant)
            linears.append(linear)
            quadratics.append(quadratic)
            coefficients.append(jnp.asarray(coefficient, dtype=x.dtype))
    return jnp.stack(constants), jnp.stack(linears), jnp.stack(quadratics), jnp.stack(coefficients)


@jax.jit
def _witness_block(z, constants, linears, quadratics, coefficients):
    """``h_i(z)`` for a block of candidate points, shape ``(N, B)``."""

    dimension = z.shape[-1]
    upper = jnp.triu_indices(dimension)
    pair = jnp.where(upper[0] == upper[1], 1.0, 2.0).astype(z.dtype)
    zz = z[:, upper[0]] * z[:, upper[1]] * pair

    def body(h, piece):
        constant, linear, quadratic, coefficient = piece
        exponent = constant[:, None] + linear @ z.T - 0.5 * (quadratic @ zz.T)
        return h + coefficient * jnp.exp(exponent), None

    h0 = jnp.zeros((constants.shape[1], z.shape[0]), dtype=z.dtype)
    h, _ = jax.lax.scan(body, h0, (constants, linears, quadratics, coefficients))
    return h


def witness_stats(
    params: ConvMMDParams,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    candidates: Array,
) -> ConvMMDWitness:
    """Witness score of the Gaussian-noise convMMD loss at candidate rows (``convmmd_g_common.py:276-342``).

    The loss falls fastest, when weight moves onto a latent point mass at
    ``z``, where ``w(z) = mean_i h_i(z)`` peaks:

        ``h_i(z) = mean_g [ G(x_i - z, S_i) - sum_k pi_k G(mu_k - z, Sigma_k + 2 S_i) ]``

    with :func:`~development.convmmd.expected_rbf_kernel`'s ``G`` at each
    bandwidth. ``params`` is the mixture the model sees (floored). Candidates
    ``z = x_c`` are data rows; the leave-one-out peak drops row ``c`` itself.
    Measurement covariances may be full.
    """

    x = jnp.asarray(observations)
    noise = jnp.asarray(measurement_covariances)
    bandwidths = jnp.asarray(bandwidths, dtype=x.dtype)
    candidates = jnp.asarray(candidates)
    n = x.shape[0]
    centre = jnp.mean(x, axis=0)
    xc = x - centre
    shifted = ConvMMDParams(params.weights, params.means - centre, params.covariances)
    terms = _witness_terms(shifted, xc, noise, bandwidths)

    total = candidates.shape[0]
    block = max(1, min(total, _WITNESS_BLOCK_ENTRIES // n))
    s1, s2, own = [], [], []
    for start in range(0, total, block):
        rows = candidates[start:start + block]
        size = rows.shape[0]
        padded = jnp.concatenate([rows, jnp.broadcast_to(rows[:1], (block - size,))])
        h = _witness_block(xc[padded], *terms)[:, :size]
        s1.append(jnp.sum(h, axis=0))
        s2.append(jnp.sum(h * h, axis=0))
        own.append(h[rows, jnp.arange(size)])
    s1, s2, own = (jnp.concatenate(a) for a in (s1, s2, own))
    raw = s1 / n
    loo = (s1 - own) / (n - 1)
    variance = ((s2 - own**2) - (n - 1) * loo**2) / (n - 2)
    se = jnp.sqrt(jnp.maximum(variance, 1e-300) / (n - 1))
    j_raw, j_loo, j_t = jnp.argmax(raw), jnp.argmax(loo), jnp.argmax(loo / se)
    return ConvMMDWitness(
        raw_peak=raw[j_raw],
        raw_row=candidates[j_raw],
        own_term_at_raw_peak=own[j_raw] / n,
        loo_peak=loo[j_loo],
        loo_row=candidates[j_loo],
        se=se[j_loo],
        t=loo[j_loo] / se[j_loo],
        t_max=loo[j_t] / se[j_t],
        t_max_row=candidates[j_t],
        raw_at_loo_row=raw[j_loo],
    )


def birth_at(
    params: ConvMMDParams,
    observations: Array,
    measurement_covariances: Array,
    row,
    floor,
) -> ConvMMDParams:
    """Move the lowest-weight component to data row ``row`` (``convmmd_g_common.py:350-361``).

    Its mean becomes the mean of the ``max(10, N // 100)`` nearest rows (every
    row when ``N < 10``), its
    covariance their sample covariance minus their mean noise, eigenvalue-floored
    at ``floor``, and its weight ``max(old, 0.05)``; the other weights are
    rescaled to sum to the rest.
    """

    x = jnp.asarray(observations)
    noise = jnp.asarray(measurement_covariances)
    weights, means, covariances = params
    n = x.shape[0]
    j = jnp.argmin(weights)
    count = min(max(10, n // 100), n)
    near = jnp.argsort(jnp.sum((x - x[row]) ** 2, axis=-1))[:count]
    local = x[near]
    mean = jnp.mean(local, axis=0)
    scatter = (local - mean).T @ (local - mean) / (count - 1)
    covariance = floor_cov(scatter - jnp.mean(noise[near], axis=0), floor)
    new_weight = jnp.maximum(weights[j], BIRTH_MIN_WEIGHT)
    others = jnp.arange(weights.shape[0]) != j
    rest = jnp.sum(jnp.where(others, weights, 0.0))
    weights = jnp.where(others, weights * (1.0 - new_weight) / rest, new_weight)
    return ConvMMDParams(
        weights=weights,
        means=means.at[j].set(mean),
        covariances=covariances.at[j].set(covariance),
    )


@jax.jit
def _row_loss_block(params, x, noise, bandwidths):
    return jax.vmap(
        lambda xi, si: convmmd_loss_analytic(params, xi[None], si[None], bandwidths)
    )(x, noise)


def row_losses(
    params: ConvMMDParams,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    *,
    chunk: int = ROW_CHUNK,
) -> Array:
    """Each row's loss term: the analytic loss of ``params`` on that row alone (``convmmd_j1_guard.py:142-160``).

    Their mean differs from the full loss by a constant that does not depend on
    ``params``, so differences of their means equal differences of the loss.
    """

    x = jnp.asarray(observations)
    noise = jnp.asarray(measurement_covariances)
    bandwidths = jnp.asarray(bandwidths, dtype=x.dtype)
    n = x.shape[0]
    size = min(chunk, n)
    out = []
    for start in range(0, n, size):
        stop = min(start + size, n)
        rows = jnp.arange(start, start + size).clip(max=n - 1)
        out.append(_row_loss_block(params, x[rows], noise[rows], bandwidths)[: stop - start])
    return jnp.concatenate(out)


def paired_z(differences: Array) -> Array:
    """z of the per-row loss change ``after - before``: positive when the loss falls (``convmmd_j1_guard.py:165-172``)."""

    differences = jnp.asarray(differences)
    n = differences.shape[0]
    mean = jnp.mean(differences)
    sd = jnp.std(differences, ddof=1)
    zero_sd = jnp.where(mean >= 0, 0.0, jnp.inf)
    return jnp.where(sd == 0.0, zero_sd, -mean / (sd / jnp.sqrt(n)))


# ---------------------------------------------------------------------------
# Recipe entry point
# ---------------------------------------------------------------------------


def _count(value, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be an integer, not boolean")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValidationError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValidationError(f"{name} must be >= {minimum}; received {result}")
    return int(result)


def _scalar(value, *, name: str, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a real number, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} must be a real number") from error
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        bound = "> 0" if positive else ">= 0"
        raise ValidationError(f"{name} must be finite and {bound}; received {value!r}")
    return result


def _require_float64(value, *, field: str) -> None:
    """Refuse a lower-precision float input instead of silently upcasting it."""

    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    if np.issubdtype(dtype, np.floating) and np.dtype(dtype) != np.float64:
        raise PrecisionError(
            f"{field} must be float64 (the recipe's certificate and curvature "
            f"thresholds are float64); received {np.dtype(dtype)}"
        )


def _float64_array(value, *, field: str) -> Array:
    """Validate and convert a real floating input to a float64 device array."""

    _require_float64(value, field=field)
    return _convert_array(
        value,
        field=field,
        requested_numpy_dtype=np.dtype(np.float64),
        requested_jax_dtype=jnp.dtype(jnp.float64),
        require_floating_source=True,
    )


def _recipe_inputs(observations, measurement_covariances, k, bandwidths, covariance_floor):
    """Eager checks at the public boundary; returns float64 device arrays and ``k``."""

    if not jax.config.x64_enabled:
        raise PrecisionError(
            "fit_convmmd_recipe runs in float64, but JAX x64 support is disabled; "
            "enable jax_enable_x64 before calling it"
        )
    k = _positive_count(k, name="k")
    x = _float64_array(observations, field="observations")
    if x.ndim != 2 or x.shape[1] < 1:
        _raise_shape("observations", tuple(x.shape), "(N, D) with D >= 1")
    n, dimension = x.shape
    if n <= k:
        raise ValidationError(
            f"observations: the K + 1 start needs more than k = {k} rows; received N = {n}"
        )
    shape = _shape(measurement_covariances, field="measurement_covariances")
    if shape != (n, dimension, dimension):
        _raise_shape("measurement_covariances", shape, f"({n}, {dimension}, {dimension})")
    _require_float64(measurement_covariances, field="measurement_covariances")
    noise = full_noise(measurement_covariances, dtype=np.float64)
    if bandwidths is not None:
        bandwidths = _float64_array(bandwidths, field="bandwidths")
        if bandwidths.ndim != 1 or bandwidths.shape[0] < 1 or not bool(jnp.all(bandwidths > 0.0)):
            raise ValidationError("bandwidths must be a non-empty 1-D array of positive values")
    if covariance_floor is not None:
        covariance_floor = _scalar(covariance_floor, name="covariance_floor", positive=False)
    return x, noise, k, bandwidths, covariance_floor


def fit_convmmd_recipe(
    observations: Array,
    measurement_covariances: Array,
    k: int,
    *,
    key: Array | None = None,
    bandwidths: Array | None = None,
    bandwidth_rows: int = BANDWIDTH_ROWS,
    covariance_floor=None,
    floor_fraction: float = FLOOR_FRACTION,
    n_steps: int = MAX_STEPS,
    chunk: int = CHUNK,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    grad_tol: float = GRAD_TOL,
    exact_limit: int = EXACT_LIMIT,
    lanczos_steps: int = LANCZOS_STEPS,
    max_escapes: int = MAX_ESCAPES,
    witness_moves: int = WITNESS_MOVES,
    candidate_rows: int = WITNESS_CANDIDATES,
    health_broad: float = HEALTH_BROAD,
    health_collapsed: float = HEALTH_COLLAPSED,
) -> ConvMMDRecipeResult:
    """Fit a ``k``-component convMMD mixture with the J1 recipe ``kplus1+floor:HU Kkz``.

    1. **Start**: :func:`kplus1_start` seeded with ``key`` (default
       ``PRNGKey(0)``).
    2. **Base fit**: :func:`fit_saddle_aware` on the floored loss (ridge ``f``,
       default :func:`floor_value`): constant-rate Adam in ``chunk``-step chunks,
       at most ``n_steps``, stopping at a certified (``max|grad| <= grad_tol``)
       point whose smallest curvature is not negative, escaping saddles at most
       ``max_escapes`` times.
    3. **Witness repair**: up to ``witness_moves`` times, while the health flag
       (H), the raw witness peak (F) or its standardized leave-one-out peak (Z)
       fires, the lowest-weight component is moved to the raw peak row
       (:func:`birth_at`) and the mixture refitted as in step 2. A move is kept
       only under Kkz (loss drop above ``KEEP_DROP``, ``kappa`` not lower, paired
       ``z >= Z_MIN``); the first move not kept ends the chain.

    Bandwidths default to :func:`subsample_bandwidths` on at most
    ``bandwidth_rows`` rows (drawn with ``fold_in(key, BANDWIDTH_STREAM)`` when
    ``N`` exceeds it); witness candidates are every row up to ``candidate_rows``
    (else drawn with ``fold_in(key, WITNESS_STREAM)``). The returned fit is the
    last kept one (the base fit if none). Its health is reported as the
    ``collapsed`` flag, which never changes ``status``; ``saddle`` likewise
    flags a certified point left at negative curvature. Set ``witness_moves=0``
    for the base fit alone.

    Inputs are checked eagerly. ``observations`` must be a finite float64
    ``(N, D)`` array with ``N > k``; ``measurement_covariances`` a finite float64
    ``(N, D, D)`` stack of symmetric positive-semidefinite matrices; optional
    ``bandwidths`` a non-empty 1-D array of positive float64 values. Counts and
    rates must be in range. A violation raises
    :class:`~development.validation.ValidationError`, and a lower-precision
    float input, or JAX x64 being disabled, raises
    :class:`~development.validation.PrecisionError`: the recipe is not run in
    float32, because its certificate and curvature thresholds are float64 ones.
    This is a host-level driver (Python control flow between jitted chunks),
    not a ``jit``-able function.
    """

    x, noise, k, bandwidths, covariance_floor = _recipe_inputs(
        observations, measurement_covariances, k, bandwidths, covariance_floor
    )
    bandwidth_rows = _count(bandwidth_rows, name="bandwidth_rows", minimum=2)
    n_steps = _count(n_steps, name="n_steps", minimum=0)
    chunk = _count(chunk, name="chunk", minimum=1)
    exact_limit = _count(exact_limit, name="exact_limit", minimum=0)
    lanczos_steps = _count(lanczos_steps, name="lanczos_steps", minimum=1)
    max_escapes = _count(max_escapes, name="max_escapes", minimum=0)
    witness_moves = _count(witness_moves, name="witness_moves", minimum=0)
    candidate_rows = _count(candidate_rows, name="candidate_rows", minimum=1)
    floor_fraction = _scalar(floor_fraction, name="floor_fraction", positive=False)
    learning_rate = _scalar(learning_rate, name="learning_rate", positive=True)
    weight_decay = _scalar(weight_decay, name="weight_decay", positive=False)
    grad_tol = _scalar(grad_tol, name="grad_tol", positive=False)
    if key is None:
        key = jax.random.PRNGKey(0)
    if bandwidths is None:
        bandwidths = subsample_bandwidths(
            jax.random.fold_in(key, BANDWIDTH_STREAM), x, rows=bandwidth_rows
        )
    bandwidths = jnp.asarray(bandwidths, dtype=x.dtype)
    if covariance_floor is None:
        covariance_floor = floor_value(noise, floor_fraction=floor_fraction)
    floor = jnp.asarray(covariance_floor, dtype=x.dtype)
    fit_options = dict(
        max_steps=n_steps,
        chunk=chunk,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_tol=grad_tol,
        exact_limit=exact_limit,
        lanczos_steps=lanczos_steps,
        max_escapes=max_escapes,
    )
    lanczos_key = jax.random.fold_in(key, LANCZOS_STREAM)

    start, start_status = kplus1_start(key, x, noise, k)
    base = fit_saddle_aware(
        to_floor_unc(start, floor), x, noise, bandwidths, floor,
        key=jax.random.fold_in(lanczos_key, 0), **fit_options,
    )

    candidates = witness_candidates(
        jax.random.fold_in(key, WITNESS_STREAM), x.shape[0], size=candidate_rows
    )
    birth_floor = START_FLOOR_FRACTION * jnp.mean(jnp.var(x, axis=0))
    nan = jnp.asarray(jnp.nan, dtype=x.dtype)
    no = jnp.asarray(False)
    current, repairs = base, []
    for move in range(witness_moves):
        witness = witness_stats(current.parameters, x, noise, bandwidths, candidates)
        before = health(current.parameters, broad=health_broad, collapsed=health_collapsed)
        fired_f = witness.raw_peak > RAW_TRIGGER
        fired_z = witness.t > Z_TRIGGER
        fired_h = before.collapsed
        attempted = fired_f | fired_z | fired_h
        if not bool(attempted):
            repairs.append(ConvMMDRepairMove(
                witness, before, fired_f, fired_z, fired_h, attempted,
                None, nan, nan, before.kappa, nan, no, no,
            ))
            break
        proposal = birth_at(current.parameters, x, noise, witness.raw_row, birth_floor)
        refit = fit_saddle_aware(
            to_floor_unc(proposal, floor), x, noise, bandwidths, floor,
            key=jax.random.fold_in(lanczos_key, move + 1), **fit_options,
        )
        changes = row_losses(refit.parameters, x, noise, bandwidths) - row_losses(
            current.parameters, x, noise, bandwidths
        )
        z = paired_z(changes)
        kappa_after = health(refit.parameters, broad=health_broad, collapsed=health_collapsed).kappa
        accepted = jnp.isfinite(refit.loss) & (refit.loss < current.loss - KEEP_DROP)
        kept = accepted & (kappa_after >= before.kappa) & (z >= Z_MIN)
        repairs.append(ConvMMDRepairMove(
            witness, before, fired_f, fired_z, fired_h, attempted, refit,
            current.loss - refit.loss, z, before.kappa, kappa_after, accepted, kept,
        ))
        if not bool(kept):
            break
        current = refit

    # A flag only: the health check never changes the returned fit's status.
    report = health(current.parameters, broad=health_broad, collapsed=health_collapsed)
    total_steps = int(base.n_iter) + sum(int(m.fit.n_iter) for m in repairs if m.fit is not None)
    return ConvMMDRecipeResult(
        parameters=current.parameters,
        unconstrained=current.unconstrained,
        loss=current.loss,
        history=current.history,
        n_iter=current.n_iter,
        status=current.status,
        converged=current.converged,
        numerical_failure=current.numerical_failure,
        max_abs_grad=current.max_abs_grad,
        grad_tol=current.grad_tol,
        collapsed=report.collapsed,
        health=report,
        covariance_floor=floor,
        bandwidths=bandwidths,
        initial_parameters=start,
        initialization_status=start_status,
        metadata=current_convmmd_metadata(),
        saddle=current.saddle,
        escapes=current.escapes,
        min_eigenvalue=current.min_eigenvalue,
        base=base,
        repairs=tuple(repairs),
        total_steps=jnp.asarray(total_steps),
    )
