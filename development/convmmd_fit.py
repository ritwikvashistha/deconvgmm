# SPDX-License-Identifier: MIT
# Provenance: convMMD is the maintainer's own method (Vashistha, Sarkar, Farahi,
# arXiv:2606.21907). This is a clean-room implementation from the model contract,
# not derived from astroML or Bovy XD code.
"""Temporary pure-JAX convMMD gradient-descent fit control (development stage).

Provides a self-contained AdamW optimizer (no ``optax`` dependency) with a
cosine-decay or constant learning-rate schedule that minimizes a convMMD loss
over the unconstrained parameters, with a documented :class:`ConvMMDFitStatus`
and rollback to the best finite state -- the convMMD analog of the
identity/general ``FitStatus`` culture. The step loop is a ``jax.lax.scan`` so
the whole fit is ``jit``-compatible; the Monte-Carlo fit threads one explicit PRNG
key through the scan. This is development evidence, not a released API, and it is
not exposed through any ``src/xdgmm_jax`` facade.

``converged`` is a **gradient-stationarity certificate** on the deterministic
path: it holds only when ``max|grad|`` of the objective at the *returned*
unconstrained parameters is at most ``grad_tol``. (It replaced a two-point
loss-change test that reported convergence at points far from stationary; see
``docs/convmmd-recipe-crosscheck.md``, finding B1.) The Monte-Carlo path never
certifies: its gradient is a single noisy estimate.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .convmmd import (
    ConvMMDParams,
    ConvMMDUnconstrained,
    convmmd_loss_analytic,
    convmmd_loss_mc,
    to_canonical,
)


Array = jax.Array

CONVMMD_CONTRACT_ID = "xdgmm-jax.convmmd"
CONVMMD_CONTRACT_VERSION = "0.6.0-draft.1"

# AdamW constants (mirroring the reference training recipe).
_BETA1 = 0.9
_BETA2 = 0.999
_ADAM_EPS = 1.0e-8
_COSINE_ALPHA = 1.0e-4


_SCHEDULES = ("cosine", "constant")


class ConvMMDFitStatus(IntEnum):
    """Terminal states of a convMMD gradient-descent fit.

    Values mirror the identity/general XD ``FitStatus``. ``COMPONENT_COLLAPSED``
    is reserved and currently never set: the recipe
    (:mod:`development.convmmd_recipe`) reports its post-fit health check as a
    ``collapsed`` flag without changing the status.
    """

    CONVERGED = 1
    MAX_ITER = 2
    NUMERICAL_FAILURE = 4
    COMPONENT_COLLAPSED = 5


class ConvMMDResultMetadata(NamedTuple):
    """Minimal identity and version of the convMMD numerical contract."""

    contract_id: str
    contract_version: str


class ConvMMDFitState(NamedTuple):
    """Array-only fit state; ``jit``-safe (no host-only metadata strings)."""

    parameters: ConvMMDParams
    unconstrained: ConvMMDUnconstrained
    loss: Array  # objective recomputed at the returned params (exact analytic; one MC estimate)
    history: Array  # (n_steps,) pre-update loss per step
    n_iter: Array
    status: Array  # ConvMMDFitStatus value
    converged: Array
    numerical_failure: Array
    learning_rate: Array
    weight_decay: Array
    clip_norm: Array
    # max|grad| of the objective w.r.t. the unconstrained parameters at the
    # returned point (exact on the analytic path; one estimate on the MC path).
    max_abs_grad: Array
    grad_tol: Array  # certificate threshold: converged iff max_abs_grad <= grad_tol


class ConvMMDFitResult(NamedTuple):
    """Host fit result: the array state plus host-only custody metadata."""

    parameters: ConvMMDParams
    unconstrained: ConvMMDUnconstrained
    loss: Array
    history: Array
    n_iter: Array
    status: Array
    converged: Array
    numerical_failure: Array
    learning_rate: Array
    weight_decay: Array
    clip_norm: Array
    max_abs_grad: Array
    grad_tol: Array
    metadata: ConvMMDResultMetadata


def _attach_metadata(state: ConvMMDFitState) -> ConvMMDFitResult:
    return ConvMMDFitResult(
        parameters=state.parameters,
        unconstrained=state.unconstrained,
        loss=state.loss,
        history=state.history,
        n_iter=state.n_iter,
        status=state.status,
        converged=state.converged,
        numerical_failure=state.numerical_failure,
        learning_rate=state.learning_rate,
        weight_decay=state.weight_decay,
        clip_norm=state.clip_norm,
        max_abs_grad=state.max_abs_grad,
        grad_tol=state.grad_tol,
        metadata=current_convmmd_metadata(),
    )


def current_convmmd_metadata() -> ConvMMDResultMetadata:
    return ConvMMDResultMetadata(
        contract_id=CONVMMD_CONTRACT_ID,
        contract_version=CONVMMD_CONTRACT_VERSION,
    )


def _global_norm(tree) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in leaves))


def _clip_by_global_norm(grads, clip_norm: Array):
    norm = _global_norm(grads)
    factor = jnp.minimum(1.0, clip_norm / (norm + 1e-12))
    return jax.tree_util.tree_map(lambda g: g * factor, grads)


def _cosine_learning_rate(base: Array, step: Array, total: int) -> Array:
    decay = 0.5 * (1.0 + jnp.cos(jnp.pi * step / total))
    return base * (_COSINE_ALPHA + (1.0 - _COSINE_ALPHA) * decay)


def _tree_where(predicate: Array, on_true, on_false):
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(predicate, a, b), on_true, on_false
    )


def _all_finite(tree) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
    )


def _max_abs(tree) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))


def _add_covariance_floor(params: ConvMMDParams, floor) -> ConvMMDParams:
    """``params`` with every component covariance replaced by ``Sigma_k + floor I``."""

    eye = jnp.eye(params.means.shape[-1], dtype=params.covariances.dtype)
    return params._replace(covariances=params.covariances + floor * eye)


def _run(
    loss_and_grad,
    initial: ConvMMDUnconstrained,
    *,
    n_steps: int,
    learning_rate: float,
    weight_decay: float,
    clip_norm: float,
    tol: float,
    key: Array | None,
    deterministic: bool,
    grad_tol: float = 1.0e-6,
    schedule: str = "cosine",
    to_params=to_canonical,
    stop_at_certificate: bool = False,
) -> ConvMMDFitState:
    """Run AdamW (cosine-decay or constant rate) with honest rollback and status.

    Rollback target is the **last finite iterate** (not a ``min`` over noisy
    per-step estimates, which would be biased for the stochastic Monte-Carlo
    loss). The reported ``loss`` and ``max_abs_grad`` are recomputed at the
    returned parameters -- exact for the deterministic analytic path, a single
    honest estimate for the Monte-Carlo path.

    ``converged`` is the gradient-stationarity certificate ``max_abs_grad <=
    grad_tol`` at the returned point, asserted only for a ``deterministic``
    objective whose fit did not fail. ``tol`` is accepted for signature
    compatibility and no longer affects the status.

    ``schedule="cosine"`` decays ``learning_rate`` to ``1e-4`` of its value over
    ``n_steps``; ``schedule="constant"`` uses ``learning_rate`` every step. The
    AdamW moments, bias correction, global-norm clip and weight decay are the
    same for both. ``to_params`` maps the unconstrained parameters to the
    reported canonical parameters; it must be the same map ``loss_and_grad``
    evaluates the objective through.

    ``stop_at_certificate`` (deterministic path only; default off) freezes the
    fit at the first iterate whose pre-update gradient already satisfies
    ``max|grad| <= grad_tol``: no further updates are applied, later ``history``
    slots repeat that iterate's loss, the remaining steps skip the objective
    (``lax.cond``), and ``n_iter`` counts the updates actually applied. Without
    it, ``n_iter`` is ``n_steps`` and a constant-rate fit may drift off a point
    it had already certified.
    """

    del tol  # superseded by the gradient certificate (``grad_tol``)
    if schedule not in _SCHEDULES:
        raise ValueError(
            f"schedule must be one of {_SCHEDULES}; received {schedule!r}"
        )
    stop = bool(stop_at_certificate) and deterministic
    dtype = initial.means.dtype
    lr = jnp.asarray(learning_rate, dtype=dtype)
    wd = jnp.asarray(weight_decay, dtype=dtype)
    clip = jnp.asarray(clip_norm, dtype=dtype)
    gtol = jnp.asarray(grad_tol, dtype=dtype)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, initial)
    if key is None:
        key = jax.random.PRNGKey(0)  # unused by deterministic losses

    def step(carry, index):
        (
            unc,
            moment1,
            moment2,
            last_finite,
            failed,
            fold_key,
            stopped,
            n_updates,
            frozen_loss,
        ) = carry
        fold_key, step_key = jax.random.split(fold_key)
        loss, grads = loss_and_grad(unc, step_key)
        finite = jnp.isfinite(loss) & _all_finite(grads)
        live_finite = finite & jnp.logical_not(failed)
        if stop:
            certified_now = live_finite & (_max_abs(grads) <= gtol)
        else:
            certified_now = jnp.asarray(False)
        do_update = live_finite & jnp.logical_not(certified_now)

        # The current iterate is a known-good rollback target only if this step
        # scored finite while the fit was still live.
        last_finite_next = _tree_where(live_finite, unc, last_finite)

        grads = _clip_by_global_norm(grads, clip)
        step_next = index + 1
        moment1_new = jax.tree_util.tree_map(
            lambda m, g: _BETA1 * m + (1.0 - _BETA1) * g, moment1, grads
        )
        moment2_new = jax.tree_util.tree_map(
            lambda v, g: _BETA2 * v + (1.0 - _BETA2) * g * g, moment2, grads
        )
        bias1 = 1.0 - _BETA1 ** step_next.astype(dtype)
        bias2 = 1.0 - _BETA2 ** step_next.astype(dtype)
        if schedule == "cosine":
            lr_step = _cosine_learning_rate(lr, index.astype(dtype), n_steps)
        else:
            lr_step = lr

        def apply(param, m, v):
            m_hat = m / bias1
            v_hat = v / bias2
            return param - lr_step * (
                m_hat / (jnp.sqrt(v_hat) + _ADAM_EPS) + wd * param
            )

        unc_updated = jax.tree_util.tree_map(
            apply, unc, moment1_new, moment2_new
        )
        unc_next = _tree_where(do_update, unc_updated, unc)
        moment1_next = _tree_where(do_update, moment1_new, moment1)
        moment2_next = _tree_where(do_update, moment2_new, moment2)
        failed_next = failed | jnp.logical_not(finite)

        carry_next = (
            unc_next,
            moment1_next,
            moment2_next,
            last_finite_next,
            failed_next,
            fold_key,
            stopped | certified_now,
            n_updates + do_update.astype(n_updates.dtype),
            jnp.where(certified_now, loss, frozen_loss),
        )
        return carry_next, loss

    if stop:
        live_step = step

        def frozen_step(carry, index):
            del index
            return carry, carry[-1]

        def step(carry, index):  # noqa: F811 - wraps the live step
            return jax.lax.cond(carry[6], frozen_step, live_step, carry, index)

    carry0 = (
        initial,
        zeros,
        zeros,
        initial,
        jnp.asarray(False),
        key,
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.nan, dtype=dtype),
    )
    carry_final, history = jax.lax.scan(step, carry0, jnp.arange(n_steps))
    current_unc, _, _, last_finite, failed, final_key, _, n_updates, _ = carry_final

    # On success return the final iterate; on failure roll back to the last
    # finite iterate. Report the objective and its gradient recomputed at the
    # returned params.
    returned_unc = _tree_where(failed, last_finite, current_unc)
    reported_loss, returned_grads = loss_and_grad(returned_unc, final_key)
    max_abs_grad = _max_abs(returned_grads)

    if deterministic:
        # NaN compares False, so a non-finite gradient never certifies.
        converged = jnp.logical_and(
            jnp.logical_not(failed), max_abs_grad <= gtol
        )
    else:
        converged = jnp.asarray(False)

    status = jnp.where(
        failed,
        jnp.asarray(int(ConvMMDFitStatus.NUMERICAL_FAILURE)),
        jnp.where(
            converged,
            jnp.asarray(int(ConvMMDFitStatus.CONVERGED)),
            jnp.asarray(int(ConvMMDFitStatus.MAX_ITER)),
        ),
    )

    return ConvMMDFitState(
        parameters=to_params(returned_unc),
        unconstrained=returned_unc,
        loss=reported_loss,
        history=history,
        n_iter=n_updates if stop else jnp.asarray(n_steps),
        status=status,
        converged=converged,
        numerical_failure=failed,
        learning_rate=lr,
        weight_decay=wd,
        clip_norm=clip,
        max_abs_grad=max_abs_grad,
        grad_tol=gtol,
    )


def fit_analytic_state(
    initial: ConvMMDUnconstrained,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    *,
    n_steps: int = 300,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 1.0e-1,
    clip_norm: float = 1.0,
    tol: float = 1.0e-6,
    grad_tol: float = 1.0e-6,
    schedule: str = "cosine",
    covariance_floor: float | Array = 0.0,
    stop_at_certificate: bool = False,
) -> ConvMMDFitState:
    """Array-only (``jit``-safe) deterministic analytic-loss fit.

    ``converged`` certifies ``max|grad| <= grad_tol`` at the returned point.
    ``schedule`` (``"cosine"`` or ``"constant"``) and ``stop_at_certificate``
    must be static under ``jit``; with ``stop_at_certificate`` the fit freezes at
    the first certified iterate (see :func:`_run`).

    ``covariance_floor`` ``f`` is a ridge **inside the objective**: the optimizer
    moves ``Sigma_k = to_canonical(unconstrained).covariances`` but the loss is
    evaluated on ``Sigma_k + f I``, and the reported ``parameters`` are that
    floored mixture (so ``loss`` recomputes exactly at ``parameters``). The
    default ``0.0`` leaves the objective unchanged.
    """

    if isinstance(covariance_floor, (int, float)) and covariance_floor == 0.0:
        to_params = to_canonical
    else:

        def to_params(candidate):
            return _add_covariance_floor(to_canonical(candidate), covariance_floor)

    def loss_and_grad(unc, _key):
        def objective(candidate):
            return convmmd_loss_analytic(
                to_params(candidate),
                observations,
                measurement_covariances,
                bandwidths,
            )

        return jax.value_and_grad(objective)(unc)

    return _run(
        loss_and_grad,
        initial,
        n_steps=n_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        tol=tol,
        key=None,
        deterministic=True,
        grad_tol=grad_tol,
        schedule=schedule,
        to_params=to_params,
        stop_at_certificate=stop_at_certificate,
    )


def fit_mc_state(
    initial: ConvMMDUnconstrained,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    key: Array,
    *,
    num_samples: int = 200,
    n_steps: int = 300,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 1.0e-1,
    clip_norm: float = 1.0,
    tol: float = 1.0e-6,
) -> ConvMMDFitState:
    """Array-only (``jit``-safe) stochastic Monte-Carlo-loss fit."""

    def loss_and_grad(unc, step_key):
        def objective(candidate):
            return convmmd_loss_mc(
                to_canonical(candidate),
                observations,
                measurement_covariances,
                bandwidths,
                step_key,
                num_samples,
            )

        return jax.value_and_grad(objective)(unc)

    return _run(
        loss_and_grad,
        initial,
        n_steps=n_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        tol=tol,
        key=key,
        deterministic=False,
    )


def fit_analytic(
    initial: ConvMMDUnconstrained,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    **kwargs,
) -> ConvMMDFitResult:
    """Host analytic-loss fit: array state plus custody metadata."""

    return _attach_metadata(
        fit_analytic_state(
            initial, observations, measurement_covariances, bandwidths, **kwargs
        )
    )


def fit_mc(
    initial: ConvMMDUnconstrained,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    key: Array,
    **kwargs,
) -> ConvMMDFitResult:
    """Host Monte-Carlo-loss fit: array state plus custody metadata."""

    return _attach_metadata(
        fit_mc_state(
            initial, observations, measurement_covariances, bandwidths, key, **kwargs
        )
    )
