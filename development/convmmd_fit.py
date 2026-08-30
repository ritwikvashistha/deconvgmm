# SPDX-License-Identifier: MIT
# Provenance: convMMD is the maintainer's own method (Vashistha, Sarkar, Farahi,
# arXiv:2606.21907). This is a clean-room implementation from the model contract,
# not derived from astroML or Bovy XD code.
"""Temporary pure-JAX convMMD gradient-descent fit control (development stage).

Provides a self-contained AdamW + cosine-decay optimizer (no ``optax``
dependency) that minimizes a convMMD loss over the unconstrained parameters, with
a documented :class:`ConvMMDFitStatus` and rollback to the best finite state --
the convMMD analog of the identity/general ``FitStatus`` culture. The step loop
is a ``jax.lax.scan`` so the whole fit is ``jit``-compatible; the Monte-Carlo fit
threads one explicit PRNG key through the scan. This is development evidence, not
a released API, and it is not exposed through any ``src/xdgmm_jax`` facade.
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
CONVMMD_CONTRACT_VERSION = "0.2.0-draft.1"

# AdamW constants (mirroring the reference training recipe).
_BETA1 = 0.9
_BETA2 = 0.999
_ADAM_EPS = 1.0e-8
_COSINE_ALPHA = 1.0e-4


class ConvMMDFitStatus(IntEnum):
    """Terminal states of a convMMD gradient-descent fit."""

    CONVERGED = 1
    MAX_ITER = 2
    NUMERICAL_FAILURE = 4


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
) -> ConvMMDFitState:
    """Run AdamW + cosine decay with honest rollback and status.

    Rollback target is the **last finite iterate** (not a ``min`` over noisy
    per-step estimates, which would be biased for the stochastic Monte-Carlo
    loss). The reported ``loss`` is the objective recomputed at the returned
    parameters -- exact for the deterministic analytic path, a single honest
    estimate for the Monte-Carlo path. Two-point convergence is only asserted for
    a ``deterministic`` objective with at least two steps.
    """

    dtype = initial.means.dtype
    lr = jnp.asarray(learning_rate, dtype=dtype)
    wd = jnp.asarray(weight_decay, dtype=dtype)
    clip = jnp.asarray(clip_norm, dtype=dtype)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, initial)
    if key is None:
        key = jax.random.PRNGKey(0)  # unused by deterministic losses

    def step(carry, index):
        unc, moment1, moment2, last_finite, failed, fold_key = carry
        fold_key, step_key = jax.random.split(fold_key)
        loss, grads = loss_and_grad(unc, step_key)
        finite = jnp.isfinite(loss) & _all_finite(grads)
        do_update = finite & jnp.logical_not(failed)

        # The current iterate is a known-good rollback target only if this step
        # scored finite while the fit was still live.
        last_finite_next = _tree_where(do_update, unc, last_finite)

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
        lr_step = _cosine_learning_rate(lr, index.astype(dtype), n_steps)

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
        )
        return carry_next, loss

    carry0 = (initial, zeros, zeros, initial, jnp.asarray(False), key)
    carry_final, history = jax.lax.scan(step, carry0, jnp.arange(n_steps))
    current_unc, _, _, last_finite, failed, final_key = carry_final

    # On success return the final iterate; on failure roll back to the last
    # finite iterate. Report the objective recomputed at the returned params.
    returned_unc = _tree_where(failed, last_finite, current_unc)
    reported_loss, _ = loss_and_grad(returned_unc, final_key)

    if deterministic and n_steps >= 2:
        last_change = jnp.abs(history[-1] - history[-2]) / (
            jnp.abs(history[-1]) + 1e-12
        )
        converged = jnp.logical_and(
            jnp.logical_not(failed), last_change < tol
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
        parameters=to_canonical(returned_unc),
        unconstrained=returned_unc,
        loss=reported_loss,
        history=history,
        n_iter=jnp.asarray(n_steps),
        status=status,
        converged=converged,
        numerical_failure=failed,
        learning_rate=lr,
        weight_decay=wd,
        clip_norm=clip,
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
) -> ConvMMDFitState:
    """Array-only (``jit``-safe) deterministic analytic-loss fit."""

    def loss_and_grad(unc, _key):
        def objective(candidate):
            return convmmd_loss_analytic(
                to_canonical(candidate),
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
