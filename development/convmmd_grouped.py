# SPDX-License-Identifier: MIT
# Provenance: convMMD is the maintainer's own method (Vashistha, Sarkar, Farahi,
# arXiv:2606.21907). This is a clean-room implementation from the model contract
# (docs/convmmd-model-contract.md §16), not derived from astroML or Bovy XD code.
"""Temporary eager grouped orchestration for masked (MAR) convMMD (development).

The projected fixed-``M`` leaves in :mod:`development.convmmd` evaluate one mask
group. This module turns a full-width collection with a boolean ``observed_mask``
into deterministic fixed-``M`` coordinate-selection groups (reusing XD's tested
mask adapter in :mod:`development.general_validation`), evaluates each group's
projected leaf, and combines the per-row contributions with informative-weight
normalization (contract §16.3). Unlike XD's grouped path there is **no M-step**:
the grouped loss is a single differentiable function of the parameters over a
fixed group structure, so ``value_and_grad``/``jit`` apply; mask grouping,
validation, and restoration are host-only and outside the JIT/autodiff contract.

This is development evidence, not a released API, and is not exposed through any
``src/deconvgmm`` facade (that is Phase 6, maintainer-gated).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .convmmd import (
    ConvMMDParams,
    PosteriorComponents,
    convmmd_loss_analytic_projected,
    convmmd_loss_mc_projected,
    denoise_projected,
    posterior_components_projected,
    to_canonical,
)
from .convmmd_fit import ConvMMDFitResult, _attach_metadata, _run
from .general_validation import (
    GroupedGeneralInputs,
    IdentityProjection,
    group_masked_general_fit_inputs,
    group_masked_general_inputs,
    restore_grouped_rows,
)


Array = jax.Array


def _latent_dimension(params) -> int:
    return int(jnp.asarray(params.means).shape[1])


def median_bandwidths_masked(
    observations,
    observed_mask,
    *,
    n_scales: int = 9,
    log10_low: float = -2.0,
    log10_high: float = 2.0,
) -> Array:
    """Single global masked bandwidth set (§16.6); host-only convenience.

    ``gamma_g = b_mask * 10**s_g`` where ``b_mask`` is the median, over pairs
    ``(i, j), i < j`` sharing at least one observed coordinate, of the Euclidean
    distance on their shared coordinates. Equals :func:`development.convmmd.
    median_bandwidths` on fully-observed data; raises when no pair shares an
    observed coordinate. Ragged over pairs, so this is deliberately a host loop
    outside the JIT/autodiff contract (like the base bandwidth heuristic).
    """

    raw = np.asarray(jax.device_get(observations))
    original_dtype = raw.dtype
    # Distances are computed in float64 (matching the oracle) so the heuristic is
    # dtype-robust; the returned bandwidths carry the observations' dtype.
    x = raw.astype(np.float64)
    mask = np.asarray(jax.device_get(observed_mask))
    if mask.dtype != np.bool_:
        raise ValueError("observed_mask must be a boolean array")
    if mask.shape != x.shape:
        raise ValueError(
            f"observed_mask shape {mask.shape} must match observations {x.shape}"
        )
    n = x.shape[0]
    distances: list[float] = []
    for i in range(n):
        coords_i = np.flatnonzero(mask[i])
        if coords_i.size == 0:
            continue
        for j in range(i + 1, n):
            coords_j = np.flatnonzero(mask[j])
            shared = np.intersect1d(coords_i, coords_j, assume_unique=True)
            if shared.size == 0:
                continue
            difference = x[i, shared] - x[j, shared]
            distances.append(float(np.sqrt(difference @ difference)))
    if not distances:
        raise ValueError(
            "median_bandwidths_masked needs at least one pair of observations "
            "sharing an observed coordinate"
        )
    base = float(np.median(np.asarray(distances, dtype=np.float64)))
    scales = jnp.logspace(log10_low, log10_high, n_scales, dtype=original_dtype)
    return base * scales


def group_masked_inputs(
    params,
    observations,
    observed_mask,
    *,
    noise,
    sample_weight=None,
    dtype,
) -> GroupedGeneralInputs:
    """Group a full-width masked collection into fixed-``M`` selection groups.

    Thin wrapper fixing ``projection=IdentityProjection(D)`` (pure coordinate
    selection); ``noise`` and any shared/per-item full-covariance forms follow the
    XD mask adapter. Valid for inference even when every group is ``M=0``.
    """

    return group_masked_general_inputs(
        params,
        observations,
        observed_mask,
        projection=IdentityProjection(_latent_dimension(params)),
        noise=noise,
        sample_weight=sample_weight,
        dtype=dtype,
    )


def group_masked_fit_inputs(
    params,
    observations,
    observed_mask,
    *,
    noise,
    sample_weight=None,
    dtype,
) -> GroupedGeneralInputs:
    """Group for fitting; raises ``no_informative_weight`` if every row is ``M=0``."""

    fit_inputs = group_masked_general_fit_inputs(
        params,
        observations,
        observed_mask,
        projection=IdentityProjection(_latent_dimension(params)),
        noise=noise,
        sample_weight=sample_weight,
        dtype=dtype,
    )
    return fit_inputs.grouped


def _informative_groups(grouped: GroupedGeneralInputs):
    """Return ``(index, group)`` pairs for groups with ``M > 0`` (static)."""

    return [
        (index, group)
        for index, group in enumerate(grouped.groups)
        if group.observations.shape[-1] > 0
    ]


def _normalize_by_informative_weight(total: Array, informative_weight: Array) -> Array:
    """Divide by the informative weight, returning 0 when that weight is 0.

    ``jit``/``grad``-safe: the degenerate zero-weight branch divides by one and is
    masked to zero, so it never forms ``0/0`` (which would be a ``NaN`` value and a
    ``NaN`` gradient). This matches the NumPy oracle, whose zero-informative-weight
    loss is exactly 0 (the fit path separately rejects it as ``no_informative_weight``).
    """

    positive = informative_weight > 0
    safe = jnp.where(positive, informative_weight, jnp.ones_like(informative_weight))
    return jnp.where(positive, total / safe, jnp.zeros_like(total))


def grouped_analytic_loss(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
) -> Array:
    """Informative-weight-normalized analytic masked loss (§16.3), scalar.

    ``params`` are the CURRENT parameters (the group structure is fixed and
    param-independent); they MUST share the dtype of the grouped arrays.
    """

    dtype = params.means.dtype
    informative = _informative_groups(grouped)
    if not informative:
        return jnp.asarray(0.0, dtype=dtype)  # all M=0: loss defined as exactly 0
    informative_weight = jnp.asarray(grouped.informative_weight, dtype=dtype)
    total = jnp.asarray(0.0, dtype=dtype)
    for _, group in informative:
        per_row = convmmd_loss_analytic_projected(
            params,
            group.observations,
            group.projection_matrices,
            group.measurement_covariances,
            bandwidths,
        )
        total = total + jnp.sum(group.sample_weight.astype(dtype) * per_row)
    return _normalize_by_informative_weight(total, informative_weight)


def grouped_mc_loss(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    key: Array,
    num_samples: int,
) -> Array:
    """Informative-weight-normalized Monte-Carlo masked loss (§16.5), scalar.

    One explicit ``key`` is split across the groups so each group draws
    independently.
    """

    dtype = params.means.dtype
    informative = _informative_groups(grouped)
    if not informative:
        return jnp.asarray(0.0, dtype=dtype)
    informative_weight = jnp.asarray(grouped.informative_weight, dtype=dtype)
    keys = jax.random.split(key, max(len(grouped.groups), 1))
    total = jnp.asarray(0.0, dtype=dtype)
    for index, group in informative:
        per_row = convmmd_loss_mc_projected(
            params,
            group.observations,
            group.projection_matrices,
            group.measurement_covariances,
            bandwidths,
            keys[index],
            num_samples,
        )
        total = total + jnp.sum(group.sample_weight.astype(dtype) * per_row)
    return _normalize_by_informative_weight(total, informative_weight)


def grouped_posterior_components(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
) -> PosteriorComponents:
    """Restored projected posterior responsibilities ``(N, K)`` and means ``(N, K, D)``.

    ``M=0`` rows take the prior weights and prior (full-``D``) component means.
    """

    n_components = params.weights.shape[0]
    dimension = params.means.shape[1]
    responsibility_groups: list[Array] = []
    mean_groups: list[Array] = []
    for group in grouped.groups:
        n_rows = len(group.original_indices)
        if group.observations.shape[-1] == 0:
            responsibility_groups.append(
                jnp.broadcast_to(params.weights, (n_rows, n_components))
            )
            mean_groups.append(
                jnp.broadcast_to(params.means, (n_rows, n_components, dimension))
            )
        else:
            components = posterior_components_projected(
                params,
                group.observations,
                group.projection_matrices,
                group.measurement_covariances,
            )
            responsibility_groups.append(components.responsibilities)
            mean_groups.append(components.component_means)
    return PosteriorComponents(
        responsibilities=restore_grouped_rows(
            grouped, responsibility_groups, field="masked responsibilities"
        ),
        component_means=restore_grouped_rows(
            grouped, mean_groups, field="masked component means"
        ),
    )


def grouped_denoise(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
) -> Array:
    """Restored projected posterior mean ``(N, D)`` in original row order (§16.4).

    ``M=0`` rows return the prior mean ``sum_k pi_k mu_k``.
    """

    dimension = params.means.shape[1]
    prior_mean = jnp.sum(params.weights[:, None] * params.means, axis=0)  # (D,)
    mean_groups: list[Array] = []
    for group in grouped.groups:
        n_rows = len(group.original_indices)
        if group.observations.shape[-1] == 0:
            mean_groups.append(jnp.broadcast_to(prior_mean, (n_rows, dimension)))
        else:
            mean_groups.append(
                denoise_projected(
                    params,
                    group.observations,
                    group.projection_matrices,
                    group.measurement_covariances,
                )
            )
    return restore_grouped_rows(grouped, mean_groups, field="masked denoised means")


def convmmd_loss_analytic_masked(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    bandwidths,
    sample_weight=None,
    dtype,
) -> Array:
    """One-shot: group a full-width masked collection and return the analytic loss."""

    grouped = group_masked_inputs(
        params,
        observations,
        observed_mask,
        noise=noise,
        sample_weight=sample_weight,
        dtype=dtype,
    )
    canonical = grouped.parameters
    return grouped_analytic_loss(
        ConvMMDParams(canonical.weights, canonical.means, canonical.covariances),
        grouped,
        jnp.asarray(bandwidths, dtype=canonical.means.dtype),
    )


def convmmd_loss_mc_masked(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    bandwidths,
    key: Array,
    num_samples: int,
    sample_weight=None,
    dtype,
) -> Array:
    """One-shot: group a full-width masked collection and return the MC loss."""

    grouped = group_masked_inputs(
        params,
        observations,
        observed_mask,
        noise=noise,
        sample_weight=sample_weight,
        dtype=dtype,
    )
    canonical = grouped.parameters
    return grouped_mc_loss(
        ConvMMDParams(canonical.weights, canonical.means, canonical.covariances),
        grouped,
        jnp.asarray(bandwidths, dtype=canonical.means.dtype),
        key,
        num_samples,
    )


def convmmd_denoise_masked(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    dtype,
) -> Array:
    """One-shot: group a full-width masked collection and return the posterior mean."""

    grouped = group_masked_inputs(
        params, observations, observed_mask, noise=noise, dtype=dtype
    )
    canonical = grouped.parameters
    return grouped_denoise(
        ConvMMDParams(canonical.weights, canonical.means, canonical.covariances),
        grouped,
    )


def convmmd_posterior_components_masked(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    dtype,
) -> PosteriorComponents:
    """One-shot: group a full-width masked collection and return posterior components."""

    grouped = group_masked_inputs(
        params, observations, observed_mask, noise=noise, dtype=dtype
    )
    canonical = grouped.parameters
    return grouped_posterior_components(
        ConvMMDParams(canonical.weights, canonical.means, canonical.covariances),
        grouped,
    )


def fit_masked_analytic_state(
    initial,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    *,
    n_steps: int = 300,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 1.0e-1,
    clip_norm: float = 1.0,
    tol: float = 1.0e-6,
):
    """Array-only deterministic analytic masked-loss fit over a fixed group set."""

    def loss_and_grad(unc, _key):
        def objective(candidate):
            return grouped_analytic_loss(to_canonical(candidate), grouped, bandwidths)

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


def fit_masked_mc_state(
    initial,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    key: Array,
    *,
    num_samples: int = 200,
    n_steps: int = 300,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 1.0e-1,
    clip_norm: float = 1.0,
    tol: float = 1.0e-6,
):
    """Array-only stochastic Monte-Carlo masked-loss fit over a fixed group set."""

    def loss_and_grad(unc, step_key):
        def objective(candidate):
            return grouped_mc_loss(
                to_canonical(candidate), grouped, bandwidths, step_key, num_samples
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


def fit_masked_analytic(
    initial, grouped: GroupedGeneralInputs, bandwidths: Array, **kwargs
) -> ConvMMDFitResult:
    """Host analytic masked-loss fit: array state plus custody metadata."""

    return _attach_metadata(
        fit_masked_analytic_state(initial, grouped, bandwidths, **kwargs)
    )


def fit_masked_mc(
    initial, grouped: GroupedGeneralInputs, bandwidths: Array, key: Array, **kwargs
) -> ConvMMDFitResult:
    """Host Monte-Carlo masked-loss fit: array state plus custody metadata."""

    return _attach_metadata(
        fit_masked_mc_state(initial, grouped, bandwidths, key, **kwargs)
    )


__all__ = [
    "convmmd_denoise_masked",
    "convmmd_loss_analytic_masked",
    "convmmd_loss_mc_masked",
    "convmmd_posterior_components_masked",
    "fit_masked_analytic",
    "fit_masked_analytic_state",
    "fit_masked_mc",
    "fit_masked_mc_state",
    "group_masked_fit_inputs",
    "group_masked_inputs",
    "grouped_analytic_loss",
    "grouped_denoise",
    "grouped_mc_loss",
    "grouped_posterior_components",
    "median_bandwidths_masked",
]
