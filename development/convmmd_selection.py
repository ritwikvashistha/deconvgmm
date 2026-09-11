# SPDX-License-Identifier: MIT
# Provenance: convMMD is the maintainer's own method (Vashistha, Sarkar, Farahi,
# arXiv:2606.21907). This is a clean-room implementation from the model contract
# (docs/convmmd-model-contract.md §17), not derived from astroML or Bovy XD code.
"""Temporary pure-JAX convMMD under a known selection function (MNAR, development).

Contract §17 adds a known, simulable **completeness** ``Omega`` (MNAR truncation /
detection incompleteness). This module implements selection on the **true value**
``Omega(z)`` (convention ``on_z``, §§17.4–17.6), selection on the **observed value**
``Omega(x-tilde)`` under a homoscedastic fully-observed restriction (§17.12), and — new
in contract ``0.4.0-draft.1`` — **heteroscedastic** ``Omega(x-tilde)`` via a per-object
transform on each detected object's own known ``S_i`` (§17.13): machine-eps analytic for
a Gaussian window, per-object SNIS for a general ``Omega``, needing **no noise model**.

The **Gaussian-window analytic** sub-case (§17.5) is the exact oracle and supported
fast path implemented here: a Gaussian window
``Omega(z) = exp(-1/2 (z - a)^T Psi_inv (z - a))`` acts on the latent mixture
*before* projection and noise convolution, mapping ``(pi, mu, Sigma)`` to a selected
Gaussian mixture ``(pi', mu', Sigma')`` via the Gaussian-product identity. Every
selected analytic operation is therefore the shipped §16 masked operator
(:mod:`development.convmmd_grouped`) evaluated verbatim on the transformed
parameters; the only new numerical kernel is the differentiable parameter transform
:func:`select_gaussian_params`. The general-``Omega`` SNIS Monte-Carlo path (§17.4)
is added separately.

Like the rest of ``development/``, this is development evidence exercised by the
contract-driven suite, not yet a released API surface. It is exposed through the
``deconvgmm.convmmd`` facade at Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .convmmd import (
    ConvMMDParams,
    PosteriorComponents,
    convmmd_loss_analytic,
    denoise,
    expected_rbf_kernel,
    to_canonical,
)
from .convmmd_fit import ConvMMDFitResult, _attach_metadata, _run
from .convmmd_grouped import (
    _informative_groups,
    _normalize_by_informative_weight,
    convmmd_denoise_masked,
    convmmd_loss_analytic_masked,
    convmmd_posterior_components_masked,
    grouped_analytic_loss,
    grouped_denoise,
    grouped_posterior_components,
    group_masked_fit_inputs,
    group_masked_inputs,
)
from .general_validation import GroupedGeneralInputs, restore_grouped_rows


Array = jax.Array


class GaussianSelection(NamedTuple):
    """A Gaussian selection window ``Omega(z)`` on the true value (§17.3).

    ``Omega(z) = exp(-1/2 (z - location)^T precision (z - location))`` (peak 1 at
    ``z = location``). ``precision`` is the window precision ``Psi^{-1}`` (symmetric
    PSD, shape ``(D, D)``); ``precision = 0`` is exactly "no selection" and reduces
    every selected operation to its §16 counterpart. Convention is ``on_z``.

    Both fields are arrays, so this is a JAX PyTree and MAY be an argument to the
    ``grad``-able transform (gradients flow to the mixture parameters, not to the
    fixed window). ``location`` has shape ``(D,)``.
    """

    location: Array
    precision: Array


@dataclass(frozen=True)
class CallableSelection:
    """A general simulable completeness ``Omega`` for the SNIS path (§17.3).

    ``omega`` is a differentiable callable mapping latent draws ``(..., D)`` to
    weights ``(...,)`` in ``[0, 1]``; it drives the general-``Omega`` SNIS
    Monte-Carlo loss (§17.4). This is a **host-level** container (it wraps a Python
    callable), not a JAX PyTree: the callable is closed over by the leaf, not passed
    through ``jit`` as an array leaf. Convention is ``on_z``.
    """

    omega: Callable[[Array], Array]
    convention: str = "on_z"


def gaussian_selection_omega(selection: GaussianSelection) -> Callable[[Array], Array]:
    """Return the differentiable ``Omega(z)`` callable for a Gaussian window.

    Lets the SNIS path (§17.4) consume a :class:`GaussianSelection`, so the SNIS
    estimator can be checked against the §17.5 analytic value for the same window.
    """

    location = selection.location
    precision = selection.precision

    def omega(latent: Array) -> Array:
        centered = latent - location.astype(latent.dtype)
        quadratic = jnp.einsum(
            "...d,de,...e->...", centered, precision.astype(latent.dtype), centered
        )
        return jnp.exp(-0.5 * quadratic)

    return omega


def _resolve_omega(selection) -> tuple[Callable[[Array], Array], str]:
    """Return ``(omega, convention)`` for a Gaussian or callable selection spec."""

    if isinstance(selection, GaussianSelection):
        return gaussian_selection_omega(selection), "on_z"
    if isinstance(selection, CallableSelection):
        if selection.convention != "on_z":
            raise NotImplementedError(
                "only the on_z selection convention is implemented (contract §17)"
            )
        return selection.omega, selection.convention
    raise TypeError(
        "selection must be a GaussianSelection or CallableSelection; "
        f"received {type(selection).__name__}"
    )


def _inverse_and_logdet(matrix: Array) -> tuple[Array, Array]:
    """Return ``(matrix^{-1}, logdet(matrix))`` for one SPD ``(D, D)`` matrix.

    Computed through the Cholesky factor (matching the kernel's numerical
    discipline); ``grad``-clean for SPD inputs.
    """

    dimension = matrix.shape[-1]
    factor = jnp.linalg.cholesky(matrix)
    identity = jnp.eye(dimension, dtype=matrix.dtype)
    inverse = jax.scipy.linalg.cho_solve((factor, True), identity)
    inverse = 0.5 * (inverse + jnp.swapaxes(inverse, -1, -2))
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(factor)))
    return inverse, log_det


def _gaussian_log_weights(
    params: ConvMMDParams, selection: GaussianSelection
) -> tuple[Array, Array, Array, Array]:
    """Return ``(pi', mu', Sigma', log_w)`` for the Gaussian-window transform (§17.5).

    Uses the ``|Psi|``-free precision-native form: with
    ``M_k = Sigma_k^{-1} + Psi^{-1} = Sigma'_k^{-1}``,
    ``b_k = Sigma_k^{-1} mu_k + Psi^{-1} a``, ``mu'_k = Sigma'_k b_k``,
    ``log_w_k = 1/2 (log|Sigma'_k| - log|Sigma_k|) - 1/2 C_k`` with
    ``C_k = mu_k^T Sigma_k^{-1} mu_k + a^T Psi^{-1} a - b_k^T Sigma'_k b_k``. This
    equals ``log w_k`` exactly (the ``(2 pi)^{D/2} |Psi|^{1/2}`` factors cancel) and
    is finite at ``Psi^{-1} = 0``. ``pi'_k = softmax_k(log pi_k + log_w_k)``.
    """

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    dtype = mus.dtype
    location = selection.location.astype(dtype)
    precision = selection.precision.astype(dtype)

    sigma_inverse, log_det_sigma = jax.vmap(_inverse_and_logdet)(sigmas)  # (K,D,D),(K,)
    precision_mean = precision + sigma_inverse  # M_k = Sigma'_k^{-1}, (K,D,D)
    sigma_prime, log_det_precision_mean = jax.vmap(_inverse_and_logdet)(precision_mean)
    log_det_sigma_prime = -log_det_precision_mean  # log|Sigma'_k|

    psi_a = precision @ location  # (D,)
    b = jnp.einsum("kde,ke->kd", sigma_inverse, mus) + psi_a[None, :]  # (K,D)
    mu_prime = jnp.einsum("kde,ke->kd", sigma_prime, b)  # (K,D)

    quad_mu = jnp.einsum("kd,kde,ke->k", mus, sigma_inverse, mus)  # mu^T Sinv mu
    quad_a = location @ psi_a  # a^T Psi^{-1} a, scalar
    quad_b = jnp.einsum("kd,kde,ke->k", b, sigma_prime, b)  # b^T Sigma' b
    c_k = quad_mu + quad_a - quad_b
    log_w = 0.5 * (log_det_sigma_prime - log_det_sigma) - 0.5 * c_k  # (K,)

    log_weights_prime = jax.nn.log_softmax(jnp.log(pis) + log_w)
    weights_prime = jnp.exp(log_weights_prime)
    return weights_prime, mu_prime, sigma_prime, log_w


def select_gaussian_params(
    params: ConvMMDParams, selection: GaussianSelection
) -> ConvMMDParams:
    """Differentiable Gaussian-window transform ``(pi,mu,Sigma) -> (pi',mu',Sigma')``.

    The selected latent mixture ``Omega(z) q_theta(z) / Z_theta`` (§17.5). At
    ``precision = 0`` it returns ``params`` unchanged (up to floating error). Pure
    and ``grad``-able; reads no PRNG key.
    """

    weights_prime, mu_prime, sigma_prime, _ = _gaussian_log_weights(params, selection)
    return ConvMMDParams(
        weights=weights_prime, means=mu_prime, covariances=sigma_prime
    )


def effective_volume_gaussian(
    params: ConvMMDParams, selection: GaussianSelection
) -> Array:
    """Effective volume ``Z_theta = int Omega(z) q_theta(z) dz`` (a diagnostic, §17.7).

    ``Z_theta = sum_k pi_k exp(log_w_k)`` using the same ``|Psi|``-free ``log_w_k``;
    well-conditioned everywhere, and ``-> 1`` as ``Omega -> 1`` (``precision -> 0``).
    Never needed by the loss or denoiser.
    """

    _, _, _, log_w = _gaussian_log_weights(params, selection)
    return jnp.sum(params.weights * jnp.exp(log_w))


# ---------------------------------------------------------------------------
# Grouped selected analytic operators (§17.5): transform then reuse §16 leaves
# ---------------------------------------------------------------------------


def grouped_analytic_selected_loss(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    selection: GaussianSelection,
) -> Array:
    """Informative-weight-normalized selected analytic loss over a fixed group set.

    Transforms ``params`` under the Gaussian window, then evaluates the §16 grouped
    analytic loss on the selected mixture. Differentiable in ``params`` through the
    transform (the group structure is fixed and param-value-independent).
    """

    return grouped_analytic_loss(
        select_gaussian_params(params, selection), grouped, bandwidths
    )


def grouped_denoise_selected(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    selection: GaussianSelection,
) -> Array:
    """Selection-aware projected posterior mean ``(N, D)`` (§17.6, Gaussian ``Omega(z)``)."""

    return grouped_denoise(select_gaussian_params(params, selection), grouped)


def grouped_posterior_components_selected(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    selection: GaussianSelection,
) -> PosteriorComponents:
    """Selection-aware responsibilities ``(N,K)`` and full-``D`` component means ``(N,K,D)``."""

    return grouped_posterior_components(
        select_gaussian_params(params, selection), grouped
    )


# ---------------------------------------------------------------------------
# One-shot selected operations (group + transform + evaluate)
# ---------------------------------------------------------------------------


def convmmd_loss_analytic_selected(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    bandwidths,
    selection: GaussianSelection,
    sample_weight=None,
    dtype,
) -> Array:
    """One-shot Gaussian-window selected analytic loss (§17.5)."""

    selected = select_gaussian_params(params, selection)
    return convmmd_loss_analytic_masked(
        selected,
        observations,
        observed_mask,
        noise=noise,
        bandwidths=bandwidths,
        sample_weight=sample_weight,
        dtype=dtype,
    )


def convmmd_denoise_selected(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    selection: GaussianSelection,
    dtype,
) -> Array:
    """One-shot selection-aware posterior mean ``(N, D)`` (§17.6)."""

    selected = select_gaussian_params(params, selection)
    return convmmd_denoise_masked(
        selected, observations, observed_mask, noise=noise, dtype=dtype
    )


def convmmd_posterior_components_selected(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    selection: GaussianSelection,
    dtype,
) -> PosteriorComponents:
    """One-shot selection-aware responsibilities and full-``D`` component means (§17.6)."""

    selected = select_gaussian_params(params, selection)
    return convmmd_posterior_components_masked(
        selected, observations, observed_mask, noise=noise, dtype=dtype
    )


# ---------------------------------------------------------------------------
# General-Omega SNIS Monte-Carlo path (§17.4): draw un-selected, weight by omega
# ---------------------------------------------------------------------------


def convmmd_loss_snis_projected(
    params: ConvMMDParams,
    observations: Array,
    projection: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    key: Array,
    num_samples: int,
    omega: Callable[[Array], Array],
) -> Array:
    """Per-row SNIS selected Monte-Carlo loss for one fixed-``M`` group (§17.4).

    Draws two independent reparameterized sets from the **un-selected** model,
    weights each latent draw by ``omega`` (convention ``on_z``), and forms the
    self-normalized cross/self terms with the per-group normalizers ``Zhat``. The
    self term is the full cross-set double sum. Returns the per-row loss averaged
    over scales, shape ``(n,)``. Requires one explicit key and a static
    ``num_samples``; **biased at finite ``num_samples``, consistent as it grows**.
    """

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    dtype = mus.dtype
    dimension = mus.shape[1]
    n_samples = observations.shape[0]
    m_dim = observations.shape[1]

    latent_factor = jnp.linalg.cholesky(sigmas)  # (K, D, D)
    noise_factor = jnp.linalg.cholesky(measurement_covariances)  # (n, M', M')

    k1, k2, k3, k4 = jax.random.split(key, 4)
    z1 = jax.random.normal(k1, (num_samples, dimension), dtype=dtype)
    z2 = jax.random.normal(k2, (num_samples, dimension), dtype=dtype)
    latent_1 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z1)  # (K,M,D)
    latent_2 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z2)
    weight_1 = omega(latent_1).astype(dtype)  # (K, M) in [0, 1]
    weight_2 = omega(latent_2).astype(dtype)

    proj_1 = jnp.einsum("npd,kmd->nkmp", projection, latent_1)  # (n, K, M, M')
    proj_2 = jnp.einsum("npd,kmd->nkmp", projection, latent_2)
    e1 = jax.random.normal(k3, (n_samples, num_samples, m_dim), dtype=dtype)
    e2 = jax.random.normal(k4, (n_samples, num_samples, m_dim), dtype=dtype)
    noise_1 = jnp.einsum("npq,nmq->nmp", noise_factor, e1)  # (n, M, M')
    noise_2 = jnp.einsum("npq,nmq->nmp", noise_factor, e2)
    tilde_1 = proj_1 + noise_1[:, None, :, :]  # (n, K, M, M')
    tilde_2 = proj_2 + noise_2[:, None, :, :]

    z_hat_1 = jnp.sum(pis * jnp.mean(weight_1, axis=1))  # scalar Zhat (on_z)
    z_hat_2 = jnp.sum(pis * jnp.mean(weight_2, axis=1))

    diff_cross = tilde_1 - observations[:, None, None, :]
    dist_cross = jnp.sum(diff_cross * diff_cross, axis=-1)  # (n, K, M)
    sq_1 = jnp.sum(tilde_1 * tilde_1, axis=-1)  # (n, K, M)
    sq_2 = jnp.sum(tilde_2 * tilde_2, axis=-1)
    gram = jnp.einsum("nkma,nlpa->nklmp", tilde_1, tilde_2)  # (n, K, K, M, M)
    dist_self = sq_1[:, :, None, :, None] + sq_2[:, None, :, None, :] - 2.0 * gram
    dist_self = jnp.maximum(dist_self, 0.0)  # guard tiny negatives from cancellation

    def per_scale(gamma: Array) -> Array:
        gamma_sq_2 = (2.0 * gamma * gamma).astype(dtype)
        kernel_cross = jnp.exp(-dist_cross / gamma_sq_2)  # (n, K, M)
        cross_num = jnp.sum(
            pis[None, :] * jnp.mean(weight_1[None, :, :] * kernel_cross, axis=2),
            axis=1,
        )  # (n,)
        cross = cross_num / z_hat_1
        kernel_self = jnp.exp(-dist_self / gamma_sq_2)  # (n, K, K, M, M)
        weight_outer = (
            weight_1[None, :, None, :, None] * weight_2[None, None, :, None, :]
        )  # (1, K, K, M, M)
        self_kk = jnp.mean(weight_outer * kernel_self, axis=(3, 4))  # (n, K, K)
        self_num = jnp.sum(
            pis[None, :, None] * pis[None, None, :] * self_kk, axis=(1, 2)
        )  # (n,)
        self_term = self_num / (z_hat_1 * z_hat_2)
        return self_term - 2.0 * cross  # (n,)

    per_scale_losses = jax.vmap(per_scale)(bandwidths)  # (G, n)
    return jnp.mean(per_scale_losses, axis=0)  # (n,)


def snis_diagnostics_latent(
    params: ConvMMDParams,
    key: Array,
    num_samples: int,
    omega: Callable[[Array], Array],
) -> tuple[Array, Array]:
    """Return ``(ESS, Zhat)`` from un-selected latent draws (§17.7 diagnostics).

    ``ESS = (sum u)^2 / sum u^2`` with ``u_{k,m} = pi_k omega(z_k^{(m)})`` over the
    ``K * num_samples`` proposal draws (range ``(0, K * num_samples]``); ``Zhat`` is
    the SNIS effective-volume estimate. Cheap: no projection, noise, or kernels.
    """

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    dtype = mus.dtype
    dimension = mus.shape[1]
    factor = jnp.linalg.cholesky(sigmas)
    z = jax.random.normal(key, (num_samples, dimension), dtype=dtype)
    latent = mus[:, None, :] + jnp.einsum("kab,mb->kma", factor, z)  # (K, M, D)
    weight = omega(latent).astype(dtype)  # (K, M)
    z_hat = jnp.sum(pis * jnp.mean(weight, axis=1))
    pooled = pis[:, None] * weight  # (K, M)
    effective_sample_size = jnp.sum(pooled) ** 2 / jnp.sum(pooled * pooled)
    return effective_sample_size, z_hat


def grouped_snis_loss(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    key: Array,
    num_samples: int,
    selection,
) -> Array:
    """Informative-weight-normalized SNIS selected Monte-Carlo loss, scalar (§17.4).

    ``params`` are the **un-selected** parameters (SNIS weights samples by ``omega``;
    it does not transform the mixture). One explicit ``key`` is split across groups.
    """

    omega, _ = _resolve_omega(selection)
    dtype = params.means.dtype
    informative = _informative_groups(grouped)
    if not informative:
        return jnp.asarray(0.0, dtype=dtype)
    informative_weight = jnp.asarray(grouped.informative_weight, dtype=dtype)
    keys = jax.random.split(key, max(len(grouped.groups), 1))
    total = jnp.asarray(0.0, dtype=dtype)
    for index, group in informative:
        per_row = convmmd_loss_snis_projected(
            params,
            group.observations,
            group.projection_matrices,
            group.measurement_covariances,
            bandwidths,
            keys[index],
            num_samples,
            omega,
        )
        total = total + jnp.sum(group.sample_weight.astype(dtype) * per_row)
    return _normalize_by_informative_weight(total, informative_weight)


def convmmd_loss_snis_selected(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    bandwidths,
    selection,
    key: Array,
    num_samples: int,
    sample_weight=None,
    dtype,
) -> Array:
    """One-shot general-``Omega`` SNIS selected Monte-Carlo loss (§17.4)."""

    grouped = group_masked_inputs(
        params,
        observations,
        observed_mask,
        noise=noise,
        sample_weight=sample_weight,
        dtype=dtype,
    )
    canonical = grouped.parameters
    return grouped_snis_loss(
        ConvMMDParams(canonical.weights, canonical.means, canonical.covariances),
        grouped,
        jnp.asarray(bandwidths, dtype=canonical.means.dtype),
        key,
        num_samples,
        selection,
    )


def snis_effective_sample_size(
    params: ConvMMDParams, selection, key: Array, num_samples: int
) -> tuple[Array, Array]:
    """One-shot ``(ESS, Zhat)`` diagnostic for a selection spec (§17.7)."""

    omega, _ = _resolve_omega(selection)
    return snis_diagnostics_latent(params, key, num_samples, omega)


# ---------------------------------------------------------------------------
# General-Omega SNIS denoiser (§17.6): sample the MAR posterior, weight by omega
# ---------------------------------------------------------------------------


def _mar_posterior_components_full(
    params: ConvMMDParams,
    observations: Array,
    projection: Array,
    measurement_covariances: Array,
) -> tuple[Array, Array, Array]:
    """Projected MAR posterior components of the **un-selected** prior, per row.

    Returns ``(responsibilities (n,K), component_means (n,K,D), component_covs
    (n,K,D,D))`` of ``p_MAR(z | x) = sum_k r_k N(z; m_k, C_k)`` with
    ``r_k ~ pi_k N(x; P mu_k, B_k)``,
    ``m_k = mu_k + Sigma_k P^T B_k^{-1}(x - P mu_k)``,
    ``C_k = Sigma_k - Sigma_k P^T B_k^{-1} P Sigma_k``, ``B_k = P Sigma_k P^T + S``.
    This is the closed-form Gaussian conditional used as the SNIS proposal (§17.6).
    """

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    dtype = mus.dtype
    m_dim = observations.shape[1]
    log_pis = jnp.log(pis)

    def per_row(x_row: Array, p_row: Array, s_row: Array):
        proj_mu = jnp.einsum("md,kd->km", p_row, mus)  # (K, M)
        proj_sigma = jnp.einsum("md,kde,ne->kmn", p_row, sigmas, p_row)  # (K, M, M)
        marginal = proj_sigma + s_row[None, :, :]  # B_k, (K, M, M)
        delta = x_row[None, :] - proj_mu  # (K, M)
        factor = jnp.linalg.cholesky(marginal)  # (K, M, M)
        whitened = jax.scipy.linalg.solve_triangular(
            factor, delta[..., None], lower=True
        )[..., 0]  # (K, M)
        log_det = 2.0 * jnp.sum(jnp.log(_diagonal_last2(factor)), axis=-1)  # (K,)
        log_norm = -0.5 * (
            m_dim * jnp.log(2.0 * jnp.pi).astype(dtype)
            + log_det
            + jnp.sum(whitened * whitened, axis=-1)
        )
        responsibilities = jax.nn.softmax(log_pis + log_norm)  # (K,)
        cross_cov = jnp.einsum("kde,me->kdm", sigmas, p_row)  # Sigma_k P^T, (K, D, M)
        gain_delta = jnp.linalg.solve(marginal, delta[..., None])[..., 0]  # (K, M)
        component_means = mus + jnp.einsum("kdm,km->kd", cross_cov, gain_delta)  # (K,D)
        # C_k = Sigma_k - (Sigma_k P^T) B_k^{-1} (P Sigma_k)
        proj_sigma_full = jnp.swapaxes(cross_cov, -1, -2)  # P Sigma_k, (K, M, D)
        marginal_solve = jnp.linalg.solve(marginal, proj_sigma_full)  # (K, M, D)
        component_covs = sigmas - jnp.einsum("kdm,kme->kde", cross_cov, marginal_solve)
        component_covs = 0.5 * (component_covs + jnp.swapaxes(component_covs, -1, -2))
        return responsibilities, component_means, component_covs

    return jax.vmap(per_row)(observations, projection, measurement_covariances)


def _diagonal_last2(matrices: Array) -> Array:
    return jnp.diagonal(matrices, axis1=-2, axis2=-1)


def _sample_mixture(
    key: Array,
    log_weights: Array,
    means: Array,
    chol_covs: Array,
    num_samples: int,
) -> Array:
    """Draw ``num_samples`` from a Gaussian mixture ``sum_k w_k N(means_k, cov_k)``.

    ``log_weights (K,)``, ``means (K, D)``, ``chol_covs (K, D, D)``; returns
    ``(num_samples, D)``. Component choice is categorical (non-differentiable), which
    is acceptable for an evaluation-time posterior-mean estimator.
    """

    dimension = means.shape[-1]
    dtype = means.dtype
    key_component, key_normal = jax.random.split(key)
    component = jax.random.categorical(key_component, log_weights, shape=(num_samples,))
    chosen_means = means[component]  # (M, D)
    chosen_factors = chol_covs[component]  # (M, D, D)
    standard = jax.random.normal(key_normal, (num_samples, dimension), dtype=dtype)
    return chosen_means + jnp.einsum("mde,me->md", chosen_factors, standard)


def _snis_weighted_mean(samples: Array, weights: Array) -> Array:
    """Self-normalized weighted mean ``sum_m w_m z_m / sum_m w_m`` over axis 0."""

    return jnp.sum(weights[:, None] * samples, axis=0) / jnp.sum(weights)


def snis_denoise_projected(
    params: ConvMMDParams,
    observations: Array,
    projection: Array,
    measurement_covariances: Array,
    key: Array,
    num_samples: int,
    omega: Callable[[Array], Array],
) -> Array:
    """Per-row general-``Omega`` SNIS posterior mean for one fixed-``M`` group (§17.6).

    Samples the MAR posterior (proposal), weights each draw by ``omega`` (``on_z``),
    and returns the self-normalized weighted mean, full ``D``, shape ``(n, D)``.
    Evaluation-time estimator (consistent as ``num_samples -> inf``); the categorical
    component choice makes it non-differentiable, as for any sampling denoiser.
    """

    dtype = params.means.dtype
    responsibilities, means, covs = _mar_posterior_components_full(
        params, observations, projection, measurement_covariances
    )
    chol_covs = jnp.linalg.cholesky(covs)  # (n, K, D, D)
    keys = jax.random.split(key, observations.shape[0])

    def per_row(r_row, m_row, l_row, key_row):
        samples = _sample_mixture(key_row, jnp.log(r_row), m_row, l_row, num_samples)
        weights = omega(samples).astype(dtype)
        return _snis_weighted_mean(samples, weights)

    return jax.vmap(per_row)(responsibilities, means, chol_covs, keys)


def _snis_selected_prior_mean(
    params: ConvMMDParams, key: Array, num_samples: int, omega: Callable[[Array], Array]
) -> Array:
    """Selected prior mean ``E_{Omega q}[z] / E_{Omega q}[Omega]`` via SNIS, ``(D,)``.

    The ``M=0`` posterior under ``Omega(z)`` selection (no observation): proposal is
    the prior ``q_theta`` itself.
    """

    dtype = params.means.dtype
    chol_covs = jnp.linalg.cholesky(params.covariances)
    samples = _sample_mixture(
        key, jnp.log(params.weights), params.means, chol_covs, num_samples
    )
    weights = omega(samples).astype(dtype)
    return _snis_weighted_mean(samples, weights)


def grouped_snis_denoise(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    key: Array,
    num_samples: int,
    selection,
) -> Array:
    """Restored general-``Omega`` SNIS posterior mean ``(N, D)`` in row order (§17.6).

    ``M=0`` rows take the SNIS **selected prior mean**. ``params`` are the un-selected
    parameters (the estimator samples the MAR posterior and weights by ``omega``).
    """

    omega, _ = _resolve_omega(selection)
    dimension = params.means.shape[1]
    keys = jax.random.split(key, max(len(grouped.groups), 1))
    mean_groups: list[Array] = []
    for index, group in enumerate(grouped.groups):
        n_rows = len(group.original_indices)
        if group.observations.shape[-1] == 0:
            prior_mean = _snis_selected_prior_mean(
                params, keys[index], num_samples, omega
            )
            mean_groups.append(jnp.broadcast_to(prior_mean, (n_rows, dimension)))
        else:
            mean_groups.append(
                snis_denoise_projected(
                    params,
                    group.observations,
                    group.projection_matrices,
                    group.measurement_covariances,
                    keys[index],
                    num_samples,
                    omega,
                )
            )
    return restore_grouped_rows(grouped, mean_groups, field="selected SNIS denoised means")


def convmmd_snis_denoise_selected(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    selection,
    key: Array,
    num_samples: int,
    dtype,
) -> Array:
    """One-shot general-``Omega`` SNIS posterior mean ``(N, D)`` (§17.6)."""

    grouped = group_masked_inputs(
        params, observations, observed_mask, noise=noise, dtype=dtype
    )
    canonical = grouped.parameters
    return grouped_snis_denoise(
        ConvMMDParams(canonical.weights, canonical.means, canonical.covariances),
        grouped,
        key,
        num_samples,
        selection,
    )


# ---------------------------------------------------------------------------
# Selection on the OBSERVED value Omega(x-tilde) (§17.11; homoscedastic, fully
# observed). The Gaussian window acts on the noise-CONVOLVED model, so the transform
# is the same Gaussian-product identity applied to the inflated covariances Sigma+S.
# ---------------------------------------------------------------------------


def _homoscedastic_noise_matrix(noise, dimension, dtype) -> Array:
    """Coerce a homoscedastic noise argument to a single ``(D, D)`` matrix.

    Accepts ``(D, D)`` or ``(N, D, D)`` with identical rows (the homoscedastic
    restriction of §17.11); rejects a genuinely heteroscedastic ``(N, D, D)``.
    """

    array = jnp.asarray(noise, dtype=dtype)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        first = array[0]
        # Host-side check (validation is outside the JIT/autodiff contract).
        if not bool(jnp.all(jnp.abs(array - first[None]) <= 1e-12 * (1.0 + jnp.abs(first[None])))):
            raise ValueError(
                "Omega(x-tilde) selection requires homoscedastic noise (§17.11); the "
                "per-item covariances differ. Use Omega(z) for heteroscedastic noise."
            )
        return first
    raise ValueError("noise must have shape (D, D) or (N, D, D)")


def select_gaussian_params_observed(
    params: ConvMMDParams, selection: GaussianSelection, noise
) -> ConvMMDParams:
    """Gaussian-window transform for selection on the OBSERVED value (§17.11).

    ``Omega(x-tilde)`` acts on the noise-convolved model
    ``sum_k pi_k N(mu_k, Sigma_k + S)``, so the selected observed model is the
    §17.5 transform applied to the **inflated** covariances ``Sigma_k + S`` (with the
    original means): ``(pi'', nu'', B'')``. Homoscedastic ``S`` only.
    """

    dtype = params.means.dtype
    dimension = params.means.shape[1]
    noise_matrix = _homoscedastic_noise_matrix(noise, dimension, dtype)
    inflated = ConvMMDParams(
        weights=params.weights,
        means=params.means,
        covariances=params.covariances + noise_matrix[None, :, :],
    )
    return select_gaussian_params(inflated, selection)


def effective_volume_gaussian_observed(
    params: ConvMMDParams, selection: GaussianSelection, noise
) -> Array:
    """Effective volume ``Z_theta = int Omega(x-tilde) p_tilde_theta(x-tilde) dx-tilde``
    for the observed-value convention (a diagnostic, §17.7)."""

    dtype = params.means.dtype
    dimension = params.means.shape[1]
    noise_matrix = _homoscedastic_noise_matrix(noise, dimension, dtype)
    inflated = ConvMMDParams(
        params.weights, params.means, params.covariances + noise_matrix[None, :, :]
    )
    return effective_volume_gaussian(inflated, selection)


def convmmd_loss_analytic_selected_observed(
    params: ConvMMDParams,
    observations,
    *,
    noise,
    bandwidths,
    selection: GaussianSelection,
) -> Array:
    """Analytic ``Omega(x-tilde)`` selected loss (§17.11; homoscedastic, fully observed).

    The selected observed model ``sum_k pi'' N(nu'', B'')`` is a Gaussian mixture in
    the observed space, so the loss is the base §4 loss on ``(pi'', nu'', B'')`` with
    **zero** measurement noise (model and data are both in observed space). Fully
    differentiable in ``params`` (no host grouping); fully observed only.
    """

    observed = jnp.asarray(observations, dtype=params.means.dtype)
    observed_params = select_gaussian_params_observed(params, selection, noise)
    n_samples = observed.shape[0]
    dimension = observed.shape[1]
    zero_noise = jnp.zeros(
        (n_samples, dimension, dimension), dtype=observed_params.means.dtype
    )
    return convmmd_loss_analytic(
        observed_params, observed, zero_noise,
        jnp.asarray(bandwidths, dtype=observed_params.means.dtype),
    )


def convmmd_denoise_selected_observed(
    params: ConvMMDParams, observations, *, noise
) -> Array:
    """Selection-aware denoiser for ``Omega(x-tilde)`` = the base MAR posterior (§17.6).

    ``P(det | z, x-tilde_i) = Omega(x-tilde_i)`` is constant in ``z`` and cancels, so
    the per-object posterior is the unchanged empirical-Bayes posterior mean. This
    wrapper documents that reduction; ``selection`` is not needed.
    """

    dtype = params.means.dtype
    observed = jnp.asarray(observations, dtype=dtype)
    dimension = observed.shape[1]
    noise_matrix = _homoscedastic_noise_matrix(noise, dimension, dtype)
    noise_full = jnp.broadcast_to(
        noise_matrix, (observed.shape[0], dimension, dimension)
    )
    return denoise(params, observed, noise_full)


def convmmd_loss_snis_selected_observed(
    params: ConvMMDParams,
    observations,
    *,
    noise,
    bandwidths,
    selection,
    key: Array,
    num_samples: int,
) -> Array:
    """General-``Omega(x-tilde)`` SNIS Monte-Carlo loss (§17.11; homoscedastic, fully
    observed). Draw ``x-tilde = z + eps`` from the observed model, weight by
    ``Omega(x-tilde)``; two independent sets, self-normalized. Biased at finite
    ``num_samples``, consistent as it grows.
    """

    omega, _ = _resolve_omega(selection)
    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    dtype = mus.dtype
    dimension = mus.shape[1]
    observed = jnp.asarray(observations, dtype=dtype)
    noise_matrix = _homoscedastic_noise_matrix(noise, dimension, dtype)

    latent_factor = jnp.linalg.cholesky(sigmas)  # (K, D, D)
    noise_factor = jnp.linalg.cholesky(noise_matrix)  # (D, D)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    z1 = jax.random.normal(k1, (num_samples, dimension), dtype=dtype)
    z2 = jax.random.normal(k2, (num_samples, dimension), dtype=dtype)
    e1 = jax.random.normal(k3, (num_samples, dimension), dtype=dtype)
    e2 = jax.random.normal(k4, (num_samples, dimension), dtype=dtype)
    latent_1 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z1)
    latent_2 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z2)
    tilde_1 = latent_1 + (e1 @ noise_factor.T)[None, :, :]  # (K, M, D)
    tilde_2 = latent_2 + (e2 @ noise_factor.T)[None, :, :]
    weight_1 = omega(tilde_1).astype(dtype)  # (K, M)
    weight_2 = omega(tilde_2).astype(dtype)
    z_hat_1 = jnp.sum(pis * jnp.mean(weight_1, axis=1))
    z_hat_2 = jnp.sum(pis * jnp.mean(weight_2, axis=1))

    diff_cross = tilde_1[None, :, :, :] - observed[:, None, None, :]  # (N, K, M, D)
    dist_cross = jnp.sum(diff_cross * diff_cross, axis=-1)  # (N, K, M)
    sq_1 = jnp.sum(tilde_1 * tilde_1, axis=-1)  # (K, M)
    sq_2 = jnp.sum(tilde_2 * tilde_2, axis=-1)
    gram = jnp.einsum("kma,lpa->klmp", tilde_1, tilde_2)  # (K, K, M, M)
    dist_self = sq_1[:, None, :, None] + sq_2[None, :, None, :] - 2.0 * gram
    dist_self = jnp.maximum(dist_self, 0.0)

    def per_scale(gamma: Array) -> Array:
        gamma_sq_2 = (2.0 * gamma * gamma).astype(dtype)
        kernel_cross = jnp.exp(-dist_cross / gamma_sq_2)  # (N, K, M)
        cross = jnp.sum(
            pis[None, :] * jnp.mean(weight_1[None, :, :] * kernel_cross, axis=2), axis=1
        ) / z_hat_1  # (N,)
        kernel_self = jnp.exp(-dist_self / gamma_sq_2)  # (K, K, M, M)
        weight_outer = weight_1[:, None, :, None] * weight_2[None, :, None, :]
        self_kk = jnp.mean(weight_outer * kernel_self, axis=(2, 3))  # (K, K)
        self_term = jnp.sum(pis[:, None] * pis[None, :] * self_kk) / (z_hat_1 * z_hat_2)
        return self_term - 2.0 * cross  # (N,)

    per_scale_losses = jax.vmap(per_scale)(bandwidths)  # (G, N)
    return jnp.mean(per_scale_losses)


# ---------------------------------------------------------------------------
# Heteroscedastic Omega(x-tilde) (§17.13): per-object transform on Sigma_k + S_i.
# A detected object's own S_i is KNOWN, so the §17.5 identity applies per object and
# the selected observed density conditional on S_i is an exact Gaussian mixture. The
# loss is the sum of per-object one-sample discrepancies; NO noise model is required
# (the contrast with pygmmis, whose EM imputation needs a covar_callback). Fully
# observed only; genuinely heteroscedastic (N, D, D) noise is accepted.
# ---------------------------------------------------------------------------


def select_gaussian_params_observed_hetero(
    params: ConvMMDParams, selection: GaussianSelection, noise
):
    """Per-object ``Omega(x-tilde)`` transform for HETEROSCEDASTIC noise (§17.13).

    For each object ``i``, apply the §17.5 transform to the inflated covariances
    ``Sigma_k + S_i`` (original means), giving that object's selected observed mixture.
    Returns stacked per-object ``(pi'', nu'', B'')`` of shapes ``(N, K)``, ``(N, K, D)``,
    ``(N, K, D, D)``. Accepts genuinely per-item ``S_i`` (unlike §17.12's homoscedastic
    :func:`select_gaussian_params_observed`); vmaps over the observation batch.
    """

    if not isinstance(selection, GaussianSelection):
        raise TypeError(
            "the heteroscedastic Omega(x-tilde) analytic path requires a "
            "GaussianSelection (§17.13)"
        )
    dtype = params.means.dtype
    per_item_noise = jnp.asarray(noise, dtype=dtype)  # (N, D, D)
    pis, mus, sigmas = params.weights, params.means, params.covariances

    def per_object(s_row):
        inflated = ConvMMDParams(pis, mus, sigmas + s_row[None, :, :])
        weights_prime, mu_prime, sigma_prime, _ = _gaussian_log_weights(
            inflated, selection
        )
        return weights_prime, mu_prime, sigma_prime

    return jax.vmap(per_object)(per_item_noise)


def convmmd_loss_analytic_selected_observed_hetero(
    params: ConvMMDParams,
    observations,
    *,
    noise,
    bandwidths,
    selection: GaussianSelection,
) -> Array:
    """Heteroscedastic ``Omega(x-tilde)`` per-object analytic loss (§17.13; exact).

    For each detected object ``i`` with its own known ``S_i``, the §17.5 transform on
    ``Sigma_k + S_i`` gives the per-object selected observed mixture
    ``(pi''_{k,i}, nu''_{k,i}, B''_{k,i})``; the loss is the mean over objects of the
    base §4 one-sample discrepancy of ``x_i`` against that mixture with **zero**
    residual noise (the noise is baked into ``B''``). Fully differentiable in ``params``
    through the per-object transform; fully observed only. **Reductions:** all ``S_i``
    equal reproduces §17.12; ``Psi^{-1} = 0`` reproduces the base §4 loss with per-item
    noise ``S_i``. No noise model.
    """

    if not isinstance(selection, GaussianSelection):
        raise TypeError(
            "the heteroscedastic Omega(x-tilde) analytic path requires a "
            "GaussianSelection (§17.13)"
        )
    dtype = params.means.dtype
    observed = jnp.asarray(observations, dtype=dtype)  # (N, D)
    per_item_noise = jnp.asarray(noise, dtype=dtype)  # (N, D, D)
    pis, mus, sigmas = params.weights, params.means, params.covariances
    scales = jnp.asarray(bandwidths, dtype=dtype)

    def per_object(x_row: Array, s_row: Array) -> Array:
        inflated = ConvMMDParams(pis, mus, sigmas + s_row[None, :, :])
        pipp, nupp, bpp, _ = _gaussian_log_weights(inflated, selection)  # (K,),(K,D),(K,D,D)
        mean_difference = nupp[:, None, :] - nupp[None, :, :]  # (K, K, D)
        self_omega = bpp[:, None, :, :] + bpp[None, :, :, :]  # (K, K, D, D)
        cross_delta = x_row[None, :] - nupp  # (K, D)
        weight_outer = pipp[:, None] * pipp[None, :]  # (K, K)

        def per_scale(gamma: Array) -> Array:
            self_kernel = expected_rbf_kernel(mean_difference, self_omega, gamma)  # (K,K)
            self_term = jnp.sum(weight_outer * self_kernel)
            cross_kernel = expected_rbf_kernel(cross_delta, bpp, gamma)  # (K,)
            cross_term = jnp.sum(pipp * cross_kernel)
            return self_term - 2.0 * cross_term

        return jnp.mean(jax.vmap(per_scale)(scales))

    return jnp.mean(jax.vmap(per_object)(observed, per_item_noise))


def convmmd_denoise_selected_observed_hetero(
    params: ConvMMDParams, observations, *, noise
) -> Array:
    """Heteroscedastic ``Omega(x-tilde)`` denoiser = base §7 MAR posterior mean (§17.13).

    ``Omega(x-tilde_i)`` is constant in ``z`` and cancels, so a *detected* object's
    posterior mean is the empirical-Bayes posterior at its **own known** ``S_i``.
    Accepts genuinely per-item ``(N, D, D)`` noise; needs no selection spec and no noise
    model.
    """

    dtype = params.means.dtype
    observed = jnp.asarray(observations, dtype=dtype)
    per_item_noise = jnp.asarray(noise, dtype=dtype)  # (N, D, D)
    return denoise(params, observed, per_item_noise)


def convmmd_loss_snis_selected_observed_hetero(
    params: ConvMMDParams,
    observations,
    *,
    noise,
    bandwidths,
    selection,
    key: Array,
    num_samples: int,
) -> Array:
    """General-``Omega(x-tilde)`` per-object SNIS loss for heteroscedastic noise (§17.13).

    For each object ``i`` draw ``z ~ q_theta`` and ``eps ~ N(0, S_i)`` with the object's
    **known** ``S_i``, form ``x-tilde = z + eps``, weight by ``Omega(x-tilde)``, and take
    the §17.4 self-normalized cross/self terms with the per-object normalizer
    ``Zhat_i`` (two independent sets, full double sum). Its expectation limit is
    :func:`convmmd_loss_analytic_selected_observed_hetero` for a Gaussian ``Omega``.
    **Biased at finite ``num_samples``, consistent as it grows.** One explicit key
    (split across objects); static ``num_samples``; **MUST NOT** read a global key. No
    noise model.
    """

    omega, _ = _resolve_omega(selection)
    dtype = params.means.dtype
    observed = jnp.asarray(observations, dtype=dtype)  # (N, D)
    per_item_noise = jnp.asarray(noise, dtype=dtype)  # (N, D, D)
    pis, mus, sigmas = params.weights, params.means, params.covariances
    scales = jnp.asarray(bandwidths, dtype=dtype)
    dimension = mus.shape[1]
    latent_factor = jnp.linalg.cholesky(sigmas)  # (K, D, D)
    keys = jax.random.split(key, observed.shape[0])

    def per_object(x_row: Array, s_row: Array, key_row: Array) -> Array:
        noise_factor = jnp.linalg.cholesky(s_row)  # (D, D)
        k1, k2, k3, k4 = jax.random.split(key_row, 4)
        z1 = jax.random.normal(k1, (num_samples, dimension), dtype=dtype)
        z2 = jax.random.normal(k2, (num_samples, dimension), dtype=dtype)
        e1 = jax.random.normal(k3, (num_samples, dimension), dtype=dtype)
        e2 = jax.random.normal(k4, (num_samples, dimension), dtype=dtype)
        latent_1 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z1)
        latent_2 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z2)
        tilde_1 = latent_1 + (e1 @ noise_factor.T)[None, :, :]  # (K, M, D)
        tilde_2 = latent_2 + (e2 @ noise_factor.T)[None, :, :]
        weight_1 = omega(tilde_1).astype(dtype)  # (K, M)
        weight_2 = omega(tilde_2).astype(dtype)
        z_hat_1 = jnp.sum(pis * jnp.mean(weight_1, axis=1))
        z_hat_2 = jnp.sum(pis * jnp.mean(weight_2, axis=1))

        diff_cross = tilde_1 - x_row[None, None, :]  # (K, M, D)
        dist_cross = jnp.sum(diff_cross * diff_cross, axis=-1)  # (K, M)
        sq_1 = jnp.sum(tilde_1 * tilde_1, axis=-1)  # (K, M)
        sq_2 = jnp.sum(tilde_2 * tilde_2, axis=-1)
        gram = jnp.einsum("kma,lpa->klmp", tilde_1, tilde_2)  # (K, K, M, M)
        dist_self = jnp.maximum(
            sq_1[:, None, :, None] + sq_2[None, :, None, :] - 2.0 * gram, 0.0
        )

        def per_scale(gamma: Array) -> Array:
            gamma_sq_2 = (2.0 * gamma * gamma).astype(dtype)
            kernel_cross = jnp.exp(-dist_cross / gamma_sq_2)  # (K, M)
            cross = jnp.sum(pis * jnp.mean(weight_1 * kernel_cross, axis=1)) / z_hat_1
            kernel_self = jnp.exp(-dist_self / gamma_sq_2)  # (K, K, M, M)
            weight_outer = weight_1[:, None, :, None] * weight_2[None, :, None, :]
            self_kk = jnp.mean(weight_outer * kernel_self, axis=(2, 3))  # (K, K)
            self_term = jnp.sum(pis[:, None] * pis[None, :] * self_kk) / (
                z_hat_1 * z_hat_2
            )
            return self_term - 2.0 * cross

        return jnp.mean(jax.vmap(per_scale)(scales))

    return jnp.mean(jax.vmap(per_object)(observed, per_item_noise, keys))


def snis_diagnostics_observed_hetero(
    params: ConvMMDParams, noise, key: Array, num_samples: int, selection
) -> tuple[Array, Array]:
    """Per-object ``(ESS, Zhat)`` for the heteroscedastic ``Omega(x-tilde)`` SNIS (§17.7).

    Cheap: draws ``z ~ q_theta`` and ``eps ~ N(0, S_i)`` per object, weights by
    ``Omega(z + eps)``; no kernels. ``ESS = (sum u)^2 / sum u^2`` with
    ``u = pi_k omega`` over the ``K * num_samples`` proposal draws; ``Zhat`` is the
    per-object SNIS effective-volume estimate. Both are diagnostics only.
    """

    omega, _ = _resolve_omega(selection)
    dtype = params.means.dtype
    per_item_noise = jnp.asarray(noise, dtype=dtype)  # (N, D, D)
    pis, mus, sigmas = params.weights, params.means, params.covariances
    dimension = mus.shape[1]
    latent_factor = jnp.linalg.cholesky(sigmas)
    keys = jax.random.split(key, per_item_noise.shape[0])

    def per_object(s_row: Array, key_row: Array) -> tuple[Array, Array]:
        noise_factor = jnp.linalg.cholesky(s_row)
        k1, k2 = jax.random.split(key_row, 2)
        z = jax.random.normal(k1, (num_samples, dimension), dtype=dtype)
        e = jax.random.normal(k2, (num_samples, dimension), dtype=dtype)
        latent = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z)
        tilde = latent + (e @ noise_factor.T)[None, :, :]
        weight = omega(tilde).astype(dtype)  # (K, M)
        z_hat = jnp.sum(pis * jnp.mean(weight, axis=1))
        pooled = pis[:, None] * weight
        effective_sample_size = jnp.sum(pooled) ** 2 / jnp.sum(pooled * pooled)
        return effective_sample_size, z_hat

    return jax.vmap(per_object)(per_item_noise, keys)


# ---------------------------------------------------------------------------
# Heteroscedastic Omega(x-tilde) COMPOSED with the §16 projection (MAR), §17.14.
# The window acts on the noise-convolved, PROJECTED model, so it must be
# MARGINALIZED onto each observed subspace R^{M_i}: the observed-subspace precision
# is (P_i Psi P_i^T)^{-1} -- the inverse of the COVARIANCE principal submatrix, NOT
# P_i Psi^{-1} P_i^T (the crux; they differ by a Schur complement). With that
# marginalized window the §17.5 identity applies per object to the projected inflated
# covariance P_i Sigma_k P_i^T + S_i, grouped by mask pattern. Reduces to §17.13 at
# P_i = I and to the base §16 masked loss at Psi^{-1} = 0. No noise model.
# ---------------------------------------------------------------------------


def _marginalized_selection(
    selection: GaussianSelection, coordinate_indices, dtype
) -> GaussianSelection:
    """Marginalize a Gaussian window onto the observed subspace ``C_i`` (§17.14).

    Returns the observed-subspace window ``(a_i, Psi_i_inv)`` with ``a_i = P_i a`` and
    ``Psi_i_inv = (P_i Psi P_i^T)^{-1} = (Psi[C_i, C_i])^{-1}`` -- the inverse of the
    **covariance** principal submatrix, **not** ``P_i Psi^{-1} P_i^T`` (they differ by a
    Schur complement). Computed **host-side in NumPy**: the window is a fixed compile-time
    constant, explicitly outside the JIT/autodiff contract (§17.14), so the branch on
    ``Psi^{-1}=0`` and the PD check are concrete Python booleans and the result is a
    constant fed into the traced leaf. The ``Psi^{-1} = 0`` (no selection) case is routed
    **without** inverting a singular precision -- the marginal of "no selection" is "no
    selection" (``Psi_i_inv = 0``). A rank-deficient ``Psi^{-1}`` (no covariance form
    ``Psi``) fails actionably at the eager boundary. Passing a **traced** ``selection``
    raises (the window must be static), consistent with host-side mask grouping.
    """

    if not isinstance(selection, GaussianSelection):
        raise TypeError(
            "the Omega(x-tilde)+MAR analytic path requires a GaussianSelection (§17.14)"
        )
    # Materialize the window on the host; the window carries no gradient and is not traced.
    location = np.asarray(jax.device_get(selection.location), dtype=np.float64)
    precision = np.asarray(jax.device_get(selection.precision), dtype=np.float64)
    coords = np.asarray([int(c) for c in coordinate_indices], dtype=np.intp)
    a_i = location[coords]  # (M,)
    m_dim = int(coords.size)
    if bool(np.all(precision == 0.0)):
        # Marginal of no selection is no selection; do not invert a singular Psi^{-1}.
        psi_i_inv = np.zeros((m_dim, m_dim), dtype=np.float64)
    else:
        # Materialize the window COVARIANCE Psi = (Psi^{-1})^{-1}; require Psi^{-1} PD so
        # Psi exists (a rank-deficient window precision has no covariance form).
        eigenvalues = np.linalg.eigvalsh(precision)
        if not bool(eigenvalues.min() > 0.0):
            raise ValueError(
                "Omega(x-tilde)+MAR requires a positive-definite window precision "
                "Psi^{-1} so the covariance Psi = (Psi^{-1})^{-1} exists to marginalize "
                "onto the observed subspace (§17.14); the supplied Psi^{-1} is "
                "rank-deficient. Use Psi^{-1}=0 for no selection, or Omega(z) selection "
                "for the projected case."
            )
        psi = np.linalg.inv(precision)  # Psi = (Psi^{-1})^{-1}
        psi = 0.5 * (psi + psi.T)
        psi_sub = psi[np.ix_(coords, coords)]  # (M, M) principal submatrix of COVARIANCE
        psi_i_inv = np.linalg.inv(psi_sub)
        psi_i_inv = 0.5 * (psi_i_inv + psi_i_inv.T)
    return GaussianSelection(
        jnp.asarray(a_i, dtype=dtype), jnp.asarray(psi_i_inv, dtype=dtype)
    )


def marginalize_gaussian_window(
    selection: GaussianSelection, coordinate_indices, *, dtype=None
) -> GaussianSelection:
    """Public: the observed-subspace window ``(a_i, Psi_i_inv)`` for a mask ``C_i`` (§17.14).

    Thin wrapper over :func:`_marginalized_selection` for the ``CMMD-SEL-OBSMAR-MARGWIN``
    gate: ``Psi_i_inv = (P_i Psi P_i^T)^{-1}``, the inverse of the covariance principal
    submatrix (**not** ``P_i Psi^{-1} P_i^T``). ``coordinate_indices`` is the ascending
    tuple of observed coordinates.
    """

    if dtype is None:
        dtype = jnp.asarray(selection.location).dtype
    return _marginalized_selection(selection, coordinate_indices, dtype)


def convmmd_loss_analytic_projected_selected_observed(
    params: ConvMMDParams,
    observations: Array,
    projection: Array,
    measurement_covariances: Array,
    marginal_selection: GaussianSelection,
    bandwidths: Array,
) -> Array:
    """Per-row analytic ``Omega(x-tilde)``+MAR loss for one fixed-``M`` group (§17.14).

    For each row (object) ``i`` the §17.5 transform is applied to the **projected,
    inflated** component ``(pi_k, P_i mu_k, B_k^{(i)})`` with ``B_k^{(i)} = P_i Sigma_k
    P_i^T + S_i`` and the group's shared **marginalized** window ``marginal_selection =
    (a_i, Psi_i_inv)``, giving that object's selected observed mixture ``(pi_{k,i},
    nu_{k,i}, C_{k,i})`` on ``R^{M_i}``; the per-row loss is the base §4 one-sample
    discrepancy of ``x_i`` against it with **zero** residual noise. Returns the per-row
    loss averaged over scales, shape ``(n,)`` -- the quantity the grouped objective
    weights and normalizes. Differentiable in ``params`` through the per-object transform
    (the marginalized window is a constant and carries no gradient).
    """

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    scales = bandwidths

    def per_row(x_row: Array, p_row: Array, s_row: Array) -> Array:
        proj_mu = jnp.einsum("md,kd->km", p_row, mus)  # (K, M)
        convolved = (
            jnp.einsum("md,kde,ne->kmn", p_row, sigmas, p_row) + s_row[None, :, :]
        )  # B_k^{(i)}, (K, M, M)
        row_params = ConvMMDParams(pis, proj_mu, convolved)
        pipp, nupp, cpp, _ = _gaussian_log_weights(
            row_params, marginal_selection
        )  # (K,), (K, M), (K, M, M)
        mean_difference = nupp[:, None, :] - nupp[None, :, :]  # (K, K, M)
        self_omega = cpp[:, None, :, :] + cpp[None, :, :, :]  # (K, K, M, M)
        cross_delta = x_row[None, :] - nupp  # (K, M)
        weight_outer = pipp[:, None] * pipp[None, :]  # (K, K)

        def per_scale(gamma: Array) -> Array:
            self_kernel = expected_rbf_kernel(mean_difference, self_omega, gamma)
            self_term = jnp.sum(weight_outer * self_kernel)
            cross_kernel = expected_rbf_kernel(cross_delta, cpp, gamma)
            cross_term = jnp.sum(pipp * cross_kernel)
            return self_term - 2.0 * cross_term

        return jnp.mean(jax.vmap(per_scale)(scales))

    return jax.vmap(per_row)(observations, projection, measurement_covariances)


def select_gaussian_params_observed_masked(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    selection: GaussianSelection,
):
    """Per-object **projected** ``Omega(x-tilde)`` transform, grouped by mask (§17.14).

    Returns a list of ``(group_index, (pi_{k,i}, nu_{k,i}, C_{k,i}))`` for every
    informative (``M > 0``) group: the marginalized window ``(a_i, Psi_i_inv)`` composed
    with the §17.5 transform on the projected inflated ``P_i Sigma_k P_i^T + S_i``. Shapes
    per group: ``(n, K)``, ``(n, K, M)``, ``(n, K, M, M)``. Diagnostic/oracle surface for
    the ``CMMD-SEL-OBSMAR-XFORM`` gate.
    """

    dtype = params.means.dtype
    pis, mus, sigmas = params.weights, params.means, params.covariances
    out = []
    for index, group in _informative_groups(grouped):
        marginal = _marginalized_selection(selection, group.coordinate_indices, dtype)

        def per_row(p_row, s_row):
            proj_mu = jnp.einsum("md,kd->km", p_row, mus)
            convolved = (
                jnp.einsum("md,kde,ne->kmn", p_row, sigmas, p_row) + s_row[None, :, :]
            )
            pipp, nupp, cpp, _ = _gaussian_log_weights(
                ConvMMDParams(pis, proj_mu, convolved), marginal
            )
            return pipp, nupp, cpp

        out.append(
            (
                index,
                jax.vmap(per_row)(
                    group.projection_matrices, group.measurement_covariances
                ),
            )
        )
    return out


def grouped_analytic_selected_observed_masked_loss(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    selection: GaussianSelection,
) -> Array:
    """Informative-weight-normalized ``Omega(x-tilde)``+MAR analytic loss (§17.14), scalar.

    Mirrors :func:`grouped_analytic_loss`, but each group's projected leaf is the
    §17.14 selected-observed leaf with that group's **marginalized** window. The mask
    grouping and per-group window marginalization are host-only; the per-group leaf is
    differentiable in ``params``. ``params`` are the CURRENT parameters (the group
    structure is fixed and param-independent) and MUST share the grouped arrays' dtype.
    """

    dtype = params.means.dtype
    informative = _informative_groups(grouped)
    if not informative:
        return jnp.asarray(0.0, dtype=dtype)  # all M=0: loss defined as exactly 0
    informative_weight = jnp.asarray(grouped.informative_weight, dtype=dtype)
    scales = jnp.asarray(bandwidths, dtype=dtype)
    total = jnp.asarray(0.0, dtype=dtype)
    for _, group in informative:
        marginal = _marginalized_selection(selection, group.coordinate_indices, dtype)
        per_row = convmmd_loss_analytic_projected_selected_observed(
            params,
            group.observations,
            group.projection_matrices,
            group.measurement_covariances,
            marginal,
            scales,
        )
        total = total + jnp.sum(group.sample_weight.astype(dtype) * per_row)
    return _normalize_by_informative_weight(total, informative_weight)


def convmmd_loss_analytic_selected_observed_masked(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    bandwidths,
    selection: GaussianSelection,
    sample_weight=None,
    dtype,
) -> Array:
    """One-shot ``Omega(x-tilde)``+MAR analytic loss (§17.14; heteroscedastic, masked).

    Groups a full-width masked collection by mask pattern (reusing the §16 adapter with a
    genuinely per-item ``(N, D, D)`` noise stack), then evaluates
    :func:`grouped_analytic_selected_observed_masked_loss`. **Reductions:** fully observed
    (``observed_mask`` all-``True``) equals the §17.13 heteroscedastic loss; ``Psi^{-1}=0``
    equals the base §16 masked loss. No noise model.
    """

    grouped = group_masked_inputs(
        params,
        observations,
        observed_mask,
        noise=noise,
        sample_weight=sample_weight,
        dtype=dtype,
    )
    return grouped_analytic_selected_observed_masked_loss(
        params, grouped, bandwidths, selection
    )


def convmmd_denoise_selected_observed_masked(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    dtype,
) -> Array:
    """``Omega(x-tilde)``+MAR denoiser = the base §16.4 masked posterior mean (§17.14).

    ``Omega(x-tilde_i)`` is constant in ``z`` and cancels, so a detected object's posterior
    is the projected empirical-Bayes posterior at its **own** ``S_i`` (full ``D``, original
    row order). This wrapper documents that reduction; no ``selection`` and no noise model
    are needed. Accepts genuinely per-item ``(N, D, D)`` noise.
    """

    return convmmd_denoise_masked(
        params, observations, observed_mask, noise=noise, dtype=dtype
    )


def _resolve_observed_omega(selection, coordinate_indices, dtype):
    """Return the observed-subspace completeness callable for group ``C_i`` (§17.14 SNIS).

    For a :class:`GaussianSelection` the window is **marginalized** onto the observed
    subspace and its Gaussian ``Omega`` is returned (so the SNIS limit is the §17.14
    analytic value -- the oracle check). For a :class:`CallableSelection` the callable is
    the per-subspace completeness and is applied to the observed-subspace draw ``R^{M_i}``
    directly (a canonical marginalization of a general ``Omega`` over unmeasured
    coordinates is model-dependent and out of scope, §17.14).
    """

    if isinstance(selection, GaussianSelection):
        marginal = _marginalized_selection(selection, coordinate_indices, dtype)
        return gaussian_selection_omega(marginal)
    if isinstance(selection, CallableSelection):
        return selection.omega
    raise TypeError(
        "selection must be a GaussianSelection or CallableSelection; "
        f"received {type(selection).__name__}"
    )


def convmmd_loss_projected_snis_selected_observed(
    params: ConvMMDParams,
    observations: Array,
    projection: Array,
    measurement_covariances: Array,
    omega,
    bandwidths: Array,
    key: Array,
    num_samples: int,
) -> Array:
    """Per-row projected ``Omega(x-tilde)``+MAR SNIS loss for one group (§17.14), ``(n,)``.

    For each row ``i`` draw ``z ~ q_theta`` (``K`` components), project ``P_i z``, add the
    object's **own** noise ``eps ~ N(0, S_i)`` to form ``x-tilde = P_i z + eps`` on
    ``R^{M_i}``, weight by ``omega`` evaluated on that observed-subspace value, and take
    the §17.4 self-normalized cross/self terms with the per-object normalizer ``Zhat_i``
    (two independent sets, full double sum). ``omega`` is the observed-subspace
    completeness (see :func:`_resolve_observed_omega`). One explicit ``key`` (split across
    the group's rows), static ``num_samples``; **MUST NOT** read a global key. Biased at
    finite ``num_samples``, consistent as it grows. No noise model.
    """

    dtype = params.means.dtype
    pis, mus, sigmas = params.weights, params.means, params.covariances
    scales = jnp.asarray(bandwidths, dtype=dtype)
    dimension = mus.shape[1]
    m_dim = observations.shape[1]
    latent_factor = jnp.linalg.cholesky(sigmas)  # (K, D, D)
    keys = jax.random.split(key, observations.shape[0])

    def per_row(x_row: Array, p_row: Array, s_row: Array, key_row: Array) -> Array:
        noise_factor = jnp.linalg.cholesky(s_row)  # (M, M)
        k1, k2, k3, k4 = jax.random.split(key_row, 4)
        z1 = jax.random.normal(k1, (num_samples, dimension), dtype=dtype)
        z2 = jax.random.normal(k2, (num_samples, dimension), dtype=dtype)
        e1 = jax.random.normal(k3, (num_samples, m_dim), dtype=dtype)
        e2 = jax.random.normal(k4, (num_samples, m_dim), dtype=dtype)
        latent_1 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z1)
        latent_2 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z2)
        proj_1 = jnp.einsum("md,ksd->ksm", p_row, latent_1)  # (K, M_s, M)
        proj_2 = jnp.einsum("md,ksd->ksm", p_row, latent_2)
        tilde_1 = proj_1 + (e1 @ noise_factor.T)[None, :, :]  # (K, M_s, M)
        tilde_2 = proj_2 + (e2 @ noise_factor.T)[None, :, :]
        weight_1 = omega(tilde_1).astype(dtype)  # (K, M_s)
        weight_2 = omega(tilde_2).astype(dtype)
        z_hat_1 = jnp.sum(pis * jnp.mean(weight_1, axis=1))
        z_hat_2 = jnp.sum(pis * jnp.mean(weight_2, axis=1))

        diff_cross = tilde_1 - x_row[None, None, :]  # (K, M_s, M)
        dist_cross = jnp.sum(diff_cross * diff_cross, axis=-1)  # (K, M_s)
        sq_1 = jnp.sum(tilde_1 * tilde_1, axis=-1)  # (K, M_s)
        sq_2 = jnp.sum(tilde_2 * tilde_2, axis=-1)
        gram = jnp.einsum("kma,lpa->klmp", tilde_1, tilde_2)  # (K, K, M_s, M_s)
        dist_self = jnp.maximum(
            sq_1[:, None, :, None] + sq_2[None, :, None, :] - 2.0 * gram, 0.0
        )

        def per_scale(gamma: Array) -> Array:
            gamma_sq_2 = (2.0 * gamma * gamma).astype(dtype)
            kernel_cross = jnp.exp(-dist_cross / gamma_sq_2)  # (K, M_s)
            cross = jnp.sum(pis * jnp.mean(weight_1 * kernel_cross, axis=1)) / z_hat_1
            kernel_self = jnp.exp(-dist_self / gamma_sq_2)  # (K, K, M_s, M_s)
            weight_outer = weight_1[:, None, :, None] * weight_2[None, :, None, :]
            self_kk = jnp.mean(weight_outer * kernel_self, axis=(2, 3))  # (K, K)
            self_term = jnp.sum(pis[:, None] * pis[None, :] * self_kk) / (
                z_hat_1 * z_hat_2
            )
            return self_term - 2.0 * cross

        return jnp.mean(jax.vmap(per_scale)(scales))

    return jax.vmap(per_row)(observations, projection, measurement_covariances, keys)


def grouped_snis_selected_observed_masked_loss(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    selection,
    key: Array,
    num_samples: int,
) -> Array:
    """Informative-weight-normalized ``Omega(x-tilde)``+MAR SNIS loss (§17.14), scalar.

    Mirrors :func:`grouped_mc_loss`: one ``key`` split across the groups, each group's
    projected SNIS leaf using its observed-subspace completeness (marginalized Gaussian
    window, or the per-subspace callable). Biased at finite ``num_samples``, consistent as
    it grows; its expectation limit is
    :func:`grouped_analytic_selected_observed_masked_loss` for a Gaussian window.
    """

    dtype = params.means.dtype
    informative = _informative_groups(grouped)
    if not informative:
        return jnp.asarray(0.0, dtype=dtype)
    informative_weight = jnp.asarray(grouped.informative_weight, dtype=dtype)
    scales = jnp.asarray(bandwidths, dtype=dtype)
    keys = jax.random.split(key, max(len(grouped.groups), 1))
    total = jnp.asarray(0.0, dtype=dtype)
    for index, group in informative:
        omega = _resolve_observed_omega(selection, group.coordinate_indices, dtype)
        per_row = convmmd_loss_projected_snis_selected_observed(
            params,
            group.observations,
            group.projection_matrices,
            group.measurement_covariances,
            omega,
            scales,
            keys[index],
            num_samples,
        )
        total = total + jnp.sum(group.sample_weight.astype(dtype) * per_row)
    return _normalize_by_informative_weight(total, informative_weight)


def convmmd_loss_snis_selected_observed_masked(
    params: ConvMMDParams,
    observations,
    observed_mask,
    *,
    noise,
    bandwidths,
    selection,
    key: Array,
    num_samples: int,
    sample_weight=None,
    dtype,
) -> Array:
    """One-shot general-``Omega(x-tilde)``+MAR per-object projected SNIS loss (§17.14).

    Groups by mask pattern, then evaluates
    :func:`grouped_snis_selected_observed_masked_loss`. Fully observed reduces to the
    §17.13 per-object SNIS; a marginalized Gaussian window converges to the §17.14
    analytic value. No noise model.
    """

    grouped = group_masked_inputs(
        params,
        observations,
        observed_mask,
        noise=noise,
        sample_weight=sample_weight,
        dtype=dtype,
    )
    return grouped_snis_selected_observed_masked_loss(
        params, grouped, bandwidths, selection, key, num_samples
    )


def snis_diagnostics_observed_masked(
    params: ConvMMDParams,
    grouped: GroupedGeneralInputs,
    selection,
    key: Array,
    num_samples: int,
):
    """Per-object ``(ESS, Zhat)`` for the ``Omega(x-tilde)``+MAR SNIS, by group (§17.14).

    Cheap (no kernels): per informative group, draw ``z ~ q_theta``, project and add each
    object's own ``S_i`` noise, weight by the observed-subspace ``Omega``; returns a list
    of ``(group_index, (ess, zhat))`` with per-row shapes ``(n,)``. ``ESS = (sum u)^2 /
    sum u^2`` with ``u = pi_k omega`` over the ``K * num_samples`` draws. Diagnostics only.
    """

    dtype = params.means.dtype
    pis, mus, sigmas = params.weights, params.means, params.covariances
    dimension = mus.shape[1]
    latent_factor = jnp.linalg.cholesky(sigmas)
    out = []
    keys = jax.random.split(key, max(len(grouped.groups), 1))
    for index, group in _informative_groups(grouped):
        omega = _resolve_observed_omega(selection, group.coordinate_indices, dtype)
        m_dim = group.observations.shape[1]
        row_keys = jax.random.split(keys[index], group.observations.shape[0])

        def per_row(p_row, s_row, key_row):
            noise_factor = jnp.linalg.cholesky(s_row)
            k1, k2 = jax.random.split(key_row, 2)
            z = jax.random.normal(k1, (num_samples, dimension), dtype=dtype)
            e = jax.random.normal(k2, (num_samples, m_dim), dtype=dtype)
            latent = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z)
            proj = jnp.einsum("md,ksd->ksm", p_row, latent)
            tilde = proj + (e @ noise_factor.T)[None, :, :]
            weight = omega(tilde).astype(dtype)  # (K, M_s)
            z_hat = jnp.sum(pis * jnp.mean(weight, axis=1))
            pooled = pis[:, None] * weight
            ess = jnp.sum(pooled) ** 2 / jnp.sum(pooled * pooled)
            return ess, z_hat

        out.append(
            (
                index,
                jax.vmap(per_row)(
                    group.projection_matrices,
                    group.measurement_covariances,
                    row_keys,
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Fit control (selected analytic): differentiate through the fixed group structure
# ---------------------------------------------------------------------------


def fit_selected_analytic_state(
    initial,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    selection: GaussianSelection,
    *,
    n_steps: int = 300,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 1.0e-1,
    clip_norm: float = 1.0,
    tol: float = 1.0e-6,
):
    """Array-only deterministic selected analytic fit over a fixed group set.

    The Gaussian window is fixed; gradients flow through the transform to the
    (unconstrained) mixture parameters, recovering the de-biased true population.
    """

    def loss_and_grad(unc, _key):
        def objective(candidate):
            return grouped_analytic_selected_loss(
                to_canonical(candidate), grouped, bandwidths, selection
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


def fit_selected_analytic(
    initial,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    selection: GaussianSelection,
    **kwargs,
) -> ConvMMDFitResult:
    """Host selected analytic fit: array state plus custody metadata."""

    return _attach_metadata(
        fit_selected_analytic_state(initial, grouped, bandwidths, selection, **kwargs)
    )


def fit_selected_mc_state(
    initial,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    selection,
    key: Array,
    *,
    num_samples: int = 200,
    n_steps: int = 300,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 1.0e-1,
    clip_norm: float = 1.0,
    tol: float = 1.0e-6,
):
    """Array-only stochastic SNIS selected fit over a fixed group set (§17.4).

    The selection spec is fixed; gradients flow through the reparameterized draws
    and ``omega`` to the (unconstrained) mixture parameters. Fixed-step (the SNIS
    objective is stochastic and biased at finite ``num_samples``); it MUST NOT report
    a spurious ``CONVERGED`` from a noisy two-point change.
    """

    def loss_and_grad(unc, step_key):
        def objective(candidate):
            return grouped_snis_loss(
                to_canonical(candidate),
                grouped,
                bandwidths,
                step_key,
                num_samples,
                selection,
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


def fit_selected_mc(
    initial,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    selection,
    key: Array,
    **kwargs,
) -> ConvMMDFitResult:
    """Host SNIS selected fit: array state plus custody metadata."""

    return _attach_metadata(
        fit_selected_mc_state(initial, grouped, bandwidths, selection, key, **kwargs)
    )


def fit_selected_observed_analytic_state(
    initial,
    observations,
    noise,
    bandwidths: Array,
    selection: GaussianSelection,
    *,
    n_steps: int = 300,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 1.0e-1,
    clip_norm: float = 1.0,
    tol: float = 1.0e-6,
):
    """Array-only deterministic ``Omega(x-tilde)`` analytic fit (fully observed,
    homoscedastic). Gradients flow through the observed-space transform."""

    observed = jnp.asarray(observations)

    def loss_and_grad(unc, _key):
        def objective(candidate):
            return convmmd_loss_analytic_selected_observed(
                to_canonical(candidate), observed,
                noise=noise, bandwidths=bandwidths, selection=selection,
            )

        return jax.value_and_grad(objective)(unc)

    return _run(
        loss_and_grad, initial, n_steps=n_steps, learning_rate=learning_rate,
        weight_decay=weight_decay, clip_norm=clip_norm, tol=tol,
        key=None, deterministic=True,
    )


def fit_selected_observed_analytic(
    initial, observations, noise, bandwidths: Array, selection: GaussianSelection, **kwargs
) -> ConvMMDFitResult:
    """Host ``Omega(x-tilde)`` analytic fit: array state plus custody metadata."""

    return _attach_metadata(
        fit_selected_observed_analytic_state(
            initial, observations, noise, bandwidths, selection, **kwargs
        )
    )


def fit_selected_observed_hetero_analytic_state(
    initial,
    observations,
    noise,
    bandwidths: Array,
    selection: GaussianSelection,
    *,
    n_steps: int = 300,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 1.0e-1,
    clip_norm: float = 1.0,
    tol: float = 1.0e-6,
):
    """Array-only deterministic **heteroscedastic** ``Omega(x-tilde)`` analytic fit
    (fully observed; per-object ``S_i``, §17.13). Gradients flow through the per-object
    observed-space transform."""

    observed = jnp.asarray(observations)
    per_item_noise = jnp.asarray(noise)

    def loss_and_grad(unc, _key):
        def objective(candidate):
            return convmmd_loss_analytic_selected_observed_hetero(
                to_canonical(candidate), observed,
                noise=per_item_noise, bandwidths=bandwidths, selection=selection,
            )

        return jax.value_and_grad(objective)(unc)

    return _run(
        loss_and_grad, initial, n_steps=n_steps, learning_rate=learning_rate,
        weight_decay=weight_decay, clip_norm=clip_norm, tol=tol,
        key=None, deterministic=True,
    )


def fit_selected_observed_hetero_analytic(
    initial, observations, noise, bandwidths: Array, selection: GaussianSelection, **kwargs
) -> ConvMMDFitResult:
    """Host heteroscedastic ``Omega(x-tilde)`` analytic fit: array state plus metadata."""

    return _attach_metadata(
        fit_selected_observed_hetero_analytic_state(
            initial, observations, noise, bandwidths, selection, **kwargs
        )
    )


def fit_selected_observed_masked_analytic_state(
    initial,
    grouped: GroupedGeneralInputs,
    bandwidths: Array,
    selection: GaussianSelection,
    *,
    n_steps: int = 300,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 1.0e-1,
    clip_norm: float = 1.0,
    tol: float = 1.0e-6,
):
    """Array-only deterministic ``Omega(x-tilde)``+MAR analytic fit over a fixed group set
    (§17.14). Gradients flow through the per-object projected observed-space transform; the
    mask grouping and per-group window marginalization are fixed and host-side. Group with
    :func:`group_masked_fit_inputs` (rejects an all-``M=0`` collection)."""

    def loss_and_grad(unc, _key):
        def objective(candidate):
            return grouped_analytic_selected_observed_masked_loss(
                to_canonical(candidate), grouped, bandwidths, selection
            )

        return jax.value_and_grad(objective)(unc)

    return _run(
        loss_and_grad, initial, n_steps=n_steps, learning_rate=learning_rate,
        weight_decay=weight_decay, clip_norm=clip_norm, tol=tol,
        key=None, deterministic=True,
    )


def fit_selected_observed_masked_analytic(
    initial, grouped: GroupedGeneralInputs, bandwidths: Array, selection: GaussianSelection, **kwargs
) -> ConvMMDFitResult:
    """Host ``Omega(x-tilde)``+MAR analytic fit: array state plus custody metadata (§17.14)."""

    return _attach_metadata(
        fit_selected_observed_masked_analytic_state(
            initial, grouped, bandwidths, selection, **kwargs
        )
    )


__all__ = [
    "CallableSelection",
    "GaussianSelection",
    "convmmd_denoise_selected",
    "convmmd_denoise_selected_observed",
    "convmmd_denoise_selected_observed_hetero",
    "convmmd_denoise_selected_observed_masked",
    "convmmd_loss_analytic_projected_selected_observed",
    "convmmd_loss_analytic_selected",
    "convmmd_loss_analytic_selected_observed",
    "convmmd_loss_analytic_selected_observed_hetero",
    "convmmd_loss_analytic_selected_observed_masked",
    "convmmd_loss_projected_snis_selected_observed",
    "convmmd_loss_snis_selected",
    "convmmd_loss_snis_selected_observed",
    "convmmd_loss_snis_selected_observed_hetero",
    "convmmd_loss_snis_selected_observed_masked",
    "convmmd_posterior_components_selected",
    "convmmd_snis_denoise_selected",
    "effective_volume_gaussian",
    "effective_volume_gaussian_observed",
    "fit_selected_analytic",
    "fit_selected_analytic_state",
    "fit_selected_mc",
    "fit_selected_mc_state",
    "fit_selected_observed_analytic",
    "fit_selected_observed_analytic_state",
    "fit_selected_observed_hetero_analytic",
    "fit_selected_observed_hetero_analytic_state",
    "fit_selected_observed_masked_analytic",
    "fit_selected_observed_masked_analytic_state",
    "gaussian_selection_omega",
    "group_masked_fit_inputs",
    "group_masked_inputs",
    "grouped_analytic_selected_loss",
    "grouped_analytic_selected_observed_masked_loss",
    "grouped_denoise_selected",
    "grouped_posterior_components_selected",
    "grouped_snis_denoise",
    "grouped_snis_loss",
    "grouped_snis_selected_observed_masked_loss",
    "marginalize_gaussian_window",
    "select_gaussian_params",
    "select_gaussian_params_observed",
    "select_gaussian_params_observed_hetero",
    "select_gaussian_params_observed_masked",
    "snis_denoise_projected",
    "snis_diagnostics_latent",
    "snis_diagnostics_observed_hetero",
    "snis_diagnostics_observed_masked",
    "snis_effective_sample_size",
]
