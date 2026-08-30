# SPDX-License-Identifier: MIT
# Provenance: convMMD is the maintainer's own method (Vashistha, Sarkar, Farahi,
# arXiv:2606.21907). This is a clean-room implementation from the model contract
# (docs/convmmd-model-contract.md), not derived from astroML or Bovy XD code.
"""Temporary pure-JAX convMMD numerical kernels (development stage).

This module implements the Gaussian-GMM convMMD contract
(``docs/convmmd-model-contract.md``) as pure JAX. Like the rest of
``development/``, it is **not** the installable library, a promised public API, or
a released namespace; it is exercised by the contract-driven test suite while the
method is validated toward the next beta. It is not yet exposed through any
``src/xdgmm_jax`` facade.

Two co-equal loss operators are provided:

* :func:`convmmd_loss_analytic` -- the exact Gaussian-integral closed form
  (deterministic, PRNG-free, ``jit``/``grad``-clean), and
* :func:`convmmd_loss_mc` -- the reparameterized Monte-Carlo estimator (requires
  one explicit PRNG key), whose ``num_samples -> inf`` limit is the analytic
  form.

The empirical-Bayes denoiser (:func:`denoise`, :func:`posterior_components`) is
the exact Gaussian-mixture posterior; it is identical in form to the XD posterior
mean. Canonical parameters ``(weights, means, covariances)`` are consumed by the
numerical core; :func:`to_canonical` maps the unconstrained optimization
parameters (``softmax`` weights, ``softplus``-diagonal Cholesky) onto them.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


Array = jax.Array

#: Fixed positive floors for the unconstrained -> canonical covariance map
#: (mirroring the reference prototype's ``get_covariances``).
EPS_SIGMA = 1.0e-5
EPS_L = 1.0e-4


class ConvMMDParams(NamedTuple):
    """Canonical convMMD mixture parameters.

    Fields have shapes ``(K,)``, ``(K, D)``, and ``(K, D, D)``. ``weights`` lie
    on the simplex and ``covariances`` are symmetric positive definite. A
    ``NamedTuple`` is a JAX PyTree without registration or host conversion.
    """

    weights: Array
    means: Array
    covariances: Array


class ConvMMDUnconstrained(NamedTuple):
    """Unconstrained optimization parameters ``(alpha, mu, Lambda)``.

    Fields have shapes ``(K,)``, ``(K, D)``, and ``(K, D, D)``.
    """

    alphas: Array
    means: Array
    unconstrained_L: Array


def _log2pi(dtype) -> Array:
    return jnp.log(2.0 * jnp.pi).astype(dtype)


def _diagonal(matrices: Array) -> Array:
    """Return the diagonals of a ``(..., D, D)`` batch as ``(..., D)``."""

    return jnp.diagonal(matrices, axis1=-2, axis2=-1)


def expected_rbf_kernel(delta: Array, omega: Array, gamma: Array) -> Array:
    r"""Exact ``E[exp(-||W||^2 / (2 gamma^2))]`` for ``W ~ N(delta, omega)``.

    ``G(delta, omega; gamma) = |I + gamma^-2 omega|^{-1/2}
    exp(-1/2 delta^T (omega + gamma^2 I)^{-1} delta)``, broadcast over any
    leading batch axes shared by ``delta`` (``..., D``) and ``omega``
    (``..., D, D``). ``gamma`` is a scalar.
    """

    dimension = delta.shape[-1]
    dtype = delta.dtype
    identity = jnp.eye(dimension, dtype=dtype)
    gamma_sq = (gamma * gamma).astype(dtype)
    total = omega + gamma_sq * identity
    factor = jnp.linalg.cholesky(total)
    log_det_total = 2.0 * jnp.sum(jnp.log(_diagonal(factor)), axis=-1)
    log_det_ratio = log_det_total - dimension * jnp.log(gamma_sq)
    whitened = jax.scipy.linalg.solve_triangular(
        factor, delta[..., None], lower=True
    )[..., 0]
    quadratic = jnp.sum(whitened * whitened, axis=-1)
    return jnp.exp(-0.5 * log_det_ratio - 0.5 * quadratic)


def convmmd_loss_analytic(
    params: ConvMMDParams,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
) -> Array:
    r"""Exact analytic convMMD loss (scalar).

    ``L = (1 / (G N)) sum_g sum_i [ sum_{k,k'} pi_k pi_k'
    G(mu_k-mu_k', A_k^i + A_k'^i; gamma_g) - 2 sum_k pi_k
    G(x_i-mu_k, A_k^i; gamma_g) ]`` with ``A_k^i = Sigma_k + S_i``. The
    theta-independent data-data term is omitted, so the value MAY be negative.
    """

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    x = observations
    noise = measurement_covariances

    # A[i, k] = Sigma_k + S_i  ->  (N, K, D, D)
    convolved = sigmas[None, :, :, :] + noise[:, None, :, :]

    mean_difference = mus[:, None, :] - mus[None, :, :]  # (K, K, D)
    self_omega = convolved[:, :, None, :, :] + convolved[:, None, :, :, :]
    n_samples = x.shape[0]
    n_components = mus.shape[0]
    self_delta = jnp.broadcast_to(
        mean_difference[None, :, :, :],
        (n_samples, n_components, n_components, x.shape[-1]),
    )
    cross_delta = x[:, None, :] - mus[None, :, :]  # (N, K, D)
    weight_outer = pis[:, None] * pis[None, :]  # (K, K)

    def per_scale(gamma: Array) -> Array:
        self_kernel = expected_rbf_kernel(self_delta, self_omega, gamma)
        self_term = jnp.sum(weight_outer[None, :, :] * self_kernel, axis=(-2, -1))
        cross_kernel = expected_rbf_kernel(cross_delta, convolved, gamma)
        cross_term = jnp.sum(pis[None, :] * cross_kernel, axis=-1)
        return jnp.mean(self_term - 2.0 * cross_term)

    per_scale_losses = jax.vmap(per_scale)(bandwidths)
    return jnp.mean(per_scale_losses)


def convmmd_loss_mc(
    params: ConvMMDParams,
    observations: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    key: Array,
    num_samples: int,
) -> Array:
    """Reparameterized Monte-Carlo convMMD loss (scalar).

    Requires one explicit PRNG ``key`` and a static ``num_samples``. Draws model
    samples ``mu_k + L_k zeta`` and full-covariance noise ``L_{S_i} eta``; its
    expectation is :func:`convmmd_loss_analytic`.
    """

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    dtype = mus.dtype
    dimension = mus.shape[1]
    n_samples = observations.shape[0]

    latent_factor = jnp.linalg.cholesky(sigmas)  # (K, D, D)
    noise_factor = jnp.linalg.cholesky(measurement_covariances)  # (N, D, D)

    k1, k2, k3, k4 = jax.random.split(key, 4)
    z1 = jax.random.normal(k1, (num_samples, dimension), dtype=dtype)
    z2 = jax.random.normal(k2, (num_samples, dimension), dtype=dtype)
    latent_1 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z1)
    latent_2 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z2)

    e1 = jax.random.normal(k3, (n_samples, num_samples, dimension), dtype=dtype)
    e2 = jax.random.normal(k4, (n_samples, num_samples, dimension), dtype=dtype)
    noise_1 = jnp.einsum("nab,nmb->nma", noise_factor, e1)
    noise_2 = jnp.einsum("nab,nmb->nma", noise_factor, e2)

    tilde_1 = latent_1[None, :, :, :] + noise_1[:, None, :, :]  # (N, K, M, D)
    tilde_2 = latent_2[None, :, :, :] + noise_2[:, None, :, :]

    diff_cross = tilde_1 - observations[:, None, None, :]
    dist_cross = jnp.sum(diff_cross * diff_cross, axis=-1)  # (N, K, M)
    diff_self = tilde_1[:, :, None, :, :] - tilde_2[:, None, :, :, :]
    dist_self = jnp.sum(diff_self * diff_self, axis=-1)  # (N, K, K, M)

    def per_scale(gamma: Array) -> Array:
        gamma_sq_2 = 2.0 * gamma * gamma
        kernel_cross = jnp.exp(-dist_cross / gamma_sq_2)
        expected_cross = jnp.mean(kernel_cross, axis=-1)  # (N, K)
        cross_term = jnp.sum(pis * expected_cross, axis=-1)  # (N,)
        kernel_self = jnp.exp(-dist_self / gamma_sq_2)
        expected_self = jnp.mean(kernel_self, axis=-1)  # (N, K, K)
        self_term = jnp.sum(
            pis[:, None] * pis[None, :] * expected_self, axis=(-2, -1)
        )
        return self_term - 2.0 * cross_term

    per_scale_losses = jax.vmap(per_scale)(bandwidths)  # (G, N)
    return jnp.mean(per_scale_losses)


class PosteriorComponents(NamedTuple):
    """Exact denoising posterior quantities."""

    responsibilities: Array  # (N, K)
    component_means: Array  # (N, K, D)


def posterior_components(
    params: ConvMMDParams,
    observations: Array,
    measurement_covariances: Array,
) -> PosteriorComponents:
    """Return exact posterior responsibilities and component means."""

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    dtype = mus.dtype
    dimension = mus.shape[1]

    marginal = sigmas[None, :, :, :] + measurement_covariances[:, None, :, :]
    delta = observations[:, None, :] - mus[None, :, :]  # (N, K, D)
    factor = jnp.linalg.cholesky(marginal)  # (N, K, D, D)
    whitened = jax.scipy.linalg.solve_triangular(
        factor, delta[..., None], lower=True
    )[..., 0]
    log_det = 2.0 * jnp.sum(jnp.log(_diagonal(factor)), axis=-1)  # (N, K)
    log_norm = -0.5 * (
        dimension * _log2pi(dtype) + log_det + jnp.sum(whitened * whitened, axis=-1)
    )
    log_joint = jnp.log(pis)[None, :] + log_norm  # (N, K)
    log_evidence = logsumexp(log_joint, axis=-1, keepdims=True)
    responsibilities = jnp.exp(log_joint - log_evidence)

    gain_delta = jnp.linalg.solve(marginal, delta[..., None])[..., 0]  # (N, K, D)
    component_means = mus[None, :, :] + jnp.einsum(
        "kde,nke->nkd", sigmas, gain_delta
    )
    return PosteriorComponents(
        responsibilities=responsibilities, component_means=component_means
    )


def denoise(
    params: ConvMMDParams,
    observations: Array,
    measurement_covariances: Array,
) -> Array:
    """Return the exact empirical-Bayes posterior mean ``(N, D)``."""

    components = posterior_components(params, observations, measurement_covariances)
    return jnp.einsum(
        "nk,nkd->nd", components.responsibilities, components.component_means
    )


def to_canonical(
    unconstrained: ConvMMDUnconstrained,
    *,
    eps_sigma: float = EPS_SIGMA,
    eps_l: float = EPS_L,
) -> ConvMMDParams:
    """Map unconstrained ``(alpha, mu, Lambda)`` to canonical ``(pi, mu, Sigma)``."""

    weights = jax.nn.softmax(unconstrained.alphas)
    lam = unconstrained.unconstrained_L
    dtype = lam.dtype
    dimension = lam.shape[-1]
    lower = jnp.tril(lam, k=-1)
    diagonal_values = jax.nn.softplus(_diagonal(lam)) + eps_l
    factor = lower + _diag_embed(diagonal_values)
    covariances = jnp.einsum("kab,kcb->kac", factor, factor) + eps_sigma * jnp.eye(
        dimension, dtype=dtype
    )
    return ConvMMDParams(
        weights=weights, means=unconstrained.means, covariances=covariances
    )


def _diag_embed(values: Array) -> Array:
    """Embed ``(..., D)`` values on the diagonal of ``(..., D, D)`` matrices."""

    dimension = values.shape[-1]
    eye = jnp.eye(dimension, dtype=values.dtype)
    return values[..., None] * eye


def median_bandwidths(
    observations: Array,
    *,
    n_scales: int = 9,
    log10_low: float = -2.0,
    log10_high: float = 2.0,
) -> Array:
    """Predeclared bandwidth set: median pairwise distance x log grid.

    Host convenience (not part of the differentiable hot path). ``gamma_g =
    median_pairwise_distance(X) * 10**s_g`` with ``s_g`` on an ``n_scales``-point
    linear grid in ``[log10_low, log10_high]``.
    """

    n = observations.shape[0]
    if n < 2:
        raise ValueError(
            "median_bandwidths needs at least two observations to form a "
            f"pairwise distance; received n={n}"
        )
    differences = observations[:, None, :] - observations[None, :, :]
    distances = jnp.sqrt(jnp.sum(differences * differences, axis=-1))
    upper = distances[jnp.triu_indices(n, k=1)]
    median = jnp.median(upper)
    scales = jnp.logspace(
        log10_low, log10_high, n_scales, dtype=observations.dtype
    )
    return median * scales


# ---------------------------------------------------------------------------
# Projected fixed-``M`` leaves for per-coordinate missing data (§16)
# ---------------------------------------------------------------------------
#
# Each leaf operates on one fixed-``M`` mask group: ``observations`` ``(n, M)``,
# per-row projection ``(n, M, D)``, and observed-space noise ``(n, M, M)``. ``M``
# is static within a group. The projected convolved covariance is
# ``B_k^i = P_i Sigma_k P_i^T + S_i``. These leaves are ``jit``/``grad``/``vmap``
# clean; the host-side mask grouping lives in :mod:`development.convmmd_grouped`.


def convmmd_loss_analytic_projected(
    params: ConvMMDParams,
    observations: Array,
    projection: Array,
    measurement_covariances: Array,
    bandwidths: Array,
) -> Array:
    r"""Per-row analytic projected convMMD loss for one fixed-``M`` group (§16.3).

    Returns the per-row loss averaged over scales, shape ``(n,)`` -- the quantity
    the grouped objective weights and normalizes by the informative weight.
    """

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    weight_outer = pis[:, None] * pis[None, :]  # (K, K)

    def per_row(x_row: Array, p_row: Array, s_row: Array) -> Array:
        proj_mu = jnp.einsum("md,kd->km", p_row, mus)  # (K, M)
        proj_sigma = jnp.einsum("md,kde,ne->kmn", p_row, sigmas, p_row)  # (K, M, M)
        convolved = proj_sigma + s_row[None, :, :]  # B_k^i, (K, M, M)
        mean_difference = proj_mu[:, None, :] - proj_mu[None, :, :]  # (K, K, M)
        self_omega = (
            convolved[:, None, :, :] + convolved[None, :, :, :]
        )  # (K, K, M, M)
        cross_delta = x_row[None, :] - proj_mu  # (K, M)

        def per_scale(gamma: Array) -> Array:
            self_kernel = expected_rbf_kernel(mean_difference, self_omega, gamma)
            self_term = jnp.sum(weight_outer * self_kernel)
            cross_kernel = expected_rbf_kernel(cross_delta, convolved, gamma)
            cross_term = jnp.sum(pis * cross_kernel)
            return self_term - 2.0 * cross_term

        return jnp.mean(jax.vmap(per_scale)(bandwidths))

    return jax.vmap(per_row)(observations, projection, measurement_covariances)


def convmmd_loss_mc_projected(
    params: ConvMMDParams,
    observations: Array,
    projection: Array,
    measurement_covariances: Array,
    bandwidths: Array,
    key: Array,
    num_samples: int,
) -> Array:
    """Per-row reparameterized projected Monte-Carlo loss (§16.5), shape ``(n,)``.

    Requires one explicit PRNG ``key`` and a static ``num_samples``. Its
    ``num_samples -> inf`` limit is :func:`convmmd_loss_analytic_projected`.
    """

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    dtype = mus.dtype
    dimension = mus.shape[1]
    n_samples = observations.shape[0]
    m_dim = observations.shape[1]

    latent_factor = jnp.linalg.cholesky(sigmas)  # (K, D, D)
    noise_factor = jnp.linalg.cholesky(measurement_covariances)  # (n, M, M)

    k1, k2, k3, k4 = jax.random.split(key, 4)
    z1 = jax.random.normal(k1, (num_samples, dimension), dtype=dtype)
    z2 = jax.random.normal(k2, (num_samples, dimension), dtype=dtype)
    latent_1 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z1)
    latent_2 = mus[:, None, :] + jnp.einsum("kab,mb->kma", latent_factor, z2)
    proj_1 = jnp.einsum("npd,kmd->nkmp", projection, latent_1)  # (n, K, M', M)
    proj_2 = jnp.einsum("npd,kmd->nkmp", projection, latent_2)

    e1 = jax.random.normal(k3, (n_samples, num_samples, m_dim), dtype=dtype)
    e2 = jax.random.normal(k4, (n_samples, num_samples, m_dim), dtype=dtype)
    noise_1 = jnp.einsum("npq,nmq->nmp", noise_factor, e1)  # (n, M', M)
    noise_2 = jnp.einsum("npq,nmq->nmp", noise_factor, e2)
    tilde_1 = proj_1 + noise_1[:, None, :, :]  # (n, K, M', M)
    tilde_2 = proj_2 + noise_2[:, None, :, :]

    diff_cross = tilde_1 - observations[:, None, None, :]
    dist_cross = jnp.sum(diff_cross * diff_cross, axis=-1)  # (n, K, M')
    diff_self = tilde_1[:, :, None, :, :] - tilde_2[:, None, :, :, :]
    dist_self = jnp.sum(diff_self * diff_self, axis=-1)  # (n, K, K, M')

    def per_scale(gamma: Array) -> Array:
        # Cast the bandwidth to the compute dtype so the reduction is not silently
        # upcast when a caller pairs an f32 fit with an f64 bandwidth (mirrors the
        # analytic path, which downcasts gamma inside ``expected_rbf_kernel``).
        gamma_sq_2 = (2.0 * gamma * gamma).astype(dtype)
        kernel_cross = jnp.exp(-dist_cross / gamma_sq_2)
        expected_cross = jnp.mean(kernel_cross, axis=-1)  # (n, K)
        cross_term = jnp.sum(pis * expected_cross, axis=-1)  # (n,)
        kernel_self = jnp.exp(-dist_self / gamma_sq_2)
        expected_self = jnp.mean(kernel_self, axis=-1)  # (n, K, K)
        self_term = jnp.sum(
            pis[:, None] * pis[None, :] * expected_self, axis=(-2, -1)
        )
        return self_term - 2.0 * cross_term  # (n,)

    per_scale_losses = jax.vmap(per_scale)(bandwidths)  # (G, n)
    return jnp.mean(per_scale_losses, axis=0)  # (n,)


def posterior_components_projected(
    params: ConvMMDParams,
    observations: Array,
    projection: Array,
    measurement_covariances: Array,
) -> PosteriorComponents:
    """Projected posterior responsibilities and full-``D`` component means (§16.4)."""

    pis = params.weights
    mus = params.means
    sigmas = params.covariances
    dtype = mus.dtype
    m_dim = observations.shape[1]
    log_pis = jnp.log(pis)

    def per_row(x_row: Array, p_row: Array, s_row: Array):
        proj_mu = jnp.einsum("md,kd->km", p_row, mus)  # (K, M)
        convolved = (
            jnp.einsum("md,kde,ne->kmn", p_row, sigmas, p_row) + s_row[None, :, :]
        )  # B_k^i, (K, M, M)
        delta = x_row[None, :] - proj_mu  # (K, M)
        factor = jnp.linalg.cholesky(convolved)  # (K, M, M)
        whitened = jax.scipy.linalg.solve_triangular(
            factor, delta[..., None], lower=True
        )[..., 0]  # (K, M)
        log_det = 2.0 * jnp.sum(jnp.log(_diagonal(factor)), axis=-1)  # (K,)
        log_norm = -0.5 * (
            m_dim * _log2pi(dtype) + log_det + jnp.sum(whitened * whitened, axis=-1)
        )
        log_joint = log_pis + log_norm  # (K,)
        log_evidence = logsumexp(log_joint)
        responsibilities = jnp.exp(log_joint - log_evidence)  # (K,)
        gain = jnp.linalg.solve(convolved, delta[..., None])[..., 0]  # (K, M)
        cross_cov = jnp.einsum("kde,me->kdm", sigmas, p_row)  # Sigma_k P^T, (K, D, M)
        component_means = mus + jnp.einsum("kdm,km->kd", cross_cov, gain)  # (K, D)
        return responsibilities, component_means

    responsibilities, component_means = jax.vmap(per_row)(
        observations, projection, measurement_covariances
    )
    return PosteriorComponents(
        responsibilities=responsibilities, component_means=component_means
    )


def denoise_projected(
    params: ConvMMDParams,
    observations: Array,
    projection: Array,
    measurement_covariances: Array,
) -> Array:
    """Projected empirical-Bayes posterior mean for one group, ``(n, D)`` (§16.4)."""

    components = posterior_components_projected(
        params, observations, projection, measurement_covariances
    )
    return jnp.einsum(
        "nk,nkd->nd", components.responsibilities, components.component_means
    )
