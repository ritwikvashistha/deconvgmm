"""Clear NumPy reference for the convMMD Gaussian-GMM equations.

This module is the independent oracle for contract ``xdgmm-jax.convmmd``
(``docs/convmmd-model-contract.md``). It is written from the contract's
mathematics, NOT from the supplied JAX/Monte-Carlo prototype: the analytic
closed form here is the exact ``num_samples -> inf`` limit that the prototype's
Monte-Carlo estimator approximates, so it can validate that estimator without
sharing its code path. It favors explicit loops and independent linear algebra.
It is test evidence, not a performance implementation or public package API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]


def _as_float64(value: npt.ArrayLike) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


# ---------------------------------------------------------------------------
# Exact expected RBF kernel (the Gaussian-integral closed form)
# ---------------------------------------------------------------------------


def expected_rbf_kernel(
    delta: npt.ArrayLike, omega: npt.ArrayLike, gamma: float
) -> float:
    r"""Return ``E[exp(-||W||^2 / (2 gamma^2))]`` for ``W ~ N(delta, omega)``.

    Closed form ``G(delta, omega; gamma) = |I + gamma^-2 omega|^{-1/2}
    exp(-1/2 delta^T (omega + gamma^2 I)^{-1} delta)`` computed through the
    Cholesky factor of ``omega + gamma^2 I`` for numerical stability.
    """

    d = _as_float64(delta)
    om = _as_float64(omega)
    g = float(gamma)
    if g <= 0.0:
        raise ValueError("bandwidth gamma must be positive")
    dimension = d.shape[0]
    identity = np.eye(dimension, dtype=np.float64)
    total = om + (g * g) * identity
    factor = np.linalg.cholesky(total)
    # logdet(I + gamma^-2 omega) = logdet(omega + gamma^2 I) - D log(gamma^2)
    log_det_total = 2.0 * np.log(np.diag(factor)).sum()
    log_det_ratio = log_det_total - dimension * np.log(g * g)
    whitened = np.linalg.solve(factor, d)
    quadratic = float(whitened @ whitened)
    return float(np.exp(-0.5 * log_det_ratio - 0.5 * quadratic))


# ---------------------------------------------------------------------------
# Analytic convMMD loss (normative core, §4 of the contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceConvMMD:
    """Per-scale and aggregate analytic convMMD loss."""

    per_scale_loss: FloatArray  # (G,)
    loss: float


def convmmd_loss(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
) -> ReferenceConvMMD:
    """Evaluate the exact analytic convMMD loss in float64.

    ``L = (1 / (G N)) sum_g sum_i [ sum_{k,k'} pi_k pi_k' G(mu_k-mu_k',
    A_k^i + A_k'^i; gamma_g) - 2 sum_k pi_k G(x_i-mu_k, A_k^i; gamma_g) ]``
    with ``A_k^i = Sigma_k + S_i``. The theta-independent data-data term is
    omitted, exactly as in the reference method, so ``loss`` MAY be negative.
    """

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    noise = _as_float64(measurement_covariances)
    gammas = _as_float64(bandwidths)

    n_samples = x.shape[0]
    n_components = pis.shape[0]
    n_scales = gammas.shape[0]

    per_scale = np.zeros(n_scales, dtype=np.float64)
    for scale_index in range(n_scales):
        gamma = float(gammas[scale_index])
        accumulator = 0.0
        for sample in range(n_samples):
            convolved = [sigmas[k] + noise[sample] for k in range(n_components)]
            self_term = 0.0
            for k in range(n_components):
                for k_prime in range(n_components):
                    self_term += (
                        pis[k]
                        * pis[k_prime]
                        * expected_rbf_kernel(
                            mus[k] - mus[k_prime],
                            convolved[k] + convolved[k_prime],
                            gamma,
                        )
                    )
            cross_term = 0.0
            for k in range(n_components):
                cross_term += pis[k] * expected_rbf_kernel(
                    x[sample] - mus[k], convolved[k], gamma
                )
            accumulator += self_term - 2.0 * cross_term
        per_scale[scale_index] = accumulator / n_samples

    return ReferenceConvMMD(
        per_scale_loss=per_scale, loss=float(per_scale.mean())
    )


# ---------------------------------------------------------------------------
# Monte-Carlo reference estimator (stochastic; documents the SBI form)
# ---------------------------------------------------------------------------


def monte_carlo_loss(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
    rng: np.random.Generator,
    num_samples: int,
) -> float:
    """Independent NumPy Monte-Carlo estimate of the convMMD loss.

    A readable, reference-only reparameterized estimator whose expectation is
    :func:`convmmd_loss`; used to bind the ``MC -> analytic`` convergence
    property. Not the packaged estimator.
    """

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    noise = _as_float64(measurement_covariances)
    gammas = _as_float64(bandwidths)

    n_samples = x.shape[0]
    n_components = pis.shape[0]
    dimension = x.shape[1]
    latent_factor = np.linalg.cholesky(sigmas)  # (K, D, D)
    noise_factor = np.linalg.cholesky(noise)  # (N, D, D)

    total = 0.0
    count = 0
    for gamma in gammas:
        gamma_sq_2 = 2.0 * float(gamma) * float(gamma)
        for sample in range(n_samples):
            # Draw model-noisy samples per component for two independent copies.
            tilde_1 = np.empty((n_components, num_samples, dimension))
            tilde_2 = np.empty((n_components, num_samples, dimension))
            for k in range(n_components):
                z1 = rng.standard_normal((num_samples, dimension))
                z2 = rng.standard_normal((num_samples, dimension))
                e1 = rng.standard_normal((num_samples, dimension))
                e2 = rng.standard_normal((num_samples, dimension))
                tilde_1[k] = (
                    mus[k]
                    + z1 @ latent_factor[k].T
                    + e1 @ noise_factor[sample].T
                )
                tilde_2[k] = (
                    mus[k]
                    + z2 @ latent_factor[k].T
                    + e2 @ noise_factor[sample].T
                )
            self_term = 0.0
            for k in range(n_components):
                for k_prime in range(n_components):
                    diff = tilde_1[k] - tilde_2[k_prime]
                    kernel = np.exp(-np.sum(diff * diff, axis=-1) / gamma_sq_2)
                    self_term += pis[k] * pis[k_prime] * kernel.mean()
            cross_term = 0.0
            for k in range(n_components):
                diff = tilde_1[k] - x[sample]
                kernel = np.exp(-np.sum(diff * diff, axis=-1) / gamma_sq_2)
                cross_term += pis[k] * kernel.mean()
            total += self_term - 2.0 * cross_term
            count += 1
    return float(total / count)


# ---------------------------------------------------------------------------
# Empirical-Bayes denoiser (exact GMM posterior, §7 of the contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceDenoise:
    """Exact posterior quantities of the fitted-prior denoiser."""

    responsibilities: FloatArray  # (N, K)
    component_posterior_means: FloatArray  # (N, K, D)
    posterior_mean: FloatArray  # (N, D)


def _log_gaussian(x: FloatArray, mean: FloatArray, cov: FloatArray) -> float:
    dimension = x.shape[0]
    factor = np.linalg.cholesky(cov)
    whitened = np.linalg.solve(factor, x - mean)
    log_det = 2.0 * np.log(np.diag(factor)).sum()
    return float(
        -0.5 * (dimension * np.log(2.0 * np.pi) + log_det + whitened @ whitened)
    )


def denoise(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
) -> ReferenceDenoise:
    """Return exact posterior responsibilities, component means, and mean."""

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    noise = _as_float64(measurement_covariances)

    n_samples = x.shape[0]
    n_components = pis.shape[0]
    dimension = x.shape[1]
    log_pis = np.log(pis)

    responsibilities = np.empty((n_samples, n_components), dtype=np.float64)
    component_means = np.empty(
        (n_samples, n_components, dimension), dtype=np.float64
    )
    posterior_mean = np.empty((n_samples, dimension), dtype=np.float64)

    for sample in range(n_samples):
        log_joint = np.empty(n_components, dtype=np.float64)
        for k in range(n_components):
            marginal_cov = sigmas[k] + noise[sample]
            log_joint[k] = log_pis[k] + _log_gaussian(
                x[sample], mus[k], marginal_cov
            )
            gain = np.linalg.solve(marginal_cov, sigmas[k]).T
            component_means[sample, k] = mus[k] + gain @ (x[sample] - mus[k])
        maximum = log_joint.max()
        unnormalized = np.exp(log_joint - maximum)
        r = unnormalized / unnormalized.sum()
        responsibilities[sample] = r
        posterior_mean[sample] = np.einsum(
            "k,kd->d", r, component_means[sample]
        )

    return ReferenceDenoise(
        responsibilities=responsibilities,
        component_posterior_means=component_means,
        posterior_mean=posterior_mean,
    )


# ---------------------------------------------------------------------------
# Parameterization transform (§6 of the contract)
# ---------------------------------------------------------------------------


EPS_SIGMA = 1.0e-5
EPS_L = 1.0e-4


def _softplus(value: FloatArray) -> FloatArray:
    return np.logaddexp(0.0, value)


def softmax(alphas: npt.ArrayLike) -> FloatArray:
    """Numerically stable softmax over the last axis."""

    a = _as_float64(alphas)
    shifted = a - a.max()
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


def unconstrained_to_canonical(
    alphas: npt.ArrayLike,
    means: npt.ArrayLike,
    unconstrained_L: npt.ArrayLike,
    *,
    eps_sigma: float = EPS_SIGMA,
    eps_l: float = EPS_L,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Map ``(alpha, mu, Lambda)`` to canonical ``(pi, mu, Sigma)``.

    Mirrors the prototype's ``get_covariances``: strictly-lower entries pass
    through, the diagonal is ``softplus(diag) + eps_l``, and
    ``Sigma = L L^T + eps_sigma I``. Weights are ``softmax(alpha)``.
    """

    mus = _as_float64(means)
    lam = _as_float64(unconstrained_L)
    n_components, dimension, _ = lam.shape
    identity = np.eye(dimension, dtype=np.float64)
    covariances = np.empty((n_components, dimension, dimension), dtype=np.float64)
    for k in range(n_components):
        lower = np.tril(lam[k], k=-1)
        diagonal = np.diag(_softplus(np.diag(lam[k])) + eps_l)
        factor = lower + diagonal
        covariances[k] = factor @ factor.T + eps_sigma * identity
    return softmax(alphas), mus, covariances


# ---------------------------------------------------------------------------
# Masked (per-coordinate MAR) projected oracle (§16 of the contract)
# ---------------------------------------------------------------------------
#
# Derived independently from contract §16, not from any JAX implementation. For
# observation ``i`` with observed coordinates ``C_i`` (ascending), the projection
# ``P_i`` is the row-subset of the identity selecting ``C_i``; the observed
# sub-vector is ``x[i, C_i]``, the observed noise ``S_i`` is the principal block
# ``noise[i][C_i, C_i]``, and ``B_k^i = Sigma_k[C_i, C_i] + S_i``. Inputs are
# supplied at full width plus a boolean mask; each row is sliced to its observed
# subspace. Fully-observed masks reduce to the full-data oracle above.


def _observed_indices(mask_row: npt.ArrayLike) -> np.ndarray:
    """Ascending observed-coordinate indices for one boolean mask row."""

    return np.flatnonzero(np.asarray(mask_row, dtype=bool))


def _as_bool_mask(
    observed_mask: npt.ArrayLike, expected_shape: tuple[int, int]
) -> np.ndarray:
    mask = np.asarray(observed_mask)
    if mask.dtype != np.dtype(np.bool_):
        raise TypeError("observed_mask must be a boolean array")
    if mask.shape != expected_shape:
        raise ValueError(
            f"observed_mask shape {mask.shape} must match observations "
            f"{expected_shape}"
        )
    return mask


def convmmd_loss_masked(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
    sample_weight: npt.ArrayLike | None = None,
) -> ReferenceConvMMD:
    """Exact analytic masked convMMD loss (§16.3), informative-row normalized.

    ``per_scale[g] = (sum_{i: M_i>0} w_i ell_i(gamma_g)) / (sum_{i: M_i>0} w_i)``
    and ``loss = mean_g per_scale[g]``. Rows with ``M_i = 0`` contribute exactly
    zero and are excluded from the denominator. A collection with no informative
    row has loss exactly zero.
    """

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    noise = _as_float64(measurement_covariances)
    gammas = _as_float64(bandwidths)

    n_samples = x.shape[0]
    n_components = pis.shape[0]
    n_scales = gammas.shape[0]
    mask = _as_bool_mask(observed_mask, x.shape)

    if sample_weight is None:
        weights_row = np.ones(n_samples, dtype=np.float64)
    else:
        weights_row = _as_float64(sample_weight)
        if weights_row.shape != (n_samples,):
            raise ValueError("sample_weight must have shape (N,)")
        if np.any(weights_row < 0.0):
            raise ValueError("sample_weight must be nonnegative")

    informative = np.array([mask[i].any() for i in range(n_samples)], dtype=bool)
    informative_weight = float(weights_row[informative].sum())

    per_scale = np.zeros(n_scales, dtype=np.float64)
    if informative_weight == 0.0:
        # No informative row: loss is defined to be exactly zero (§16.3).
        return ReferenceConvMMD(per_scale_loss=per_scale, loss=0.0)

    for scale_index in range(n_scales):
        gamma = float(gammas[scale_index])
        accumulator = 0.0
        for sample in range(n_samples):
            coords = _observed_indices(mask[sample])
            if coords.size == 0:
                continue  # M_i = 0 contributes exactly zero
            block = np.ix_(coords, coords)
            x_obs = x[sample, coords]
            s_obs = noise[sample][block]
            convolved = [sigmas[k][block] + s_obs for k in range(n_components)]
            proj_mu = [mus[k][coords] for k in range(n_components)]
            self_term = 0.0
            for k in range(n_components):
                for k_prime in range(n_components):
                    self_term += (
                        pis[k]
                        * pis[k_prime]
                        * expected_rbf_kernel(
                            proj_mu[k] - proj_mu[k_prime],
                            convolved[k] + convolved[k_prime],
                            gamma,
                        )
                    )
            cross_term = 0.0
            for k in range(n_components):
                cross_term += pis[k] * expected_rbf_kernel(
                    x_obs - proj_mu[k], convolved[k], gamma
                )
            accumulator += weights_row[sample] * (self_term - 2.0 * cross_term)
        per_scale[scale_index] = accumulator / informative_weight

    return ReferenceConvMMD(
        per_scale_loss=per_scale, loss=float(per_scale.mean())
    )


def denoise_masked(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
) -> ReferenceDenoise:
    """Exact projected empirical-Bayes posterior (§16.4); full-``D`` output.

    ``r_ik ~ pi_k N(x_obs; P_i mu_k, B_k^i)``,
    ``m_ik = mu_k + Sigma_k P_i^T (B_k^i)^{-1} (x_obs - P_i mu_k)`` (full ``D``),
    ``zhat_i = sum_k r_ik m_ik``. For ``M_i = 0`` the responsibilities are the
    prior weights and ``zhat_i`` is the prior mean ``sum_k pi_k mu_k``.
    """

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    noise = _as_float64(measurement_covariances)

    n_samples = x.shape[0]
    n_components = pis.shape[0]
    dimension = x.shape[1]
    mask = _as_bool_mask(observed_mask, x.shape)
    log_pis = np.log(pis)
    prior_mean = np.einsum("k,kd->d", pis, mus)

    responsibilities = np.empty((n_samples, n_components), dtype=np.float64)
    component_means = np.empty(
        (n_samples, n_components, dimension), dtype=np.float64
    )
    posterior_mean = np.empty((n_samples, dimension), dtype=np.float64)

    for sample in range(n_samples):
        coords = _observed_indices(mask[sample])
        if coords.size == 0:
            responsibilities[sample] = pis
            component_means[sample] = mus
            posterior_mean[sample] = prior_mean
            continue
        block = np.ix_(coords, coords)
        x_obs = x[sample, coords]
        s_obs = noise[sample][block]
        log_joint = np.empty(n_components, dtype=np.float64)
        for k in range(n_components):
            proj_mu = mus[k][coords]
            marginal = sigmas[k][block] + s_obs  # B_k^i, (M_i, M_i)
            log_joint[k] = log_pis[k] + _log_gaussian(x_obs, proj_mu, marginal)
            cross_cov = sigmas[k][:, coords]  # Sigma_k P_i^T, (D, M_i)
            gain = np.linalg.solve(marginal, x_obs - proj_mu)  # (M_i,)
            component_means[sample, k] = mus[k] + cross_cov @ gain
        maximum = log_joint.max()
        unnormalized = np.exp(log_joint - maximum)
        r = unnormalized / unnormalized.sum()
        responsibilities[sample] = r
        posterior_mean[sample] = np.einsum(
            "k,kd->d", r, component_means[sample]
        )

    return ReferenceDenoise(
        responsibilities=responsibilities,
        component_posterior_means=component_means,
        posterior_mean=posterior_mean,
    )


def median_bandwidths_masked(
    observations: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    *,
    n_scales: int = 9,
    log10_low: float = -2.0,
    log10_high: float = 2.0,
) -> FloatArray:
    """Single global masked bandwidth set (§16.6).

    ``gamma_g = b_mask * 10**s_g`` where ``b_mask`` is the median, over pairs
    ``(i, j), i < j`` sharing at least one observed coordinate, of the Euclidean
    distance on their shared coordinates. Equals :func:`median_bandwidths` (via
    the full pairwise distance) on fully-observed data; raises when no pair shares
    an observed coordinate.
    """

    x = _as_float64(observations)
    mask = _as_bool_mask(observed_mask, x.shape)
    n = x.shape[0]

    distances: list[float] = []
    for i in range(n):
        coords_i = _observed_indices(mask[i])
        if coords_i.size == 0:
            continue
        for j in range(i + 1, n):
            coords_j = _observed_indices(mask[j])
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
    scales = np.logspace(log10_low, log10_high, n_scales)
    return base * scales


def monte_carlo_loss_masked(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
    rng: np.random.Generator,
    num_samples: int,
) -> float:
    """Independent NumPy Monte-Carlo estimate of the masked loss (§16.5).

    Unweighted reference estimator whose expectation is
    :func:`convmmd_loss_masked` with uniform weights; used to bind the
    ``MC -> analytic`` convergence property under missingness. Model draws are
    projected to the observed subspace before the observed-space noise is added.
    """

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    noise = _as_float64(measurement_covariances)
    gammas = _as_float64(bandwidths)

    n_samples = x.shape[0]
    n_components = pis.shape[0]
    dimension = x.shape[1]
    mask = _as_bool_mask(observed_mask, x.shape)
    latent_factor = np.linalg.cholesky(sigmas)  # (K, D, D)

    total = 0.0
    count = 0
    for gamma in gammas:
        gamma_sq_2 = 2.0 * float(gamma) * float(gamma)
        for sample in range(n_samples):
            coords = _observed_indices(mask[sample])
            if coords.size == 0:
                continue
            block = np.ix_(coords, coords)
            x_obs = x[sample, coords]
            noise_factor = np.linalg.cholesky(noise[sample][block])  # (M_i, M_i)
            m_dim = coords.size
            tilde_1 = np.empty((n_components, num_samples, m_dim))
            tilde_2 = np.empty((n_components, num_samples, m_dim))
            for k in range(n_components):
                z1 = rng.standard_normal((num_samples, dimension))
                z2 = rng.standard_normal((num_samples, dimension))
                e1 = rng.standard_normal((num_samples, m_dim))
                e2 = rng.standard_normal((num_samples, m_dim))
                latent_1 = mus[k] + z1 @ latent_factor[k].T  # (num_samples, D)
                latent_2 = mus[k] + z2 @ latent_factor[k].T
                tilde_1[k] = latent_1[:, coords] + e1 @ noise_factor.T
                tilde_2[k] = latent_2[:, coords] + e2 @ noise_factor.T
            self_term = 0.0
            for k in range(n_components):
                for k_prime in range(n_components):
                    diff = tilde_1[k] - tilde_2[k_prime]
                    kernel = np.exp(-np.sum(diff * diff, axis=-1) / gamma_sq_2)
                    self_term += pis[k] * pis[k_prime] * kernel.mean()
            cross_term = 0.0
            for k in range(n_components):
                diff = tilde_1[k] - x_obs
                kernel = np.exp(-np.sum(diff * diff, axis=-1) / gamma_sq_2)
                cross_term += pis[k] * kernel.mean()
            total += self_term - 2.0 * cross_term
            count += 1
    if count == 0:
        return 0.0
    return float(total / count)
