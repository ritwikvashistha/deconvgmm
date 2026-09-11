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


# ---------------------------------------------------------------------------
# Known-selection (MNAR) oracle: Gaussian window Omega(z) (§17 of the contract)
# ---------------------------------------------------------------------------
#
# Selection on the TRUE value with a Gaussian window
# ``Omega(z) = exp(-1/2 (z-a)^T Psi_inv (z-a))`` acts on the latent mixture BEFORE
# projection/noise (contract §17.5). This oracle uses the independent TEXTBOOK route
# ``w_k = (2 pi)^{D/2} |Psi|^{1/2} N(mu_k; a, Sigma_k + Psi)`` (Psi = inv(Psi_inv)),
# a different algebra from the packaged |Psi|-free precision-native form, so the two
# must agree at machine epsilon. The selected analytic loss/denoiser are then the
# §16 masked oracle evaluated verbatim on the transformed ``(pi', mu', Sigma')``.


@dataclass(frozen=True)
class ReferenceSelectionTransform:
    """Gaussian-window selected latent mixture (§17.5)."""

    weights: FloatArray  # (K,) pi'
    means: FloatArray  # (K, D) mu'
    covariances: FloatArray  # (K, D, D) Sigma'
    effective_volume: float  # Z_theta = int Omega q_theta


def gaussian_window_transform(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    location: npt.ArrayLike,
    precision: npt.ArrayLike,
) -> ReferenceSelectionTransform:
    """Transform ``(pi, mu, Sigma)`` under a Gaussian window ``Omega(z)`` (§17.5).

    Textbook route (independent of the packaged precision-native form):
    ``Sigma'_k = (Sigma_k^{-1} + Psi_inv)^{-1}``,
    ``mu'_k = Sigma'_k (Sigma_k^{-1} mu_k + Psi_inv a)``,
    ``w_k = (2 pi)^{D/2} |Psi|^{1/2} N(mu_k; a, Sigma_k + Psi)``,
    ``pi'_k = pi_k w_k / Z``, ``Z = sum_k pi_k w_k = int Omega q_theta``. Requires a
    positive-definite ``precision`` (invertible ``Psi``).
    """

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    a = _as_float64(location)
    psi_inv = _as_float64(precision)

    n_components, dimension = mus.shape
    psi = np.linalg.inv(psi_inv)
    log_det_psi = np.linalg.slogdet(psi)[1]
    const = 0.5 * dimension * np.log(2.0 * np.pi) + 0.5 * log_det_psi

    mu_prime = np.empty((n_components, dimension), dtype=np.float64)
    sigma_prime = np.empty((n_components, dimension, dimension), dtype=np.float64)
    log_w = np.empty(n_components, dtype=np.float64)
    for k in range(n_components):
        sigma_inv = np.linalg.inv(sigmas[k])
        cov = np.linalg.inv(sigma_inv + psi_inv)
        cov = 0.5 * (cov + cov.T)
        sigma_prime[k] = cov
        mu_prime[k] = cov @ (sigma_inv @ mus[k] + psi_inv @ a)
        # log w_k = const + log N(mu_k; a, Sigma_k + Psi)
        log_w[k] = const + _log_gaussian(mus[k], a, sigmas[k] + psi)

    log_unnorm = np.log(pis) + log_w
    shift = log_unnorm.max()
    unnorm = np.exp(log_unnorm - shift)
    weights_prime = unnorm / unnorm.sum()
    effective_volume = float(np.exp(shift) * unnorm.sum())  # = sum_k pi_k w_k
    return ReferenceSelectionTransform(
        weights=weights_prime,
        means=mu_prime,
        covariances=sigma_prime,
        effective_volume=effective_volume,
    )


def convmmd_loss_selected(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
    location: npt.ArrayLike,
    precision: npt.ArrayLike,
    sample_weight: npt.ArrayLike | None = None,
) -> ReferenceConvMMD:
    """Exact analytic selected loss for a Gaussian window ``Omega(z)`` (§17.5).

    Transform to ``(pi', mu', Sigma')`` then evaluate the §16 masked oracle: the
    selected analytic loss is the projected loss on the selected latent mixture.
    """

    transform = gaussian_window_transform(
        weights, means, covariances, location, precision
    )
    return convmmd_loss_masked(
        transform.weights,
        transform.means,
        transform.covariances,
        observations,
        observed_mask,
        measurement_covariances,
        bandwidths,
        sample_weight,
    )


def denoise_selected(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    location: npt.ArrayLike,
    precision: npt.ArrayLike,
) -> ReferenceDenoise:
    """Selection-aware denoiser for a Gaussian window ``Omega(z)`` (§17.6).

    The tilted posterior ``p(z | x, det) ~ Omega(z) q_theta(z) N(x; P z, S)`` is the
    §16.4 projected GMM posterior of the transformed prior ``(pi', mu', Sigma')``.
    """

    transform = gaussian_window_transform(
        weights, means, covariances, location, precision
    )
    return denoise_masked(
        transform.weights,
        transform.means,
        transform.covariances,
        observations,
        observed_mask,
        measurement_covariances,
    )


def gaussian_window_transform_observed(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    noise: npt.ArrayLike,
    location: npt.ArrayLike,
    precision: npt.ArrayLike,
) -> ReferenceSelectionTransform:
    """Observed-value ``Omega(x-tilde)`` transform (§17.11; homoscedastic).

    The window acts on the noise-convolved model ``sum_k pi_k N(mu_k, Sigma_k + S)``,
    so it is the §17.5 transform applied to the inflated covariances ``Sigma_k + S``.
    """

    sigmas = _as_float64(covariances)
    noise_matrix = _as_float64(noise)
    if noise_matrix.ndim == 3:
        noise_matrix = noise_matrix[0]  # homoscedastic
    inflated = sigmas + noise_matrix[None, :, :]
    return gaussian_window_transform(weights, means, inflated, location, precision)


def convmmd_loss_selected_observed(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    noise: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
    location: npt.ArrayLike,
    precision: npt.ArrayLike,
) -> ReferenceConvMMD:
    """Analytic ``Omega(x-tilde)`` selected loss (§17.11; homoscedastic, fully observed).

    The selected observed model is a Gaussian mixture ``(pi'', nu'', B'')``, so the loss
    is the base §4 loss on those parameters with **zero** measurement noise.
    """

    transform = gaussian_window_transform_observed(
        weights, means, covariances, noise, location, precision
    )
    x = _as_float64(observations)
    zero_noise = np.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float64)
    return convmmd_loss(
        transform.weights, transform.means, transform.covariances, x, zero_noise, bandwidths
    )


def convmmd_loss_selected_observed_hetero(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    noise: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
    location: npt.ArrayLike,
    precision: npt.ArrayLike,
) -> ReferenceConvMMD:
    """Heteroscedastic ``Omega(x-tilde)`` per-object analytic loss (§17.13).

    ``noise`` is a genuinely per-object ``(N, D, D)`` stack of known covariances
    ``S_i``. For each object the textbook §17.5 transform is applied to the inflated
    covariances ``Sigma_k + S_i`` (original means), giving the per-object selected
    observed mixture ``(pi''_i, nu''_i, B''_i)``; the loss is the mean over objects of
    the base §4 one-sample discrepancy of ``x_i`` against that mixture (zero residual
    noise). Independent textbook algebra; the packaged per-object leaf must agree at
    machine epsilon. No noise model.
    """

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    per_item_noise = _as_float64(noise)  # (N, D, D)
    gammas = _as_float64(bandwidths)

    n_samples = x.shape[0]
    n_components = pis.shape[0]
    n_scales = gammas.shape[0]

    # The per-object transform is invariant across bandwidths; compute it once per
    # object (identical summation order as the naive scale-outer loop -> same value).
    transforms = [
        gaussian_window_transform(
            pis, mus, sigmas + per_item_noise[sample][None, :, :], location, precision
        )
        for sample in range(n_samples)
    ]

    per_scale = np.zeros(n_scales, dtype=np.float64)
    for scale_index in range(n_scales):
        gamma = float(gammas[scale_index])
        accumulator = 0.0
        for sample in range(n_samples):
            transform = transforms[sample]
            pipp = transform.weights
            nupp = transform.means
            bpp = transform.covariances
            self_term = 0.0
            for k in range(n_components):
                for k_prime in range(n_components):
                    self_term += (
                        pipp[k]
                        * pipp[k_prime]
                        * expected_rbf_kernel(
                            nupp[k] - nupp[k_prime], bpp[k] + bpp[k_prime], gamma
                        )
                    )
            cross_term = 0.0
            for k in range(n_components):
                cross_term += pipp[k] * expected_rbf_kernel(
                    x[sample] - nupp[k], bpp[k], gamma
                )
            accumulator += self_term - 2.0 * cross_term
        per_scale[scale_index] = accumulator / n_samples

    return ReferenceConvMMD(per_scale_loss=per_scale, loss=float(per_scale.mean()))


def monte_carlo_loss_selected_observed_hetero(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    noise: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
    omega,
    rng: np.random.Generator,
    num_samples: int,
) -> float:
    """Independent NumPy per-object SNIS estimate of the heteroscedastic ``Omega(x-tilde)``
    loss (§17.13), using each object's **known** ``S_i`` and no noise model.

    Draws two independent sets ``z ~ q_theta``, ``eps ~ N(0, S_i)`` per object, forms
    ``x-tilde = z + eps``, weights by ``omega(x-tilde)``, and forms the self-normalized
    cross/self terms with the per-object normalizer ``Zhat_i`` (full cross-set double
    sum). Its expectation limit is :func:`convmmd_loss_selected_observed_hetero` for a
    Gaussian ``omega``; **biased at finite ``num_samples``, consistent as it grows.**
    """

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    per_item_noise = _as_float64(noise)  # (N, D, D)
    gammas = _as_float64(bandwidths)

    n_samples = x.shape[0]
    dimension = x.shape[1]
    latent_factor = np.linalg.cholesky(sigmas)  # (K, D, D)

    total = 0.0
    for sample in range(n_samples):
        noise_factor = np.linalg.cholesky(per_item_noise[sample])  # (D, D)
        z1 = rng.standard_normal((mus.shape[0], num_samples, dimension))
        z2 = rng.standard_normal((mus.shape[0], num_samples, dimension))
        e1 = rng.standard_normal((num_samples, dimension))
        e2 = rng.standard_normal((num_samples, dimension))
        latent_1 = mus[:, None, :] + np.einsum("kde,kme->kmd", latent_factor, z1)
        latent_2 = mus[:, None, :] + np.einsum("kde,kme->kmd", latent_factor, z2)
        tilde_1 = latent_1 + (e1 @ noise_factor.T)[None, :, :]  # (K, M, D)
        tilde_2 = latent_2 + (e2 @ noise_factor.T)[None, :, :]
        weight_1 = np.asarray(omega(tilde_1), dtype=np.float64)  # (K, M)
        weight_2 = np.asarray(omega(tilde_2), dtype=np.float64)
        z_hat_1 = float(np.sum(pis * weight_1.mean(axis=1)))
        z_hat_2 = float(np.sum(pis * weight_2.mean(axis=1)))

        diff_cross = tilde_1 - x[sample][None, None, :]
        dist_cross = np.sum(diff_cross * diff_cross, axis=-1)  # (K, M)
        sq_1 = np.sum(tilde_1 * tilde_1, axis=-1)  # (K, M)
        sq_2 = np.sum(tilde_2 * tilde_2, axis=-1)
        gram = np.einsum("kma,lpa->klmp", tilde_1, tilde_2)  # (K, K, M, M)
        dist_self = np.maximum(
            sq_1[:, None, :, None] + sq_2[None, :, None, :] - 2.0 * gram, 0.0
        )

        object_loss = 0.0
        for gamma in gammas:
            gamma_sq_2 = 2.0 * float(gamma) * float(gamma)
            kernel_cross = np.exp(-dist_cross / gamma_sq_2)  # (K, M)
            cross = float(np.sum(pis * np.mean(weight_1 * kernel_cross, axis=1))) / z_hat_1
            kernel_self = np.exp(-dist_self / gamma_sq_2)  # (K, K, M, M)
            weight_outer = weight_1[:, None, :, None] * weight_2[None, :, None, :]
            self_kk = np.mean(weight_outer * kernel_self, axis=(2, 3))  # (K, K)
            self_term = float(
                np.sum(pis[:, None] * pis[None, :] * self_kk)
            ) / (z_hat_1 * z_hat_2)
            object_loss += self_term - 2.0 * cross
        total += object_loss / len(gammas)

    return total / n_samples


@dataclass(frozen=True)
class ReferenceSelectedMC:
    """SNIS selected loss with importance-sampling diagnostics (§17.4/§17.7)."""

    loss: float
    ess: float  # mean effective sample size across informative rows/scales
    effective_volume: float  # mean SNIS Zhat across informative rows/scales


def monte_carlo_loss_selected(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
    omega,
    rng: np.random.Generator,
    num_samples: int,
    *,
    convention: str = "on_z",
    sample_weight: npt.ArrayLike | None = None,
) -> ReferenceSelectedMC:
    """Independent NumPy SNIS estimate of the selected loss (§17.4, ``on_z``).

    Draws two independent reparameterized sets from the un-selected model, weights
    each latent draw by ``omega`` (a callable ``R^D -> [0, 1]`` applied to the
    latent value), and forms the self-normalized cross/self terms with the
    per-observation normalizer ``Zhat``. The self term is the full cross-set double
    sum (the SNIS ratio normalizer does not factor across a paired diagonal). Its
    expectation limit is :func:`convmmd_loss_selected`; it is **biased at finite
    ``num_samples``, consistent as ``num_samples -> inf``**. Also returns the mean
    effective sample size ``(sum u)^2 / sum u^2`` (``u = pi_k omega``) and the mean
    ``Zhat`` as diagnostics.
    """

    if convention != "on_z":
        raise NotImplementedError("only on_z selection is implemented")

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    noise = _as_float64(measurement_covariances)
    gammas = _as_float64(bandwidths)

    n_samples = x.shape[0]
    n_components = pis.shape[0]
    dimension = x.shape[1]
    n_scales = gammas.shape[0]
    mask = _as_bool_mask(observed_mask, x.shape)
    latent_factor = np.linalg.cholesky(sigmas)  # (K, D, D)

    if sample_weight is None:
        weights_row = np.ones(n_samples, dtype=np.float64)
    else:
        weights_row = _as_float64(sample_weight)

    informative = np.array([mask[i].any() for i in range(n_samples)], dtype=bool)
    informative_weight = float(weights_row[informative].sum())
    if informative_weight == 0.0:
        return ReferenceSelectedMC(loss=0.0, ess=0.0, effective_volume=0.0)

    total = 0.0
    ess_accumulator = 0.0
    z_accumulator = 0.0
    diagnostic_count = 0
    for gamma in gammas:
        gamma_sq_2 = 2.0 * float(gamma) * float(gamma)
        for sample in range(n_samples):
            coords = _observed_indices(mask[sample])
            if coords.size == 0:
                continue
            block = np.ix_(coords, coords)
            x_obs = x[sample, coords]
            noise_factor = np.linalg.cholesky(noise[sample][block])
            m_dim = coords.size
            tilde_1 = np.empty((n_components, num_samples, m_dim))
            tilde_2 = np.empty((n_components, num_samples, m_dim))
            omega_1 = np.empty((n_components, num_samples))
            omega_2 = np.empty((n_components, num_samples))
            for k in range(n_components):
                z1 = rng.standard_normal((num_samples, dimension))
                z2 = rng.standard_normal((num_samples, dimension))
                latent_1 = mus[k] + z1 @ latent_factor[k].T  # (M, D)
                latent_2 = mus[k] + z2 @ latent_factor[k].T
                omega_1[k] = np.asarray(omega(latent_1), dtype=np.float64)
                omega_2[k] = np.asarray(omega(latent_2), dtype=np.float64)
                e1 = rng.standard_normal((num_samples, m_dim))
                e2 = rng.standard_normal((num_samples, m_dim))
                tilde_1[k] = latent_1[:, coords] + e1 @ noise_factor.T
                tilde_2[k] = latent_2[:, coords] + e2 @ noise_factor.T
            z_hat_1 = float(sum(pis[k] * omega_1[k].mean() for k in range(n_components)))
            z_hat_2 = float(sum(pis[k] * omega_2[k].mean() for k in range(n_components)))
            cross_term = 0.0
            for k in range(n_components):
                diff = tilde_1[k] - x_obs
                kernel = np.exp(-np.sum(diff * diff, axis=-1) / gamma_sq_2)
                cross_term += pis[k] * (omega_1[k] * kernel).mean()
            cross_term /= z_hat_1
            self_term = 0.0
            for k in range(n_components):
                for k_prime in range(n_components):
                    diff = tilde_1[k][:, None, :] - tilde_2[k_prime][None, :, :]
                    kernel = np.exp(-np.sum(diff * diff, axis=-1) / gamma_sq_2)
                    outer = np.outer(omega_1[k], omega_2[k_prime])
                    self_term += pis[k] * pis[k_prime] * (outer * kernel).mean()
            self_term /= z_hat_1 * z_hat_2
            total += weights_row[sample] * (self_term - 2.0 * cross_term)
            u = np.concatenate([pis[k] * omega_1[k] for k in range(n_components)])
            ess_accumulator += float((u.sum() ** 2) / (u * u).sum())
            z_accumulator += z_hat_1
            diagnostic_count += 1

    loss = total / (informative_weight * n_scales)
    return ReferenceSelectedMC(
        loss=float(loss),
        ess=ess_accumulator / diagnostic_count,
        effective_volume=z_accumulator / diagnostic_count,
    )


def snis_denoise_selected(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    omega,
    rng: np.random.Generator,
    num_samples: int,
    *,
    convention: str = "on_z",
) -> FloatArray:
    """Independent NumPy SNIS posterior mean under selection (§17.6, ``on_z``).

    Samples the MAR posterior ``p_MAR(z | x)`` (the closed-form projected GMM
    posterior of the un-selected prior), weights each draw by ``omega``, and returns
    the self-normalized weighted mean, full ``D``, shape ``(N, D)``. ``M=0`` rows use
    the prior as the proposal (the selected prior mean). Its limit as
    ``num_samples -> inf`` is :func:`denoise_selected` for a Gaussian ``Omega``.
    """

    if convention != "on_z":
        raise NotImplementedError("only on_z selection is implemented")

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    noise = _as_float64(measurement_covariances)

    n_samples = x.shape[0]
    n_components = pis.shape[0]
    dimension = x.shape[1]
    mask = _as_bool_mask(observed_mask, x.shape)
    prior_factor = np.linalg.cholesky(sigmas)  # (K, D, D)
    posterior_mean = np.empty((n_samples, dimension), dtype=np.float64)

    for sample in range(n_samples):
        coords = _observed_indices(mask[sample])
        if coords.size == 0:
            component = rng.choice(n_components, size=num_samples, p=pis)
            draws = np.empty((num_samples, dimension))
            for k in range(n_components):
                selected = component == k
                count = int(selected.sum())
                if count:
                    draws[selected] = (
                        mus[k]
                        + rng.standard_normal((count, dimension)) @ prior_factor[k].T
                    )
            weight = np.asarray(omega(draws), dtype=np.float64)
            posterior_mean[sample] = (weight[:, None] * draws).sum(0) / weight.sum()
            continue

        block = np.ix_(coords, coords)
        x_obs = x[sample, coords]
        s_obs = noise[sample][block]
        log_joint = np.empty(n_components, dtype=np.float64)
        component_means = np.empty((n_components, dimension), dtype=np.float64)
        component_covs = np.empty((n_components, dimension, dimension), dtype=np.float64)
        for k in range(n_components):
            proj_mu = mus[k][coords]
            marginal = sigmas[k][block] + s_obs
            log_joint[k] = np.log(pis[k]) + _log_gaussian(x_obs, proj_mu, marginal)
            cross = sigmas[k][:, coords]  # Sigma_k P^T, (D, M)
            marginal_inv = np.linalg.inv(marginal)
            component_means[k] = mus[k] + cross @ marginal_inv @ (x_obs - proj_mu)
            component_covs[k] = sigmas[k] - cross @ marginal_inv @ cross.T
        maximum = log_joint.max()
        responsibilities = np.exp(log_joint - maximum)
        responsibilities /= responsibilities.sum()

        component = rng.choice(n_components, size=num_samples, p=responsibilities)
        draws = np.empty((num_samples, dimension))
        for k in range(n_components):
            selected = component == k
            count = int(selected.sum())
            if count:
                draws[selected] = rng.multivariate_normal(
                    component_means[k], component_covs[k], size=count
                )
        weight = np.asarray(omega(draws), dtype=np.float64)
        posterior_mean[sample] = (weight[:, None] * draws).sum(0) / weight.sum()

    return posterior_mean


# ---------------------------------------------------------------------------
# Heteroscedastic Omega(x-tilde) + MAR projection oracle (§17.14 of the contract)
# ---------------------------------------------------------------------------
#
# Independent textbook route for the MAR-composed selection-on-observed-value case.
# For object ``i`` observed in coordinates ``C_i`` with its own noise ``S_i``, the
# Gaussian window is MARGINALIZED onto the observed subspace -- location ``a[C_i]`` and
# precision ``(Psi[C_i, C_i])^{-1}`` (the inverse of the COVARIANCE principal submatrix,
# ``Psi = precision^{-1}``), NOT ``precision[C_i, C_i]``. The §17.5 transform then runs
# per object on the projected inflated covariance ``Sigma_k[C_i, C_i] + S_i`` and the loss
# is the informative-weight-normalized sum of per-object one-sample discrepancies.


def gaussian_window_marginal(
    location: npt.ArrayLike, precision: npt.ArrayLike, coords: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Marginalized observed-subspace Gaussian window ``(a_i, Psi_i_inv)`` (§17.14).

    ``a_i = a[coords]`` and ``Psi_i_inv = (Psi[coords, coords])^{-1}`` with
    ``Psi = precision^{-1}`` -- the inverse of the covariance principal submatrix, **not**
    ``precision[coords, coords]`` (they differ by a Schur complement). At ``precision = 0``
    (no selection) returns ``Psi_i_inv = 0`` without inverting a singular precision.
    Independent textbook route for the ``CMMD-SEL-OBSMAR-MARGWIN`` gate.
    """

    a = _as_float64(location)
    psi_inv = _as_float64(precision)
    coord_index = np.asarray(coords, dtype=int)
    a_i = a[coord_index]
    if np.all(psi_inv == 0.0):
        m_dim = int(coord_index.size)
        return a_i, np.zeros((m_dim, m_dim), dtype=np.float64)
    psi = np.linalg.inv(psi_inv)
    psi_sub = psi[np.ix_(coord_index, coord_index)]
    return a_i, np.linalg.inv(psi_sub)


def convmmd_loss_selected_observed_masked(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
    location: npt.ArrayLike,
    precision: npt.ArrayLike,
    sample_weight: npt.ArrayLike | None = None,
) -> ReferenceConvMMD:
    """Heteroscedastic ``Omega(x-tilde)`` + MAR projection per-object analytic loss (§17.14).

    ``noise`` is a genuinely per-object ``(N, D, D)`` stack of known ``S_i``. For each
    detected object ``i`` (observed coordinates ``C_i``, ``M_i > 0``), the window is
    marginalized onto ``R^{M_i}`` and the textbook §17.5 transform is applied to the
    projected inflated covariance ``Sigma_k[C_i, C_i] + S_i`` (projected means
    ``mu_k[C_i]``), giving that object's selected observed mixture ``(pi_i, nu_i, C_i)``;
    the loss is the informative-weight-normalized sum of the base §4 one-sample
    discrepancies (zero residual noise). Rows with ``M_i = 0`` contribute exactly zero.
    Independent textbook algebra; the packaged leaf must agree at machine epsilon. No noise
    model.
    """

    if np.all(_as_float64(precision) == 0.0):
        # Psi^{-1} = 0 (no selection): the marginalized window vanishes on every subspace,
        # so the loss is exactly the base §16 masked (MAR) loss (§17.14 reduction). Routed
        # here because the textbook transform needs an invertible marginal precision.
        return convmmd_loss_masked(
            weights, means, covariances, observations, observed_mask,
            measurement_covariances, bandwidths, sample_weight,
        )

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
        return ReferenceConvMMD(per_scale_loss=per_scale, loss=0.0)

    # Per-object transform is invariant across bandwidths; compute it once per object.
    transforms: list = []
    for sample in range(n_samples):
        coords = _observed_indices(mask[sample])
        if coords.size == 0:
            transforms.append(None)
            continue
        block = np.ix_(coords, coords)
        a_i, psi_i_inv = gaussian_window_marginal(location, precision, coords)
        s_obs = noise[sample][block]
        proj_mu = np.stack([mus[k][coords] for k in range(n_components)])  # (K, M)
        inflated = np.stack(
            [sigmas[k][block] + s_obs for k in range(n_components)]
        )  # B_k^{(i)}, (K, M, M)
        transform = gaussian_window_transform(pis, proj_mu, inflated, a_i, psi_i_inv)
        transforms.append(
            (x[sample, coords], transform.weights, transform.means, transform.covariances)
        )

    for scale_index in range(n_scales):
        gamma = float(gammas[scale_index])
        accumulator = 0.0
        for sample in range(n_samples):
            record = transforms[sample]
            if record is None:
                continue
            x_obs, pipp, nupp, bpp = record
            self_term = 0.0
            for k in range(n_components):
                for k_prime in range(n_components):
                    self_term += (
                        pipp[k]
                        * pipp[k_prime]
                        * expected_rbf_kernel(
                            nupp[k] - nupp[k_prime], bpp[k] + bpp[k_prime], gamma
                        )
                    )
            cross_term = 0.0
            for k in range(n_components):
                cross_term += pipp[k] * expected_rbf_kernel(
                    x_obs - nupp[k], bpp[k], gamma
                )
            accumulator += weights_row[sample] * (self_term - 2.0 * cross_term)
        per_scale[scale_index] = accumulator / informative_weight

    return ReferenceConvMMD(per_scale_loss=per_scale, loss=float(per_scale.mean()))


def monte_carlo_loss_selected_observed_masked(
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    covariances: npt.ArrayLike,
    observations: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    measurement_covariances: npt.ArrayLike,
    bandwidths: npt.ArrayLike,
    location: npt.ArrayLike,
    precision: npt.ArrayLike,
    rng: np.random.Generator,
    num_samples: int,
) -> float:
    """Independent NumPy per-object projected SNIS estimate of the §17.14 loss.

    Builds the **marginalized** Gaussian window per detected object and draws
    ``z ~ q_theta``, projects ``P_i z`` to the observed subspace, adds the object's own
    ``eps ~ N(0, S_i)``, weights by the marginalized window, and forms the self-normalized
    cross/self terms with the per-object ``Zhat_i`` (full cross-set double sum). Its
    expectation limit is :func:`convmmd_loss_selected_observed_masked`; biased at finite
    ``num_samples``, consistent as it grows. Informative-weight-normalized (uniform rows).
    """

    pis = _as_float64(weights)
    mus = _as_float64(means)
    sigmas = _as_float64(covariances)
    x = _as_float64(observations)
    noise = _as_float64(measurement_covariances)
    gammas = _as_float64(bandwidths)
    mask = _as_bool_mask(observed_mask, x.shape)

    n_samples = x.shape[0]
    n_components = pis.shape[0]
    latent_factor = np.linalg.cholesky(sigmas)  # (K, D, D)
    informative = [i for i in range(n_samples) if mask[i].any()]
    if not informative:
        return 0.0

    total = 0.0
    for sample in informative:
        coords = _observed_indices(mask[sample])
        m_dim = int(coords.size)
        block = np.ix_(coords, coords)
        a_i, psi_i_inv = gaussian_window_marginal(location, precision, coords)

        def omega(value: FloatArray) -> FloatArray:
            delta = value - a_i
            return np.exp(-0.5 * np.einsum("...a,ab,...b->...", delta, psi_i_inv, delta))

        noise_factor = np.linalg.cholesky(noise[sample][block])  # (M, M)
        z1 = rng.standard_normal((n_components, num_samples, mus.shape[1]))
        z2 = rng.standard_normal((n_components, num_samples, mus.shape[1]))
        e1 = rng.standard_normal((num_samples, m_dim))
        e2 = rng.standard_normal((num_samples, m_dim))
        latent_1 = mus[:, None, :] + np.einsum("kde,kme->kmd", latent_factor, z1)
        latent_2 = mus[:, None, :] + np.einsum("kde,kme->kmd", latent_factor, z2)
        proj_1 = latent_1[:, :, coords]  # (K, M_s, M)  coordinate selection
        proj_2 = latent_2[:, :, coords]
        tilde_1 = proj_1 + (e1 @ noise_factor.T)[None, :, :]  # (K, M_s, M)
        tilde_2 = proj_2 + (e2 @ noise_factor.T)[None, :, :]
        weight_1 = np.asarray(omega(tilde_1), dtype=np.float64)  # (K, M_s)
        weight_2 = np.asarray(omega(tilde_2), dtype=np.float64)
        z_hat_1 = float(np.sum(pis * weight_1.mean(axis=1)))
        z_hat_2 = float(np.sum(pis * weight_2.mean(axis=1)))

        diff_cross = tilde_1 - x[sample, coords][None, None, :]
        dist_cross = np.sum(diff_cross * diff_cross, axis=-1)  # (K, M_s)
        sq_1 = np.sum(tilde_1 * tilde_1, axis=-1)
        sq_2 = np.sum(tilde_2 * tilde_2, axis=-1)
        gram = np.einsum("kma,lpa->klmp", tilde_1, tilde_2)
        dist_self = np.maximum(
            sq_1[:, None, :, None] + sq_2[None, :, None, :] - 2.0 * gram, 0.0
        )

        object_loss = 0.0
        for gamma in gammas:
            gamma_sq_2 = 2.0 * float(gamma) * float(gamma)
            kernel_cross = np.exp(-dist_cross / gamma_sq_2)
            cross = float(np.sum(pis * np.mean(weight_1 * kernel_cross, axis=1))) / z_hat_1
            kernel_self = np.exp(-dist_self / gamma_sq_2)
            weight_outer = weight_1[:, None, :, None] * weight_2[None, :, None, :]
            self_kk = np.mean(weight_outer * kernel_self, axis=(2, 3))
            self_term = float(
                np.sum(pis[:, None] * pis[None, :] * self_kk)
            ) / (z_hat_1 * z_hat_2)
            object_loss += self_term - 2.0 * cross
        total += object_loss / len(gammas)

    return total / len(informative)
