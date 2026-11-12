"""Minimal convMMD workflow on generated data (DeconvGMM).

convMMD fits a latent Gaussian mixture by minimizing a convolutional Maximum Mean
Discrepancy between the noisy observations and the noise-convolved model, then
denoises each observation with the fitted prior. This example uses the exact
analytic loss (deterministic, PRNG-free) two ways:

1. The **robust fit recipe** (``fit_convmmd_recipe``) fits from the data alone. It
   chooses the start, a covariance floor, when to stop, and any repairs.
2. A **manual fit** (``fit_analytic``) starts from parameters you supply. Its
   ``converged`` is a gradient certificate, so a fit stopped early reports
   ``MAX_ITER`` instead of claiming convergence.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # the recipe runs in float64 only

import jax.numpy as jnp  # noqa: E402

from deconvgmm import convmmd  # noqa: E402


def make_data(key, n=400):
    """Two latent clusters seen through known, per-observation diagonal noise."""

    k_label, k_latent, k_variance, k_noise = jax.random.split(key, 4)
    truth_means = jnp.asarray([[-2.0, 0.0], [2.0, 1.0]])
    labels = jax.random.bernoulli(k_label, 0.4, (n,)).astype(jnp.int32)
    latent = truth_means[labels] + 0.5 * jax.random.normal(k_latent, (n, 2))
    variances = jax.random.uniform(k_variance, (n, 2), minval=0.05, maxval=0.2)
    noise = jax.vmap(jnp.diag)(variances)  # (n, 2, 2) measurement covariances
    observations = latent + jnp.sqrt(variances) * jax.random.normal(k_noise, (n, 2))
    return observations, noise, latent, truth_means


def describe(name, result):
    status = convmmd.ConvMMDFitStatus(int(result.status)).name
    print(f"{name}: status={status} max|grad|={float(result.max_abs_grad):.1e} "
          f"loss={float(result.loss):.6f}")


def main() -> None:
    observations, noise, latent, truth_means = make_data(jax.random.PRNGKey(0))

    # 1. Robust fit recipe: start, floor, stopping and repairs are chosen for you.
    fit = convmmd.fit_convmmd_recipe(observations, noise, 2, key=jax.random.PRNGKey(1))
    describe("recipe", fit)
    print(f"recipe: collapsed={bool(fit.collapsed)} saddle={bool(fit.saddle)} "
          f"steps={int(fit.total_steps)}")  # flags: they never change status
    weights = [round(w, 3) for w in fit.parameters.weights.tolist()]
    print(f"recipe: weights={weights} (truth about 0.6 and 0.4)")

    # Denoise: the exact empirical-Bayes posterior mean under the fitted prior.
    denoised = convmmd.denoise(fit.parameters, observations, noise)
    before = float(jnp.mean(jnp.sum((observations - latent) ** 2, axis=1)))
    after = float(jnp.mean(jnp.sum((denoised - latent) ** 2, axis=1)))
    print(f"denoise: mean squared error {before:.4f} -> {after:.4f}")

    # 2. Manual fit from a start you supply (here near the truth, for illustration).
    bandwidths = convmmd.median_bandwidths(observations)  # predeclared heuristic
    init = convmmd.ConvMMDUnconstrained(
        alphas=jnp.zeros(2),
        means=truth_means + 0.5,
        unconstrained_L=jnp.tile(0.5 * jnp.eye(2), (2, 1, 1)),
    )
    # The defaults (300 cosine-decayed steps) usually stop before the certificate,
    # so this honestly reports MAX_ITER.
    describe("manual, defaults", convmmd.fit_analytic(init, observations, noise, bandwidths))
    # A constant learning rate reaches max|grad| <= grad_tol (1e-6) and stops there.
    certified = convmmd.fit_analytic(
        init, observations, noise, bandwidths,
        n_steps=1000, schedule="constant", learning_rate=3e-2, weight_decay=0.0,
        stop_at_certificate=True,
    )
    describe("manual, constant rate", certified)
    print(f"manual, constant rate: certified after {int(certified.n_iter)} steps")


if __name__ == "__main__":
    main()
