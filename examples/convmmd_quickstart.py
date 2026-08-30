"""Minimal convMMD workflow on generated data (DeconvGMM).

convMMD fits a latent Gaussian mixture by minimizing a convolutional Maximum Mean
Discrepancy between the observed noisy data and the noise-convolved model, then
denoises each observation with the fitted prior. This example uses the exact
analytic loss (deterministic, PRNG-free).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from deconvgmm import convmmd


def main() -> None:
    key = jax.random.PRNGKey(0)
    n, dimension, n_components = 200, 2, 2

    # A two-component latent truth, observed through additive Gaussian noise.
    truth_means = jnp.asarray([[-2.0, 0.0], [2.0, 1.0]])
    latent = jnp.concatenate(
        [
            truth_means[0] + 0.5 * jax.random.normal(key, (n // 2, dimension)),
            truth_means[1] + 0.5 * jax.random.normal(key, (n - n // 2, dimension)),
        ]
    )
    noise = jnp.tile(0.1 * jnp.eye(dimension), (n, 1, 1))
    observations = latent + 0.3 * jax.random.normal(
        jax.random.fold_in(key, 1), (n, dimension)
    )

    # Predeclared bandwidth set (median pairwise distance x log grid).
    bandwidths = convmmd.median_bandwidths(observations)

    # Initialize an unconstrained K-component GMM and fit with the analytic loss.
    init = convmmd.ConvMMDUnconstrained(
        alphas=jnp.zeros(n_components),
        means=truth_means + 0.5,
        unconstrained_L=jnp.tile(0.5 * jnp.eye(dimension), (n_components, 1, 1)),
    )
    result = convmmd.fit_analytic(
        init, observations, noise, bandwidths, n_steps=300
    )

    # Denoise: exact empirical-Bayes posterior mean under the fitted prior.
    denoised = convmmd.denoise(result.parameters, observations, noise)

    print(f"status={convmmd.ConvMMDFitStatus(int(result.status)).name}")
    print(f"final_loss={float(result.loss):.6f}")
    print(f"weights={result.parameters.weights.tolist()}")
    print(f"mean_denoise_shift={float(jnp.mean(jnp.abs(denoised - observations))):.4f}")


if __name__ == "__main__":
    main()
