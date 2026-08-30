"""Minimal Extreme-Deconvolution (XD) workflow on generated data (DeconvGMM)."""

from __future__ import annotations

import jax.numpy as jnp

from deconvgmm import Params, identity


def main() -> None:
    """Run one explicit fixed-step identity-projection fit on generated data."""

    params = Params(
        weights=jnp.asarray([1.0], dtype=jnp.float32),
        means=jnp.asarray([[0.0]], dtype=jnp.float32),
        covariances=jnp.asarray([[[1.0]]], dtype=jnp.float32),
    )
    observations = jnp.asarray(
        [[-0.5], [0.0], [0.5], [1.0]],
        dtype=jnp.float32,
    )
    measurement_covariances = identity.fit_isotropic_noise(
        [0.10, 0.20, 0.10, 0.20],
        dimension=1,
        dtype=jnp.float32,
    )
    inputs = identity.canonicalize_fit_inputs(
        params,
        observations,
        measurement_covariances,
        dtype=jnp.float32,
    )

    fit = identity.fit_fixed_steps(
        inputs.parameters,
        inputs.observations,
        inputs.measurement_covariances,
        n_steps=1,
    )
    scores = identity.score_samples(
        fit.parameters,
        inputs.observations,
        inputs.measurement_covariances,
    )
    denoised = identity.posterior_mean(
        fit.parameters,
        inputs.observations,
        inputs.measurement_covariances,
    )

    print(f"status={identity.FitStatus(int(fit.status)).name}")
    print(f"accepted_steps={int(fit.n_iter)}")
    print(f"mean_score={float(jnp.mean(scores)):.6f}")
    print(f"posterior_means={denoised[:, 0].tolist()}")


if __name__ == "__main__":
    main()

