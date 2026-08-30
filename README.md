# DeconvGMM

**Gaussian-mixture density deconvolution in JAX — two ways.**

DeconvGMM recovers a latent density and denoises individual signals when the
observations are corrupted by **known, heterogeneous measurement noise**
(`x_i = z_i + ε_i`, `ε_i ~ N(0, S_i)` with full-covariance `S_i`). It offers two
co-equal estimators behind a shared, differentiable, accelerator-ready API:

- **Extreme Deconvolution (XD)** — exact expectation-maximization on the Gaussian
  convolution (`deconvgmm.identity`, `deconvgmm.general`).
- **convMMD** — a likelihood-free / simulation-based objective that matches the
  noise-convolved model to the data via a convolutional Maximum Mean Discrepancy
  (`deconvgmm.convmmd`). It also handles **per-coordinate missing-at-random (MAR)**
  observations through an explicit boolean observed-mask path.

Both share the same full-covariance Gaussian-mixture model and the same
empirical-Bayes denoiser, so they differ only in **how the prior is fit**. The
whole package is pure JAX: `jit`/`vmap`/`grad`-able, device-agnostic (CPU/GPU/TPU),
explicit about PRNG, and correct at float32 and float64.

> **Status: pre-1.0 beta.** The API and the on-disk artifact format may change
> before 1.0. Every numerical operation is validated against an independent NumPy
> reference, but the recorded evidence is **CPU-only on the qualified stack**
> (Python 3.10–3.13, JAX 0.6.x, NumPy ≥ 1.26). GPU/TPU and cross-platform behavior
> are not yet qualified, every capability is **Pending**, and **no performance
> claim is made** (`performance_claim: none`).

## Install

This beta is distributed from GitHub (not yet published to PyPI). It requires
Python 3.10–3.13, JAX 0.6.x, and NumPy ≥ 1.26; `jax`/`jaxlib` must be installed
with the build appropriate for your CPU or accelerator.

Install the tagged release from GitHub:

```bash
pip install "git+https://github.com/ritwikvashistha/deconvgmm.git@v0.2.0b1"
```

Or from a source checkout:

```bash
pip install .
```

## Quickstart

Extreme Deconvolution (see [`examples/xd_quickstart.py`](examples/xd_quickstart.py)):

```python
import jax
jax.config.update("jax_enable_x64", True)  # this example uses float64 inputs

import jax.numpy as jnp
from deconvgmm import Params, identity

params = Params(
    weights=jnp.asarray([1.0]),
    means=jnp.asarray([[0.0]]),
    covariances=jnp.asarray([[[1.0]]]),
)
noise = identity.fit_isotropic_noise([0.1, 0.2, 0.1, 0.2], dimension=1, dtype=jnp.float64)
observations = jnp.asarray([[-0.5], [0.0], [0.5], [1.0]])
fit = identity.fit_converged(params, observations, noise)
denoised = identity.posterior_mean(fit.parameters, observations, noise)
```

convMMD (see [`examples/convmmd_quickstart.py`](examples/convmmd_quickstart.py)):

```python
from deconvgmm import convmmd

bandwidths = convmmd.median_bandwidths(observations)          # predeclared heuristic
init = convmmd.ConvMMDUnconstrained(alphas=..., means=..., unconstrained_L=...)
result = convmmd.fit_analytic(init, observations, noise, bandwidths, n_steps=300)
denoised = convmmd.denoise(result.parameters, observations, noise)
```

A worked, fair comparison of the two methods on shared ground-truth tasks is in
[`notebooks/convmmd_xdgmm_comparison.ipynb`](notebooks/convmmd_xdgmm_comparison.ipynb).

## Which method should I use?

- **XD** is the maximum-likelihood estimator when a Gaussian mixture is a good
  model for the latent density; it is exact and needs no sampling.
- **convMMD** is likelihood-free — it only needs to *sample* the noise-convolved
  model — so it extends to settings where the convolved likelihood is intractable
  and is more robust under model misspecification. In this release both are fit as
  Gaussian mixtures; flexible/implicit generators are future work.

## Citing

If you use DeconvGMM, please cite the software (see [`CITATION.cff`](CITATION.cff))
and the relevant method papers: Bovy, Hogg & Roweis (2011) for XD, and Vashistha,
Sarkar & Farahi (arXiv:2606.21907) for convMMD.

## License and provenance

DeconvGMM is released under the [MIT License](LICENSE). Its Extreme Deconvolution
implementation is conservatively classified as adapted from
[astroML](https://github.com/astroML/astroML) (BSD-2-Clause) and informed by
[jobovy/extreme-deconvolution](https://github.com/jobovy/extreme-deconvolution)
(BSD-3-Clause); their complete notices and citation are retained in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). convMMD is original work of the
maintainer.
