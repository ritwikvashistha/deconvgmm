# Changelog

All notable changes to DeconvGMM are documented here. This project follows
[Semantic Versioning](https://semver.org/) with PEP 440 pre-release suffixes.
Every convMMD capability below is **Pending** (specified and exercised by the
development test suite, not advertised as Supported) and carries **no performance
claim** (`performance_claim: none`).

## 0.4.0b1 — 2026-11-11 (GitHub beta; tag pending maintainer approval)

### Added — convMMD robust fit recipe

The convMMD contract `xdgmm-jax.convmmd` and its capability matrix are bumped
`0.5.0-draft.1 → 0.6.0-draft.1`. All new rows are **Pending**, with
`performance_claim: none`.

- **`deconvgmm.convmmd.fit_convmmd_recipe(observations, measurement_covariances, k, *, key)`**
  (contract §18) fits `k` components from the data alone. It uses the recipe the
  convMMD robustness campaign adopted (`kplus1+floor:HU Kkz`, experiment J1):
  - a K + 1 greedy k-means++ start polished by zero-noise EM;
  - a covariance floor applied as a ridge inside the loss;
  - constant-rate Adam in 200-step chunks, stopped at a gradient certificate whose
    curvature is checked (exact for P ≤ 80, Lanczos above) and saddles escaped;
  - witness repair (moving the lowest-weight component to the witness peak), kept only
    when the loss falls, conditioning does not fall, and the paired z is at least 3.

  The health check (`collapsed`) and an unescaped saddle (`saddle`) are **flags**;
  `status` is the gradient certificate alone. The function runs in float64 only and
  validates its inputs eagerly. Its result types are re-exported too:
  `ConvMMDRecipeResult`, `ConvMMDHealth`, `ConvMMDSaddleFit`, `ConvMMDRepairMove` and
  `ConvMMDWitness`.
- **Evidence.** Over all 390 of J1's datasets, the recipe given J1's start reproduces
  J1's decisions and losses (to 1e-9) on every dataset. As shipped it recovers the
  truth on 387/390, as J1 does. Unit tests:
  `tests/development/test_convmmd_recipe{,_phase2,_inputs}.py`. Cross-check:
  `scripts/convmmd_j1_crosscheck/`.
- **Analytic fits** gain `schedule="constant"` (fixed-rate AdamW),
  `stop_at_certificate`, and a `grad_tol` control. Their defaults are unchanged.

### Changed

- **`converged` is now a gradient certificate** for the deterministic analytic fits:
  `fit_analytic`, the masked fits and the analytic selection fits (contract §9). They
  report `converged` / `CONVERGED` iff `max|grad| <= grad_tol` (default `1e-6`) at the
  returned parameters. The old two-point loss-change test reported convergence at
  points far from stationary. Expect fewer `CONVERGED` and more `MAX_ITER` at the same
  `n_steps`. The `tol` keyword is still accepted but no longer used. The Monte-Carlo
  fits still never report `CONVERGED`.
- `ConvMMDFitState` and `ConvMMDFitResult` gain the fields `max_abs_grad` and
  `grad_tol`, before `metadata`. Code that unpacks a result by position must be updated.
- `ConvMMDFitStatus.COMPONENT_COLLAPSED = 5` is added, reserved, and never set.

### Fixed

- A 0.3.0b1 test (`test_snis_callable_on_observed_subspace_agrees_with_marginalized_gaussian`)
  compared two different computations of the same SNIS loss with `==`. On some CI runners
  they differ by 1 ulp, so it failed intermittently. It now uses a `1e-12` relative gate.
  The package code is unchanged.

## 0.3.0b1 — 2026-09-10 (GitHub beta; tag pending maintainer approval)

### Added — convMMD known-selection (MNAR) extension

Building on the per-coordinate missing-at-random (MAR) path shipped in 0.2.0b1,
`deconvgmm.convmmd` gains the **known selection function (MNAR)** operators. The
convMMD contract `xdgmm-jax.convmmd` is bumped `0.2.0-draft.1 → 0.5.0-draft.1`
and the capability matrix `0.2.0-draft.1 → 0.5.0-draft.1`. All new rows are
**Pending**, `performance_claim: none`.

- **Selection on the true value `Ω(z)`** (contract §17; `CMMD-SEL-*`): the
  Gaussian-window analytic parameter transform (§17.5) with a machine-eps oracle
  gate, and a general-`Ω` self-normalized-importance-sampling (SNIS) Monte-Carlo
  path with `Z_θ`/ESS diagnostics held to a statistical gate only. No machine-eps
  parity is claimed for a general `Ω`; the SNIS estimator is biased at finite
  `M`, consistent as `M→∞`. Adds a selection-aware empirical-Bayes denoiser.
- **Selection on the observed value `Ω(x̃)`**: homoscedastic fully-observed
  (§17.12; `CMMD-SEL-OBS-*`), **heteroscedastic** per-object noise `S_i`
  (§17.13; `CMMD-SEL-HET-*`), and **composed with the §16 projection (MAR)**
  (§17.14; `CMMD-SEL-OBSMAR-*`). Each detected object's own known `S_i` is used
  exactly through a per-object transform, so **no measurement-noise model is
  required**; the Gaussian window is machine-eps analytic and a general `Ω` uses
  per-object (projected) SNIS held to the statistical gate only. The §17.14
  Gaussian window is marginalized onto each observed subspace (precision
  `(P_iΨP_iᵀ)⁻¹`), and the loss reduces exactly to §17.13 at `P_i=I` and to the
  base §16 masked loss at `Ψ⁻¹=0`.
- **Facade** (`deconvgmm.convmmd`): re-exports of the `*_selected` family
  (`GaussianSelection`, `CallableSelection`, `fit_selected_observed_hetero_analytic`,
  `convmmd_denoise_selected_observed_masked`, `effective_volume_gaussian`,
  `snis_effective_sample_size`, …); every `__all__` name is a faithful
  re-export verified by `test_convmmd_facade.py`.

### Changed

- convMMD contract/matrix bumped to `0.5.0-draft.1`; new selection sections
  §17–§17.14 and the `CMMD-SEL-*` rows added, all **Pending**.

### Custody / qualification

- Validated only on the qualified stack (Python 3.10–3.12, JAX 0.6.x, NumPy
  1.26.4 / SciPy 1.12.0); warning-strict, oracle-gated, with byte-exact
  host-locked fixture custody (`convmmd_selection_001`).
- Frozen identifiers unchanged: contract ID `xdgmm-jax.convmmd`, record-field
  key `"xdgmm_jax"`.
- Unknown `Ω` remains out of scope.

## 0.2.0b1

- First public GitHub beta of DeconvGMM (XD + convMMD), including the
  per-coordinate missing-at-random (MAR) convMMD path (contract §16), CPU-only
  qualified stack.
