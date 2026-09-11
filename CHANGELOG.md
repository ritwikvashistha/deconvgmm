# Changelog

All notable changes to DeconvGMM are documented here. This project follows
[Semantic Versioning](https://semver.org/) with PEP 440 pre-release suffixes.
Every convMMD capability below is **Pending** (specified and exercised by the
development test suite, not advertised as Supported) and carries **no performance
claim** (`performance_claim: none`).

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
