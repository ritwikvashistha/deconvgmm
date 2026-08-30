# convMMD under a known selection function (MNAR) — design note

- Status: **development-stage design note for a planned future revision.** NOT
  normative, NOT in scope for the current MAR (`observed_mask` / projection) task,
  and NOT a capability claim. It records the mathematics and validation strategy so
  MNAR is a real future phase rather than a hand-wave. See the MAR task plan in
  [`convmmd_missing_data_plan.md`](convmmd_missing_data_plan.md).
- Notation follows `docs/convmmd-model-contract.md`:
  `q_θ(z) = Σ_k π_k N(z; μ_k, Σ_k)`; observed `x̃_i = P_i z_i + ε_i`,
  `ε_i ~ N(0, S_i)`; noise-convolved-projected component
  `N(P_iμ_k, B_k^i)` with `B_k^i = P_i Σ_k P_iᵀ + S_i`; expected RBF kernel
  `G(δ, Ω; γ) = |I + γ^{-2}Ω|^{-1/2} exp(−½ δᵀ(Ω + γ² I)^{-1} δ)`.
- **Selection = MNAR** here means a *whole object* is retained with a known
  probability that depends on its value. This composes with MAR (per-coordinate
  projection `P_i`): an object may be selected AND observed in only some coordinates.

## 1. The generative model with selection

Add a known **completeness / detection probability** `Ω(·) ∈ [0, 1]` and keep only
detected objects. Two conventions, both simulable and differentiable:

- **Selection on the true value, `Ω(z)`** (physical: detection depends on the true
  flux, then you measure with error). Global, `θ`-only effective volume
  `Z_θ = ∫ Ω(z) q_θ(z) dz`.
- **Selection on the observed value, `Ω(x̃)`** (detection depends on the measured
  quantity). With heteroscedastic `S_i` the effective volume becomes
  per-observation, `Z_{θ,i} = ∫ Ω(x) p̃_{θ,i}(x) dx`, `p̃_{θ,i} = Σ_k π_k N(P_iμ_k, B_k^i)`.

In both cases the density of the **observed (detected) sample** is

`p_obs(x) = Ω(x) p̃_θ(x) / Z_θ`  (renormalized; `Z_θ` is the "eaten" mass).

The fitted prior `q_θ` targets the **true population** density — deconvolve *and*
de-bias — so the recovered `q_θ` is what a downstream denoiser uses.

## 2. Why convMMD absorbs this naturally: the MC/SBI path never needs `Z_θ`

MMD compares distributions **through samples**, and convMMD's MC loss is a full
forward simulator. Selection is one more forward stage:

`z ~ q_θ` → `x̃ = P z + ε` → **detect by `Ω`** → compare survivors to the (already
selected) real data.

A simulator that applies `Ω` produces correctly-normalized draws from `p_obs` by
construction, so **the MC estimator needs no explicit `Z_θ`**. Hard accept/reject is
non-differentiable, so use **self-normalized importance sampling (SNIS)**: draw
reparameterized samples from the *un-selected* model and weight by `ω = Ω(x̃)`.

Per observation `i`, mirroring the base MC loss (sum over components with `π_k`
weights, reparameterized draws `x̃_k^{(m)}(θ)` from component `k`; write
`ω_k^{(m)} = Ω(x̃_k^{(m)})`):

- **normalizer estimate** `Ẑ_i = Σ_k π_k (1/M) Σ_m ω_k^{(m)}`  (≈ `Z_{θ,i} = E_{p̃}[Ω]`)
- **cross** `E_{p_obs}[k(W, x̃_i)] ≈ (1/Ẑ_i) Σ_k π_k (1/M) Σ_m ω_k^{(m)} k(x̃_k^{(m)}, x̃_i)`
- **self** `E_{p_obs}[k(W,W')] ≈ (1/Ẑ_i²) Σ_{k,k'} π_kπ_{k'} (1/M²) Σ_{m,m'} ω_k^{(m)} ω_{k'}^{(m')} k(x̃_k^{(m)}, x̃_{k'}^{(m')})`
- `ℓ_i(γ) = self − 2·cross` (drop the θ-independent data self-kernel, as in the base
  contract).

Both terms share the **same** `Ẑ_i`. This is fully differentiable in `θ` through the
reparameterized draws and through `ω = Ω(x̃(θ))` (requiring `Ω` differentiable).

**Honesty note.** Unlike the base MC loss (an *unbiased* estimator of the analytic
loss), the selected loss is a **ratio (SNIS) estimator: biased at finite `M`,
consistent as `M → ∞`.** The Gaussian-`Ω` analytic form (§4) is the exact reference.

## 3. Selection composes with MAR projection

The draws `x̃_k^{(m)}` already carry the per-observation projection `P_i` and noise
`S_i` from the MAR path (§2 of the MAR plan). Selection just multiplies in `ω`. So an
object can be simultaneously (a) observed in a coordinate subset via `P_i` and (b)
value-selected via `Ω`. The grouped, host-side orchestration is unchanged; each
fixed-`M` leaf gains the `ω`-weighting and the per-group/per-observation `Ẑ`.

## 4. The analytic closed form survives **only** for Gaussian-family `Ω`

The base analytic loss exists because `Σ_k π_k N(ν_k, B_k)` is a Gaussian mixture and
RBF-MMD of Gaussians is `G`. Multiplying by a general `Ω` and renormalizing makes
`p_obs` **not** a Gaussian mixture — no `G`, no exact oracle. Two tractable cases:

**Gaussian-family `Ω` (the analytic on-ramp / oracle).** Let
`Ω(x) = exp(−½ (x−a)ᵀ Ψ^{-1} (x−a))` (a Gaussian window; peak 1 at `x=a`; a sum of
such bumps handles Gaussian-mixture completeness). Using the Gaussian-product identity
`N(x;ν_k,B_k)·Ω(x) = w̃_k · N(x; ν'_k, B'_k)` with

- `B'_k = (B_k^{-1} + Ψ^{-1})^{-1}`,
- `ν'_k = B'_k (B_k^{-1} ν_k + Ψ^{-1} a)`,
- `w̃_k = (2π)^{M/2} |Ψ|^{1/2} · N(ν_k; a, B_k + Ψ)`,

the **selected model is again a Gaussian mixture** with weights
`π'_k = π_k w̃_k / Z`, `Z = Σ_k π_k w̃_k`, means `ν'_k`, covariances `B'_k`. The full
convMMD closed form applies verbatim on `(π'_k, ν'_k, B'_k)`:

`ℓ_i(γ) = Σ_{k,k'} π'_kπ'_{k'} G(ν'_k − ν'_{k'}, B'_k + B'_{k'}; γ)
          − 2 Σ_k π'_k G(x̃_i − ν'_k, B'_k; γ)`.

(For selection on `z`: apply the same identity in `z`-space to `q_θ(z)Ω(z)` first,
giving modified `(π'_k, μ'_k, Σ'_k)`, then noise-convolve/project as usual — also a
Gaussian mixture, also closed-form.) This makes a **clean-room NumPy oracle** for a
machine-eps correctness gate.

**Hard cuts / half-spaces** (`Ω = 1{x_1 > t}`, survey footprints): `∫ Ω·N` yields
Gaussian-CDF (`erf`) terms; the self term needs multivariate normal CDFs.
Semi-closed-form in low `D`, not a clean general form — treat via MC.

## 5. Denoiser under selection

Condition on a **specific detected object** measured at `x̃_i`:
`p(z | x̃_i, detected) ∝ q_θ(z) N(x̃_i; P_i z, S_i) · P(detected | z, x̃_i)`.

- **Selection on the observed value `Ω(x̃)`:** `P(det | z, x̃_i) = Ω(x̃_i)` is
  constant in `z` → **cancels**. The per-object posterior is the **unchanged MAR
  posterior** (§2 of the MAR plan). Selection biases *which objects we see*, not the
  denoising of an object we did see.
- **Selection on the true value `Ω(z)`:** `P(det | z) = Ω(z)` tilts the posterior:
  `p(z | x̃_i, det) ∝ Ω(z) q_θ(z) N(x̃_i; P_i z, S_i)`. For Gaussian `Ω(z)` this is a
  Gaussian mixture (via §4 in `z`-space) → closed-form responsibilities/means; for
  general `Ω(z)`, an SNIS posterior-mean estimator.

Either way the denoiser uses the recovered **true** `q_θ`.

## 6. Modeling subtleties (why this is a *careful* extension, not a trivial add-on)

1. **`Ω(z)` vs `Ω(x̃)`.** `Ω(z)` gives a global, `θ`-only `Z_θ` (Gaussian-closed for
   Gaussian `Ω`) and cleanly composes with per-object noise (thin in `z`, then add
   each survivor's own `S_i`). `Ω(x̃)` makes `Z_{θ,i}` per-observation once noise is
   heteroscedastic. `Ω(z)` is the recommended first formulation.
2. **Noise of undetected objects.** With heteroscedastic `S_i` and `Ω(x̃)`, the
   population observed density mixes over the noise distribution, and the noise level
   of objects you did *not* detect is generally unknown. `Ω(z)`-selection sidesteps
   this. pygmmis is careful here; a future revision must be too, or restrict to
   homoscedastic noise / `Ω(z)`.
3. **IS variance.** SNIS effective sample size `ESS = (Σω)² / Σω²` collapses when `Ω`
   is very selective (few model draws land in the observed region), inflating variance
   and bias of the ratio estimator. Mitigations: more samples, a proposal concentrated
   on the selected region, or the Gaussian-`Ω` analytic form where applicable. `log`
   the ESS as a fit diagnostic.

## 7. Validation strategy (preserves the project's gate culture)

- **Gaussian-family `Ω` (and Gaussian-mixture `Ω`):** clean-room NumPy oracle from §4
  → **f64 machine-eps gate** (rtol 5e-8 / atol 5e-10) + declared f32 profile, exactly
  like `CMMD-LOSS-001`.
- **General `Ω`:** no closed form → **statistical gate only.** (a) When `Ω` is
  Gaussian, the SNIS MC loss must converge to the §4 analytic value at the MC rate
  (`∝ M^{-1/2}`, fixed key discipline). (b) For non-Gaussian `Ω`, compare to a
  high-`M` / large-sample MC reference and check the convergence trend and the ESS
  diagnostic; never claim machine-eps parity for general `Ω`.
- **Denoiser:** exact-oracle gate for the `Ω(x̃)` case (equals MAR) and Gaussian
  `Ω(z)`; statistical gate for general `Ω(z)`.
- Deterministic `Generator(PCG64(<seed>))`; SHA-256-pinned fixtures/records; all
  `-W error` in the `cv` lane. `performance_claim` stays `none`.

## 8. Relationship to XD and pygmmis (honest positioning)

- **XD (exact EM)** needs the effective-volume `Z_θ = ∫ Ω p̃` in its normalized
  marginal likelihood; this integral is analytic only for special `Ω`, so XD cannot
  do general known-selection deconvolution exactly. **pygmmis** handles known `Ω` by
  imputing the unobserved complement inside EM (a Monte-Carlo correction) — a genuine
  contribution, and its distinctive capability.
- **convMMD-MC** avoids `Z_θ` entirely (samples are self-normalized), so a known,
  simulable, differentiable `Ω` is a **natural** fit — arguably more so than analytic
  XD, and aligned with the reference method's SBI thesis (arXiv:2606.21907). This is a
  real, honest differentiator to document.
- This does **not** demote XD: where XD is exact (Gaussian truth, no selection) it
  remains preferable. The head-to-head stays fair — both consume identical data,
  noise, selection, seed, split, and metrics.

## 9. What a future contract revision would add

- A normative "known selection function" section: the two conventions, the SNIS MC
  loss (default), the Gaussian-`Ω` analytic sub-case (oracle + supported fast path),
  the selection-aware denoiser, `Z_θ`/ESS diagnostics, and the honest bias/variance
  statement.
- An `Ω` input API: a differentiable, simulable callable plus a declared selection
  convention (`on_z` / `on_observed`) and any parametric-window parameters.
- New **Pending** capability rows (`CMMD-SEL-LOSS-*`, `CMMD-SEL-GAUSS-*`,
  `CMMD-SEL-DENOISE-*`, `CMMD-SEL-ESS-*`, `CMMD-SEL-CMP-*`), a Gaussian-`Ω` fixture +
  oracle, and optionally a fair comparison vs pygmmis on a known-selection task.
- Contract/version discipline unchanged: minor revision (new input form), contract ID
  frozen, no row → Supported without qualification, `performance_claim: none`.

## 10. Out of scope (genuinely, for both methods)

- An **unknown** selection function `Ω`. Recovering `Ω` jointly with `q_θ` is a
  different, ill-posed-without-anchoring problem (needs a reference/complete sample or
  a parametric selection model with identifiability constraints). Out for XD, pygmmis,
  and convMMD alike.
- Any `Ω` that cannot be evaluated or simulated, or is non-differentiable in a way
  that defeats both the SNIS gradient and a finite-difference fallback.
