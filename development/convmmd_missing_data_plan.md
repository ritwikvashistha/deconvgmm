# convMMD per-coordinate missing-data (MAR) — approved design plan

- Status: **maintainer-approved design** (forks resolved 2026-08-30); pre-implementation.
- Scope owner: convMMD (`xdgmm-jax.convmmd`; package `deconvgmm`, import `deconvgmm`).
- This is development-stage planning material, not a released artifact or a public
  contract. The normative statements land in the contract revision (Phase 0).

This plan extends **convMMD** to handle per-coordinate **missing-at-random (MAR)**
observations by exact marginalization through a per-observation projection `P_i`,
the same mechanism XD's general path uses (not the noise-inflation shortcut). It
was written after reading the current convMMD artifacts and XD's mask-grouping
adapter, and after an independent re-derivation of the projected mathematics.

## 1. Scope (stated honestly)

- **IN (this revision):** per-coordinate MAR via a per-observation selection/
  projection `P_i` (an `M_i × D` row-subset of the identity for pure missing data).
  The numerical leaf is projection-generic (accepts any `M_i × D` `P_i`), but the
  **public API is keyed on `observed_mask`** and advertises only coordinate-selection
  / MAR.
- **DEFERRED to a future revision (feasible, not a fundamental limit): MNAR with a
  known, simulable, differentiable selection function `Ω`.** A completeness/detection
  probability `Ω ∈ [0,1]` makes the observed sample a biased draw
  `p_obs(x) ∝ Ω(x) p̃_θ(x)`. Because convMMD's MC path is a full forward simulator,
  this is a **natural** extension — a differentiable self-normalized-importance-
  sampling estimator (draw the un-selected model, weight by `Ω`) never needs the
  effective-volume `Z_θ` explicitly. This is a capability convMMD supports **more
  naturally than analytic XD** (which needs `Z_θ` in closed form; pygmmis instead
  imputes the unobserved complement inside EM). It is **out of this task's scope**
  only because it changes the generative model (`Z_θ`) and needs its own oracle
  strategy — the general-`Ω` analytic form is *not* closed (so no machine-eps oracle;
  gate statistically), while a **Gaussian-family `Ω` sub-case stays fully closed-form**
  and is the analytic on-ramp/oracle. Captured in
  [`convmmd_mnar_design_note.md`](convmmd_mnar_design_note.md) as a planned phase.
- **OUT (genuinely, for both methods):** an **unknown** selection function, and
  general truncation/completeness `Ω(x)` where `Ω` cannot be evaluated/simulated.
  pygmmis's distinctive contribution is handling known selection at all (Melchior &
  Goulding 2018, arXiv:1611.05806); recovering an *unknown* `Ω` is out for XD and
  convMMD alike.
- **Contrast with pygmmis on per-coordinate missing:** pygmmis handles per-coordinate
  missing by *inflating the missing feature's covariance* (an approximation);
  DeconvGMM's XD and this extension do the **exact** marginalization via projection.

## 2. Confirmed mathematics (observed subspace `R^{M_i}`)

Observation `i`: `x̃_i = P_i z_i + ε_i`, `ε_i ~ N(0, S_i)`, `P_i` is `M_i × D`,
`S_i` is `M_i × M_i` PSD. Latent `z_i ~ Σ_k π_k N(μ_k, Σ_k)`. Define
`B_k^i = P_i Σ_k P_iᵀ + S_i` (`M_i × M_i`).

- **Expected RBF kernel (unchanged, dimension-generic):**
  `G(δ, Ω; γ) = |I_{M_i} + γ^{-2}Ω|^{-1/2} exp(−½ δᵀ(Ω + γ² I_{M_i})^{-1} δ)`.
  `development.convmmd.expected_rbf_kernel` already computes this for any last-dim
  size; the projected path reuses it verbatim.
- **Per-observation, per-scale loss:**
  `ℓ_i(γ) = Σ_{k,k'} π_kπ_{k'} G(P_i(μ_k−μ_{k'}), B_k^i+B_{k'}^i; γ)
            − 2 Σ_k π_k G(x̃_i − P_iμ_k, B_k^i; γ)`.
- **Aggregate loss (normalize by informative rows; approved):**
  `L = (1/(G·N_inf)) Σ_g Σ_{i: M_i>0} ℓ_i(γ_g)`, `N_inf = #{i: M_i>0}`
  (weighted: `Σ_{i:M_i>0} w_i` in the denominator, `w_i ℓ_i` in the numerator).
  Reduces to today's `(1/(G·N)) Σ_g Σ_i ℓ_i` exactly when every `M_i>0`.
- **`M_i = 0`:** contributes **exactly 0** to the loss and is **excluded** from the
  denominator. Rationale: in `R^0` the RBF data self-kernel `k(x̃,x̃)=1` is the term
  we drop, so the naive per-row value is `1 − 2 = −1` (θ-independent); excluding the
  row from both numerator and denominator gives XD's add/remove-`M=0` invariance
  (`XD-GEN-M0-001`). The `M=0` leaf is not executed for the loss.
- **Empirical-Bayes denoiser (full `D`-dim output):**
  `r_{ik} ∝ π_k N(x̃_i; P_iμ_k, B_k^i)` (sum to 1),
  `m_{ik} = μ_k + Σ_k P_iᵀ (B_k^i)^{-1}(x̃_i − P_iμ_k)`,
  `ẑ_i = Σ_k r_{ik} m_{ik}`. For `M_i = 0`, `ẑ_i = Σ_k π_k μ_k` (prior mean).
  Identical in form to the XD general posterior mean.
- **MC path:** `x̃_{k,i}^{(m)} = P_iμ_k + P_i L_k ζ^{(m)} + L_{S_i} η^{(m)}`
  (`Σ_k=L_kL_kᵀ`, `S_i=L_{S_i}L_{S_i}ᵀ`), kernels evaluated in `M_i`-space,
  reparameterized/differentiable, one explicit PRNG key. Its `M→∞` limit is the
  analytic loss.
- **PD-ness:** for selection `P_i` (row-subset of `I`) and PD `Σ_k`, `P_iΣ_kP_iᵀ`
  is a principal block (PD); with PSD `S_i`, `B_k^i` is PD. `G` adds `γ²I` inside
  both the determinant and the inverse, so the loss is well-defined for `γ>0` even
  where `Ω` is only PSD. The denoiser Cholesky needs `B_k^i` PD (holds here).

## 3. Resolved design forks

1. **Projection scope — projection-generic leaf, MAR-only public API.** Reuse XD's
   `group_masked_general_inputs(..., projection=IdentityProjection(D), noise=…)` to
   turn `observed_mask` into ascending-coordinate selection rows + principal noise
   blocks. General `R_i` is reachable at the leaf but not advertised.
2. **Observed-space noise — mirror XD exactly.** Reuse `PerItemFullNoise` /
   `SharedFullNoise` (+ shared isotropic/diagonal). Mirror XD's exclusion of
   per-item isotropic/diagonal *masked* forms (as in `_validate_mask_modes`).
3. **Contract revision — minor `xdgmm-jax.convmmd 0.2.0-draft.1`** (adds a
   backward-compatible input form). Missing data is a new normative section; a
   subsection states the MNAR positioning (planned future revision with known/
   simulable/differentiable `Ω`, MC-first, Gaussian-`Ω` analytic on-ramp — *not* a
   flat exclusion; only unknown selection is genuinely out) + the pygmmis contrast.
   §§1–15 stay put; contract ID frozen. Capability matrix bumps in parallel; new rows
   start Pending.
4. **Bandwidth — single global scalar Γ (`median_bandwidths_masked`).** Median over
   pairs sharing ≥1 observed coordinate of the Euclidean distance on their shared
   coordinates, × the same `logspace(-2, 2, 9)` grid. Reduces **exactly** to
   `median_bandwidths` on fully-observed data; raises if no pair shares a coordinate.
   Predeclared and pinned (rule + any subsample seed recorded in fixture custody).
5. **Module layout — new `development/convmmd_grouped.py`** (mirrors
   `general_grouped.py`, much simpler: no M-step, no Chan merge, no collapse). Small
   `*_projected` leaf additions in `development/convmmd.py`. Grouped fit reuses
   `convmmd_fit._run` (swap in the grouped objective). Facade adds `fit_masked_*`,
   `denoise_masked`, `median_bandwidths_masked`, the grouped loss/denoiser, and
   re-exports the reused projection/noise tags.
6. **Comparison — add masked Task C now.** Masked version of Task A (Gaussian truth,
   XD home turf) on identical latent/noise/seed with a predeclared MAR mask; both
   methods via their grouped paths under the same fair protocol (common random
   numbers, oracle-gated endpoints, paired win counts, fit diagnostics),
   `performance_claim: none`.
7. **JIT contract — per-group fixed-`M` leaf is `jit`/`grad`/`vmap`-clean; mask
   grouping is host-side; no whole-operation guarantee over the ragged set** (matches
   `XD-GEN-JIT-001`). The grouped loss over a fixed group structure is a single
   differentiable function (Python loop over fixed-`M` leaves as closed-over arrays).

## 4. Phased implementation

- **P0 — Contract + matrix.** Add the normative missing-data section to
  `docs/convmmd-model-contract.md` (projected loss/denoiser/MC, shapes/dtypes,
  `M=0` + informative-normalization semantics, failure modes, MAR-only scope for this
  revision, MNAR positioning as a planned future revision (per the design note) +
  pygmmis contrast, host-side-grouping JAX note). Author matching
  Pending rows in `docs/convmmd-capability-matrix.md` (`CMMD-MISS-LOSS-*`,
  `CMMD-MISS-DENOISE-*`, `CMMD-M0-*`, `CMMD-MISS-BW-*`, `CMMD-MISS-JIT/GRAD/VMAP/PRNG-*`,
  `CMMD-MISS-CMP-*`) + explicit non-claims. Bump both to `0.2.0-draft.1`.
- **P1 — Oracle + fixtures.** Extend `tests/reference/convmmd.py` with the projected
  loss, projected denoiser, and `median_bandwidths_masked` reference (clean-room from
  the contract). Add `scripts/generate_convmmd_missing_fixture.py` writing
  `tests/fixtures/convmmd_missing_001.npz` (+ `.metadata.json`) via
  `scripts/deterministic_npz.py`: known latent truth, mixed masks including `M=0`
  and a fully-observed group, full-covariance heteroscedastic noise,
  `Generator(PCG64(<seed>))`, SHA-256-pinned, test-bound.
- **P2 — Clean-room JAX.** `development/convmmd.py`: `convmmd_loss_analytic_projected`,
  `convmmd_loss_mc_projected`, `posterior_components_projected`, `denoise_projected`,
  `median_bandwidths_masked`. `development/convmmd_grouped.py`: grouped analytic/MC
  loss, grouped denoiser (restore to full `D`, original order), grouped fit wrappers.
  Validate vs the oracle at **f64 rtol 5e-8 / atol 5e-10** and a **declared f32
  profile** (≈ rtol 1e-4 / atol 1e-5, tightened/loosened per measured evidence and
  recorded before acceptance).
- **P3 — JAX-contract tests.** `jit`/`vmap`/`grad`, explicit PRNG (MC), dtype/device,
  `M=0` handling, honest failure/status (degenerate → documented status or visible
  NaN, never NaN-as-success). Everything `-W error` in the `cv` lane.
- **P4 — Notebook demo + record.** Extend the comparison notebook with masked Task C
  vs XDGMM on shared ground truth; oracle-gate each method's endpoints before
  comparing; emit a schema-validated, SHA-pinned record with `performance_claim: none`;
  send a chart.
- **P5 — Adversarial review.** Math, numerical stability/edge cases (empty groups,
  `M=0`, singular projected covariance, single-row groups, no-shared-coordinate
  bandwidth), JAX contract, API/facade naming, comparison fairness. Fix real
  findings; re-certify.
- **P6 — Integration (maintainer-gated).** Expose via `deconvgmm.convmmd`, assign the
  contract/matrix revision, add Pending rows, update ROADMAP ledger,
  `docs/ai-usage-log.md`, `docs/compatibility.md`, `docs/deferred-0.1.0b2.md`,
  provenance docs (§14). No version bump, no row → Supported, no wheel without
  explicit approval.

## 5. Guardrails (non-negotiable)

Derive-then-implement; validate JAX only against the independent NumPy oracle;
correctness gates (f64 near machine-eps + declared f32) before any comparison; fair
comparison preserving every observation; deterministic `Generator(PCG64(<seed>))` +
SHA-256-pinned fixtures/records bound in tests; JAX contract (`jit`/`vmap`/`grad`,
device-agnostic, explicit PRNG, f32/f64 correct, honest status); `performance_claim`
stays `none`, new rows start Pending; warning-strict `cv` lane.

**Stop and ask before:** any version bump, flipping a row to Supported, or
building/publishing a wheel; modifying any frozen `0.1.0b1`-era artifact, retained
record, pinned fixture, `xdgmm-jax.*` ID, or the `"xdgmm_jax"` record-field key;
any public performance/capability claim; deleting/overwriting retained evidence.
