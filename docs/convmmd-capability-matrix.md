# convMMD capability and quality matrix

- Matrix ID: `xdgmm-jax.convmmd.matrix`
- Matrix version: `0.5.0-draft.1`
- Applies to: [`xdgmm-jax.convmmd` contract `0.5.0-draft.1`](convmmd-model-contract.md)
- Last updated: 2026-08-31

This matrix turns the convMMD contract into named, reproducible acceptance rows.
Every row is **Pending**: specified and exercised by development tests, but **not**
advertised as a Supported capability. No row is flipped to Supported without
maintainer qualification, and `performance_claim` remains `none`. convMMD is
exposed publicly as `deconvgmm.convmmd`, but being importable is not a support
claim — its capabilities remain Pending until qualified. (The `xdgmm-jax.convmmd`
contract ID is a historical identifier retained after the package rename to
DeconvGMM.)

Matrix version `0.2.0-draft.1` adds the **per-coordinate missing-at-random (MAR)**
rows (contract §16): the `CMMD-MISS-*` and `CMMD-M0-*` families below. Their
evidence files are authored during the missing-data development effort; every such
row starts **Pending**.

Matrix version `0.3.0-draft.1` adds the **known selection function (MNAR)** rows
(contract §17): the `CMMD-SEL-*` family below, covering selection on the true value
\(\Omega(z)\) — the Gaussian-window analytic path (machine-eps oracle gate), the
general-\(\Omega\) SNIS Monte-Carlo path (statistical gate only), the selection-aware
denoiser, the \(Z_\theta\)/ESS diagnostics, and the JAX-contract/fit rows. Every such
row starts **Pending**; **no machine-eps parity is claimed for a general \(\Omega\)**,
and `performance_claim` stays `none`.

Matrix version `0.4.0-draft.1` adds the **heteroscedastic** \(\Omega(\tilde x)\) rows
(contract §17.13): the `CMMD-SEL-HET-*` family below. convMMD admits per-object noise
\(S_i\) **exactly** — a Gaussian window is machine-eps analytic via a **per-object**
transform on \(\Sigma_k+S_i\), and a general \(\Omega\) uses a per-object SNIS with the
detected object's own **known** \(S_i\) — so **no measurement-noise model** is needed
(the contrast with pygmmis, whose EM imputation requires a `covar_callback`). Every such
row starts **Pending**; the analytic path carries the machine-eps oracle gate and the
§17.12 / \(\Psi^{-1}=0\) reductions, the SNIS path carries the statistical gate only, and
`performance_claim` stays `none`.

Matrix version `0.5.0-draft.1` adds the **heteroscedastic \(\Omega(\tilde x)\) composed with
the §16 projection (MAR)** rows (contract §17.14): the `CMMD-SEL-OBSMAR-*` family below. The
Gaussian window is **marginalized onto each observed subspace** — precision
\((P_i\Psi P_i^\top)^{-1}\), the inverse of the covariance principal submatrix, **not**
\(P_i\Psi^{-1}P_i^\top\) — and the §17.5 transform runs per object on the projected inflated
covariance \(P_i\Sigma_kP_i^\top+S_i\); a Gaussian window is machine-eps analytic, a general
\(\Omega\) uses per-object projected SNIS, and neither needs a noise model. The loss reduces
**exactly** to §17.13 at \(P_i=I\) and to the base §16 masked loss at \(\Psi^{-1}=0\). This is
the composition pygmmis handles only by covariance inflation (its exact `R` route raises
`NotImplementedError` under a selection callback). Every such row starts **Pending**; the
analytic path carries the machine-eps oracle gate (including the direct marginalized-window
identity and both reductions), the SNIS path the statistical gate only, and
`performance_claim` stays `none`.

## Status meanings

| Mark | Meaning |
|---|---|
| **Pending** | specified and locally exercised; not a Supported public capability |
| **Supported** | (none yet) advertised only after maintainer qualification |

The required-execution dtype is part of each row. A row naming both float64 and
float32 fails if either dtype fails. All rows run warning-as-error under the
pinned development lane (conda env `cv`; Python 3.10.11; JAX/jaxlib 0.6.2; NumPy
1.26.4).

## Correctness rows (oracle parity — the gate before any comparison)

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-LOSS-001` | Analytic convMMD loss equals the independent NumPy oracle | float64 (rtol 5e-8/atol 5e-10), float32 (declared profile) | Pending | `tests/development/test_convmmd_parity.py` |
| `CMMD-LOSS-002` | Per-scale losses equal the oracle for every bandwidth | float64, float32 | Pending | `tests/development/test_convmmd_parity.py` |
| `CMMD-DENOISE-001` | Empirical-Bayes posterior mean equals the exact GMM posterior oracle | float64, float32 | Pending | `tests/development/test_convmmd_parity.py` |
| `CMMD-DENOISE-002` | Responsibilities and component posterior means equal the oracle; rows sum to one | float64, float32 | Pending | `tests/development/test_convmmd_parity.py`, `tests/reference/test_convmmd_reference.py` |
| `CMMD-PARAM-001` | `to_canonical` (softmax weights, softplus-Cholesky) equals the oracle and yields SPD covariances | float64 | Pending | `tests/development/test_convmmd_parity.py`, `tests/reference/test_convmmd_reference.py` |
| `CMMD-KERNEL-001` | `expected_rbf_kernel` reduces to the plain RBF at zero covariance and to one at zero argument | float64 | Pending | `tests/reference/test_convmmd_reference.py` |
| `CMMD-MC-001` | Monte-Carlo loss converges to the analytic value at the Monte-Carlo rate | float64 | Pending | `tests/development/test_convmmd_jax_contract.py`, `tests/reference/test_convmmd_reference.py` |

## JAX-contract rows

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-JIT-001` | Loss operators and denoiser are callback-free and do not retrace on same-shape inputs | float64, float32 | Pending | `tests/development/test_convmmd_jax_contract.py` |
| `CMMD-DTYPE-001` | float32/float64 dtype is preserved end to end (no silent upcast) | float64, float32 | Pending | `tests/development/test_convmmd_jax_contract.py` |
| `CMMD-GRAD-001` | Autodiff through `to_canonical` matches central differences | float64 | Pending | `tests/development/test_convmmd_jax_contract.py` |
| `CMMD-VMAP-001` | Denoiser vmaps over the observation batch to the batched result | float64 | Pending | `tests/development/test_convmmd_jax_contract.py` |
| `CMMD-PRNG-001` | Monte-Carlo loss reads only the passed key: reuse-identical, split-differs, missing-key errors | float64 | Pending | `tests/development/test_convmmd_jax_contract.py` |
| `CMMD-DEVICE-001` | Outputs reside on the default device (device-agnostic core) | float64 | Pending | `tests/development/test_convmmd_jax_contract.py` |

## Fit-control and status rows

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-FIT-001` | A valid fit reports non-failure, reduces the loss, and returns SPD covariances | float64 | Pending | `tests/development/test_convmmd_jax_contract.py` |
| `CMMD-FIT-002` | Divergence reports `NUMERICAL_FAILURE` and rolls back to the last finite iterate | float64 | Pending | `tests/development/test_convmmd_jax_contract.py` |
| `CMMD-FIT-003` | Analytic reported loss recomputes exactly at the returned parameters (no biased min) | float64 | Pending | `tests/development/test_convmmd_jax_contract.py` |
| `CMMD-FIT-004` | Monte-Carlo fit never reports spurious `CONVERGED`; `n_steps=1` is not converged | float64 | Pending | `tests/development/test_convmmd_jax_contract.py` |
| `CMMD-STATUS-001` | A degenerate covariance surfaces as a visible NaN, never a finite success | float64 | Pending | `tests/development/test_convmmd_jax_contract.py` |
| `CMMD-FIT-JIT-001` | Analytic and Monte-Carlo fit state kernels are jit-compatible | float64 | Pending | `tests/development/test_convmmd_jax_contract.py` |

## Comparison-evidence rows (fairness, not performance)

| ID | Behavior | Status | Evidence |
|---|---|---|---|
| `CMMD-CMP-001` | The retained convMMD-vs-XDGMM record conforms to its authoritative schema, both methods' endpoints pass their oracle gates, every observation is preserved, and `performance_claim` is `none` | Pending | `tests/benchmarks/test_convmmd_comparison.py` |
| `CMMD-CMP-002` | The record is SHA-256-pinned; the schema validator rejects a performance claim, a failed gate, a tampered winner, and a summary inconsistent with observations | Pending | `tests/benchmarks/test_convmmd_comparison.py` |
| `CMMD-CMP-003` | The comparison notebook compiles, references the contract/schema, gates endpoints before comparing, and states the GMM-only scope caveat | Pending | `tests/benchmarks/test_convmmd_comparison.py` |

## Missing-data (MAR) correctness rows (contract §16)

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-MISS-LOSS-001` | Projected analytic masked loss equals the independent NumPy oracle on mixed masks (including a fully-observed group and an `M=0` group) | float64 (rtol 5e-8/atol 5e-10), float32 (declared profile) | Pending | `tests/development/test_convmmd_missing_parity.py` |
| `CMMD-MISS-LOSS-002` | Per-scale masked losses equal the oracle for every bandwidth | float64, float32 | Pending | `tests/development/test_convmmd_missing_parity.py` |
| `CMMD-MISS-REDUCE-001` | A fully-observed masked call equals the base §4 `convmmd_loss_analytic` (informative-row normalization reduces to `1/N`) | float64, float32 | Pending | `tests/development/test_convmmd_missing_parity.py` |
| `CMMD-MISS-DENOISE-001` | Projected empirical-Bayes posterior mean equals the exact projected GMM posterior oracle, returned at full `D` in original row order | float64, float32 | Pending | `tests/development/test_convmmd_missing_parity.py` |
| `CMMD-MISS-DENOISE-002` | Masked responsibilities and full-`D` component posterior means equal the oracle; rows sum to one | float64, float32 | Pending | `tests/development/test_convmmd_missing_parity.py`, `tests/reference/test_convmmd_missing_reference.py` |
| `CMMD-M0-001` | `M=0` rows contribute exactly zero and are excluded from the informative denominator; the denoiser returns the prior mean; adding/removing `M=0` rows changes no loss, gradient, parameter, or status; an all-`M=0` loss is exactly zero and an all-`M=0` fit fails `no_informative_weight` | float64, float32 | Pending | `tests/reference/test_convmmd_missing_reference.py`, `tests/development/test_convmmd_missing_parity.py`, `tests/development/test_convmmd_missing_jax_contract.py` |
| `CMMD-MISS-BW-001` | `median_bandwidths_masked` is a single global `Γ`, equals `median_bandwidths` exactly on fully-observed data, and raises when no pair shares an observed coordinate | float64 | Pending | `tests/reference/test_convmmd_missing_reference.py`, `tests/development/test_convmmd_missing_parity.py` |
| `CMMD-MISS-MC-001` | Masked Monte-Carlo loss converges to the masked analytic value at the Monte-Carlo rate | float64 | Pending | `tests/reference/test_convmmd_missing_reference.py`, `tests/development/test_convmmd_missing_jax_contract.py` |

## Missing-data grouping and validation rows

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-MISS-GROUP-001` | Deterministic mask grouping (ascending lexicographic boolean-tuple order, preserved relative row order, ascending coordinate selection, exact projection rows/principal noise blocks) and row restoration; masked loss/denoiser match direct per-row NumPy | float64, float32 | Pending | `tests/development/test_convmmd_missing_parity.py` |
| `CMMD-MISS-VAL-001` | Non-boolean/mismatched masks fail; every NaN/Inf (including masked positions) fails; large-noise values are not treated as missing; per-item isotropic/diagonal masked noise is rejected; each selected principal noise block is re-validated PSD at its own scale | float64, float32 | Pending | `tests/development/test_convmmd_missing_jax_contract.py` |

## Missing-data JAX-contract and fit rows

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-MISS-JIT-001` | Each per-group fixed-`M` masked leaf is callback-free and does not retrace on same-shape inputs; grouping/validation are explicitly outside this row | float64, float32 | Pending | `tests/development/test_convmmd_missing_jax_contract.py` |
| `CMMD-MISS-GRAD-001` | Autodiff through the grouped masked loss (fixed group structure) matches central differences | float64 | Pending | `tests/development/test_convmmd_missing_jax_contract.py` |
| `CMMD-MISS-VMAP-001` | The masked denoiser vmaps over a group's observation rows to the batched result | float64 | Pending | `tests/development/test_convmmd_missing_jax_contract.py` |
| `CMMD-MISS-PRNG-001` | The masked Monte-Carlo loss reads only the passed key: reuse-identical, split-differs, missing-key errors | float64 | Pending | `tests/development/test_convmmd_missing_jax_contract.py` |
| `CMMD-MISS-DTYPE-001` | float32/float64 dtype is preserved end to end across the masked path (no silent upcast) | float64, float32 | Pending | `tests/development/test_convmmd_missing_jax_contract.py` |
| `CMMD-MISS-FIT-001` | A valid masked fit reports non-failure, reduces the loss, and returns SPD covariances | float64 | Pending | `tests/development/test_convmmd_missing_jax_contract.py` |
| `CMMD-MISS-STATUS-001` | A degenerate projected covariance surfaces as a visible NaN or documented status, never a finite success | float64 | Pending | `tests/development/test_convmmd_missing_jax_contract.py` |

## Missing-data comparison-evidence rows (fairness, not performance)

| ID | Behavior | Status | Evidence |
|---|---|---|---|
| `CMMD-MISS-CMP-001` | The retained masked convMMD-vs-XDGMM record (Task C) conforms to its authoritative schema, both methods' endpoints pass their oracle gates, every observation is preserved, and `performance_claim` is `none` | Pending | `tests/benchmarks/test_convmmd_comparison.py` |
| `CMMD-MISS-CMP-002` | The masked comparison notebook compiles, references the contract/schema, gates endpoints before comparing, and states the MAR-only scope with the MNAR-deferred caveat | Pending | `tests/benchmarks/test_convmmd_comparison.py` |

## Known-selection (MNAR) correctness rows (contract §17)

These cover selection on the true value \(\Omega(z)\). The Gaussian-window analytic
rows carry a machine-eps oracle gate; the SNIS rows are **statistical only** and
**never** claim machine-eps parity for a general \(\Omega\).

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-SEL-XFORM-001` | The Gaussian-window transform \((\pi,\mu,\Sigma)\to(\pi',\mu',\Sigma')\) equals the oracle, yields SPD \(\Sigma'_k\), uses the \(|\Psi|\)-free robust log-weight, and gives the identity at \(\Psi^{-1}=0\) | float64, float32 | Pending | `tests/development/test_convmmd_selection_parity.py`, `tests/reference/test_convmmd_selection_reference.py` |
| `CMMD-SEL-GAUSS-001` | Gaussian-\(\Omega(z)\) analytic selected loss equals the independent NumPy oracle (transform ∘ §16 masked oracle) on mixed masks (incl. a fully-observed and an `M=0` group) | float64 (rtol 5e-8/atol 5e-10), float32 (declared profile) | Pending | `tests/development/test_convmmd_selection_parity.py` |
| `CMMD-SEL-GAUSS-002` | Per-scale selected analytic losses equal the oracle for every bandwidth | float64, float32 | Pending | `tests/development/test_convmmd_selection_parity.py` |
| `CMMD-SEL-REDUCE-001` | \(\Psi^{-1}=0\) (no selection) reduces the selected analytic loss and denoiser to the §16 masked operators exactly; fully observed additionally equals the base §4/§7 operators | float64, float32 | Pending | `tests/development/test_convmmd_selection_parity.py` |
| `CMMD-SEL-DENOISE-001` | Selection-aware denoiser (Gaussian \(\Omega(z)\)) equals the projected GMM posterior oracle at \((\pi',\mu',\Sigma')\), returned at full `D` in original row order | float64, float32 | Pending | `tests/development/test_convmmd_selection_parity.py` |
| `CMMD-SEL-DENOISE-002` | Selected responsibilities and full-`D` component posterior means equal the oracle; rows sum to one | float64, float32 | Pending | `tests/development/test_convmmd_selection_parity.py`, `tests/reference/test_convmmd_selection_reference.py` |
| `CMMD-SEL-DENOISE-003` | The general-`Ω` SNIS denoiser (sample the MAR posterior, weight by `Ω`) converges to the §17.5 analytic selected denoiser for a Gaussian `Ω` (statistical: spread shrinks, within noise); `M=0` rows give the selected prior mean | float64 | Pending | `tests/reference/test_convmmd_selection_reference.py`, `tests/development/test_convmmd_selection_jax_contract.py` |
| `CMMD-SEL-MC-001` | For a Gaussian \(\Omega\), the SNIS Monte-Carlo selected loss converges to the §17.5 analytic value at the Monte-Carlo rate (\(\propto M^{-1/2}\), fixed key) — statistical, not machine-eps | float64 | Pending | `tests/reference/test_convmmd_selection_reference.py`, `tests/development/test_convmmd_selection_jax_contract.py` |
| `CMMD-SEL-MC-002` | For a non-Gaussian \(\Omega\), the SNIS loss is finite and its error to a high-\(M\) reference shrinks with \(M\) (convergence trend); no machine-eps parity claimed | float64 | Pending | `tests/reference/test_convmmd_selection_reference.py` |
| `CMMD-SEL-ESS-001` | The ESS diagnostic \((\sum u)^2/\sum u^2\) (\(u_{k,m}=\pi_k\,\omega_k^{(m)}\) over the \(K\,M\) proposal draws) is finite, in \((0, KM]\), and decreases as \(\Omega\) becomes more selective; exposed as a fit diagnostic | float64 | Pending | `tests/reference/test_convmmd_selection_reference.py`, `tests/development/test_convmmd_selection_jax_contract.py` |
| `CMMD-SEL-Z-001` | The effective-volume diagnostic \(Z_\theta\) agrees between the SNIS \(\widehat Z\) and the analytic closed form for a selective Gaussian \(\Omega\), and \(Z_\theta\to 1\) as \(\Omega\to 1\) | float64 | Pending | `tests/reference/test_convmmd_selection_reference.py` |

## Known-selection JAX-contract and fit rows

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-SEL-JIT-001` | The selected analytic and SNIS fixed-`M` leaves and the Gaussian-window transform are callback-free and do not retrace on same-shape inputs | float64, float32 | Pending | `tests/development/test_convmmd_selection_jax_contract.py` |
| `CMMD-SEL-GRAD-001` | Autodiff through the Gaussian-window transform and the grouped selected analytic loss (fixed group structure) matches central differences | float64 | Pending | `tests/development/test_convmmd_selection_jax_contract.py` |
| `CMMD-SEL-PRNG-001` | The SNIS selected loss reads only the passed key: reuse-identical, split-differs, missing-key errors | float64 | Pending | `tests/development/test_convmmd_selection_jax_contract.py` |
| `CMMD-SEL-DTYPE-001` | float32/float64 dtype is preserved end to end across the selected path (no silent upcast) | float64, float32 | Pending | `tests/development/test_convmmd_selection_jax_contract.py` |
| `CMMD-SEL-VMAP-001` | The selection-aware denoiser vmaps over a group's observation rows to the batched result | float64 | Pending | `tests/development/test_convmmd_selection_jax_contract.py` |
| `CMMD-SEL-FIT-001` | A valid selected (Gaussian-\(\Omega(z)\)) analytic fit reports non-failure, reduces the loss, and returns SPD covariances | float64 | Pending | `tests/development/test_convmmd_selection_jax_contract.py` |

## Known-selection on the observed value \(\Omega(\tilde x)\) rows (contract §17.12)

Homoscedastic, fully-observed selection on the observed value. The Gaussian window acts
on the noise-convolved model, so the transform is the §17.5 identity on the inflated
covariances \(\Sigma+S\), the analytic loss is the base §4 loss on \((\pi'',\nu'',B'')\)
with zero noise, and the denoiser is the base MAR posterior (selection cancels).

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-SEL-OBS-XFORM-001` | The observed-value transform (§17.5 on inflated covariances) equals the oracle and yields SPD \(B''_k\) | float64, float32 | Pending | `tests/development/test_convmmd_selection_observed.py` |
| `CMMD-SEL-OBS-GAUSS-001` | Analytic \(\Omega(\tilde x)\) selected loss equals the independent NumPy oracle | float64 (rtol 5e-8/atol 5e-10), float32 (declared profile) | Pending | `tests/development/test_convmmd_selection_observed.py` |
| `CMMD-SEL-OBS-REDUCE-001` | \(\Psi^{-1}=0\) reduces the observed-value analytic loss to the base §4 loss with noise \(S\) | float64, float32 | Pending | `tests/development/test_convmmd_selection_observed.py` |
| `CMMD-SEL-OBS-DENOISE-001` | The \(\Omega(\tilde x)\) denoiser equals the base §7 MAR posterior mean (selection cancels in \(z\)) | float64, float32 | Pending | `tests/development/test_convmmd_selection_observed.py` |
| `CMMD-SEL-OBS-MC-001` | The general-\(\Omega(\tilde x)\) SNIS loss converges to the §17.12 analytic value (statistical: spread shrinks, within noise); Gaussian and callable specs agree at a fixed key | float64 | Pending | `tests/development/test_convmmd_selection_observed.py` |
| `CMMD-SEL-OBS-VAL-001` | Genuinely heteroscedastic noise is rejected; an \((N,D,D)\) stack with identical rows is accepted | float64 | Pending | `tests/development/test_convmmd_selection_observed.py` |
| `CMMD-SEL-OBS-JIT-001` | The observed-value analytic loss is callback-free, does not retrace on same-shape inputs, and is differentiable through the observed-space transform | float64 | Pending | `tests/development/test_convmmd_selection_observed.py` |
| `CMMD-SEL-OBS-FIT-001` | A valid \(\Omega(\tilde x)\) analytic fit reports non-failure, reduces the loss, returns SPD covariances, and recomputes its reported loss exactly | float64 | Pending | `tests/development/test_convmmd_selection_observed.py` |

## Known-selection heteroscedastic \(\Omega(\tilde x)\) rows (contract §17.13)

Per-object measurement noise \(S_i\), fully observed. Each detected object's \(S_i\) is
**known**, so the §17.5 identity applies to the **per-object** inflated covariances
\(\Sigma_k+S_i\): a Gaussian window is machine-eps analytic (a sum of per-object one-sample
discrepancies against \((\pi''_{k,i},\nu''_{k,i},B''_{k,i})\)), a general \(\Omega\) uses a
per-object SNIS with the object's own \(S_i\), and **no noise model is required**. The
all-\(S_i\)-equal case reduces to §17.12.

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-SEL-HET-XFORM-001` | The per-object observed-value transform (§17.5 on the inflated \(\Sigma_k+S_i\)) equals the oracle for **arbitrary per-object** \(S_i\) and yields SPD \(B''_{k,i}\) | float64, float32 | Pending | `tests/development/test_convmmd_selection_hetero.py`, `tests/reference/test_convmmd_selection_hetero_reference.py` |
| `CMMD-SEL-HET-GAUSS-001` | The heteroscedastic \(\Omega(\tilde x)\) per-object analytic loss equals the independent NumPy oracle (per-object transform ∘ base §4 one-sample discrepancy) | float64 (rtol 5e-8/atol 5e-10), float32 (declared profile) | Pending | `tests/development/test_convmmd_selection_hetero.py`, `tests/reference/test_convmmd_selection_hetero_reference.py` |
| `CMMD-SEL-HET-REDUCE-001` | All-\(S_i\)-equal reduces the per-object analytic loss to the §17.12 homoscedastic value; \(\Psi^{-1}=0\) reduces it to the base §4 loss with per-item noise \(S_i\) | float64, float32 | Pending | `tests/development/test_convmmd_selection_hetero.py`, `tests/reference/test_convmmd_selection_hetero_reference.py` |
| `CMMD-SEL-HET-DENOISE-001` | The heteroscedastic \(\Omega(\tilde x)\) denoiser equals the base §7 MAR posterior mean at each object's own \(S_i\) (selection cancels in \(z\)); genuinely heteroscedastic \(S_i\) is **accepted** | float64, float32 | Pending | `tests/development/test_convmmd_selection_hetero.py` |
| `CMMD-SEL-HET-MC-001` | The general-\(\Omega(\tilde x)\) per-object SNIS loss (known \(S_i\), no noise model) converges to the per-object analytic value at the Monte-Carlo rate; per-object ESS is logged | float64 | Pending | `tests/development/test_convmmd_selection_hetero.py`, `tests/reference/test_convmmd_selection_hetero_reference.py` |
| `CMMD-SEL-HET-NOMODEL-001` | The heteroscedastic loss and denoiser consume only the detected objects' own \(S_i\) and require **no noise-model input** (the pygmmis-contrast invariant of §17.13) | float64 | Pending | `tests/development/test_convmmd_selection_hetero.py` |
| `CMMD-SEL-HET-JIT-001` | The per-object analytic and SNIS leaves are callback-free, `vmap` over the observation batch, do not retrace on same-shape inputs, and are differentiable through the per-object transform; the SNIS leaf reads only the passed key | float64 | Pending | `tests/development/test_convmmd_selection_hetero.py` |
| `CMMD-SEL-HET-FIT-001` | A valid heteroscedastic \(\Omega(\tilde x)\) analytic fit reports non-failure, reduces the loss, returns SPD covariances, and recomputes its reported loss exactly | float64 | Pending | `tests/development/test_convmmd_selection_hetero.py` |

## Known-selection heteroscedastic \(\Omega(\tilde x)\) + projection (MAR) rows (contract §17.14)

Per-object measurement noise \(S_i\) **and** per-coordinate missingness (projection \(P_i\)).
The Gaussian window is marginalized onto each observed subspace — precision
\((P_i\Psi P_i^\top)^{-1}\), **not** \(P_i\Psi^{-1}P_i^\top\) — and the §17.5 transform runs
per object on the projected inflated covariance \(P_i\Sigma_kP_i^\top+S_i\), grouped by mask
pattern (§16.7). A Gaussian window is machine-eps analytic; a general \(\Omega\) uses per-object
projected SNIS on the observed-subspace draw; neither needs a noise model. Reduces to §17.13 at
\(P_i=I\) and to the base §16 masked loss at \(\Psi^{-1}=0\).

| ID | Behavior | Required dtype | Status | Evidence |
|---|---|---|---|---|
| `CMMD-SEL-OBSMAR-MARGWIN-001` | The marginalized observed-subspace window precision equals \((P_i\Psi P_i^\top)^{-1}=(\Psi_{C_iC_i})^{-1}\) (inverse of the covariance principal submatrix), **not** \(P_i\Psi^{-1}P_i^\top\) — checked directly on a \(D\ge3\) object with a genuinely off-diagonal \(\Psi\) against a brute-force marginalization; the \(\Psi^{-1}=0\) branch yields \(\Psi_i^{-1}=0\) without inverting a singular precision | float64 | Pending | `tests/development/test_convmmd_selection_obsmar.py`, `tests/reference/test_convmmd_selection_obsmar_reference.py` |
| `CMMD-SEL-OBSMAR-XFORM-001` | The per-object **projected** transform (marginalized window ∘ §17.5 on the projected inflated \(P_i\Sigma_kP_i^\top+S_i\)) equals the oracle for arbitrary per-object \(S_i\) and mask, and yields SPD \(C_{k,i}\) | float64, float32 | Pending | `tests/development/test_convmmd_selection_obsmar.py`, `tests/reference/test_convmmd_selection_obsmar_reference.py` |
| `CMMD-SEL-OBSMAR-GAUSS-001` | The heteroscedastic \(\Omega(\tilde x)\)+MAR per-object analytic loss equals the independent NumPy oracle on mixed masks (incl. a fully-observed and an `M=0` group), informative-weight-normalized | float64 (rtol 5e-8/atol 5e-10), float32 (declared profile) | Pending | `tests/development/test_convmmd_selection_obsmar.py`, `tests/reference/test_convmmd_selection_obsmar_reference.py` |
| `CMMD-SEL-OBSMAR-REDUCE-001` | \(P_i=I\) reduces the projected loss to the §17.13 heteroscedastic value; \(\Psi^{-1}=0\) reduces it to the base §16 masked (MAR) loss — both exactly | float64, float32 | Pending | `tests/development/test_convmmd_selection_obsmar.py`, `tests/reference/test_convmmd_selection_obsmar_reference.py` |
| `CMMD-SEL-OBSMAR-DENOISE-001` | The \(\Omega(\tilde x)\)+MAR denoiser equals the base §16.4 masked posterior mean at each object's own \(S_i\) (selection cancels in \(z\)), returned at full `D` in original row order | float64, float32 | Pending | `tests/development/test_convmmd_selection_obsmar.py` |
| `CMMD-SEL-OBSMAR-MC-001` | The general-\(\Omega(\tilde x)\)+MAR per-object projected SNIS loss (known \(S_i\), observed-subspace weight, no noise model) converges to the per-object analytic value at the Monte-Carlo rate; per-object ESS is logged | float64 | Pending | `tests/development/test_convmmd_selection_obsmar.py`, `tests/reference/test_convmmd_selection_obsmar_reference.py` |
| `CMMD-SEL-OBSMAR-NOMODEL-001` | The projected loss and denoiser consume only the detected objects' own \(S_i\) and require **no noise-model input** (the pygmmis-contrast invariant carried into the MAR-composed case) | float64 | Pending | `tests/development/test_convmmd_selection_obsmar.py` |
| `CMMD-SEL-OBSMAR-JIT-001` | The per-object projected analytic and SNIS leaves are callback-free, `vmap` over a group's rows, do not retrace on same-shape inputs, and are differentiable through the per-object transform (the window inversion is host-side and carries no gradient); the SNIS leaf reads only the passed key | float64 | Pending | `tests/development/test_convmmd_selection_obsmar.py` |
| `CMMD-SEL-OBSMAR-FIT-001` | A valid \(\Omega(\tilde x)\)+MAR analytic fit reports non-failure, reduces the loss, returns SPD covariances, and recomputes its reported loss exactly | float64 | Pending | `tests/development/test_convmmd_selection_obsmar.py` |

## Known-selection comparison-evidence rows (fairness, not performance)

The convMMD-vs-pygmmis comparison (`development/convmmd_pygmmis_comparison_plan.md`) now
spans **five** predeclared tasks: Design B (selection on a **noise-free coordinate**, so
`Ω(x̃) ≡ Ω(z)`), Design A (homoscedastic `Ω(x̃)`, §17.12), Design A-het (**heteroscedastic**
`Ω(x̃)`, §17.13 — convMMD uses each object's own known `S_i` with **no noise model**,
pygmmis gets the true noise field as `covar_callback`), Design A-mar (**heteroscedastic
`Ω(x̃)` composed with missing coordinates**, §17.14 — convMMD marginalises the missing part
**exactly** via `P_i`, pygmmis can only inflate the missing coordinate, its exact `R` route
being unimplemented under a selection callback), and Task E (non-Gaussian truth, both fit a
misspecified `K`-GMM). Both start from the same init, fit the true population, and are scored
with the same denoiser; each endpoint is gated against its own oracle before comparison. It is
a **capability/fairness demonstration, not a performance claim** (`performance_claim` stays
`none`).

| ID | Behavior | Status | Evidence |
|---|---|---|---|
| `CMMD-SEL-CMP-001` | The retained convMMD-vs-pygmmis selection record (all five tasks) conforms to its authoritative schema, **both methods' endpoints pass their own oracle gates** (the selection-aware denoiser machinery near machine epsilon), every observation is preserved, and `performance_claim` is `none` | Pending | `tests/benchmarks/test_convmmd_pygmmis_comparison.py` |
| `CMMD-SEL-CMP-002` | The record is SHA-256-pinned; the schema validator rejects a performance claim, a failed/inconsistent/hollow gate, a tampered winner, a summary inconsistent with observations, and a missing selection block | Pending | `tests/benchmarks/test_convmmd_pygmmis_comparison.py` |
| `CMMD-SEL-CMP-003` | The record and the predeclared plan disclose every design's convention (`Ω(x̃)≡Ω(z)` for B; genuine `Ω(x̃)` for A; heteroscedastic for A-het; non-Gaussian for E), the honest scope, and the STOP-and-ask gates; the plan names all designs | Pending | `tests/benchmarks/test_convmmd_pygmmis_comparison.py` |
| `CMMD-SEL-HET-CMP-001` | The heteroscedastic `Ω(x̃)` task (Design A-het) carries convMMD's **machine-eps** per-object analytic gate, discloses the **no-noise-model asymmetry** (convMMD uses each object's own `S_i`; pygmmis is given the true noise field its EM imputation requires), and Task E discloses the non-Gaussian misspecification with **no predicted winner** — all `performance_claim: none` | Pending | `tests/benchmarks/test_convmmd_pygmmis_comparison.py` |
| `CMMD-SEL-OBSMAR-CMP-001` | The `Ω(x̃)` + missing-coordinates task (Design A-mar, §17.14) carries convMMD's **machine-eps** §17.14 analytic + masked-denoiser gate, and the record + plan disclose the **exact-marginalisation-vs-inflation** asymmetry (convMMD marginalises the missing part exactly via `P_i`; pygmmis can only inflate it, its exact `R` route raising `NotImplementedError` under a selection callback — verified from source) and the no-noise-model asymmetry — all `performance_claim: none` | Pending | `tests/benchmarks/test_convmmd_pygmmis_comparison.py` |

## Explicit non-claims

- No timing, throughput, memory, or accelerator claim follows from these rows;
  `performance_claim` is `none` everywhere.
- The convMMD-vs-XDGMM comparison holds the model class fixed to a GMM; it
  isolates the fitting objective and does **not** exercise the flexible/implicit
  models or higher-dimensional/misspecified regimes where convMMD's reference
  advantages live. Those are out of this contract revision's scope.
- Normalizing-flow or non-Gaussian-kernel convMMD is deferred to a later contract
  revision.
- The `CMMD-MISS-*` / `CMMD-M0-*` rows cover **missing-at-random (MAR)** only:
  per-coordinate missingness via an exact projection `P_i` (contract §16). The
  `CMMD-SEL-*` rows add **missing-not-at-random (MNAR)** selection via a known
  completeness `Ω` **on the true value** `Ω(z)` (contract §17): a Gaussian-window
  analytic path (machine-eps oracle) and a general-`Ω` SNIS Monte-Carlo path
  (statistical only). **No machine-eps parity is claimed for a general `Ω`**, and the
  SNIS estimator is biased at finite `M`, consistent as `M→∞`. Selection on the
  **observed** value `Ω(x̃)` is implemented for **homoscedastic, fully-observed** noise
  (the `CMMD-SEL-OBS-*` rows; contract §17.12) and, in `0.4.0-draft.1`, for
  **heteroscedastic** per-object noise (the `CMMD-SEL-HET-*` rows; contract §17.13), and,
  in `0.5.0-draft.1`, **composed with the §16 projection** (the `CMMD-SEL-OBSMAR-*` rows;
  contract §17.14) — **exactly**, via a per-object transform on each detected object's own
  known `S_i`, needing **no noise model** (a Gaussian window is machine-eps analytic; a
  general `Ω` uses a per-object SNIS held to the statistical gate only). Only an
  **unknown** `Ω` remains out of scope. No masked row implies any MNAR capability, and no
  `CMMD-SEL-*` / `CMMD-SEL-HET-*` / `CMMD-SEL-OBSMAR-*` row is flipped to Supported without
  maintainer qualification.
- The masked comparison (Task C) demonstrates the MAR capability head-to-head on
  Gaussian truth; it is a fairness/parity demonstration, not a performance claim, and
  `performance_claim` stays `none`.
