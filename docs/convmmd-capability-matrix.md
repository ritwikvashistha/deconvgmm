# convMMD capability and quality matrix

- Matrix ID: `xdgmm-jax.convmmd.matrix`
- Matrix version: `0.2.0-draft.1`
- Applies to: [`xdgmm-jax.convmmd` contract `0.2.0-draft.1`](convmmd-model-contract.md)
- Last updated: 2026-08-30

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

## Explicit non-claims

- No timing, throughput, memory, or accelerator claim follows from these rows;
  `performance_claim` is `none` everywhere.
- The convMMD-vs-XDGMM comparison holds the model class fixed to a GMM; it
  isolates the fitting objective and does **not** exercise the flexible/implicit
  models or higher-dimensional/misspecified regimes where convMMD's reference
  advantages live. Those are out of this contract revision's scope.
- Normalizing-flow or non-Gaussian-kernel convMMD is deferred to a later contract
  revision.
- The missing-data rows cover **missing-at-random (MAR)** only: per-coordinate
  missingness via an exact projection `P_i` (contract §16). **Missing-not-at-random
  (MNAR)** selection via a known completeness `Ω` is **not** implemented in this
  revision — it is a *planned future revision, not a fundamental limitation* (§16.11;
  `development/convmmd_mnar_design_note.md`). Only an **unknown** selection function
  is genuinely out of scope. No masked row implies any MNAR capability.
- The masked comparison (Task C) demonstrates the MAR capability head-to-head on
  Gaussian truth; it is a fairness/parity demonstration, not a performance claim, and
  `performance_claim` stays `none`.
