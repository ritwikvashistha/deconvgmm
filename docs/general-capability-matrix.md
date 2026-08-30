# General-projection capability and quality matrix

- Matrix ID: `xdgmm-jax.general-xd.matrix`
- Matrix version: `0.2.0-draft.1`
- Applies to: [`xdgmm-jax.general-xd` contract
  `0.2.0-draft.1`](general-model-contract.md)
- Status: specified; temporary fixed-`M`, eager-boundary, mask-grouping, grouped
  control/inference, observed-sampling, serialization, and recovery subsets have
  local development evidence, but every formal row below remains Pending
- Last updated: 2026-08-28

This matrix converts the Phase 3 mathematical contract into named and
measurable acceptance tests. It records intended evidence, not current
public capability. Neither the preserved prototype nor the temporary identity
kernel implements this contract. Temporary general modules now exercise the
canonical fixed-`M` numerical leaf, explicit projection/noise tags and selected-
dtype validation, deterministic boolean-mask grouping/restoration, global
grouped updates and host fit controllers, fixed/grouped inference, and fixed-
`M` observed sampling. They are partial design evidence, not a conforming public
implementation.

## 1. Gate and status meanings

| Mark | Meaning |
|---|---|
| **Gate** | required before the `0.2.0a1` general-projection alpha is advertised |
| **Pending** | specified, but no conforming public implementation evidence exists |
| **Qualified** | tested only inside the stated dtype, backend, size, and conditioning envelope |
| **Deferred** | outside this contract; accepting or ignoring the feature silently is a failure |

All rows in Sections 5–9 are positive **Gate / Pending** rows. Section 10 rows
marked `Gate / Pending` are required negative sentinels for features whose
implementation remains deferred; its GPU row remains deferred platform
evidence. A row becomes verified only when an immutable evidence record links
the public-package test, fixture digest, supported environment, and observed
maximum errors. A similarly named temporary development test is useful design
evidence but does not change formal status.

The `XD-GEN-REF-BOVY-001` row is specifically pending creation and audit of its
numeric reference fixture. Its source revision, source-custody/output-schema
generator, and offline Linux/amd64 CPU container recipe are now pinned and
locally audited. The source archive SHA-256 is
`c1882ea6be58c4f08a9d66c539504f1a1c4bc892fa2a5adad7abe47fcaf165fc`
and its 66-file manifest SHA-256 is
`7111a901ba02094f16f3fb44f748da4b831ab9fa5c4e6dea75ce336b624081b8`.
Docker and Podman were unavailable locally, so the container has not produced
the numeric archive. No direct endpoint or parity result is claimed by this
document.

Native (non-container) parity EVIDENCE now exists (2026-08-29): Bovy's pinned
C/OpenMP reference was built natively (conda GSL 2.8, gcc 13.3, no container) and
one general-projection EM step on the official `N=19,K=3,D=4,M=2` `build_inputs`
fixture agrees across Bovy, the independent NumPy oracle, and XDGMM-JAX to
~1e-13 in float64 (retained as internal maintainer evidence). This is genuine numeric
evidence at the qualified JAX 0.6.2 / NumPy 1.26.4 lane, but it is NOT the pinned
Linux/amd64 container-custody archive and asserts no container provenance, so
`XD-GEN-REF-BOVY-001` remains **Pending** until the reproducible container run
and its audited archive/metadata exist.

As of the 2026-08-28 living-record refresh, development tests cover identity
equivalence, `M != D`, dense nonorthogonal projections, weights, `M=0`, eager/JIT
fixed-`M` status, zero-sized inference batches, selected gradients, rollback,
extreme representable weight scales, covariance invariants, every explicit
projection/noise tag, mask grouping/restoration, selected-block local covariance
revalidation, restored grouped posteriors, and stable global grouped M-steps
with second-pass objectives. Host fixed-step and dynamically converged grouped
controllers now preserve accepted-state history, common normalized stopping,
controls, initialization custody, and exact rollback. Fixed-`M` and grouped
inference cover scoring, exact-shape weighted reductions, probabilities,
predictions, and posterior moments; the reducers are hardened for the tested
subnormal/overflow-cancellation cases and explicitly fail unrecoverable scale
separation. Fixed-`M` general observed sampling covers every explicit projection/
noise mode, singular PSD noise, PRNG semantics, and `n=0`/`M=0`.

Temporary deterministic CPU artifact tests cover general-tagged parameters and
grouped fit results, and an independent audit found no remaining CPU correctness
blocker. A deterministic complete-fit recovery fixture separately exercises one
weighted, dense-projection, variable-`M` workload with permutation-invariant
parameters and independent latent-holdout density/moment metrics; it is basin-
conditioned evidence, not one-step parity or global-optimum robustness. The
authoritative repository-wide local CPU suite passes 1,131 tests in 618.84
seconds (0:10:18) on Python 3.10.11 and JAX/JAXlib 0.6.2.

Whole-group JIT/autodiff, stable public scoring/results/exceptions, grouped or
ragged sampling, all negative sentinels, the numeric Bovy fixture, an immutable
package commit, hosted matrix evidence, and controlled GPU evidence are absent.
Explicit shared inputs are currently expanded to canonical per-item buffers
rather than remaining physically shared. Consequently every formal row in this
matrix remains Pending.

## 2. Target support envelope

| Capability | `0.2.0a1` target | Required rows |
|---|---|---|
| General per-item linear projection | Gate | `XD-GEN-R-001`, `XD-GEN-REF-BOVY-001` |
| Different latent/observed dimensions | Gate | `XD-GEN-ANALYTIC-001`, `XD-GEN-REDUCE-001` |
| Explicit shared and per-item projection modes | Gate | `XD-GEN-SHAPE-001`, `XD-GEN-PROJ-001` |
| Identity fast-path equivalence | Gate | `XD-GEN-ID-001` |
| Full/diagonal/isotropic known noise | Gate | `XD-GEN-NOISE-001` |
| Observation weights | Gate | `XD-GEN-WEIGHT-001`, `XD-GEN-WEIGHT-002`, `XD-GEN-WEIGHT-SCORE-001`, `XD-GEN-WEIGHT-CONV-001` |
| Exact grouped missing coordinates | Gate | `XD-GEN-MISSING-001`, `XD-GEN-M0-001` |
| General latent posterior | Gate | `XD-GEN-ANALYTIC-001`, `XD-GEN-R-001`, `XD-GEN-INFER-FAIL-001` |
| General observed sampling | Gate | `XD-GEN-SAMPLE-001`, `XD-GEN-SAMPLE-002`, `XD-GEN-PRNG-001` |
| Log-space tail behavior | Gate | `XD-GEN-RESP-TAIL-001` |
| Prediction/ties/failures | Gate | `XD-GEN-PRED-001`, `XD-GEN-INFER-FAIL-001` |
| Pure dense JAX kernels | Gate | `XD-GEN-JIT-001`, `XD-GEN-VMAP-001`, `XD-GEN-GRAD-001` |
| float64 CPU scientific-reference path | Gate | every row marked f64/CPU |
| float32 CPU | Qualified alpha support | every row marked f32/CPU |
| GPU | Unadvertised | later controlled-hardware matrix |
| TPU, multi-GPU, multi-host | Unsupported/unadvertised | later matrix revision |
| Fixed parameters, priors, split-and-merge | Deferred | later contract revision |

The initial correctness domain is $1\leq N\leq4096$, $1\leq K\leq8$,
$1\leq D\leq32$, and $0\leq M\leq32$ per group. This is not a performance
or scaling claim. Variable-$M$ inputs are evaluated as fixed-$M$ dense groups.

## 3. Reproducible fixture policy

1. Small analytic fixtures are literal arrays checked into `tests/fixtures/`.
2. Generated fixtures use NumPy `Generator(PCG64(20260825))` once, then store
   deterministic `.npz` bytes, SHA-256, generator metadata, and a committed
   generator script. Ordinary tests load the archive rather than regenerate it.
3. Independent expected values use a loop-oriented NumPy/SciPy float64 oracle
   that imports no production or temporary JAX kernel. Expected values may not
   be copied from implementation output.
4. Every stored fixture records $N,K,D,M$, dtype, covariance construction,
   maximum $\kappa_2(T_{ik})$, weight normalization, and matrix/contract IDs.
5. JAX sampling uses explicit recorded keys. Timing is never inferred from a
   correctness test, and all completion/timing reports synchronize device work.
6. External reference archives record source commit, source and license hashes,
   build flags, compiler and numerical-library versions, platform, precision,
   wrapper/instrumentation patches, and generated archive digest.
7. A fixture generator that consults two references MUST fail unless they agree
   on their shared endpoints before writing the archive.

## 4. Numerical profiles

Unless a row overrides them, `allclose(actual, reference)` uses:

| Profile | `rtol` | `atol` | Intended use |
|---|---:|---:|---|
| `GEN-REF64` | `8e-10` | `8e-12` | well-conditioned float64 values and one-step parameters |
| `GEN-LOG64` | `8e-10` | `8e-10` | float64 log densities/objectives |
| `GEN-REF32` | `2e-4` | `2e-5` | well-conditioned float32 values |
| `GEN-LOG32` | `3e-4` | `3e-5` | float32 log densities/objectives |
| `GEN-GRAD64` | `3e-5` | `3e-6` | central-difference gradient comparisons |

Ordinary fixtures MUST have finite representable entries,
$\kappa_2(T_{ik})\leq10^4$, and absolute component log densities below `1e3`.
In addition to `allclose`, ordinary float64 log comparisons require maximum
absolute error `<=2e-8`, and ordinary float32 log comparisons require maximum
absolute error `<=8e-3`.

The initially qualified effective-covariance conditioning domain is

```text
float64: kappa_2(T_eff) <= 1e8
float32: kappa_2(T_eff) <= 1e4
```

An out-of-domain case may return a valid result satisfying every invariant or
an explicit numerical-failure status. It may not return a successful nonfinite
or materially indefinite result.

Responsibility and mixture-weight normalization bounds are:

```text
float64: max absolute row/sum residual <= 5e-13
float32: max absolute row/sum residual <= 2e-5
```

For a covariance $A$, let

\[
s(A)=\max(1,\lVert A\rVert_2),\qquad
e_{\rm sym}(A)=\frac{\lVert A-A^\mathsf T\rVert_\infty}{s(A)}.
\]

| Dtype | symmetry | posterior PSD residual | model covariance requirement |
|---|---:|---:|---|
| float64 | `e_sym <= 2e-13` | `lambda_min >= -2e-11*s(A)` | selected-dtype Cholesky succeeds without hidden jitter |
| float32 | `e_sym <= 2e-6` | `lambda_min >= -5e-5*s(A)` | selected-dtype Cholesky succeeds under documented policy |

Input covariance symmetry and PSD validation use the same dtype-specific bounds.
No test tolerance authorizes material eigenvalue clipping.

## 5. Shapes, adapters, validation, and metadata gates

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-GEN-SHAPE-001` | `K=3,D=4,M=2`; one observation and batch `B=(5,)`; then inference batch `B=(2,3)` | f64/CPU and f32/CPU | Every required public score, probability, conditional, and marginalized-posterior output has exactly the shape in Contract §3; any diagnostic gain has the specified optional shape. A one-item result equals the corresponding native batch item under `GEN-REF*`. No length-one axis is retained or invented. |
| `XD-GEN-PROJ-001` | `N=7,D=4,M=2`; one literal shared `(M,D)` projection and its exactly repeated `(N,M,D)` form; repeat with shared/per-item full `S` | f64/CPU and f32/CPU | Explicit shared and per-item modes agree under `GEN-REF*` for likelihood, posterior, statistics, and one update. Raw rank-two `R` and raw rank-two `S` passed to a batch without shared tags fail with received/expected shapes. A leading length-one batch is not broadcast. |
| `XD-GEN-NOISE-001` | `N=4,D=3,M=2`; isotropic `[0,.1,.5,2]`, unequal diagonal variances, correlated full covariances, and explicit shared versions | f64/CPU and f32/CPU | Adapter matrices equal literal expected arrays; isotropic off-diagonals are exactly zero. All modes produce the same result when their mathematical `S` is equal. Negative variances, raw ambiguous shared forms, and dimension mismatches fail before compilation. |
| `XD-GEN-VAL-001` | In turn: transposed `x/R/S`; mismatched batch axes; scalar/length-one fitting weights; NaN/Inf; bool/complex arrays; integer `R/S/parameters`; nonnormalized mixture weights; non-PD `V`; asymmetric/indefinite `S`; negative/NaN sample weights; a positive weight that underflows to zero in float32; all-zero informative weights; and float64 with x64 disabled | f64/CPU and f32/CPU, plus x64-disabled subprocess | Every invalid public input fails actionably without returning parameters. Messages name the offending input and received/expected shape or domain. Integer `x` and representable integer nonnegative sample weights convert to the selected float dtype without truncating posterior output. Positive-weight underflow is a precision error rather than silent row removal. |
| `XD-GEN-META-001` | Successful weighted general converged and fixed-step results, plus identity-valued `R` invoked through the general API | f64/CPU | Results carry exact ID `xdgmm-jax.general-xd` and version `0.2.0-draft.1`. Strict metadata round-trip preserves both; unknown/identity IDs, unknown versions, missing/extra fields, and relabeling fail closed unless an explicit migration is invoked. Fit diagnostics identify the weighted-informative objective, informative weight sum, jitter, and ridge. |

## 6. Mathematical and identity-equivalence gates

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-GEN-ANALYTIC-001` | Literal `K=1,D=3,M=2`: `mu=(.2,-.4,.7)`, `V=((1,.2,-.1),(.2,.8,.15),(-.1,.15,.6))`, `R=((1,.3,-.2),(.1,-.5,.8))`, `S=((.4,.08),(.08,.25))`, `x=(1.1,-.3)` | f64/CPU and f32/CPU | `T`, residual, log density, gain, `q=[1]`, conditional/marginal mean and covariance match independent solve formulas under `GEN-LOG*`/`GEN-REF*`. Joseph and subtractive NumPy covariance forms agree under `GEN-REF64`; returned covariances meet invariants. |
| `XD-GEN-ID-001` | Existing identity likelihood (`N=11,K=3,D=2`) and one-step (`N=7,K=3,D=2`) fixtures; general calls use both explicit identity adapter and repeated per-item identity matrices, weights all one | f64/CPU and f32/CPU | General and conforming identity APIs agree for log components, scores, responsibilities, gains, component/marginal posteriors, sufficient statistics, update, status, and objective under `GEN-REF*`/`GEN-LOG*`. Jitter `0` and `1e-6` and ridge `0` and `1e-4` are included. Metadata follows the API used rather than the internal fast path. |
| `XD-GEN-REDUCE-001` | Literal `K=2,D=4,M=2`, coordinate-selection `R=((1,0,0,0),(0,0,1,0))`, zero and correlated `S`, then a rotated full-row-rank `R`; one observation and `N=9` batch | f64/CPU and f32/CPU | Observed scores and latent posterior moments agree with an independent block/linear-Gaussian calculation. Unobserved latent coordinates retain the correct conditional uncertainty; dimensions are not silently dropped from the returned latent posterior. |
| `XD-GEN-R-001` | Stored ordinary fixture `N=17,K=3,D=4,M=3` with heterogeneous dense nonorthogonal `R_i`, correlated `V_k/S_i`, and positive initial weights; exactly one unweighted update | f64/CPU and f32/CPU | `T`, residuals, component log densities, `q`, gains, `b`, `B`, sufficient statistics, and updated parameters agree with the independent NumPy oracle under `GEN-REF*`/`GEN-LOG*`. Weights normalize and every covariance meets invariants. |
| `XD-GEN-RESP-TAIL-001` | Stored `N=4,K=3,D=16,M=8` fixture with literal dense nonidentity full-row-rank projections, diagonal-plus-correlated `V/S`, unequal mixture weights, and observed coordinates near `+1000`, `-1000`, zero, and alternating signs | f64/CPU and f32/CPU | Every component log density is finite while direct probability-space exponentiation underflows for at least one pair. Responsibilities remain finite/nonnegative, no row is all zero, row sums meet the dtype bound, and values match independent log-softmax under `GEN-REF*` absolute tolerances. |
| `XD-GEN-EM-001` | Stored synthetic fixture `N=96,K=3,D=4` split between `M=1,2,3`; 12 exact grouped updates from fixed parameters, zero jitter/ridge | f64/CPU and f32/CPU | Every accepted objective/parameter is finite. Normalized increments are at least `-2e-10*max(1,abs(previous))` in f64 and `-3e-5*max(1,abs(previous))` in f32. The final grouped objective is independently recomputed from returned parameters. |

## 7. Weighting and missing-coordinate gates

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-GEN-WEIGHT-001` | Stored `N=11,K=3,D=3,M=2` arbitrary-`R` fixture with weights `(0,.25,1,2,.5,3,1.5,.75,4,.125,2.25)` | f64/CPU and f32/CPU | Responsibilities/posteriors equal the unweighted per-row values. Weighted log likelihood, informative mean, `(n,h,Q)`, and one update match an independent weighted NumPy oracle under `GEN-REF*`/`GEN-LOG*`; weights normalize and covariances meet invariants. |
| `XD-GEN-WEIGHT-002` | Previous fixture; multiply all weights by `13.5`, and by `2^+/-200` in f64 / `2^+/-20` in f32 while all required values remain representable. Separately use integer weights `(1,2,0,3,1,2,1,0,4,2,1)` and explicitly replicate rows | f64/CPU and f32/CPU | Global scaling leaves normalized objective, raw statistics after the same common scaling, one-step parameters, fixed-step trajectory, collapse mask, and converged stopping decision unchanged under `GEN-REF*`. Integer weighting agrees with row replication for objective/parameters under `GEN-REF*`; total log likelihood and raw statistics scale as mathematically expected. Deliberate overflow/positive-underflow cases fail explicitly. |
| `XD-GEN-WEIGHT-SCORE-001` | Inference batch `B=(2,3)`, `K=2,D=3,M=2`, per-item `R/S`, literal `sample_weight.shape==(2,3)` including zero entries; then scalar, `(1,)`, `(2,1)`, all-zero, and mixed-dimension zero-informative-weight cases | f64/CPU and f32/CPU | `score_samples` is unchanged by weights. Weighted total and informative mean reduce all batch axes and match independent flattened calculations under `GEN-LOG*`. Every nonexact weight shape fails without broadcasting. A batch containing informative rows with `W=0` fails `no_informative_weight`; a structurally all-`M=0` collection scores exactly zero. |
| `XD-GEN-WEIGHT-CONV-001` | Stored two-group `D=3,M in {1,2}` fixture; `max_iter=5` and large `tol` for one accepted step. Separately use previous normalized objective `-10`, `tol=1e-6`, `decrease_tol=1e-10`, and candidates `-10-1.1e-9`, `-10-1e-9`, `-9.99999`, `-9.999989`; repeat with globally scaled weights | f64/CPU | Returned state/history semantics match identity Contract §8 using the informative weighted mean. The four candidates classify as material decrease, converged, converged, and continue. Scale changes neither `n_iter` nor status. A material decrease rolls back globally; all-zero informative weight fails before fitting. |
| `XD-GEN-CONV-CONTROL-001` | Ordinary grouped float32 fit; supply each of `tol` and `decrease_tol` as finite float64 `1e300`, positive float64 `1e-300`, and exact zero | f32/CPU | Overflow and positive-nonzero-to-zero conversion fail before the first grouped update. Exact zero remains valid. Identity and grouped convergence controls share one eager selected-dtype preparation boundary. |
| `XD-GEN-MISSING-001` | Literal `N=9,P=4,D=3` dense finite data with masks `1111,1010,0100,0000,1010,1111,0011,0000,0100`; per-item dense `R/S` and weights | f64/CPU and f32/CPU | Eager adapter emits groups in ascending lexicographic boolean-tuple order with `False < True`, preserves relative row order, selects ascending coordinates, forms exact projection rows/principal covariance blocks, and restores outputs to original order. Reconstructed score/posterior and one grouped update match direct per-row NumPy calculations. Nonboolean/mismatched masks and every NaN/Inf—including masked positions—fail; large-noise values are not treated as missing. |
| `XD-GEN-M0-001` | `K=3,D=3`; a four-row `M=0` group, then append/prepend it with arbitrary positive weights to the informative fixture from `XD-GEN-WEIGHT-001`. Repeat with accepted stored mixture weights `(.2,.3,.5000000000002)` f64 and `(.2,.3,.50001)` f32 | f64/CPU and f32/CPU | For every empty row, component log density and score are exactly zero, `q` is exactly the stored `alpha`, `b` equals `mu`, and `B` equals `V`; marginalized moments equal independent stored-weight prior-mixture moments. The branch does not execute log-sum-exp or factorization. Adding/removing empty rows changes no fit statistic, normalized objective, accepted parameter, status, or stopping iteration. An all-empty fit fails with `no_informative_weight`; all-empty `score` is exactly zero. |

## 8. Numerical-control and failure gates

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-GEN-JITTER-001` | `XD-GEN-R-001` with fixed `factor_jitter=1e-6`, then zero | f64/CPU | Density, responsibilities, gain, posterior, and one update match an independent calculation using `S+1e-6*I_M` everywhere under `GEN-REF64`/`GEN-LOG64`. Joseph form uses the same effective noise. Returned `V` receives no direct jitter addition; diagnostics label an effective objective. Zero reproduces `XD-GEN-R-001`. |
| `XD-GEN-RIDGE-001` | `XD-GEN-R-001` with `covariance_ridge=1e-4`, zero jitter | f64/CPU | Means/weights equal the exact update and each covariance equals `V_EM+1e-4*I_D` under `GEN-REF64`. Diagnostics distinguish ridge from jitter and make no monotonicity/MAP claim. |
| `XD-GEN-CONTROL-001` | On ordinary and `M=0` inputs, supply jitter and ridge in turn as negative, NaN, `+/-Inf`, boolean, complex, length-one, vector, a negative source value that rounds to `-0.0` in f32, and a positive nonzero source value that rounds to `0.0` in f32 | f64/CPU and f32/CPU | Every invalid type/shape/domain or disappearing nonzero control fails actionably; it cannot be reported as a successful zero-control update. Static type/shape validity for both controls is established before either value-domain failure becomes device rollback status. Valid scalar zero and positive controls retain `XD-GEN-JITTER-001`/`RIDGE-001` semantics. |
| `XD-GEN-FAIL-001` | `N=16,K=3,D=5,M=3`, fixed literal orthogonal bases; construct effective `T` at in-domain `kappa=1e8` f64 / `1e4` f32 and out-of-domain `1e12` / `1e7`; include a rank-deficient `R` with singular `S` at zero jitter | f64/CPU and f32/CPU | In-domain cases succeed and all posterior/model covariances satisfy invariants. Out-of-domain cases either satisfy all invariants or return explicit numerical failure. Singular exact `T` fails without presenting parameters; documented positive jitter may make it succeed and then must satisfy `XD-GEN-JITTER-001`. |
| `XD-GEN-INFER-FAIL-001` | `K=3,D=3,M=2`; use a tagged device-result harness for one failed component pair, plus a real rank-deficient-`R`/singular-`S` row whose three `T_ik` fail, alongside a valid row; exercise component posterior and every convenience inference leaf | f64/CPU and f32/CPU | `failed_pairs` and global status identify exact failures. Pair sentinels and all-failed fallback match Contract §11.1. For any failed row, score/probability/posterior leaves are NaN and label is `-1`; the valid row is unchanged. Aggregate score/likelihood propagates NaN. An eager default wrapper raises actionably rather than presenting fallback values as success. |
| `XD-GEN-ROLLBACK-001` | Two observed-dimension groups where the first produces valid statistics and the second has a tagged factorization failure; also component-collapse harnesses in converged and fixed-step modes and a material-decrease harness in converged mode | f64/CPU and f32/CPU | Numerical failure/collapse rejects the entire candidate in both modes and terminates the logical fixed-step history. Material decrease rolls back only dynamically converged mode; a fixed-step finite decreasing candidate is accepted/recorded. Returned state/history/iteration and diagnostics follow Contract §12, with no partial group update or public NaN/Inf parameter. |

### Multiple initialization and restart selection

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-GEN-RESTART-001` | Ordered user-supplied candidates through grouped converged and fixed-step wrappers, then the same candidates/data through identity-valued projections with unit weights | f64/CPU and f32/CPU | Candidates are validated before fitting and replace only the common mixture state. Eligibility, strict maximum, and lowest-index exact ties follow the companion restart contract. General and identity wrappers select the same index and numerically equivalent endpoint under the identity-equivalence profile. |
| `XD-GEN-RESTART-002` | Every grouped terminal status, all-candidate failure, deterministic mask groups, common positive rescaling of observation weights, and an explicit prior-result parameter candidate | f64/CPU and selected f32/CPU sentinels | A failed candidate does not terminate the collection. All-failed diagnostic selection never relabels failure; grouping/restoration and common controls remain identical across candidates. Valid common weight scaling preserves the selected index except for a documented selected-dtype rounding tie. Warm starts reset all grouped trajectory and attempted-state diagnostics. The wrapper remains sequential and host-only. |
| `XD-GEN-RESTART-REC-001` | Stored Phase 4 grouped recovery workload with an ordered adverse duplicate-component candidate and the existing perturbed recovery candidate under identical groups, weights, and controls | f64/CPU | The adverse symmetric candidate is a valid lower-objective local solution and fails the independent recovery envelope; the selected eligible candidate has the greater independently recomputed informative-weight normalized objective and passes the existing permutation-invariant parameter/density/moment envelope. This is candidate-set-conditioned evidence, not a global-optimum claim. |

## 9. JAX transformation, sampling, and reference gates

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-GEN-JIT-001` | `XD-GEN-ANALYTIC-001`, `XD-GEN-R-001`, one fixed EM step, and the `M=0` leaf; eager and `jax.jit` | f64/CPU and f32/CPU | Eager/compiled shapes, statuses, and values agree under `GEN-REF*`. Jaxprs contain no host callback; a second same-shape call does not retrace. Validation and missing-mask grouping are explicitly outside this row. |
| `XD-GEN-VMAP-001` | Apply a single-observation score/posterior to all 17 fixed-`M` rows from `XD-GEN-R-001`; separately close over an explicit shared `R` | f64/CPU and f32/CPU | `jax.vmap` results match native batch results under `GEN-REF*`; there is no host conversion or implicit projection/noise broadcast. |
| `XD-GEN-GRAD-001` | Well-conditioned literal `N=4,K=2,D=3,M=2` fixture. Differentiate total likelihood with respect to `X`, `mu`, and dense `R`; then differentiate post-update likelihood through one fixed weighted EM step. Central difference `h=1e-5` | f64/CPU | Every advertised gradient is finite and agrees under `GEN-GRAD64`. No claim is made for eager adapters, masks, serialization, dynamic convergence, failures, labels, sampling, or unsupported parameter domains. |
| `XD-GEN-PRED-001` | `K=3,D=3,M=2`; construct exact equal component log joints under a dense nonidentity `R`, then repeat at `M=0` with equal stored weights | f64/CPU and f32/CPU | `predict_proba` returns equal finite probabilities within normalization tolerance and `predict` returns lowest index `0`. The behavior is identical in eager/JIT paths; numerical-failure labels remain governed by `XD-GEN-INFER-FAIL-001`. |
| `XD-GEN-SAMPLE-001` | `K=1,D=3,M=2`, literal `mu/V/R/S`, `n=150000`, key `jax.random.key(20260825)`, explicit shared projection/noise | f64/CPU and f32/CPU | Output shape is `(n,M)` and dtype is selected dtype. Each empirical mean coordinate is within six analytic standard errors plus `2e-3` f64 / `3e-3` f32 of `R@mu`, and covariance relative Frobenius error from `R@V@R.T+S` is `<=0.025` f64 / `<=0.035` f32. |
| `XD-GEN-SAMPLE-002` | `K=2,D=3,M=2`, `n=100000`; exercise all four shared/per-item projection/noise combinations and explicit isotropic, diagonal, correlated full, zero, and singular PSD noise modes; also `n=0` and `M=0` | f64/CPU and f32/CPU | Every mode has exact `(n,M)` shape and selected dtype, with no implicit singleton broadcast. For each homogeneous moment case, every empirical mean coordinate is within six analytic standard errors plus `2e-3` f64 / `3e-3` f32, and covariance relative Frobenius error is `<=0.03` f64 / `<=0.04` f32. Singular/zero noise succeeds by the documented square root. Zero-sized cases return exact shapes and still validate all static inputs. |
| `XD-GEN-PRNG-001` | Call general observed sampling without a key; then with invalid legacy/typed key shape/type, reused key, split key, boolean/negative/nonintegral/rank-positive/traced `n`, and per-item leading axis unequal to `n` | CPU | Missing/invalid key or `n` is an actionable signature/validation error, including at `n=0` and `M=0`. No random API constructs an internal default key. Same key/parameters/modes/version/backend/shape reproduce exactly; a split key changes at least one nonempty draw. |
| `XD-GEN-REF-BOVY-001` | **Pending fixture:** pinned `jobovy/extreme-deconvolution` commit `a8a5988d2ab3ceeecbe7f0c23e0554d8a3a4222c`; stored `N=19,K=3,D=4,M=2` per-item dense `R`, correlated full `S`, identical literal initialization, no data weights/fixed parameters/prior regularization, one EM update | f64/CPU | The independently built reference and independent NumPy oracle must agree on shared likelihood/update endpoints before archive creation. Production observed likelihood and one-step weights/means/covariances then agree with the stored Bovy endpoints under `rtol=5e-8, atol=5e-10`, max absolute log error `<=5e-8`. Responsibilities, gains, posterior moments, and statistics agree with the independent oracle under `GEN-REF64`. Source/license/build/fixture hashes are recorded. |

The Bovy reference row compares the supported intersection only. It does not
claim drop-in parity for Bovy's fixed parameters, prior regularization,
split-and-merge machinery, diagonal-error storage convention, or convergence
loop. Instrumenting upstream code to expose an endpoint must be a minimal,
recorded fixture-generation patch and must not be copied into the production
package.

## 10. Required rejection gates for deferred features

The features named here remain deferred, but each row marked `Gate / Pending`
is a required `0.2.0a1` rejection gate. A public signature that accepts and
ignores one of these inputs is nonconforming. The platform-evidence row records
an unadvertised target and is not an alpha blocker.

| Test ID | Input | Status | Required behavior under this contract |
|---|---|---|---|
| `XD-GEN-FIXPARAM-001` | fixed-weight/mean/covariance controls | Gate / Pending | explicit unsupported-feature error until a revised contract defines coupled M-step normalization |
| `XD-GEN-PRIOR-001` | conjugate-prior or MAP arguments | Gate / Pending | explicit unsupported-feature error; covariance ridge is not labeled a prior |
| `XD-GEN-EXTENSION-001` | conditioning, split-and-merge, selection-function, background-population, or minibatch/online controls | Gate / Pending | each unsupported extension fails explicitly under this version and is never accepted then ignored |
| `XD-GEN-RAGGED-001` | Python ragged/object arrays passed to a numerical kernel | Gate / Pending | validation error directing the user to dense groups or the mask adapter |
| `XD-GEN-NANMISS-001` | NaN-coded missing coordinates | Gate / Pending | validation error directing the user to the explicit boolean-mask adapter |
| `XD-GEN-HUGENOISE-001` | a flag requesting huge covariance as missingness | Gate / Pending | unsupported-feature error; exact coordinate removal uses grouping |
| `XD-GEN-PLAT-GPU-001` | general numerical suite on GPU | Deferred platform evidence | no GPU claim until exact environment/hardware and results are recorded; this row does not block a CPU-only `0.2.0a1` whose documentation says GPU unadvertised |

## 11. Evidence record template

When a row passes, append or link an immutable record containing:

```text
test_id:
contract_id: xdgmm-jax.general-xd
contract_version: 0.2.0-draft.1
matrix_id: xdgmm-jax.general-xd.matrix
matrix_version: 0.2.0-draft.1
package_commit:
test_file_and_node:
fixture_digest:
oracle_or_reference_digest:
python_version:
jax_version:
jaxlib_version:
x64_enabled:
platform/backend/device:
command:
result:
maximum_observed_error_or_residual:
date:
```

Changing a tolerance after observing a failure requires a written numerical
justification and a matrix-version change. CPU evidence does not support a GPU
claim, and identity-fast-path evidence does not replace arbitrary-projection
evidence.
