# Initial capability and quality matrix

- Matrix ID: `xdgmm-jax.identity-xd.matrix`
- Matrix version: `0.1.0-draft.1`
- Applies to: [`xdgmm-jax.identity-xd` contract
  `0.1.0-draft.1`](model-contract.md)
- Last updated: 2026-08-28

This matrix turns the initial identity-projection contract into named,
reproducible acceptance tests. It describes intended evidence; it does **not**
claim that the preserved prototype passes. A capability is advertised only after
its required rows pass on the stated dtype and backend and the evidence is linked
from this file or the roadmap.

## 1. Status and release meanings

| Mark | Meaning |
|---|---|
| **Gate** | required for the `0.1.0a1` identity-projection alpha |
| **Qualified** | tested but advertised with the limitations in this matrix |
| **Deferred** | not part of `0.1.0a1`; accepting it silently is a failure |
| **Baseline** | characterizes the preserved prototype; not conformance evidence |

All Gate rows remain **Pending: specified but not yet verified against a
production implementation**. Every row in §5 is a Gate row;
every row in §6 is a Gate row for the Phase 2 API required by `0.1.0a1`; and §7
rows are Deferred sentinels. Baseline behavior is recorded separately in [the
prototype report](baseline-report.md). Requiring float32 executions at the gate
means that the bounded domain below is tested; the public label remains
Qualified because no broader conditioning or backend claim follows from it.
Failure of any required float32 execution still blocks `0.1.0a1`; “Qualified”
describes the limited public support envelope, not an optional release test.

Each named row has this individual release class:

- **Gate — Phase 1:** `XD-IP-SHAPE-001`, `XD-IP-SHAPE-002`,
  `XD-IP-VAL-001`, `XD-IP-NOISE-001`, `XD-IP-DTYPE-001`,
  `XD-IP-DTYPE-002`, `XD-IP-LL-001`, `XD-IP-RESP-001`,
  `XD-IP-RESP-002`, `XD-IP-POST-001`, `XD-IP-POST-002`,
  `XD-IP-ZERO-001`, `XD-IP-EM-001`, `XD-IP-EM-002`,
  `XD-IP-MIXWEIGHT-001`, `XD-IP-COV-001`, `XD-IP-JITTER-001`,
  `XD-IP-RIDGE-001`, `XD-IP-COLLAPSE-001`, `XD-IP-CONV-001`,
  `XD-IP-CONV-002`, `XD-IP-CONV-003`, `XD-IP-CONV-004`,
  `XD-IP-FIXED-001`, and `XD-IP-REF-001`.
- **Gate — Phase 2:** `XD-IP-JIT-001`, `XD-IP-VMAP-001`,
  `XD-IP-GRAD-001`, `XD-IP-SAMPLE-001`, `XD-IP-SAMPLE-002`,
  `XD-IP-PRNG-001`, `XD-IP-PRED-001`, `XD-IP-META-001`,
  `XD-IP-RESTART-001`, `XD-IP-RESTART-002`, and
  `XD-IP-RESTART-REC-001`.
- **Deferred:** `XD-GEN-R-001`, `XD-GEN-WEIGHT-001`,
  `XD-GEN-MISSING-001`, and `XD-PLAT-GPU-001`.

The required-execution cell on each row is part of its gate. If it names both
float64 and float32, failure on either dtype fails that row.

As of the 2026-08-28 living-record refresh, local tests exercise
workloads corresponding to the Phase 1 and Phase 2 rows, including conditioning,
pinned astroML endpoint plus independent-oracle composite parity,
callback/retrace/vmap/gradient behavior, prediction, explicit-key latent and
observed sampling (including singular PSD noise), strict contract metadata, and
the operation-specific isotropic adapters required by `XD-IP-NOISE-001`.
Temporary CPU tests also exercise deterministic parameter/identity-fit artifact
round trips and basin-conditioned complete-fit statistical recovery with
permutation-invariant parameters and independent latent-holdout density metrics.
The authoritative repository-wide local CPU suite passes 1,131 tests in 618.84
seconds (0:10:18) on Python 3.10.11 and JAX/JAXlib 0.6.2. This changes
no row's formal Pending status: the code is outside a public namespace on an
unborn local branch, there is no package commit or released artifact, and hosted
supported-version/backend evidence has not run. The astroML fixture's direct
versus oracle-derived scope is recorded in
[`reference-fixtures/astroml-1.0.2.post1.md`](reference-fixtures/astroml-1.0.2.post1.md).

The same local tree now also contains a separate temporary fixed-`M` general
leaf, an explicit tagged general boundary, deterministic mask grouping, grouped
fixed/converged control, fixed/grouped inference, and fixed-`M` observed
sampling. That work is governed by
`general-model-contract.md` and `general-capability-matrix.md`; it does not turn
the deferred general inputs below into accepted identity-contract arguments.
The chunked identity path has local reduced-endpoint parity, keeps the original
`N`-row inputs as scan constants, gathers `C` rows per step, and stages no global
`P`-row input copy. Its regenerated checked record uses schema
`0.1.0-draft.3` and reports `global_padded_input_staging: false`. Its refreshed
raw SHA-256 is
`c5a5328ec7bb4bd37b84a457f43ea060093acd691f2b9293f5b910d1375ce081`
and it retains `performance_claim: none`. The structural workspace record is not
an allocator trace, total-memory ceiling, or performance claim. A representable
float32 final covariance may still trigger conservative
rollback when its raw centered numerator overflows. Neither addition changes
any identity row's formal status.

## 2. Initial support envelope

| Capability | `0.1.0a1` target | Evidence required |
|---|---|---|
| Identity projection \(R_i=I_D\) | Gate | all `XD-IP-*` Gate rows |
| Full latent covariances \(V_k\) | Gate | `XD-IP-VAL-001`, `XD-IP-EM-001`, `XD-IP-COV-001` |
| Per-observation full \(S_i\) | Gate | `XD-IP-NOISE-001`, `XD-IP-POST-001`, `XD-IP-REF-001` |
| Isotropic and diagonal \(S_i\) adapters | Gate | `XD-IP-NOISE-001` |
| Zero measurement error | Gate | `XD-IP-ZERO-001` |
| float64 CPU scientific-reference path | Gate | every Gate row bearing `f64/CPU` |
| float32 CPU | Qualified alpha support | every Gate row bearing `f32/CPU`; limitations documented |
| JIT-compiled pure kernels | Gate after Phase 2 API exists | `XD-IP-JIT-001` |
| Single-observation inference | Gate | `XD-IP-SHAPE-001` |
| Posterior denoising | Gate | `XD-IP-POST-001`, `XD-IP-DTYPE-001` |
| Latent sampling | Gate after Phase 2 API exists | `XD-IP-SAMPLE-001` |
| Dynamically converged fit | Gate | `XD-IP-CONV-001`, `XD-IP-CONV-002` |
| Fixed-step fit | Gate | `XD-IP-FIXED-001` |
| Ordered user-supplied multiple initializations | Gate after Phase 2 API exists | `XD-IP-RESTART-001`, `XD-IP-RESTART-002` |
| General projection matrices | Deferred to `0.2.0a1` | later general-model matrix |
| Exact missing-coordinate workflow | Deferred to `0.2.0a1` | later general-model matrix |
| Observation/sample weights | Deferred to `0.2.0a1` | `XD-GEN-WEIGHT-001` in a revised contract |
| GPU claims | Deferred until controlled hardware exists | `XD-PLAT-GPU-001` plus all advertised numerical rows |
| TPU, multi-GPU, multi-host | Unsupported/unadvertised | future matrix revision |

The provisional problem-size domain for correctness tests is
\(1\leq N\leq4096\), \(1\leq K\leq8\), and \(1\leq D\leq32\). This is not a
scalability ceiling. Large-\(N\) memory and performance claims require the Phase 5
benchmark matrix and are not inferred from these tests or the temporary chunked
one-step development record.

## 3. Reproducible fixture policy

Tests MUST avoid undocumented random state:

1. Small analytic fixtures are literal arrays checked into `tests/fixtures/`.
2. Generated fixtures use NumPy `Generator(PCG64(20260825))` once and are then
   stored as literal `.npz` data with SHA-256 and generator metadata. Tests read
   the stored arrays rather than regenerating them across NumPy versions.
3. JAX sampling tests use an explicit typed or legacy key whose representation is
   recorded with the tested JAX version.
4. Independent expected values are computed in NumPy/SciPy float64 without
   importing the production numerical kernel. Expected values may not be
   generated by copying production output.
5. Reference-software fixtures record package version or commit, build options,
   precision, platform, and license/provenance.

All comparisons synchronize JAX work before timing or reporting completion.
Performance is never inferred from a correctness test.

## 4. Numerical tolerance profiles

For `allclose(actual, reference)`, unless a row overrides them:

| Profile | `rtol` | `atol` | Intended use |
|---|---:|---:|---|
| `REF64` | `5e-10` | `5e-12` | well-conditioned float64 values and one-step parity |
| `LOG64` | `5e-10` | `5e-10` | float64 log densities/objectives |
| `REF32` | `1e-4` | `1e-5` | well-conditioned float32 values |
| `LOG32` | `2e-4` | `2e-5` | float32 log densities/objectives |

The ordinary reference fixtures MUST have
\(\kappa_2(V_k+S_i)\leq10^4\) and absolute component log densities below
`1e3`. Near-singular and deliberate tail tests use their row-specific bounds,
not unconstrained `REF*`/`LOG*` relative comparisons. In addition to `allclose`,
ordinary `LOG64` comparisons require maximum absolute error `<=1e-8`, and
ordinary `LOG32` comparisons require maximum absolute error `<=5e-3`.

The initial successful conditioning domain for every effective covariance
\(T_{ik}\) is \(\kappa_2(T_{ik})\leq10^8\) in float64 and
\(\kappa_2(T_{ik})\leq10^4\) in float32, with finite representable entries and
positive scale. Matrices beyond those bounds may succeed, but are not supported
by this matrix; if they do not, the implementation must fail explicitly rather
than return a successful nonfinite state.

Responsibility normalization uses

```text
float64: max_i abs(sum_k(q[i,k]) - 1) <= 5e-13
float32: max_i abs(sum_k(q[i,k]) - 1) <= 2e-5
```

For a returned covariance \(A\), define

\[
s(A)=\max(1,\|A\|_2),\qquad
e_{\rm sym}(A)=\frac{\|A-A^\mathsf T\|_\infty}{s(A)}.
\]

The covariance invariant profile is:

| Dtype | symmetry \(e_{\rm sym}\) | PSD residual for posterior covariances | PD requirement for model covariances |
|---|---:|---:|---:|
| float64 | `<= 2e-13` | `lambda_min >= -2e-11 * s(A)` | Cholesky succeeds without hidden jitter |
| float32 | `<= 2e-6` | `lambda_min >= -5e-5 * s(A)` | Cholesky succeeds under the documented policy |

An invariant tolerance permits only rounding-scale residuals; it does not permit
returning a materially indefinite covariance and calling it PSD.

Input covariance symmetry uses the same dtype-specific symmetry threshold.
Input \(S\) is numerically PSD only when its smallest eigenvalue meets the
corresponding PSD-residual bound; a more negative value is rejected. An input
inside the symmetry threshold may be symmetrized, but no material eigenvalue
clipping is permitted without an explicit adapter option and diagnostic. Input
mixture-weight sums use the responsibility-normalization bound for their dtype.

## 5. Phase 1 numerical-core gates

### Shapes, validation, and noise construction

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-IP-SHAPE-001` | `K=3, D=2`; one input `(D,)` and batch `(5,D)` with corresponding full `S` | f64/CPU and f32/CPU | All inference outputs have exactly the shapes in Contract §2; the one-item values equal the first batch item under `REF64`/`REF32`. |
| `XD-IP-SHAPE-002` | Fit shapes `N=7,K=3,D=2`; deliberately transpose each non-square input in turn | f64/CPU | Valid shapes pass. Every transposed or incompatible shape fails before compilation with a message naming the received and expected shape; no implicit transpose/broadcast occurs. |
| `XD-IP-VAL-001` | Each of: `NaN`/`Inf` data, zero or negative mixture weight, mixture-weight sum error `1e-2`, asymmetric covariance, negative-eigenvalue `S`, non-PD `V`, complex/bool inputs, `N=0`, and a sample-without-replacement initializer with `K>N` | f64/CPU | Every invalid public input raises the documented exception or returns an explicit invalid-input status. No case returns model parameters. Symmetry/PSD perturbations inside the stated validation tolerance are handled exactly as documented. |
| `XD-IP-NOISE-001` | `N=4,D=3`; `fit_isotropic_noise` on `[0,0.1,0.5,2]` and rejected scalar/empty/`(N,1)`/multi-axis inputs; `inference_isotropic_noise` on scalar, `(3,)`, `B=(2,1)`, and `B=(2,1,3)` inputs; selected-dtype minimum-subnormal values; diagonal variances with unequal coordinates; correlated full covariances; ambiguous `N=D=3` rank-two full input; boolean/complex/nonnumeric/nonfinite/negative and positive-underflow isotropic sentinels | f64/CPU and f32/CPU | Fit isotropic input is exactly nonempty `(N,)`; inference interprets the complete variance shape literally as `B`, including `B=(2,1)`. Outputs equal `s[...,None,None] * eye(D)` exactly for representable values, including a selected-dtype subnormal diagonal, and every off-diagonal is zero. An inference-only `(N,1,D,D)` result fails fit validation rather than broadcasting. Invalid source values and nonzero-to-zero selected-dtype conversion fail actionably. Diagonal and full adapters equal literal expected arrays. A raw `(3,3)` is not inferred as shared full covariance for a fit; only the explicit shared adapter may broadcast it. |
| `XD-IP-DTYPE-001` | Integer `x=[0,1]`, floating full `S`, `K=1,D=2`; posterior mean is nonintegral | f64/CPU and f32/CPU | Output dtype is selected computation dtype and values match the analytic posterior under `REF64`/`REF32`; no truncation occurs. |
| `XD-IP-DTYPE-002` | Request float64 in a subprocess with `jax_enable_x64=False` | CPU | Call fails with an actionable precision error before fitting; it does not return float32 values labeled float64. |

Temporary local red-to-green evidence for `XD-IP-NOISE-001` on 2026-08-28:
the identity validation file passes 119 tests, including the explicit adapter
cases above, and a warning-strict validation/inference/functional/JAX selection
passes 169 tests on Python 3.10.11 with JAX/JAXlib 0.6.2 CPU. This does not change
the row's formal Pending status: the implementation remains in the temporary
namespace without an immutable public package revision or hosted supported-
environment evidence.

### Likelihood, responsibilities, and posterior

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-IP-LL-001` | Stored well-conditioned fixture `N=11,K=3,D=2` with correlated `V` and heterogeneous correlated `S` | f64/CPU and f32/CPU | Per-component log densities, `score_samples`, total likelihood, and mean score match an independent SciPy/NumPy Cholesky calculation under `LOG64`/`LOG32`. `sum(score_samples)` equals total and `mean(score_samples)` equals score to the same profile. |
| `XD-IP-RESP-001` | Same `N=11,K=3,D=2` fixture | f64/CPU and f32/CPU | `q` matches independent log-softmax under `REF64`/`REF32`; all values are finite and nonnegative and rows meet the dtype normalization bound. |
| `XD-IP-RESP-002` | Tail fixture `N=4,K=3,D=32`: `alpha=(0.2,0.3,0.5)`, component means constant by component in `{-2,0,2}`, diagonal `V` scales `{0.5,1,2}`, diagonal `S` in `[0.01,0.2]`, and observations with coordinates near `+1000`, `-1000`, zero, and alternating signs | f64/CPU and f32/CPU | All component log densities are finite, although direct density exponentiation underflows for at least one row. Returned `q` remains finite/nonnegative, no row is all zero, rows meet normalization bounds, and values match independent log-softmax under `REF64`/`REF32` (absolute tolerance controls near zero). |
| `XD-IP-POST-001` | Analytic `K=1,D=2`: `mu=(-0.2,0.7)`, `V=[[2,.4],[.4,1]]`, `S=[[.5,.1],[.1,.3]]`, `x=(1.1,-.5)` | f64/CPU and f32/CPU | `q=[1]`; conditional and marginalized means/covariances match independent linear-solve formulas under `REF64`/`REF32`. Covariances satisfy the invariant profile. Joseph and subtractive NumPy formulas agree under `REF64`. |
| `XD-IP-POST-002` | One observation, `K=3,D=2`, unequal weights/means and correlated `V`/`S` | f64/CPU and f32/CPU | Marginalized posterior mean and covariance match an independent law-of-total-covariance calculation under `REF64`/`REF32`; they differ from every single component's moments and the covariance satisfies the invariant profile. |
| `XD-IP-ZERO-001` | Stored `N=13,K=3,D=3` fixture with `S=0` | f64/CPU and f32/CPU | Log densities, responsibilities, and score equal an independent ordinary full-covariance GMM under `REF64`/`REF32`; posterior has `b=x` and `B=0` within the same absolute tolerances. |

### EM updates and numerical state

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-IP-EM-001` | Stored well-conditioned fixture `N=7,K=3,D=2`, positive initial weights, heterogeneous correlated errors; exactly one update with zero jitter/ridge | f64/CPU and f32/CPU | `q`, `b`, `B`, sufficient statistics `(n,h,G)`, and updated `(alpha,mu,V)` match an independent two-pass NumPy implementation under `REF64`/`REF32`. Weights meet the responsibility normalization bound and covariances meet the invariant profile. |
| `XD-IP-EM-002` | Stored synthetic fixture `N=128,K=3,D=3`; 15 exact updates from fixed valid parameters, zero jitter/ridge, no collapse | f64/CPU and f32/CPU | Every objective and parameter is finite. For normalized objectives, each increment is at least `-1e-10 * max(1,abs(previous))` in float64 and `-2e-5 * max(1,abs(previous))` in float32. The final objective is independently recomputed from the returned parameters. |
| `XD-IP-MIXWEIGHT-001` | The fixtures from `XD-IP-EM-001` and `XD-IP-RESP-002`, plus a one-component far-tail case | f64/CPU and f32/CPU | Every public mixture weight is finite and strictly positive and its sum meets the responsibility normalization bound. In the one-component case, every responsibility and the updated weight equal one within the dtype normalization bound, even when the Gaussian density would underflow in probability space. |
| `XD-IP-COV-001` | `N=32,K=3,D=5`, zero jitter/ridge. Use fixed orthogonal DCT matrix `Q`, `T=Q@diag(geomspace(1,kappa,5))@Q.T`, `V=0.5*I`, `S=T-V`, with `kappa=1e8` (f64), `1e4` (f32), and out-of-domain `1e12`/`1e7` | f64/CPU and f32/CPU | All in-domain cases succeed and posterior/updated covariances satisfy the invariant profile. Each out-of-domain case may either produce a valid result satisfying the profile or an explicit numerical-failure status; it may not return a successful `NaN` or indefinite model covariance. Literal `Q` and arrays are stored with the fixture. |
| `XD-IP-JITTER-001` | `XD-IP-EM-001` fixture with fixed scalar `factor_jitter=1e-6`, then with zero jitter | f64/CPU | With nonzero jitter, likelihood, responsibilities, posterior moments, and one update match an independent calculation using `S + 1e-6*I` under `REF64`/`LOG64`; diagnostics identify the effective objective and jitter. Returned `V` does not receive a direct `1e-6*I` addition. Zero jitter reproduces `XD-IP-EM-001`. |
| `XD-IP-RIDGE-001` | `XD-IP-EM-001` fixture with `covariance_ridge=1e-4` and zero factor jitter | f64/CPU | Means/weights equal the exact update, and each returned covariance equals `V_EM + 1e-4*I` under `REF64`. Diagnostics record the ridge separately from jitter and make no monotonicity or penalized-objective claim. Setting the ridge to zero reproduces `XD-IP-EM-001`. |
| `XD-IP-COLLAPSE-001` | `N=8,K=2,D=2`; one component placed sufficiently far away that its accumulated responsibility is numerically zero, default `on_collapse="error"` | f64/CPU and f32/CPU | Collapse is detected before division. Kernel status names the component and iteration; host API fails actionably. No partially updated fit is marked successful and no public result contains `NaN`/`Inf`. |

### Convergence and execution semantics

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-IP-CONV-001` | Stored `N=40,K=2,D=2` fixture, `max_iter=5`, `tol=1e6`, `decrease_tol=1e-10`, zero jitter/ridge | f64/CPU | Fit stops after the first accepted update: `n_iter=1`, `converged=True`, history length `2`. Returned parameters equal a direct one-step update under `REF64`, and returned objective equals an independent evaluation of those parameters under `LOG64`. They must differ from a forced five-step result on this fixture. |
| `XD-IP-CONV-002` | Previous objective `-10`, `tol=1e-6`, `decrease_tol=1e-10`; current objectives `-10-1.1e-9`, `-10-1e-9`, `-9.99999`, and `-9.999989`, plus nonfinite objective and infinite/negative tolerances. For each stopping field in turn, select float32 with finite float64 sources `1e300`, `1e-300`, and exact zero. | f64/CPU and f32/CPU | The normalized changes are `-1.1e-10`, `-1e-10`, `1e-6`, and `1.1e-6`: respectively `objective_decreased`, converged, converged, and continue. Nonfinite objective yields numerical failure. Infinite or negative tolerances fail validation. A finite source that overflows or a positive source that becomes zero in the selected dtype fails before fitting; intentional exact zero remains valid. |
| `XD-IP-CONV-003` | Valid initialized `N=10,K=2,D=2` fixture with `max_iter=0` | f64/CPU and f32/CPU | Returned parameters are exactly the validated initialization, `n_iter=0`, `converged=False`, status is `max_iter`, history contains only the independently verified initial objective. |
| `XD-IP-CONV-004` | Fit-loop state harness with tagged valid state `theta0`, candidate `theta1`, objectives `-10` then `-11`, `tol=1e-6`, `decrease_tol=1e-10`; also negative/nonintegral `max_iter` and `n_steps` | f64/CPU | Material decrease returns `theta0`, initial objective/history, `n_iter=0`, unsuccessful status, and attempted objective `-11` only in diagnostics. Invalid iteration counts fail validation. |
| `XD-IP-FIXED-001` | Same fixture as `XD-IP-EM-002`, `n_steps` in `{0,1,5}` | f64/CPU and f32/CPU | Exactly the requested number of updates is represented. Step 0 equals initialization; step 1 equals an independent one-step calculation using the `XD-IP-EM-001` equations; history length is `n_steps+1`; mode is `fixed_steps` and no convergence claim is made. |

### Independent parity

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-IP-REF-001` | A pinned astroML identity-projection fixture `N=25,K=3,D=2`, identical literal parameters/data/errors, exactly one E/M update, zero-equivalent jitter/ridge on both sides; reference commit/version and license recorded | f64/CPU | Observed log likelihood, responsibilities, conditional moments, sufficient statistics, and updated parameters agree under `rtol=2e-8, atol=2e-10`. The reference values are stored, not recomputed during ordinary CI. |

`XD-IP-REF-001` does not replace the independent NumPy tests: implementation
similarity or common bugs are possible. Final-fit comparisons are Phase 4 work
and require label alignment and compatible stopping policies.

## 6. Phase 2 API and JAX gates

These are alpha Gate rows. They remain specified/pending until their Phase 2
public operations exist; `0.1.0a1` cannot pass its release gate without them.

| Test ID | Workload | Required execution | Acceptance criterion |
|---|---|---|---|
| `XD-IP-JIT-001` | `XD-IP-LL-001`, `XD-IP-POST-001`, and one fixed EM step, eager and `jax.jit` | f64/CPU and f32/CPU | Eager and compiled outputs have identical shapes/statuses and agree under `REF64`/`REF32`. The jaxpr contains no host callback; a second call with unchanged shapes does not retrace. |
| `XD-IP-VMAP-001` | Apply single-observation score/posterior to the 11 observations from `XD-IP-LL-001` using `jax.vmap` | f64/CPU and f32/CPU | Results match the native batch operation under `REF64`/`REF32`; there is no host conversion. |
| `XD-IP-GRAD-001` | Well-conditioned `N=4,K=2,D=2` literal fixture. Differentiate scalar total likelihood with respect to `X` and `mu`; then differentiate the scalar post-update total likelihood `ell(theta_1(X,mu0); X,S)` through one fixed EM step. Central-difference step `h=1e-5` | f64/CPU | Each advertised `jax.grad` is finite and agrees with the specified central difference under `rtol=2e-5, atol=2e-6`. No gradient claim is made for dynamically converged fitting or validation/adapters. |
| `XD-IP-SAMPLE-001` | `alpha=(.35,.65)`, `mu=((-1,.5),(2,-.75))`, `V=(((1,.2),(.2,.5)),((.6,-.1),(-.1,1.2)))`, `D=2`, `n=200000`, key `jax.random.key(20260825)` and a split key | f64/CPU and f32/CPU | Reusing the same key gives exactly equal arrays; the split key changes at least one draw. With analytic mixture mean `m` and covariance `C`, each mean error is at most `6*sqrt(C[d,d]/n) + dtype_atol`, and relative covariance Frobenius error is `<=0.02`. Output is floating and has shape `(n,D)`. |
| `XD-IP-SAMPLE-002` | `K=1,D=2`, `mu=(.2,-.4)`, `V=((1,.25),(.25,.7))`, `n=100000`, key `jax.random.key(20260825)`, and repeated `S`: zero, `((.4,.15),(.15,.3))`, and singular `((.2,.2),(.2,.2))` | f64/CPU | For zero `S`, empirical observed moments satisfy the latent analytic bounds from `XD-IP-SAMPLE-001`; bitwise equality to a separate latent call is not required. For nonzero `S`, empirical mean is within six standard errors and covariance relative Frobenius error from `V+S` is `<=0.025`. Singular `S` succeeds through the documented PSD square-root policy. |
| `XD-IP-PRNG-001` | Call every random public functional API without a key, then with explicitly reused/split keys | CPU | Missing key is a signature/type error. No random API internally creates `PRNGKey(0)`. Reused/split behavior is documented and matches `XD-IP-SAMPLE-001`. |
| `XD-IP-PRED-001` | One `K=3,D=2` observation with independently constructed equal component log-joints | f64/CPU and f32/CPU | `predict_proba` returns three equal probabilities within the normalization tolerance and `predict` returns component index `0`. |
| `XD-IP-META-001` | Successful converged and fixed-step fit results | f64/CPU | Both results record contract ID `xdgmm-jax.identity-xd` and version `0.1.0-draft.1`; serialized metadata round-trips without changing them. |
| `XD-IP-RESTART-001` | Ordered canonical candidates including duplicates, label permutations, `R=1`, a higher-objective `max_iter` result, and exact objective ties; both converged and fixed-step wrappers | f64/CPU and f32/CPU | The complete nonempty sequence is validated before fitting; candidate arrays are retained in order. Eligibility follows Companion Contract §15, the greatest eligible normalized objective is selected, and exact ties choose the lowest index. The selected single-fit result is unchanged and its initial parameters are bitwise equal to the selected candidate. |
| `XD-IP-RESTART-002` | Harness every terminal single-fit status, finite and invalid rollback objectives, all-candidate failure, explicit prior-result parameters as a new candidate, and invalid mixed `K`/`D`/dtype/covariance candidate collections | f64/CPU and selected f32/CPU sentinels | Ineligible failures do not stop later candidates. All-failed selection is unsuccessful and chooses the greatest finite valid diagnostic objective, or index zero when none exists, without relabeling status. Invalid collections start no fit. Explicit warm starts recompute objective and reset all trajectory/control state. The wrapper is sequential host-only and is rejected by the current single-fit serializers. |
| `XD-IP-RESTART-REC-001` | Stored Phase 4 identity recovery workload with an ordered adverse duplicate-component candidate and the existing perturbed recovery candidate under identical controls | f64/CPU | The duplicate-component candidate remains a valid lower-objective symmetric local solution and fails the independent recovery envelope; the selected eligible candidate has the greater independently recomputed objective and passes the existing permutation-invariant parameter/density/moment envelope. This is candidate-set-conditioned evidence, not a global-optimum claim. |

Sampling moment rows are marked slow and need not run in every small pull-request
job; they MUST run in the release gate and scheduled numerical suite.

## 7. Deferred capability sentinels

The following tests prevent roadmap features from being accidentally implied by
a permissive identity-contract signature. They remain required rejection
sentinels even while a separately versioned temporary general leaf is developed:

| Test ID | Input | Current required behavior |
|---|---|---|
| `XD-GEN-R-001` | Any nonidentity projection matrix `R` | Explicit unsupported-feature error under this contract; no silent ignoring of `R`. |
| `XD-GEN-WEIGHT-001` | Any `sample_weight` array | Explicit unsupported-feature error; weights are not silently discarded. |
| `XD-GEN-MISSING-001` | NaN-coded or ragged missing coordinates | Validation error with a pointer to the future missing-data API; no implicit imputation. |
| `XD-PLAT-GPU-001` | Run the advertised numerical suite on a pinned GPU/JAX environment | No GPU support or performance claim until the exact rows, hardware, precision, and results are recorded. |

## 8. Evidence record template

When a row passes, append or link an immutable record containing:

```text
test_id:
contract_version:
matrix_version:
package_commit:
fixture_digest:
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
justification and a matrix-version change. A backend is supported only by rows
that actually ran there; CPU success is not indirect evidence for GPU success.
