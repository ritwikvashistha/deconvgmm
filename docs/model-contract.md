# Identity-projection Extreme Deconvolution model contract

- Contract ID: `xdgmm-jax.identity-xd`
- Contract version: `0.1.0-draft.1`
- Status: normative design target for Phases 1 and 2
- Model scope: identity projection, \(R_i=I_D\)
- Last updated: 2026-08-29

This document defines the mathematical and behavioral contract that the
production implementation must satisfy before the identity-projection API is
released. It is intentionally more precise than the preserved prototype. The
prototype is **not** claimed to conform to this contract.

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. A
behavior not defined here is not a public guarantee. The measurable tests for
this version are listed in [the capability matrix](capability-matrix.md).

## 1. Versioning and scope

The contract version is independent of the Python package version:

- a patch change clarifies language without changing accepted results;
- a minor change adds backward-compatible operations or input forms; and
- a major change changes an equation, normalization, shape, or established
  result.

Every public fit result and serialized model MUST record the applicable contract
ID and version. Before the Phase 2 result schema exists, development test reports
MUST record them; absence of an interim result object is not treated as a public
API guarantee.

This version covers the observation model \(R_i=I_D\). General projections,
different latent and observed dimensions, exact missing-coordinate adapters,
and observation weights are deferred to a later contract revision. Implementing
one of those features without first revising this contract is nonconforming.

## 2. Symbols and canonical shapes

Let:

- \(N\geq1\) be the number of observations;
- \(K\geq1\) be the number of mixture components; and
- \(D\geq1\) be both the latent and observed dimension.

The canonical fitting representation is:

| Quantity | Symbol | Shape | Meaning |
|---|---:|---:|---|
| observations | \(X=(x_i)\) | `(N, D)` | noisy observed vectors |
| measurement covariance | \(S=(S_i)\) | `(N, D, D)` | known covariance for each observation |
| mixture weights | \(\alpha\) | `(K,)` | component probabilities |
| latent means | \(\mu\) | `(K, D)` | component means |
| latent covariances | \(V\) | `(K, D, D)` | component covariances |
| responsibilities | \(q\) | `(N, K)` | posterior component probabilities |
| conditional means | \(b\) | `(N, K, D)` | \(E[z_i\mid x_i,c_i=k]\) |
| conditional covariances | \(B\) | `(N, K, D, D)` | \(\operatorname{Cov}(z_i\mid x_i,c_i=k)\) |

The last axis is always a feature axis and the penultimate axis of a matrix is
its row axis. Core fitting functions MUST NOT infer a transpose from values.

Inference operations MAY accept either one observation or a batch. For a batch
shape `B` (which may be empty), canonical full-covariance input and output
shapes are:

```text
x                 B + (D,)
S                 B + (D, D)
score_samples      B
component_prob     B + (K,)
conditional_mean  B + (K, D)
conditional_cov   B + (K, D, D)
posterior_mean     B + (D,)
posterior_cov      B + (D, D)
```

This rule makes a single `x.shape == (D,)` a supported inference case without
silently inserting or retaining a length-one batch axis. Fitting still requires
the explicit `(N, D)` representation.

## 3. Parameter and data domains

At a validated public boundary:

1. All observations and parameters MUST be real and finite.
2. Every weight MUST be finite and strictly positive, and weights MUST sum to
   one within the validation tolerance. A constructor MAY explicitly normalize
   positive unnormalized masses, but the canonical parameter object MUST contain
   normalized weights.
3. Each \(V_k\) MUST be symmetric positive definite (SPD).
4. Each known \(S_i\) MUST be symmetric positive semidefinite (PSD). Exact zero
   measurement covariance is valid.
5. Consequently \(T_{ik}=V_k+S_i\) MUST be SPD. A failed factorization is a
   numerical/data error, not permission to propagate `NaN`.
6. Boolean and complex inputs MUST be rejected. Integer observations MAY be
   converted to the requested floating computation dtype; all posterior and fit
   outputs MUST remain floating. Covariance and parameter arrays MUST be
   floating at the canonical boundary.
7. Empty datasets, nonpositive dimensions, and a component count incompatible
   with the selected initializer MUST produce an actionable validation error.

Validation MAY replace a covariance \(A\) by
\((A+A^\mathsf{T})/2\) only when its asymmetry is within the documented dtype
tolerance. A materially asymmetric or indefinite input MUST be rejected rather
than silently repaired.

Mixture labels have no intrinsic order. Fitting and sampling provide no label
ordering guarantee unless a future API explicitly requests one.

## 4. Probability model and likelihood

For independent observations,

\[
\begin{aligned}
c_i &\sim \operatorname{Categorical}(\alpha_1,\ldots,\alpha_K),\\
z_i\mid c_i=k &\sim \mathcal N(\mu_k,V_k),\\
x_i &= z_i+\epsilon_i,\\
\epsilon_i &\sim \mathcal N(0,S_i),
\end{aligned}
\]

where \(z_i\), \(c_i\), and \(\epsilon_i\) are independent across observations
and measurement noise is independent of the latent component draw. Marginally,

\[
x_i\mid c_i=k\sim\mathcal N(\mu_k,T_{ik}),
\qquad T_{ik}=V_k+S_i.
\]

Define

\[
a_{ik}=\log\alpha_k+\log\mathcal N(x_i\mid\mu_k,T_{ik})
\]

and

\[
\ell_i=\operatorname{logsumexp}_{k=1}^K(a_{ik}).
\]

The unregularized observed-data log likelihood and its per-observation form are

\[
\ell(\theta;X,S)=\sum_{i=1}^N\ell_i,
\qquad \bar\ell=\frac{1}{N}\ell.
\]

The objective contains no covariance prior, entropy term, or implicit penalty.
Known measurement covariances are conditioned on and are not learned.

## 5. Density evaluation and responsibilities

For \(r_{ik}=x_i-\mu_k\) and a Cholesky factor
\(L_{ik}L_{ik}^{\mathsf T}=T_{ik}\),

\[
\log\mathcal N(x_i\mid\mu_k,T_{ik})=
-\frac{1}{2}\left[
D\log(2\pi)+2\sum_d\log (L_{ik})_{dd}
+\|L_{ik}^{-1}r_{ik}\|_2^2
\right].
\]

The production kernel MUST use factorization and triangular/linear solves; it
MUST NOT form \(T_{ik}^{-1}\) explicitly. A factor SHOULD be reused for the log
determinant, Mahalanobis distance, and posterior solves.

Responsibilities are

\[
\log q_{ik}=a_{ik}-\operatorname{logsumexp}_{j=1}^K(a_{ij}),
\qquad q_{ik}=\exp(\log q_{ik}).
\]

Normalization MUST take place in log space. Adding a constant to a probability-
space denominator is forbidden. For every valid observation, returned
responsibilities MUST be finite, nonnegative, and sum to one within the relevant
dtype tolerance. Individual nonwinning responsibilities may round to exactly
zero in finite precision; an entire row may not.

## 6. Conditional and mixture posterior moments

Let the Kalman gain for observation \(i\) and component \(k\) be

\[
K_{ik}=V_kT_{ik}^{-1}.
\]

The component-conditional latent posterior is Gaussian:

\[
z_i\mid x_i,c_i=k\sim\mathcal N(b_{ik},B_{ik}),
\]

with

\[
b_{ik}=\mu_k+K_{ik}(x_i-\mu_k)
\]

and the mathematically equivalent covariance forms

\[
B_{ik}=V_k-V_kT_{ik}^{-1}V_k
\]

and

\[
B_{ik}=(I-K_{ik})V_k(I-K_{ik})^{\mathsf T}
       +K_{ik}S_iK_{ik}^{\mathsf T}.
\]

The Joseph form in the second expression is the preferred implementation where
it improves positive-semidefinite behavior. No explicit inverse is required:
the action of \(T_{ik}^{-1}\) MUST be computed with the reused factor. Returned
covariances MUST be symmetrized as
\((B+B^{\mathsf T})/2\) before validation.

If factorization jitter is active, every \(T_{ik}\) in these equations is
replaced by \(T_{ik}^{\mathrm{eff}}\), and the Joseph form uses
\(S_i+\delta_{ik}I\). Mixing the exact and effective matrices would not describe
a valid conditional Gaussian and is forbidden.

The posterior mean marginalized over components is

\[
m_i=E[z_i\mid x_i]=\sum_k q_{ik}b_{ik},
\]

and its covariance is

\[
C_i=\sum_k q_{ik}
\left[B_{ik}+(b_{ik}-m_i)(b_{ik}-m_i)^{\mathsf T}\right].
\]

A public operation named `posterior` MUST make clear whether it returns the
component-conditional quantities \((q,b,B)\), marginalized moments \((m,C)\),
or both. `posterior_mean` means \(m\), never just the most likely component's
conditional mean.

## 7. One exact EM update

All E-step quantities in an iteration are evaluated using the same old
parameters \(\theta^{(t-1)}\). Define the component sufficient statistics

\[
\begin{aligned}
n_k &= \sum_i q_{ik},\\
h_k &= \sum_i q_{ik}b_{ik},\\
G_k &= \sum_i q_{ik}\left(B_{ik}+b_{ik}b_{ik}^{\mathsf T}\right).
\end{aligned}
\]

For a noncollapsed component, the M-step is

\[
\begin{aligned}
\alpha_k^{(t)} &= \frac{n_k}{\sum_j n_j},\\
\mu_k^{(t)} &= \frac{h_k}{n_k},\\
V_k^{(t)} &= \frac{G_k}{n_k}
             -\mu_k^{(t)}\mu_k^{(t)\mathsf T}\\
&=\frac{1}{n_k}\sum_i q_{ik}
\left[B_{ik}+(b_{ik}-\mu_k^{(t)})
(b_{ik}-\mu_k^{(t)})^{\mathsf T}\right].
\end{aligned}
\]

The second covariance expression is preferred as a two-pass or equivalent
stable accumulation. The implementation MUST symmetrize each updated covariance.
The division by \(\sum_j n_j\), rather than a floating representation of \(N\),
ensures that returned weights use the same accumulated mass and remain
normalized; the expressions are identical in exact arithmetic.

Chunked and unchunked implementations MUST accumulate quantities equivalent to
\((n,h,G)\) and MUST implement the same update. Chunking may change rounding but
not the objective or estimator.

### 7.1 Collapsed components

A component is collapsed when its effective mass is nonfinite or nonpositive, or
when its proposed parameters are nonfinite or fail the covariance domain. An API
MAY expose an additional positive minimum-mass threshold, but it MUST be stated
in effective-count or weight-fraction units.

The initial Phase 1 default is `on_collapse="error"`:

- the numerical kernel returns a failure status identifying the component and
  iteration before performing an invalid division;
- a host-facing wrapper raises an actionable exception or returns an explicitly
  unsuccessful fit result; and
- partial parameters MUST NOT be presented as a successful fit.

Dynamic removal MUST NOT change \(K\) inside compiled code. A future deterministic
in-place reinitialization policy MAY be added, but its selection rule, PRNG use,
and effect on monotonicity must first be added to this contract and matrix.

## 8. Converged and fixed-step fitting

Initialization produces \(\theta^{(0)}\). Its objective \(\bar\ell^{(0)}\) is
evaluated before any EM update. The next E/M step proposes a candidate
\(\theta^{(t)}\) and candidate objective. Define

\[
g_t=\frac{\bar\ell^{(t)}-\bar\ell^{(t-1)}}
          {\max(1,|\bar\ell^{(t-1)}|)}.
\]

For user-supplied finite, nonnegative `tol` and `decrease_tol`:

- a nonfinite objective is a numerical failure;
- if \(g_t<-\texttt{decrease_tol}\), the status is `objective_decreased`, not
  convergence;
- if \(-\texttt{decrease_tol}\leq g_t\leq\texttt{tol}\), the returned state is
  \(\theta^{(t)}\) with `converged=True`; and
- if \(g_t>\texttt{tol}\), fitting continues.

The source values for both stopping tolerances are validated before conversion
to the selected computation dtype. Each converted value MUST remain finite.
Exact source zero is valid, but a positive nonzero source value that becomes
zero in the selected dtype MUST fail validation rather than silently changing
the stopping rule.

An update is accepted only after these checks. On a material objective decrease,
nonfinite candidate, or component-collapse failure, the returned parameters,
objective, history, and `n_iter` describe the last valid state
\(\theta^{(t-1)}\). Diagnostics also record the attempted iteration and candidate
objective when finite. A failed candidate is never appended to accepted history
or presented as a successful fitted state.

Thus a tiny negative change attributable to documented rounding can satisfy the
stopping rule, while a material decrease cannot. Tests of theoretical EM
monotonicity use `decrease_tol` values from the capability matrix.

The returned fields have one meaning:

- `n_iter` is the number of EM updates actually represented by returned
  parameters;
- `objective` is the objective of those exact returned parameters;
- if history is requested, it begins with \(\bar\ell^{(0)}\), ends with the
  returned objective, and has length `n_iter + 1`; and
- no update may modify the state after convergence has been recorded.

With `max_iter=0`, converged fitting returns the validated initial parameters,
their objective, `n_iter=0`, `converged=False`, and status `max_iter`. Reaching a
positive `max_iter` without satisfying the criterion has the same status and is
not reported as convergence.

A separate fixed-step operation executes exactly `n_steps` EM updates in the
absence of a numerical failure, for JIT/autodiff composition. It reports
`mode="fixed_steps"`; convergence is not applicable. Its parameters and
objective correspond to step `n_steps`. On nonfinite arithmetic or component
collapse it stops or signals and returns only the last valid state under the
same rules as converged fitting.
Differentiability of fixed-step code does not imply differentiability through a
data-dependent converged fit.

Exact EM guarantees a nondecreasing unregularized likelihood only when the
updates are exact and there is no covariance ridge, iteration-varying factor
jitter, component reinitialization, or other constrained/postprocessed step. A
fixed factor-jitter value defines one consistent effective likelihood for the
entire fit and retains the ordinary EM argument for that effective model.

## 9. Noise covariance representations

The canonical numerical kernels accept full \(S\) arrays only. The initial
public input layer MUST provide explicit isotropic, diagonal, and full adapters
with exactly these shapes:

| Representation | Batch shape | Definition |
|---|---:|---|
| isotropic variance | `B` | \(S= sI_D\) |
| diagonal variances | `B + (D,)` | \(S=\operatorname{diag}(s_1,\ldots,s_D)\) |
| full covariance | `B + (D,D)` | \(S\) as supplied after validation |

All supplied values are variances/covariances, not standard deviations. Scalar
variance expands onto the diagonal only; it MUST NOT broadcast into every matrix
entry. The identity input layer MUST expose distinct
`fit_isotropic_noise` and `inference_isotropic_noise` entry points; an
operation-agnostic isotropic constructor is not part of the public contract.
The fitting entry point accepts exactly `variances.shape == (N,)` with `N >= 1`
and returns `(N,D,D)`. The inference entry point interprets the complete supplied
variance shape literally as batch shape `B` and returns `B + (D,D)`. Thus a
scalar is the single-observation form, `B=(2,1)` is a valid inference batch, and
`(N,1)` remains invalid at the fitting entry point. Neither entry point squeezes,
inserts a missing batch axis, or broadcasts a singleton axis.

Isotropic adapter values MAY use an integer or floating real source dtype and
MUST produce the selected floating computation dtype. Boolean, complex,
nonnumeric, nonfinite, and negative values MUST be rejected before they can be
silently repaired by conversion. A nonzero source variance that becomes zero in
the selected dtype MUST raise a precision error. A representable selected-dtype
subnormal MUST remain nonzero in the constructed adapter output; eager adapter
construction MUST NOT silently flush it through a device multiplication. Every
accepted value is written only to the corresponding matrix diagonal;
off-diagonal entries are exact zero.

Shape inference MUST NOT reinterpret an ambiguous rank-two array as a shared
covariance. Each noncanonical form uses an explicit covariance-kind argument or
distinct constructor.

`S=None` MAY be documented as exact zero measurement covariance. If offered, it
must produce the same values as an explicit zero full covariance. Shared noise
covariances MAY be supported only through explicit broadcasting at an adapter
boundary; a rank-two `(D,D)` input to a fit is never inferred to be shared,
including when `N == D`. The canonical kernel receives the expanded or otherwise
unambiguous representation.

## 10. Scoring, prediction, and posterior semantics

For supplied \(S\):

- `score_samples` returns \((\ell_i)\), the noisy observed-data log density;
- `log_likelihood` returns \(\sum_i\ell_i\);
- `score` returns the arithmetic mean \(N^{-1}\sum_i\ell_i\);
- `predict_proba` returns \(q\); and
- `predict` returns `argmax(q, axis=-1)`, with the lowest component index winning
  an exact tie.

With exact zero \(S\), scoring is the latent mixture density and must agree with
an ordinary full-covariance Gaussian mixture using the same parameters.

Posterior/denoising operations return floating arrays in the computation dtype.
They MUST NOT allocate results from an integer observation template. All scoring
and posterior operations use the same density, normalization, and effective
factorization policy as fitting.

For an observation whose component factorization fails, the convenience
inference leaves report the failure through sentinels rather than presenting a
successful value: `score_samples`, `posterior`, and `posterior_mean` return
`NaN` for the affected row, and `predict` returns integer label `-1`. The
device operation `posterior_components` remains the authoritative failure status
(mirroring the general contract §11.1). This deliberate sentinel is consistent
with §3's rule that a *kernel* MUST NOT present a failed factorization as a
successful result: the sentinel is an explicit, detectable failure signal, not a
silently corrupted success. A researcher-facing eager wrapper MAY instead raise
an actionable exception by default.

## 11. Sampling and PRNG semantics

Sampling has two distinct meanings:

1. `sample_latent(params, key, n)` accepts integer \(n\geq0\), draws
   \(c_j\sim\operatorname{Categorical}(\alpha)\), then
   \(z_j\sim\mathcal N(\mu_{c_j},V_{c_j})\), and returns shape `(n,D)`.
2. `sample_observed(params, key, S)` accepts canonical `S.shape == (n,D,D)`,
   infers \(n\), and draws \(x_j=z_j+\epsilon_j\),
   \(\epsilon_j\sim\mathcal N(0,S_j)\), returning shape `(n,D)`.

A generic method called only `sample` MUST document which of these it implements;
the preferred default for an XD model is latent sampling. Component labels MAY
be returned only as an explicit option.

Every pure-JAX sampling call MUST require an explicit PRNG key. It MUST NOT create
an internal constant key. Calling with the same key, parameters, software
version, backend, and shape is reproducible; callers are responsible for
splitting keys before independent draws. Bitwise equality across JAX versions or
different accelerator backends is not guaranteed unless separately tested.
Sampling from PSD \(S\), including exactly singular or zero matrices, uses a
documented deterministic matrix square root \(A\) satisfying
\(AA^{\mathsf T}=S\) within the covariance tolerance; it MUST NOT require \(S\)
to be Cholesky-positive-definite. No bitwise coupling between `sample_latent`
and `sample_observed` is guaranteed beyond their stated key semantics.

## 12. Dtype and numerical policy

The computation dtype is an explicit property of a fit or inference call:

- float64 on CPU is the normative reference path for scientific parity;
- float32 is a qualified path and becomes supported only for matrix rows that
  pass their float32 tolerances;
- float16 and bfloat16 are not supported for fitting in this contract; and
- a requested float64 calculation when JAX x64 is disabled MUST fail clearly
  rather than silently downcast.

Mixed real inputs are explicitly converted to the selected computation dtype.
Core results use that dtype. Public validation occurs outside compiled kernels
where necessary; compiled kernels may assume they received a validated canonical
representation but still MUST surface a numerical failure status instead of
silently returning nonfinite parameters.

Computed covariance matrices are symmetrized before return. Their admissible
rounding-level asymmetry and minimum-eigenvalue residuals are defined separately
for float32 and float64 in the capability matrix. Meeting a float64 test does not
establish float32 or GPU support.

## 13. Numerical jitter versus statistical regularization

These mechanisms MUST be separate configuration fields and diagnostics.

### 13.1 Factorization jitter

Factorization jitter \(\delta I\) is a numerical device added to the covariance
being factored. It is not a prior and MUST NOT be silently stored in \(V_k\).
Exact reference mode uses \(\delta=0\). The initial conforming option is a finite
scalar \(\delta\geq0\) fixed across all observations, components, and iterations
of one fit.

If a nonzero jitter is used, it MUST be visible in fit diagnostics. Likelihood,
responsibilities, posterior solves, and posterior covariance MUST all use the
same effective matrix, equivalent here to

\[
T_{ik}^{\mathrm{eff}}=V_k+S_i+\delta_{ik}I.
\]

The reported value is then identified as an effective/stabilized objective, not
silently labeled the exact objective from Section 4. Applying jitter to a solve
but not to the corresponding log determinant is forbidden.

Iteration-varying adaptive jitter is outside this contract. A future adaptive
schedule must be deterministic, record every value used, and either hold the
effective matrices fixed across objective comparisons or disable ordinary
convergence/monotonicity claims; objectives from changing effective models are
not directly comparable.

### 13.2 Covariance ridge or update regularization

The initial conforming update option is a finite scalar covariance ridge
\(\lambda\geq0\), applied after the exact M-step as
\(V_k\leftarrow V_k^{\mathrm{EM}}+\lambda I\). It persists in the model parameters
and changes the estimator. It is not numerical factorization jitter. Its value
and application rule MUST be user-visible and serialized. A nonzero ridge
generally removes the ordinary EM monotonicity guarantee; the reported
likelihood is evaluated at the returned parameters and is not a penalized/MAP
objective.

No MAP prior or penalized objective is defined in this contract. A future prior
must state its probability model, hyperparameters, and reported objective before
being called statistical regularization.

## 14. Observation weights are deferred

Version `0.1.0-draft.1` does not accept `sample_weight`; a supplied weight array
MUST raise an unsupported-feature error rather than be ignored.

The planned, non-normative extension is: finite \(w_i\geq0\), not all zero,
weighted objective \(\sum_iw_i\ell_i\), unchanged per-observation
responsibilities, statistics \(n_k=\sum_iw_iq_{ik}\), and normalization by
\(\sum_iw_i\). Convergence would use the objective divided by \(\sum_iw_i\), so
scaling all weights by a common positive constant would not change the EM update
or stopping decision. This proposal becomes binding only in the general-model
contract revision.

## 15. Companion multiple-initialization selection contract

The temporary host restart layer is governed by the separate companion
contract ID `xdgmm-jax.restart-selection`, version `0.1.0-draft.1`.  Adding
this companion operation does not change the identity single-fit contract ID
or version above.  A restart wrapper result is not a single-fit result and the
current draft serialization format does not support it.

Version 1 of the companion contract accepts only a nonempty, explicitly ordered
sequence of user-supplied initial parameter states.  The eager constructor MUST
canonicalize and validate the complete sequence before any fit starts.  Every
candidate MUST have the same positive `K` and `D` and the same selected
computation dtype.  Each candidate independently satisfies Sections 2, 3, and
12, including finite strictly positive normalized weights and model
covariances that pass the selected-dtype positive-definite factorization policy.
Duplicate candidates and label permutations are valid and remain distinct;
candidate order is semantic.  No candidate is generated, deduplicated,
reordered, or relabeled, and no PRNG is used in this version.

For a restart collection, every single fit uses the same observations,
measurement covariances, computation dtype, factor jitter, covariance ridge,
mode, stopping tolerances, and iteration limit.  Candidates run sequentially
in ascending index and every candidate receives its complete logical budget;
one failed candidate does not prevent later candidates from running.  The
wrapper MUST validate the candidate collection and common static controls
before starting the first candidate.  Candidate-specific controls and early
termination after the first successful candidate are outside this version.

For dynamically converged fits, only terminal statuses `converged` and
`max_iter` are eligible for selection.  For fixed-step fits, only
`fixed_steps_complete` is eligible.  `objective_decreased`, numerical failure,
and component collapse are ineligible even when rollback leaves a finite
accepted state.  Among eligible candidates, the wrapper selects the greatest
finite final normalized objective from Section 8.  Comparison uses the selected
dtype without a tolerance, updates the selected index only for a strict greater
value, and therefore resolves exact ties in favor of the lowest candidate
index.  It does not prefer `converged` over a higher-objective `max_iter`
result.  Mixture labels are not aligned for this comparison because the scalar
objective is permutation invariant.

If no candidate is eligible, the wrapper returns an explicitly unsuccessful
selection rather than relabeling a failed fit.  Its deterministic diagnostic
representative is the ineligible result with the greatest finite objective
marked valid, with exact ties again resolved by lowest index.  If no result has
a valid finite objective, candidate zero is the representative.  The selected
single-fit result retains its original terminal status and every field
unchanged.  Its `initial_parameters` MUST be bitwise equal to the selected
canonical candidate.

The wrapper retains the complete ordered canonical candidate arrays, bounded
per-candidate objective/status summaries, the selected full single-fit result,
the selected index, the selection-success state, and the companion contract and
selection-rule identities.  It does not retain every nonselected trajectory or
final parameter state.  Contradictory single-fit result invariants are an
internal error, not a candidate-selection outcome.

Warm starting is explicit rather than mutable: a caller supplies a prior
result's valid parameters as one of the new candidates.  The new call
revalidates that state, recomputes its initial objective, and resets history,
iteration counts, attempted-state diagnostics, controls, tolerances, and
budget.  Nothing is inherited implicitly from the prior fit.

Restart construction, sequential orchestration, result collection, and
selection are host operations with no JIT, `vmap`, `pmap`, or autodiff claim.
The existing fixed-step numerical kernel retains its own transformation scope,
but discrete best-result selection is outside it.  This operation establishes
only the best eligible endpoint among the supplied candidates; it is not a
global-optimum or arbitrary-initialization robustness guarantee.

## 16. Conformance

An implementation conforms to this contract only when:

1. every alpha-gate row in the capability matrix passes on its required dtype
   and backend;
2. returned public metadata names this exact contract version;
3. unsupported general-projection and weighted operations fail explicitly; and
4. documentation does not advertise deferred rows as implemented.

A plausible recovered fit, a single passing smoke test, or agreement with the
preserved prototype is not conformance evidence.

## References

The model and EM construction follow Bovy, Hogg, and Roweis (2011), “Extreme
deconvolution: Inferring complete distribution functions from noisy,
heterogeneous and incomplete observations,” *Annals of Applied Statistics*,
[arXiv:0905.2979](https://arxiv.org/abs/0905.2979). Software provenance and
license obligations are tracked separately from this mathematical citation.
