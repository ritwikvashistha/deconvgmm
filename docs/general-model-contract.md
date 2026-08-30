# General-projection Extreme Deconvolution model contract

- Contract ID: `xdgmm-jax.general-xd`
- Contract version: `0.2.0-draft.1`
- Status: normative design target for Phase 3; no implementation conformance is
  claimed
- Model scope: fixed latent dimension, general linear projections, grouped
  observed dimensions, and observation weights
- Last updated: 2026-08-29

This document defines the mathematical and behavioral target for the full
linear-observation Extreme Deconvolution model. It extends the
[identity-projection contract](model-contract.md), but it does not claim that
the preserved prototype or the temporary identity kernel implements this
contract. Measurable evidence requirements are specified in the
[general capability matrix](general-capability-matrix.md).

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. A
behavior not defined here is not a public guarantee.

## 1. Versioning, scope, and relationship to the identity contract

The contract version is independent of the Python package version. Patch,
minor, and major contract changes have the meanings defined by the identity
contract: a change to an equation, normalization, accepted shape, missingness
rule, or established result requires a major contract-version change.

Every public fit result and every serialized general-projection model MUST
record exactly:

```text
contract_id      = "xdgmm-jax.general-xd"
contract_version = "0.2.0-draft.1"
```

An unknown contract ID or version MUST fail closed at a deserialization
boundary. Loading an identity-contract record into a general-contract API, or
the reverse, requires an explicit, tested migration; relabeling metadata is not
a migration. The numerical-array transport schema remains a separate Phase 2
design task, but no eventual transport may omit these two fields.

This contract covers:

- arbitrary known linear projections from a fixed latent dimension $D$ to a
  fixed observed dimension $M$ within one compiled group;
- different observed dimensions across eagerly prepared groups;
- explicit per-item, shared, and identity projection representations;
- full, diagonal, and isotropic known measurement covariances;
- exact missing-coordinate grouping through a boolean-mask host adapter;
- nonnegative observation weights; and
- latent posterior moments and exact weighted EM updates.

Fixed-component controls, conjugate priors, conditioning utilities,
split-and-merge updates, selection functions, background populations, and
minibatch/online EM are outside this version. They MUST NOT be inferred from a
permissive argument or silently ignored.

## 2. Probability model

Let $c_i$ be a component label and $z_i\in\mathbb R^D$ the latent value.
Observation $i$ has $M_i\geq0$ measured coordinates:

\[
\begin{aligned}
c_i &\sim \operatorname{Categorical}(\alpha_1,\ldots,\alpha_K),\\
z_i\mid c_i=k &\sim \mathcal N(\mu_k,V_k),\\
x_i &= R_i z_i+\epsilon_i,\\
\epsilon_i &\sim \mathcal N(0,S_i).
\end{aligned}
\]

Here $R_i\in\mathbb R^{M_i\times D}$ and
$S_i\in\mathbb R^{M_i\times M_i}$ are known and conditioned on; they are not
learned. The mixture parameters are
$\theta=(\alpha_k,\mu_k,V_k)_{k=1}^K$. Marginally,

\[
x_i\mid c_i=k\sim\mathcal N(R_i\mu_k,T_{ik}),
\qquad
T_{ik}=R_iV_kR_i^\mathsf T+S_i.
\]

These are equations (1), (6), and (7) of Bovy, Hogg, and Roweis (2011), with
symbols renamed consistently with the identity contract.

## 3. Canonical shapes and compiled groups

Let $K\geq1$, $D\geq1$, and let one dense group contain $N\geq1$
observations sharing one observed dimension $M\geq0$. The canonical fitting
shapes for a per-item projection and full noise covariance are:

| Quantity | Symbol | Shape |
|---|---:|---:|
| observations | $X$ | `(N, M)` |
| per-item projections | $R$ | `(N, M, D)` |
| per-item full noise | $S$ | `(N, M, M)` |
| observation weights | $w$ | `(N,)` |
| mixture weights | $\alpha$ | `(K,)` |
| latent means | $\mu$ | `(K, D)` |
| latent covariances | $V$ | `(K, D, D)` |
| log component density |  | `(N, K)` |
| responsibilities | $q$ | `(N, K)` |
| gains (internal; optionally exposed) | $G$ | `(N, K, D, M)` |
| conditional means | $b$ | `(N, K, D)` |
| conditional covariances | $B$ | `(N, K, D, D)` |

The latent dimension $D$, component count $K$, and observed dimension $M$
are static for one compiled kernel invocation. Different $M_i$ values are
represented by separate dense groups and separate compiled specializations;
ragged arrays do not enter a numerical kernel. Group sizes may differ. A future
chunked executor may pad or split the leading observation axis, but it MUST
preserve this model and the sufficient statistics in Section 9.

For inference over an arbitrary batch shape `B`, the per-item canonical forms
are:

```text
x                   B + (M,)
R                   B + (M, D)
S                   B + (M, M)
score_samples        B
component_prob       B + (K,)
gain                 B + (K, D, M)
conditional_mean     B + (K, D)
conditional_cov      B + (K, D, D)
posterior_mean       B + (D,)
posterior_cov        B + (D, D)
```

For a single observation, `B` is empty and the canonical shapes are `(M,)`,
`(M,D)`, and `(M,M)`. Fitting always requires an explicit leading `N` axis.
No numerical kernel infers a transpose or broadcasts a missing batch axis.

### 3.1 Draft canonical functional boundary

The temporary fixed-$M$ core and the future public core use one parameter
PyTree with weights, latent means, and latent covariances. The contract-facing
canonical signatures are:

```text
posterior_components_general(params, x, R, S, *, factor_jitter=0)
sufficient_statistics_general(
    params, x, R, S, sample_weight, *, factor_jitter=0
)
one_em_step_general(
    params, x, R, S, sample_weight,
    *, factor_jitter=0, covariance_ridge=0
)
```

Here `R` and `S` are canonical per-item arrays with exactly the batch axes of
`x`; these leaf kernels perform no implicit broadcasting. `sample_weight` is a
canonical `(N,)` array. A host convenience boundary implements
`sample_weight=None` by constructing ones before calling a fitting kernel.
Explicit shared/identity projection and shared/noise adapters in Section 4 may
dispatch to specialized compiled kernels or produce canonical arrays, but their
choice is never inferred from a raw lower-rank batch argument.

The posterior result contains component log densities/joints, scores,
responsibilities, conditional latent means/covariances, and device-resident
pair/global failure status. A gain MAY be exposed as a diagnostic, but it is not
a required public posterior field. Sufficient-statistic and one-step results
contain device arrays and status; they do not raise from inside compiled code.
The names remain draft until a public namespace is selected, but changing their
mathematics, argument roles, or shape rules requires a contract revision.

## 4. Projection and noise representations

### 4.1 Projection modes are explicit

The public boundary MUST distinguish these projection modes by a tagged wrapper,
a distinct constructor, or a distinct function entry point:

| Mode | Supplied shape for batch `B` | Meaning |
|---|---:|---|
| per item | `B + (M,D)` | one $R_i$ for every item |
| shared | `(M,D)` | one declared matrix used for every item |
| identity | dimension/tag only, with `M == D` | $R_i=I_D$ for every item |

A raw rank-two `(M,D)` array accompanying batched observations MUST NOT be
silently interpreted as shared. It is canonical only for a single observation.
Likewise, a leading length-one projection batch is not broadcast. The user must
choose the shared adapter explicitly. A shared implementation MAY avoid
materializing repeated matrices.

Projection matrices need not be square, orthogonal, coordinate selectors, or
full rank. They MUST be real and finite. Regardless of rank, every effective
observed covariance used by a positive-weight informative row MUST satisfy the
factorization domain in Section 6.

An identity fast-path adapter is explicit. An implementation MUST NOT inspect
traced projection values on the host to decide that arbitrary input happens to
equal the identity.

### 4.2 Noise modes are explicit

For batch shape `B`, the adapters are:

| Mode | Supplied shape | Definition |
|---|---:|---|
| per-item isotropic variance | `B` | $S_i=s_i I_M$ |
| per-item diagonal variances | `B + (M,)` | $S_i=\operatorname{diag}(s_i)$ |
| per-item full covariance | `B + (M,M)` | $S_i$ as supplied |
| shared isotropic variance | scalar through explicit shared adapter | one $sI_M$ |
| shared diagonal variances | `(M,)` through explicit shared adapter | one diagonal $S$ |
| shared full covariance | `(M,M)` through explicit shared adapter | one full $S$ |

All values are variances or covariances, not standard deviations. Raw lower-rank
arrays are never inferred to be shared for a batch, including when dimensions
happen to be equal. Isotropic construction writes the scalar only on the
diagonal. A shared representation MAY remain shared inside a compiled kernel;
canonical does not imply physically expanding it to `(N,M,M)`.

## 5. Eager missing-coordinate adapter

Exact missingness uses an explicit boolean mask and grouped dense selection at
an eager host boundary. NaN values, infinite covariance, and merely large noise
are not missingness encodings.

For a common potential observed dimension $P\geq0$, the mask adapter accepts:

```text
x_full             (N, P)
observed_mask       (N, P), boolean
R_full              per item (N, P, D) or explicit shared (P, D)
S_full              per item (N, P, P) or an explicit shared noise form
sample_weight       (N,), optional
```

All numeric input, including values in masked-out positions, MUST be finite;
the mask is the sole missingness indicator. Full covariances supplied to this
adapter MUST satisfy their declared covariance domain before slicing. A user who
only possesses already-reduced covariance blocks MAY construct dense groups
directly instead of inventing a full covariance.

For each distinct mask $m\in\{0,1\}^P$, let
$C_m\in\{0,1\}^{M\times P}$ contain, in ascending source-coordinate order,
the rows of $I_P$ selected by the true mask entries. The adapter constructs

\[
x_{g,i}=C_mx_{{\rm full},i},\qquad
R_{g,i}=C_mR_{{\rm full},i},\qquad
S_{g,i}=C_mS_{{\rm full},i}C_m^\mathsf T
\]

row by row, and operationally:

1. selects observation indices carrying exactly that mask, preserving their
   original relative order;
2. selects observed coordinate indices in ascending source-coordinate order;
3. gathers `x_full` on those coordinates;
4. gathers the corresponding rows of $R_{\rm full}$;
5. gathers the corresponding principal submatrix of $S_{\rm full}$; and
6. carries observation weights and original indices alongside the group.

The result for a mask with $M$ true entries has shapes `(N_g,M)`,
`(N_g,M,D)`, and `(N_g,M,M)`. Groups MUST have a deterministic order defined by
the adapter. The initial required order is lexicographic order of the boolean
tuples `(mask[0], ..., mask[P-1])`, with `False < True`.
Per-row inference results are restored to original input order.

Grouping inspects mask values, may synchronize a device mask, and changes the
number or shapes of compiled calls when the mask pattern changes. The adapter is
therefore deliberately outside JIT, `vmap`, and autodiff guarantees. The dense
numeric kernels invoked after grouping remain eligible for those guarantees.

The adapter MUST reject:

- nonboolean masks;
- mask, observation, projection, covariance, or weight shape mismatches;
- NaN or infinite values, including in masked-out positions;
- using a large finite/infinite variance as a request to remove a coordinate.

This version does not define a second public schema for user-supplied grouped
collections with restoration indices. Users with already reduced data call the
fixed-$M$ dense boundary separately for each group and own any output
reordering. A future grouped-collection schema must define index coverage and
ordering before it is accepted by a public adapter.

## 6. Parameter, data, dtype, and covariance domains

At a validated public boundary:

1. $\alpha$, $\mu$, $V$, $X$, $R$, $S$, and weights MUST be real
   and finite after conversion to the selected computation dtype.
2. Mixture weights MUST be strictly positive and normalized within the selected
   dtype tolerance. Latent covariances $V_k$ MUST be symmetric positive
   definite (SPD).
3. Measurement covariances $S_i$ MUST be symmetric positive semidefinite
   (PSD); exact zero and singular PSD matrices are valid.
4. For $M>0$, each covariance actually used in density evaluation MUST be
   SPD after the documented factor-jitter policy. Failure to factor it is an
   explicit numerical failure.
5. Boolean and complex numeric arrays are rejected. Integer observations and
   integer observation weights MAY be converted to the selected float dtype.
   Parameters, projections, and covariance inputs MUST be floating.
6. The supported computation dtypes are float64 and qualified float32. A
   float64 request with JAX x64 disabled MUST fail before computation; float16
   and bfloat16 fitting are unsupported.
7. Every array result uses the selected computation dtype. Integer observations
   can never cause an integer posterior buffer.
8. Observation weights are checked both before and after conversion. A strictly
   positive source weight that becomes zero, or a finite source weight that
   becomes nonfinite, in the selected dtype MUST raise an actionable precision
   error rather than silently removing or infinitely weighting a row.

Covariance symmetrization and PSD/PD residual tolerances are those in the
general capability matrix. Validation may symmetrize only a rounding-level
asymmetry; it may not repair a materially asymmetric or indefinite matrix. A
projection may make $R_iV_kR_i^\mathsf T+S_i$ singular even when $V_k$ is
SPD and $S_i$ is PSD. Such a row is outside exact mode unless documented
jitter makes the effective matrix SPD.

## 7. Observed likelihood and responsibilities

For $M_i>0$, define the observed residual and effective covariance

\[
r_{ik}=x_i-R_i\mu_k,
\qquad
T_{ik}^{\rm eff}=R_iV_kR_i^\mathsf T+S_i+\delta I_{M_i},
\]

where the default factor jitter is $\delta=0$. With
$L_{ik}L_{ik}^\mathsf T=T_{ik}^{\rm eff}$,

\[
\log\mathcal N(x_i\mid R_i\mu_k,T_{ik}^{\rm eff})=
-\frac12\left[M_i\log(2\pi)+2\sum_m\log(L_{ik})_{mm}
+\lVert L_{ik}^{-1}r_{ik}\rVert_2^2\right].
\]

Define

\[
a_{ik}=\log\alpha_k+
\log\mathcal N(x_i\mid R_i\mu_k,T_{ik}^{\rm eff}),
\qquad
\ell_i=\operatorname{logsumexp}_k(a_{ik}),
\]

and normalize responsibilities in log space:

\[
\log q_{ik}=a_{ik}-\ell_i,
\qquad q_{ik}=\exp(\log q_{ik}).
\]

The implementation MUST use factorizations and solves, MUST NOT form an inverse,
and SHOULD reuse one factor for density and posterior calculations. Every valid
responsibility row is finite, nonnegative, and normalized within the selected
dtype tolerance.

## 8. Gain and latent posterior moments

For component $k$, define the gain

\[
G_{ik}=V_kR_i^\mathsf T(T_{ik}^{\rm eff})^{-1}.
\]

The inverse notation states the mathematics only; the implementation computes
its action with the reused factor. The conditional latent posterior is

\[
z_i\mid x_i,c_i=k\sim\mathcal N(b_{ik},B_{ik}),
\]

with

\[
b_{ik}=\mu_k+G_{ik}(x_i-R_i\mu_k)
\]

and the subtractive form

\[
B_{ik}=V_k-V_kR_i^\mathsf T(T_{ik}^{\rm eff})^{-1}R_iV_k.
\]

The preferred stable covariance is the generalized Joseph form

\[
B_{ik}=(I_D-G_{ik}R_i)V_k(I_D-G_{ik}R_i)^\mathsf T
       +G_{ik}(S_i+\delta I_{M_i})G_{ik}^\mathsf T.
\]

Using $T^{\rm eff}$ in the gain while using unjittered $S_i$ in the Joseph
form is forbidden: those quantities would not describe the same conditional
Gaussian. Returned covariances are symmetrized before validation.

The component-marginalized moments are

\[
m_i=\sum_kq_{ik}b_{ik},
\qquad
C_i=\sum_kq_{ik}\left[B_{ik}+
(b_{ik}-m_i)(b_{ik}-m_i)^\mathsf T\right].
\]

`posterior_components` denotes $(q,b,B)$; `posterior` denotes or contains the
marginalized $(m,C)$; and `posterior_mean` means $m$, never the most likely
component's conditional mean.

## 9. Observation weights, sufficient statistics, and one EM update

An optional observation weight has the domain

\[
w_i\in[0,\infty),\quad w_i\text{ finite}.
\]

Its canonical fitting shape is exactly `(N,)`; scalars and length-one arrays are
not broadcast. `sample_weight=None` means one for every row. Weights do not
change responsibilities or any single-row posterior. For an inference batch
shape `B`, weighted `log_likelihood` and `score` require
`sample_weight.shape == B` exactly and reduce over every batch axis. For a
single observation, `B` is empty and the corresponding weight is a scalar with
shape `()`. No scalar, singleton, or lower-rank weight is broadcast to a
nonempty batch. A grouped host operation accepts one `(N,)` weight array in
original row order and gathers it with the groups.

For informative rows,

\[
\mathcal I=\{i:M_i>0\},
\qquad
W=\sum_{i\in\mathcal I}w_i,
\]

and a fit requires finite $W>0$ in the selected computation dtype. A mixed-
dimension weighted scoring reduction likewise requires finite $W>0$; the sole
exception is the structurally all-$M=0$ score in Section 10. The weighted
observed-data log likelihood and the
normalized fit objective are

\[
\ell_w=\sum_{i\in\mathcal I}w_i\ell_i,
\qquad
\bar\ell_w=\frac{\ell_w}{W}.
\]

The sufficient statistics are

\[
\begin{aligned}
n_k &= \sum_{i\in\mathcal I} w_iq_{ik},\\
h_k &= \sum_{i\in\mathcal I} w_iq_{ik}b_{ik},\\
Q_k &= \sum_{i\in\mathcal I} w_iq_{ik}
      (B_{ik}+b_{ik}b_{ik}^\mathsf T).
\end{aligned}
\]

Statistics from all fixed-$M$ groups are summed before one global M-step. For
each noncollapsed component,

\[
\begin{aligned}
\alpha_k^{\rm new}&=\frac{n_k}{\sum_jn_j},\\
\mu_k^{\rm new}&=\frac{h_k}{n_k},\\
V_k^{\rm new}&=\frac1{n_k}\sum_{i\in\mathcal I}w_iq_{ik}
\left[B_{ik}+(b_{ik}-\mu_k^{\rm new})
(b_{ik}-\mu_k^{\rm new})^\mathsf T\right].
\end{aligned}
\]

The centered covariance expression, or a numerically equivalent stable
two-pass accumulation, is preferred over subtracting two large raw moments.
Every updated covariance is symmetrized.

Multiplying all weights by one positive constant MUST leave responsibilities,
updated parameters, the normalized objective, convergence decisions, and
collapse decisions unchanged within the matrix tolerance. Integer weights MUST
agree with explicit row replication within the corresponding accumulation
tolerance. Zero-weight rows have no statistical effect, but their public inputs
still undergo ordinary shape, dtype, and finiteness validation. Scale-invariance
claims apply only while the original and scaled arrays are representable without
overflow or positive-to-zero underflow in the selected dtype.

The accumulated informative weight, weighted objective, component masses, and
moment statistics MUST all be finite. A common positive rescaling MAY be used
internally for stable accumulation, provided raw statistics explicitly promised
by an API retain their documented scale. If no such scaling can keep the
contracted outputs finite and nonzero where required, fitting fails with an
explicit precision/numerical status rather than changing the estimator
silently.

## 10. Fully missing observations: $M=0$

The zero-dimensional Gaussian uses the standard empty-product convention:
its determinant and density are one and its log density is zero. For an
observation with shapes

```text
x       (0,)
R       (0,D)
S       (0,0)
```

the exact result is

\[
\log\mathcal N=0,\quad
a_{ik}=\log\alpha_k,\quad
\ell_i=0,\quad
q_{ik}=\alpha_k,\quad
b_{ik}=\mu_k,\quad
B_{ik}=V_k.
\]

This is an explicit finite-precision branch, not an evaluation of the generic
log-sum-exp path. Once the boundary has accepted the stored mixture weights as
normalized within tolerance, the branch copies those stored weights to $q_i$
and returns an exact zero score. It does not renormalize them again. Thus these
equalities remain exact even when the floating sum of accepted stored weights
differs from one by a permitted rounding residual.

The marginalized posterior is therefore exactly the mixture prior: its mean is
$\sum_k\alpha_k\mu_k$ and its covariance follows the ordinary law of total
covariance. No zero-by-zero Cholesky or solve is attempted, and factor jitter
has no effect.

Fully missing rows are valid for `score_samples`, prediction, and posterior
operations, but they are deliberately **excluded from fitting statistics and
from $W$**. They carry no observed-likelihood information; excluding them
prevents their count or weights from damping an otherwise identical EM
trajectory or changing its stopping iteration. Adding or removing any number of
fully missing rows MUST leave a fit unchanged. A dataset with no positive-weight
row having $M>0$ is valid for posterior inference but MUST fail fitting with an
actionable `no_informative_weight` validation error.

For grouped scoring, `score_samples` returns zero for every $M=0$ row and
`log_likelihood` receives zero contribution. `score` uses the informative
weight denominator $W$; for an all-$M=0$ scoring collection it returns zero
by definition. This convention is explicit because log densities from different
observed dimensions use different reference measures and should not be compared
as if dimension-normalized.

A scoring collection that contains an informative-dimension row but has
$W=0$ fails with `no_informative_weight`; the all-$M=0$ result above is the only
zero-denominator special case. Consistent with §11.1, this raising behavior is
the responsibility of the *fitting* boundary and of a future researcher-facing
eager scoring wrapper: the pure device scoring leaves themselves do not raise
from traced code, so for such a collection they return the ordinary reduction
sentinel (`NaN` for the normalized `score`, `0` for the summed
`log_likelihood`). The `0.1.0b1` private beta ships only these device leaves; the
raising eager scoring wrapper is deferred.

## 11. Scoring, prediction, and sampling surface

For canonical general-projection inputs:

- `score_samples` returns each observed log density $\ell_i$;
- `log_likelihood` returns $\ell_w$ when weights are supplied and the ordinary
  sum otherwise;
- `score` returns the informative weighted mean defined above;
- `predict_proba` returns $q$;
- `predict` returns `argmax(q, axis=-1)`, with the lowest component index winning
  an exact tie;
- `posterior_components` returns component probabilities and conditional latent
  moments; and
- `posterior`/`posterior_mean` follow Section 8.

### 11.1 Device and public inference failures

`posterior_components_general` is the authoritative device-status operation.
For every failed observation/component pair it sets `failed_pairs=True`, uses
`-inf` for that component's log density and log joint, and uses zero placeholders
for that component's conditional mean and covariance. If at least one component
of the row succeeds, its internal score/responsibilities normalize over the
successful components and failed-component responsibility is zero. If every
component fails, its internal score is `-inf` and its fallback responsibilities
are the stored mixture weights. In all cases `numerical_failure` is the reduction
of `failed_pairs`, and status—not a placeholder value—is authoritative.

The convenient numerical leaves deliberately refuse to make partial-pair
failure look successful. If any component fails for a row:

- `score_samples` returns `NaN` for that row;
- `predict_proba` returns an all-`NaN` probability row;
- `predict` returns integer label `-1`;
- `posterior` and `posterior_mean` return all-`NaN` row moments; and
- `log_likelihood` and `score` propagate `NaN` if the failed row participates
  in their reduction.

The pure device functions do not raise from traced code. A researcher-facing
eager wrapper MUST raise an actionable numerical exception by default, or MAY
return a documented result object carrying the same status when the caller
explicitly requests nonraising behavior. It MUST NOT return the internal
fallback responsibilities as a successful public prediction. The $M=0$ branch
is valid and never enters this failure pathway.

### 11.2 General observed sampling

Latent sampling is unchanged from the identity contract. The conceptual general
signature is

```text
sample_observed_general(params, key, n, projection, noise)
```

`key` is a required positional typed or legacy JAX key. `n` is a static integer
at least zero; booleans, negative values, nonintegral values, and traced dynamic
counts fail actionably. Projection is either an explicit per-item array with
shape `(n,M,D)` or an explicit shared adapter with shape `(M,D)`. Noise uses any
explicit Section 4 mode: a per-item mode has batch shape exactly `(n,)`, and a
shared mode has no batch axis. After covariance construction this corresponds
to full shapes `(n,M,M)` or `(M,M)`. All four shared/per-item projection/noise
combinations are valid. `n` is authoritative: a per-item leading axis must
equal it exactly, and all supplied modes must agree on $M$ and $D$. No singleton
or lower-rank array is broadcast implicitly.

For each item the operation draws
a latent mixture value and then

\[
x_i=R_i z_i+\epsilon_i,\qquad \epsilon_i\sim\mathcal N(0,S_i).
\]

For shared $R,S$, if the latent mixture has mean

\[
m_z=\sum_k\alpha_k\mu_k
\]

and covariance

\[
C_z=\sum_k\alpha_k\left[V_k+
(\mu_k-m_z)(\mu_k-m_z)^\mathsf T\right],
\]

then observed samples have mean $Rm_z$ and covariance
$RC_zR^\mathsf T+S$. These are the analytic moment targets for sampling
conformance.

Its output shape is `(n,M)`. With `M=0` it returns the unique empty array of
shape `(n,0)` after validating the key and static arguments. Singular PSD
$S_i$, including zero, uses the
documented deterministic symmetric square-root policy. A pure random function
never creates an internal default key; key reuse/splitting has the identity
contract's semantics.

## 12. Convergence, collapse, and fitting modes

Converged fitting uses the normalized weighted objective $\bar\ell_w$. For
two accepted states,

\[
g_t=\frac{\bar\ell_w^{(t)}-\bar\ell_w^{(t-1)}}
          {\max(1,|\bar\ell_w^{(t-1)}|)}.
\]

The identity contract's `tol`, `decrease_tol`, accepted-history, rollback,
zero-iteration, and status rules apply unchanged. Because $W$ excludes fully
missing rows and global weight scaling cancels, neither operation can change a
stopping decision in exact arithmetic.

This includes the identity contract's selected-dtype representability check for
both stopping tolerances: an exact source zero remains valid, while overflow or
positive-nonzero-to-zero conversion fails before the first grouped update.

A component is collapsed when its aggregated $n_k$ is nonfinite or
nonpositive, or when its proposed parameters are nonfinite or outside the
covariance domain. The default remains `on_collapse="error"`; a failure rolls
back the entire global update, not merely one observed-dimension group. A future
positive threshold MUST be expressed as a fraction of $W$ if global
weight-scale invariance is claimed.

Fixed-step fitting executes exactly the requested number of global E/M updates
on a successful path. A factorization/arithmetic failure, nonfinite objective,
invalid candidate, or component collapse terminates the logical trajectory and
returns the last valid state with a failure status; these are the exceptions to
“exactly.” Later physical scan slots, if present for static shape, repeat the
last valid objective and are outside the logical history. Fixed-step fitting
does **not** reject a finite candidate merely because its objective decreases:
it records and differentiates through the requested update. This distinction is
necessary when ridge or finite-precision rounding removes the monotonicity
guarantee.

Only dynamically converged fitting applies `decrease_tol` and rejects a
materially decreasing candidate. Its rejected candidate is absent from accepted
history and the entire grouped state rolls back. The dense fixed-$M$ fixed-step
kernel may be JIT/autodiff compatible; the eager variable-$M$ grouping and
host-orchestrated aggregation in this version carry no whole-fit JIT or autodiff
guarantee.

## 13. Jitter, ridge, and failure semantics

Factorization jitter $\delta\geq0$ and covariance ridge $\lambda\geq0$ are
distinct scalar controls:

- jitter changes every occurrence of the observed effective covariance to
  $T_{ik}^{\rm eff}=R_iV_kR_i^\mathsf T+S_i+\delta I_M$, including density,
  responsibilities, gain, and the Joseph covariance through
  $S_i+\delta I_M$; it is recorded as an effective-objective control and is
  never stored in $V_k$;
- ridge is applied after the exact global M-step as
  $V_k\leftarrow V_k^{\rm EM}+\lambda I_D$, persists in the learned model,
  is serialized, and removes an ordinary unregularized monotonicity guarantee.

Both controls are finite, scalar, nonnegative, and converted without allowing a
negative or nonzero source value to disappear through dtype conversion.
Boolean, complex, length-one, and other nonscalar controls are rejected. Static
type/shape validity is established for both controls before either value-domain
failure is translated into device status, including when zero updates are
requested or every group has $M=0$. Adaptive or per-row jitter is outside this
version.

An eager public boundary reports invalid shapes, dtypes, nonfinite values,
covariance domains, projection modes, masks, and weights before compilation
where practical. A compiled numerical kernel carries explicit status for:

- observed-covariance factorization/arithmetic failure, identifying group, row,
  and component where representable;
- component collapse;
- nonfinite candidate parameters or objective; and
- material objective decrease in dynamically converged mode.

On any global-update failure, returned parameters, normalized objective,
history, and iteration count describe the exact last accepted state. No partial
group accumulation or partially updated component is presented as successful.
An $M=0$ group never produces a factorization failure because it does not
factor a matrix.

## 14. Identity fast-path equivalence

When $M=D$, every $R_i=I_D$, all observation weights equal one, and the same
$X,S,\theta,\delta,\lambda$, convergence controls, and dtype are used, the
general contract MUST reproduce the identity contract for:

- component log densities and observed scores;
- responsibilities, gains, component posteriors, and marginalized posteriors;
- sufficient statistics and every accepted EM update;
- collapse/failure classifications; and
- normalized objective histories, convergence decisions, and returned states.

Agreement is numerical under the matrix's dtype-specific tolerance; bitwise
identity is not required. Both an explicit identity adapter and ordinary
per-item arrays whose values equal $I_D$ are tested. A result obtained through
the general public API records the general contract ID/version even if an
identity implementation path is selected internally.

## 15. JAX transformation boundary

Canonical dense numerical operations SHOULD be pure, device resident, and
callback free. The matrix separately gates:

- eager versus `jax.jit` value/status agreement;
- same-shape no-retrace behavior;
- single-row versus native-batch `jax.vmap` agreement; and
- explicitly advertised float64 gradients through likelihood and a fixed EM
  step.

No transformation claim applies to eager validation, mask discovery/grouping,
result serialization, dynamic convergence, or exception formatting. No gradient
claim is made for discrete masks, component labels, PRNG draws, failed
factorizations, or the stopping iteration of a converged fit.

## 16. Companion multiple-initialization selection contract

Grouped general fitting adopts the companion restart-selection contract defined
in Identity Contract Section 15, with contract ID
`xdgmm-jax.restart-selection` and version `0.1.0-draft.1`.  This does not change
the general single-fit contract ID or version above, and restart wrapper results
are explicitly outside the current draft serialization schema.

The ordered candidate collection contains latent mixture parameters only.  It
MUST match the grouped input's `K`, latent `D`, and selected computation dtype;
the observations, deterministic mask groups, projections, noise covariances,
sample weights, informative-weight normalization, jitter, and ridge are common
to every candidate.  No candidate changes grouping or row restoration.  The
wrapper replaces only the canonical mixture state and runs the existing grouped
single-fit controller sequentially in ascending candidate index.

Eligibility, strict greatest-objective selection, lowest-index exact ties,
all-failed diagnostic selection, unchanged selected single-fit result, explicit
warm-start reset, retained versus omitted diagnostics, and the host-only
JIT/autodiff boundary are exactly those in Identity Contract Section 15.  The
objectives compared here are the common informative-weight normalized
objectives from Section 12; consequently, valid common positive scaling of all
observation weights cannot change the selected index except where selected-dtype
rounding changes an otherwise exact tie.  When every projection is identity and
all weights equal one, candidate ordering and selected index MUST agree with the
identity restart wrapper within the general identity-equivalence tolerance.

## 17. Conformance

An implementation conforms only when:

1. every Gate row in the matching general capability matrix passes in the
   required dtype/backend;
2. general results carry this exact contract identity/version;
3. the identity-equivalence gate passes against a conforming identity
   implementation;
4. the pinned Bovy and independent-oracle reference gate is complete; and
5. every required negative-sentinel gate rejects deferred feature inputs rather
   than accepting or ignoring them; and
6. documentation does not advertise deferred controls or untested platforms.

A formula review, a temporary development test, or a plausible recovered fit is
not conformance evidence.

## References

The general projection and EM equations follow Jo Bovy, David W. Hogg, and
Sam T. Roweis (2011), “Extreme deconvolution: Inferring complete distribution
functions from noisy, heterogeneous and incomplete observations,” *The Annals
of Applied Statistics* 5(2B), 1657–1677,
[doi:10.1214/10-AOAS439](https://doi.org/10.1214/10-AOAS439),
[arXiv:0905.2979](https://arxiv.org/abs/0905.2979). Equations (13), (14), and
(16) are the direct source for the gain/posterior/EM construction. Software
source provenance and license obligations remain governed separately by
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md); this mathematical contract
does not authorize copying or redistributing upstream code.
