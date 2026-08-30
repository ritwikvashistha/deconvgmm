# Temporary numerical development area

This directory is **not** the installable library, the final package namespace,
or a promised public API. It exists only so the contract-driven JAX numerical
kernel can be exercised while the maintainer decides the project, distribution,
and import names and while provenance/licensing gates remain open.

`identity_xd.py` currently contains pure JAX kernels for the identity-projection
model in `docs/model-contract.md`. It assumes canonical, already-validated full
covariance inputs. Future public input validation, fit-loop behavior,
serialization, and compatibility promises do not live here.

`general_xd.py` contains a temporary fixed-observed-dimension numerical leaf
for canonical per-item projections, full noise covariances, and observation
weights. It reuses one Cholesky factor per observation/component pair for the
observed density and generalized latent posterior, uses the generalized Joseph
covariance with the same jitter-adjusted noise, and rolls back a weighted
one-step update on numerical failure or component collapse. Its general
sufficient-statistics result carries active pair failures and arithmetic status;
factor failures on zero-weight rows remain visible in the E-step but do not
affect fit statistics or top-level fit status. Nonfinite moment arithmetic or an
overflowed total weight/component mass is a numerical failure, never component
collapse. A narrow eager check preserves NumPy source-weight sign and underflow
information before an x64-disabled JAX conversion. Python and NumPy scalar
jitter/ridge sources receive the same eager sign and conversion checks. Inputs
that are already traced can only be checked in their selected dtype. Effective
weights are accumulated per component from `log(w) + log(q)`, with `log(q)`
taken from the component log joint and row score rather than an exponentiated
responsibility. Within-component normalized weights drive the moments and
candidate parameters, but their log values are retained so weight-times-mean,
weight-times-covariance, and half-weighted outer terms are formed before any
subnormal probability can flush to zero. Raw statistics and candidate mixture
weights are reconstructed from component log masses. Bit-aware gradual log/exp
handling preserves representable subnormal aggregates without introducing a
component-mass floor. These discrete bit-reconstruction branches provide value
semantics in the subnormal tail, not useful derivatives through that tail; the
ordinary normal-range path retains JAX autodiff. A static `M=0` specialization
performs no factorization, returns the exact mixture prior for inference, and
contributes zero fitting statistics. Because the inherited temporary result
schema has no `no_informative_weight` status, an all-`M=0` one-step fit is
currently represented as all components collapsed with exact rollback; a
future public boundary must instead raise the contract's actionable validation
error.

The general gain never switches paths by comparing projection values, preserving
the advertised derivative with respect to an identity-valued generic `R`. For a
square, successfully factored projection with exactly zero effective noise, the
posterior covariance uses the noise-only mathematical limit and is returned as
exact zero; dimension-reducing projections retain the generalized Joseph path.
Shared projections, missing-mask grouping, public validation, fit control, and
metadata remain outside this leaf.

`general_validation.py` is the corresponding temporary eager boundary. Frozen
tags make per-item, explicitly shared, and identity projection modes distinct,
and separately distinguish per-item/shared isotropic, diagonal, and full noise
forms. The boundary validates selected dtype, source dtype, exact shapes,
finiteness, mixture parameters, covariance domains, and observation weights
before producing the canonical per-item arrays consumed by `general_xd.py`.
It supports the exact `M=0` inference shapes. Fixed-`M` fitting and grouped
fitting raise the typed `no_informative_weight` validation error when no
positive-weight row has an observed coordinate.

The same module implements the contract's eager boolean-mask adapter. It
validates every full-coordinate value (including masked positions) and each
full measurement covariance before slicing, orders distinct boolean masks
lexicographically with `False < True`, preserves relative row order, selects
coordinates in ascending order, and stores an inverse permutation for exact
row restoration. Mask grouping accepts per-item full covariance or an explicit
shared noise tag; per-item isotropic/diagonal mask forms remain outside the
current contract. All-empty groups remain inference-valid and contribute zero
informative weight. Group discovery, host validation, tuple orchestration,
restoration, and exception formatting are deliberately outside JIT/autodiff
claims; only the resulting fixed-`M` numerical calls retain those properties.
`general_grouped.py` is the matching temporary variable-`M` orchestration
layer. It restores every row-leading E-step leaf to original input order while
retaining raw per-group inference failure status. For fitting, failures on
zero-weight rows remain observable in that raw posterior but are excluded from
the active sufficient-statistic status. Fully missing `M=0` groups contribute
no objective weight or moments. Informative groups are reduced to log component
masses, local component means, and local centered covariances, then combined by
a weighted Chan merge before one global M-step; no per-group M-step is run. The
covariance ridge is applied once after that merge. A second pass evaluates the
candidate grouped objective, and any current-statistic failure, global component
collapse, or active candidate failure rolls the entire state back with an
explicit failure stage and restored row/component masks. This host-orchestrated
one-step path is development evidence only: grouped dynamic/fixed-step fit
control now wraps it, but public exceptions/results, serialization, and a
whole-operation JIT/autodiff guarantee remain absent.

`general_fit_control.py` adds the corresponding eager fixed-step and dynamically
converged host loops. Both modes perform only global grouped updates and return
accepted-only objective histories with exact whole-state rollback on numerical
failure or component collapse. Fixed-step fitting accepts a finite decreasing
candidate; dynamically converged fitting classifies the informative weighted
mean objective and rejects a material decrease. The initial objective is
evaluated through a score-only pass, so a finite `theta(0)` remains a valid
zero-step result even if the sufficient-statistic moments needed by a later
update would overflow. If an update is requested, that moment failure is
reported at the attempted iteration with exact rollback to the retained initial
state. Results retain the terminating group/row/component diagnostics,
informative-weight total, bit-exact initial parameter custody, iteration limit,
selected-dtype stopping controls, jitter, ridge, and explicit user-supplied
initialization provenance. Invalid initial objectives have an empty host history
and nonfinite objective. Any attempt marked invalid is exposed with a nonfinite
host sentinel, while a valid rejected decrease retains its finite diagnostic.
Results use the exact general contract metadata ID `xdgmm-jax.general-xd` with
version `0.2.0-draft.1`. There is intentionally no compiled whole-group fit
function: mask discovery and variable-`M` tuple orchestration remain outside JIT
and autodiff guarantees. The result schema and names are still temporary and do
not serialize numerical arrays.

`chunked.py` contains a temporary bounded-memory one-step variant. It scans over
chunk indices, safely gathers `chunk_size` rows from the original inputs inside
each iteration, masks logical final-chunk slots from likelihood and sufficient
statistics, and returns no stacked E-step values. It does not construct a
globally padded input copy. Its largest posterior-covariance intermediate is
therefore
`chunk_size x K x D x D`, while its result contains only component-level
reductions. Per-chunk centered moments are combined with the weighted Chan
merge identity, avoiding the cancellation-prone subtraction of raw second
moments. `chunk_size` must be closed over or marked static by compiled callers.
The original `N`-row observations and measurement covariances remain resident,
and each iteration adds `chunk_size`-row gathered inputs and posterior
workspace. This structural result is not yet an end-to-end measured peak-memory
ceiling or a public memory-budget guarantee.

The temporary E-step reports a scalar numerical-failure flag and a per
observation/component failure mask. The one-step result distinguishes those
factorization/configuration failures from component collapse and rolls the full
parameter state back on either condition. These device-resident flags are kernel
status, not the final public exception or diagnostics design. Scalar numerical
controls are checked before and after conversion to the selected computation
dtype; a negative/nonfinite value or nonzero value lost to underflow is reported
as numerical failure and triggers exact rollback.

`fit_control.py` layers two temporary execution modes over that one-step kernel.
The converged mode uses a host-controlled dynamic loop, commits only accepted
states, and trims history at convergence or failure. The compiled fixed-step
kernel uses a static-length JAX scan and returns a fixed history buffer plus its
logical length; it is JIT-compatible when `n_steps` is static and freezes after
a reported failure. The host wrapper trims that buffer into the same
contract-level result shape as converged fitting and retains the initial
parameters, iteration limit, effective jitter/ridge, converged-mode tolerances,
and initialization provenance required by the draft serialization contract.
The compiled fixed-step buffer schema is unchanged and intentionally omits that
host-only custody metadata. Neither result schema is yet a public compatibility
promise.

`metadata.py` centrally defines distinct identity/general contract records and
the current `user_supplied` initialization provenance. Identity host results use
contract ID `xdgmm-jax.identity-xd` and version `0.1.0-draft.1`; grouped general
results use `xdgmm-jax.general-xd` and `0.2.0-draft.1`. The existing identity
metadata-only JSON helpers reject unknown IDs, versions, or extra fields. They
remain metadata-only helpers.

`serialization.py` implements the temporary draft numerical-artifact contract
for tagged parameters and identity/grouped host fit results. It writes
deterministic uncompressed ZIP containers with canonical JSON and no-pickle NPY
1.0 members, validates resource limits and numerical/result invariants before
device placement, and supports one explicit JAX device on load. This is a
development wire format with exact draft-version compatibility; current
round-trip evidence is CPU-only and establishes no GPU or public API claim.

`restarts.py` implements the separate temporary companion contract
`xdgmm-jax.restart-selection` version `0.1.0-draft.1`. It accepts only a
nonempty ordered collection of user-supplied parameter candidates, validates
and stacks the complete collection in one explicit selected dtype, and runs the
identity or grouped-general fixed/converged controller sequentially for every
candidate. Successful selection uses the greatest eligible normalized
objective with exact ties resolved by lowest index. If every candidate fails,
the result remains explicitly unsuccessful and retains a deterministic failed
representative without changing its single-fit status. Duplicate and
label-permuted candidates are kept in order. A prior result becomes a warm
start only when its parameters are explicitly supplied as a fresh candidate;
history and controls are not inherited. The restart wrappers are host-only,
have no JIT/autodiff claim, generate no random candidates, retain only the
selected full trajectory plus bounded summaries, and are deliberately not
accepted by the current single-fit serializer.

`validation.py` is the temporary eager boundary. It validates selected
float32/float64 precision, exact fit versus inference shapes, parameter and
covariance domains, and explicit measurement-noise construction before arrays
enter compiled kernels. It may synchronize device arrays and is not a JIT
operation. It never rescales accepted mixture weights, clips covariance
eigenvalues, infers transposes, or broadcasts a shared fit covariance unless the
explicit shared adapter is used.

`inference.py` provides temporary pure-JAX scoring, marginalized-posterior,
`predict_proba`/`predict`, and distinct latent and observed sampling operations.
Numerical-leaf inference helpers replace a failed observation row with NaNs
(`predict` uses label `-1`) so the E-step's finite fallback probabilities cannot
look like success; callers needing detailed status use `posterior_components`.
Every sampling call requires one explicit PRNG key, including a zero-count call;
reusing a key repeats a draw and callers split keys for independent draws.
Latent sample counts are static nonnegative integers and categorical sampling
uses JAX's high-dynamic-range mode. Observed sampling accepts only canonical
`(n,D,D)` measurement covariances and infers `n` from that axis. It uses the
deterministic symmetric eigendecomposition square root
`Q @ diag(sqrt(max(lambda, 0)))`, so validated PSD covariances may be singular
or exactly zero. The clipping is a roundoff guard for already-validated PSD
inputs, not an adapter for indefinite matrices; a materially indefinite row
produces NaNs, while the eager boundary supplies the actionable exception.

`general_inference.py` supplies the matching temporary fixed-`M` scoring,
prediction, and marginalized-posterior leaves plus eager grouped conveniences.
The fixed-`M` functions retain arbitrary batch axes and remain callback-free
under JIT and `vmap`; weighted reductions require the exact batch weight shape.
An exact-zero-weight failed row is excluded without evaluating `0 * NaN`, while
a positive-weight failure poisons the aggregate. Grouped reductions restore row
order, use the weights stored with each mask group, and exclude `M=0` rows from
their informative denominator. A structurally all-`M=0` collection has exact
zero total and score unless the global jitter control is invalid. The grouped
tuple loop and restoration remain eager and carry no whole-operation JIT or
autodiff guarantee.

`general_sampling.py` adds temporary fixed-`M` observed-space sampling. Its
pure canonical leaf requires exact per-item `(n,M,D)` projections and
`(n,M,M)` full noise covariances and remains callback-free when `n` is static.
The eager companion accepts only the explicit projection/noise tags from
`general_validation.py`, validates parameters and PSD noise in the selected
dtype, and expands shared inputs to canonical per-item arrays. Every call
requires a typed or legacy JAX key and validates/splits it before returning an
`n=0` or `M=0` empty result. The leaf reuses `inference.sample_latent` rather
than defining another mixture sampler. Measurement noise uses the same
deterministic symmetric eigendecomposition square root as identity observed
sampling, accepting exact zero and singular PSD covariances. A materially
indefinite raw canonical row produces NaNs, while the eager tagged boundary
rejects it actionably. No grouped, ragged, or public sampling API is implied.

Explicit isotropic and diagonal variance adapters check nonnegativity before
precision conversion. Full covariance validation uses overflow-safe
symmetrization, so finite matrices near the selected dtype's upper range are
not made nonfinite merely by boundary repair.

`validate_controls` is an eager host boundary that preserves raw Python scalar
values long enough to detect sign, finiteness, overflow, and nonzero-to-zero
conversion. The host converged and fixed-step fit functions use its prepared
controls. Both controls receive static type/shape validation before either
value-domain failure is translated into rollback status. Raw compiled kernels
still assume canonical JAX inputs and cannot
recover source information already lost before tracing. Isotropic construction
accepts multi-axis inference batches while continuing to reject the ambiguous
two-dimensional `(N, 1)` form.

Once the naming and release-policy decisions are resolved, conforming behavior
will move into the selected `src/` namespace. Downstream users should not import
from `development` or cite this temporary module as a released library.
