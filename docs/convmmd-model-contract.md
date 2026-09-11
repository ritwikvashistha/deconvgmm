# convMMD density-deconvolution model contract

- Contract ID: `xdgmm-jax.convmmd` (historical identifier retained after the
  package rename to DeconvGMM; do not rename)
- Contract version: `0.5.0-draft.1`
- Status: normative design target for the convMMD development effort (next beta)
- Method: convolutional Maximum Mean Discrepancy (convMMD), a likelihood-free
  / simulation-based density deconvolution and empirical-Bayes denoiser
- Reference: Vashistha, Sarkar, Farahi, "Nonparametric Deconvolution and
  Denoising using Simulation Based Inference", arXiv:2606.21907 (maintainer's own
  work; see the provenance note in §14)
- Applies to: [`xdgmm-jax.convmmd` capability
  matrix](convmmd-capability-matrix.md)
- Last updated: 2026-08-31

Revision `0.2.0-draft.1` adds per-coordinate **missing-at-random (MAR)** data as a
backward-compatible input form (§16); §§1–15 are unchanged except the comparison
plan (§15), which gains a masked task.

Revision `0.3.0-draft.1` adds a **known selection function / completeness**
\(\Omega(\cdot)\in[0,1]\) — **missing-not-at-random (MNAR)** truncation / detection
incompleteness — as a further backward-compatible input form (new §17). Selection on
the **true value** \(\Omega(z)\) is normative in this revision, with a general-\(\Omega\)
self-normalized-importance-sampling (SNIS) Monte-Carlo objective as the default and a
Gaussian-window analytic sub-case as the exact oracle and supported fast path;
selection on the **observed value** \(\Omega(\tilde x)\) is additionally normative under a
**homoscedastic, fully-observed** restriction (§17.12). This supersedes the "out of scope"
language of §§1, 16.11 for the \(\Omega(z)\) case. §§1–16 are otherwise unchanged;
selection **composes** with the §16 projection.

Revision `0.4.0-draft.1` lifts the **heteroscedastic** restriction on selection on the
**observed value** \(\Omega(\tilde x)\) (new §17.13). Per-object measurement noise \(S_i\)
is admitted **exactly**: the §17.12 Gaussian-window transform applied to the inflated
covariances \(\Sigma_k+S_i\) **per object** yields that object's selected observed density
\(\sum_k\pi''_{k,i}\mathcal N(\nu''_{k,i},B''_{k,i})\), so the loss is the **sum of
per-object one-sample discrepancies** against each detected object's own selected mixture.
Because a *detected* object's \(S_i\) is **known**, this needs **no measurement-noise
model** — a Gaussian-window \(\Omega(\tilde x)\) is **machine-eps analytic** (per-object
transform; the homoscedastic §17.12 form is the all-\(S_i\)-equal special case), and a
general non-Gaussian \(\Omega\) uses a **per-object SNIS with the known \(S_i\)**
(statistical gate; biased at finite \(M\), consistent as \(M\to\infty\)). The effective
volume is per-object \(Z_{\theta,i}\), used only as a diagnostic; the loss's normalized
\(\pi''_{k,i}\) never needs it. The **denoiser is unchanged** — \(\Omega(\tilde x_i)\)
cancels, so it is the §7 MAR posterior with per-object \(S_i\). **Relationship to pygmmis:**
because convMMD matches detected conditional densities and **never imputes the undetected
complement**, it needs no noise model, whereas pygmmis's EM must impute and therefore
requires a `covar_callback`; de-biasing that *does* depend on an assumed noise model is a
property of the imputation approach, not of the problem. Composing \(\Omega(\tilde x)\)
with the §16 projection (the window on each observed subspace) remains reserved (§17.13,
"reserved"). No performance claim follows.

Revision `0.5.0-draft.1` lifts the **last** reserved restriction: selection on the
**observed value** \(\Omega(\tilde x)\) now **composes with the §16 projection** —
heteroscedastic \(\Omega(\tilde x)\) acting when the object is observed in only a
coordinate subset \(C_i\) (new §17.14). The one genuine subtlety is that the Gaussian
window must be **marginalized onto each observed subspace** \(\mathbb R^{M_i}\): the
observed-subspace window precision is the **inverse of the covariance principal submatrix**
\(\Psi_i^{-1}=(P_i\Psi P_i^\top)^{-1}\) — **not** \(P_i\Psi^{-1}P_i^\top\) (they differ by a
Schur complement). With that marginalized window, the §17.5 identity applied per object to
the **projected inflated** covariance \(B_k^{(i)}=P_i\Sigma_kP_i^\top+S_i\) gives that
object's selected observed mixture on \(\mathbb R^{M_i}\), and the loss is again the **sum
of per-object one-sample discrepancies** grouped by mask pattern. It reduces **exactly** to
§17.13 at \(P_i=I\) and to the base §16 masked (MAR) loss at \(\Psi^{-1}=0\). A Gaussian
window is **machine-eps analytic**; a general \(\Omega\) uses **per-object projected SNIS**
weighting the observed-subspace draw (statistical gate). The **denoiser is unchanged** —
\(\Omega(\tilde x_i)\) cancels, so it is the §16.4 masked posterior at the object's own
\(S_i\), needing no selection spec and no noise model. The **marginal-completeness** semantics
adopt the canonical Gaussian Lebesgue-marginal \(\Psi_i=P_i\Psi P_i^\top\) as normative (a
conditional-expectation completeness over unmeasured coordinates is model-dependent and out
of scope). This closes the one composition pygmmis alone handled; convMMD does the missing
part by **exact marginalization** (\(P_i\)) rather than covariance inflation, and needs no
undetected-noise model. No performance claim follows.

This document defines the mathematical and behavioral contract the production
convMMD implementation must satisfy. It is derived **independently** from the
method's mathematics and the reference paper, not transcribed from the supplied
prototype code; the supplied Monte-Carlo prototype is treated as material to be
verified against this contract, never as the contract itself. It is intentionally
more precise than any prototype. No prototype is claimed to conform.

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. A
behavior not defined here is not a public guarantee. Measurable acceptance tests
live in the capability matrix.

## 1. Versioning and scope

The contract version is independent of the Python package version:

- a patch change clarifies language without changing accepted results;
- a minor change adds backward-compatible operations or input forms; and
- a major change changes an equation, normalization, shape, or established
  result.

This version covers the **Gaussian-mixture latent model with additive Gaussian
measurement error and a Gaussian (RBF) kernel**, for which the convMMD objective
and the empirical-Bayes denoiser both admit exact closed forms. Normalizing-flow
or otherwise implicit latent models, and non-Gaussian kernels, are out of scope
for this revision and are deferred to a later contract. Implementing one of those
without first revising this contract is nonconforming.

Revision `0.2.0-draft.1` additionally covers **per-coordinate missing-at-random
(MAR)** observations, in which each observation is seen in only a subset of its
coordinates and the missing coordinates are **exactly marginalized** through a
per-observation projection \(P_i\). This is a backward-compatible input form
specified in the separate normative §16; fully-observed inputs reduce to §§3–7
exactly. Revision `0.3.0-draft.1` additionally covers **missing-not-at-random
(MNAR)** selection via a **known, simulable completeness** \(\Omega\) on the true
value \(z\) (§17): because convMMD's Monte-Carlo path is a full forward simulator, a
known differentiable \(\Omega\) is incorporated by a self-normalized importance
sampling (SNIS) estimator that never needs the effective volume \(Z_\theta\)
explicitly, and the analytic closed form additionally survives for a Gaussian-family
\(\Omega\). Selection on the **observed** value \(\Omega(\tilde x)\) is normative under a
homoscedastic, fully-observed restriction (§17.12), which revision `0.4.0-draft.1`
generalizes **exactly** to **heteroscedastic** per-object noise via a per-object transform
on each detected object's own known \(S_i\) (§17.13; machine-eps analytic for a Gaussian
window, per-object SNIS for a general \(\Omega\), needing no noise model); composing
\(\Omega(\tilde x)\) with the §16 projection is normative as of `0.5.0-draft.1` (§17.14,
the window marginalized onto each observed subspace). Only an **unknown**
selection function — or one that cannot be
evaluated/simulated — is genuinely out of scope, for XD and convMMD alike.

Two co-equal loss operators are in scope (maintainer decision, 2026-08-30):

- an **analytic** convMMD loss (exact Gaussian-integral closed form, PRNG-free,
  deterministic); and
- a **Monte-Carlo (MC)** convMMD loss (reparameterized simulation estimator,
  matching the reference method's general SBI form).

The analytic loss is the exact `num_samples -> ∞` limit of the MC loss; the MC
loss is validated to converge to it (§13).

## 2. Symbols and canonical shapes

Let:

- \(N\geq1\) be the number of observations;
- \(K\geq1\) be the number of mixture components;
- \(D\geq1\) be both the latent and observed dimension;
- \(G\geq1\) be the number of kernel bandwidth scales; and
- \(M\geq1\) be the Monte-Carlo sample count (MC loss only).

The canonical representation is:

| Quantity | Symbol | Shape | Meaning |
|---|---:|---:|---|
| observations | \(X=(x_i)\) | `(N, D)` | noisy observed vectors |
| measurement covariance | \(S=(S_i)\) | `(N, D, D)` | known **full PSD** covariance per observation |
| mixture weights | \(\pi\) | `(K,)` | component probabilities, \(\pi_k>0,\ \sum_k\pi_k=1\) |
| latent means | \(\mu\) | `(K, D)` | component means |
| latent covariances | \(\Sigma\) | `(K, D, D)` | component covariances, symmetric PD |
| kernel bandwidths | \(\Gamma=(\gamma_g)\) | `(G,)` | RBF length scales, \(\gamma_g>0\) |
| convMMD loss | \(\mathcal L\) | scalar | training objective (§4) |
| denoised means | \(\hat z\) | `(N, D)` | posterior means \(E[z_i\mid x_i]\) (§7) |

The last axis is always a feature axis; the penultimate axis of a matrix is its
row axis. Core functions **MUST NOT** infer a transpose from values. \(S_i\)
generalizes the diagonal `diag(σ_i²)` form of the supplied prototype to full PSD
covariance; the diagonal case is `S_i = diag(σ_i²)`.

## 3. Generative model

Latent signals are drawn from a full-covariance Gaussian mixture and observed
through additive, per-observation Gaussian measurement error:

\[
z_i \sim q_\theta(z)=\sum_{k=1}^K \pi_k\,\mathcal N(z;\mu_k,\Sigma_k),\qquad
x_i = z_i + \varepsilon_i,\qquad \varepsilon_i\sim\mathcal N(0,S_i).
\]

The **noise-convolved** model for observation \(i\) is
\(\tilde q_{\theta,i}=q_\theta * \mathcal N(0,S_i)
=\sum_k \pi_k\,\mathcal N(\cdot;\mu_k,\Sigma_k+S_i)\).
Deconvolution recovers \(q_\theta\); denoising recovers each \(z_i\).

## 4. convMMD objective (normative)

Let \(k_\gamma(a,b)=\exp\!\big(-\lVert a-b\rVert^2/(2\gamma^2)\big)\) be the
unnormalized Gaussian (RBF) kernel. Define the exact expected-kernel function for
\(W\sim\mathcal N(\delta,\Omega)\):

\[
G(\delta,\Omega;\gamma)\;=\;\mathbb E\big[k_\gamma(W,0)\big]
\;=\;\big|I+\gamma^{-2}\Omega\big|^{-1/2}\,
\exp\!\Big(-\tfrac12\,\delta^\top(\Omega+\gamma^2 I)^{-1}\delta\Big).
\]

Write \(A_k^{(i)}=\Sigma_k+S_i\) for the noise-convolved component covariance.
The **per-observation, per-scale** convMMD loss is

\[
\ell_i(\gamma)\;=\;
\underbrace{\sum_{k,k'}\pi_k\pi_{k'}\,G\big(\mu_k-\mu_{k'},\,A_k^{(i)}+A_{k'}^{(i)};\gamma\big)}_{\text{model–model self term}}
\;-\;2\underbrace{\sum_{k}\pi_k\,G\big(x_i-\mu_k,\,A_k^{(i)};\gamma\big)}_{\text{model–data cross term}} .
\]

The **analytic convMMD loss** is the mean over scales and observations:

\[
\boxed{\;\mathcal L^{\mathrm{an}}(\theta)\;=\;\frac{1}{G\,N}\sum_{g=1}^{G}\sum_{i=1}^{N}\ell_i(\gamma_g).\;}
\]

**Dropped constant.** The data–data term \(\tfrac1{N^2}\sum_{ij}k_\gamma(x_i,x_j)\)
of the full \(\mathrm{MMD}^2\) is θ-independent and is **omitted** from
\(\mathcal L\) (as in the reference method). For the RBF kernel the per-observation
data self-kernel is \(k_\gamma(x_i,x_i)=1\), so \(\mathcal L^{\mathrm{an}}\) equals
the mean \(\mathrm{MMD}^2\) minus \(1\); it therefore **MAY be negative** and is
**not** itself a valid \(\mathrm{MMD}^2\) magnitude. It shares the exact minimizer
and gradient of the full objective. Consumers **MUST NOT** interpret its sign or
scale as a distributional distance.

**Heteroscedastic reduction.** Under homoscedastic noise (\(S_i\equiv S\)) the
self term is independent of \(i\) and \(\mathcal L^{\mathrm{an}}\) reduces to the
paper's Eq. (2) objective minus the same dropped constant. Under heteroscedastic
noise, the per-observation self term (each observation convolved with its own
\(S_i\)) is the normative generalization.

**Monte-Carlo loss.** With one explicit PRNG key and reparameterized draws
\(z_k^{(m)}=\mu_k+L_k\,\zeta^{(m)}\) (\(\Sigma_k=L_kL_k^\top\)) and independent
noise \(\varepsilon_i^{(m)}=L_{S_i}\,\eta^{(m)}\) (\(S_i=L_{S_i}L_{S_i}^\top\)),
\(\tilde x_{k,i}^{(m)}=z_k^{(m)}+\varepsilon_i^{(m)}\), the MC loss replaces each
\(G(\cdot)\) expectation by its sample mean over \(M\) draws (paired,
independent-across-terms), then averages over scales and observations. It is an
unbiased estimator of \(\mathcal L^{\mathrm{an}}\) up to the finite-\(M\) variance
and **MUST** converge to it (§13). It is the appropriate operator when the noise
draws, rather than a Gaussian closed form, are available.

Both operators are `jit`/`grad`-able and correct at float32 and float64. The
analytic operator **MUST NOT** consume a PRNG key; the MC operator **MUST**
require one explicit key and **MUST NOT** read a global PRNG.

## 5. Kernel bandwidth protocol (predeclared)

The bandwidth set is data-driven but declared before results are observed
(maintainer decision, 2026-08-30), so a gate cannot be silently retuned:

\[
\gamma_g \;=\; \operatorname{median\_pairwise\_distance}(X)\times 10^{s_g},\qquad
s_g\in\operatorname{linspace}(-2,2,9),\quad G=9,
\]

where the median pairwise distance is the median Euclidean distance over a
deterministic, seeded subsample of the observations (subsample size and seed
recorded in the fixture/record custody). \(\Gamma\) is an **input** to the loss
operators; the heuristic is a host convenience, not part of the differentiable
core. Callers **MAY** supply any positive \(\Gamma\).

## 6. Parameterization (canonical ↔ unconstrained)

The numerical core consumes **canonical** parameters \((\pi,\mu,\Sigma)\) with
\(\pi\) on the simplex and \(\Sigma_k\) symmetric PD. Optimization uses an
**unconstrained** parameter tuple \((\alpha,\mu,\Lambda)\):

- \(\pi=\operatorname{softmax}(\alpha)\); and
- \(\Sigma_k=L_kL_k^\top+\epsilon_\Sigma I\), where \(L_k\) has strictly-lower
  part \(\operatorname{tril}(\Lambda_k,-1)\) and diagonal
  \(\operatorname{softplus}(\operatorname{diag}\Lambda_k)+\epsilon_L\).

Fixed positive floors \(\epsilon_\Sigma,\epsilon_L\) are recorded constants. The
transform is a pure, `grad`-able bijection onto its image and is applied before
the loss so gradients flow to \((\alpha,\mu,\Lambda)\). The name `sparsemax` in
the prototype denotes ordinary `softmax`; the contract uses `softmax` and **MUST
NOT** advertise sparse weights.

## 7. Empirical-Bayes denoising (normative)

For the fitted prior \(q_\theta\) and known \(S_i\), the posterior
\(p(z\mid x_i)\propto q_\theta(z)\,\mathcal N(x_i;z,S_i)\) is a Gaussian mixture
with

- responsibilities \(r_{ik}\propto \pi_k\,\mathcal N(x_i;\mu_k,\Sigma_k+S_i)\),
  \(\sum_k r_{ik}=1\);
- component posterior means
  \(m_{ik}=\mu_k+\Sigma_k(\Sigma_k+S_i)^{-1}(x_i-\mu_k)\); and
- posterior mean \(\hat z_i=\sum_k r_{ik}\,m_{ik}\).

This is exact and is **identical in form to the XD posterior mean**; the methods
differ only in how \(q_\theta\) is fit (convMMD by loss §4; XD by exact EM). A
self-normalized importance-sampling variant (the prototype's `batch_posterior_mean`)
is a stochastic approximation to \(\hat z_i\) and, if provided, **MUST** require an
explicit key and converge to the closed form.

## 8. Public operations, inputs, outputs

| Operation | Inputs | Output | Notes |
|---|---|---|---|
| `convmmd_loss_analytic` | canonical \((\pi,\mu,\Sigma)\), \(X\), \(S\), \(\Gamma\) | scalar \(\mathcal L^{\mathrm{an}}\) | deterministic, PRNG-free |
| `convmmd_loss_mc` | canonical params, \(X\), \(S\), \(\Gamma\), key, \(M\) | scalar \(\mathcal L^{\mathrm{mc}}\) | explicit key; static \(M\) |
| `convmmd_denoise` | canonical params, \(X\), \(S\) | \(\hat z\) `(N,D)` | exact posterior mean |
| `convmmd_posterior_components` | canonical params, \(X\), \(S\) | \(r,m\) `(N,K)`,`(N,K,D)` | responsibilities + component means |
| `to_canonical` / `to_unconstrained` | params | params | §6 bijection |
| `median_bandwidths` | \(X\), subsample, seed | \(\Gamma\) `(G,)` | §5 host heuristic |

Inference operations **MAY** accept a batch axis. All operations require a single
declared compute dtype (float32 or float64) and **MUST NOT** silently upcast.

## 9. Numerical robustness and status

- All covariance factorizations use Cholesky with a recorded jitter floor;
  \(A_k^{(i)}+\gamma^2 I\) and \(S_i\) are symmetric PD under valid inputs.
- The loss and denoiser **MUST** be finite for valid inputs. A non-finite loss,
  a failed factorization, or a collapsed/degenerate component is reported through
  a documented status object in the `FitStatus` family (fit control, Phase 3),
  never as a silent `NaN` masquerading as success. `NaN`-as-success is a failure.
- The denoiser **MUST** return finite \(\hat z\) for PSD \(S_i\); a materially
  indefinite \(S_i\) is rejected at the eager boundary with an actionable error,
  not silently repaired inside the differentiable core.
- Fitting **MUST** commit only accepted states and roll back to the **last
  finite iterate** on numerical failure, mirroring the identity/general
  `FitStatus`/rollback semantics. The reported fit loss is the objective
  recomputed at the returned parameters (exact for the analytic path; a single
  honest estimate for the Monte-Carlo path) — it **MUST NOT** be a minimum over
  noisy per-step estimates. A two-point convergence claim is only valid for the
  deterministic analytic objective with at least two steps; the stochastic
  Monte-Carlo fit is fixed-step and **MUST NOT** report `CONVERGED` from a noisy
  two-point change.

## 10. JAX contract

The loss operators, denoiser, and parameterization transform **MUST**:

- be `jit`-compatible and `grad`-able w.r.t. the (unconstrained) parameters;
- be `vmap`-able over the observation batch where the operation is per-observation;
- be device-agnostic (CPU/GPU/TPU) and correct at float32 and float64;
- take PRNG keys explicitly (MC loss / SNIS denoiser) and never read a global key;
  reusing a key reproduces a draw and independent draws come from split keys; and
- avoid host synchronization in the differentiable hot path (bandwidth heuristic,
  validation, and sklearn-style initialization are explicitly host-only and carry
  no JIT/autodiff guarantee).

## 11. Initialization

Initialization is host-side and outside the JAX/autodiff contract. A
scikit-learn `GaussianMixture` warm start (the prototype's approach) is a
supported host convenience with recorded `random_state`; the canonical numerical
core also accepts explicit user-supplied initial parameters (`user_supplied`
provenance), which is the reproducible path used by fixtures and tests.

## 12. Relationship to Extreme Deconvolution

| Aspect | XD (identity/general) | convMMD |
|---|---|---|
| Latent model | Gaussian mixture | Gaussian mixture (this contract) or flow (deferred) |
| Noise | full PSD \(S_i\) | full PSD \(S_i\) |
| Objective | exact marginal log-likelihood | convMMD loss (§4), likelihood-free |
| Fit | closed-form EM (E/M steps) | gradient descent on §4 |
| Denoiser | GMM posterior mean | **same** GMM posterior mean (§7) |
| Needs tractable convolved likelihood | yes | no (works for implicit models) |

A head-to-head is **fair** exactly where both target the same object: recover a
known latent density and denoise individual signals from noisy, full-covariance
Gaussian observations. Both consume identical data, noise, seed, split, and
metrics.

## 13. Tolerances and validation gates

- **f64 analytic parity (machine-eps gate).** `convmmd_loss_analytic`,
  `convmmd_denoise`, and `convmmd_posterior_components` in float64 **MUST** agree
  with the independent NumPy oracle to `rtol 5e-8`, `atol 5e-10`, reusing the
  project parity culture. Gradients agree with oracle finite differences to a
  documented looser tolerance.
- **f32 profile.** The same quantities in float32 agree with the oracle to a
  declared float32 profile (`rtol ~1e-4`, `atol ~1e-5`; tightened/loosened per
  measured evidence and recorded before acceptance).
- **MC convergence gate (statistical).** `convmmd_loss_mc` is **not** held to
  machine-eps parity. Its error to the analytic value **MUST** shrink at the
  Monte-Carlo rate (\(\propto M^{-1/2}\)) across a declared grid of \(M\) at a
  fixed key discipline, with every observation preserved.
- All gates run warning-as-error under the pinned development lane.

## 14. Provenance note

convMMD is the maintainer's own method (Vashistha, Sarkar & Farahi,
arXiv:2606.21907), implemented clean-room from this contract. It is not derived
from the astroML or Bovy Extreme Deconvolution code, so no third-party
code-origin obligation applies to it; the convMMD source files carry an own-work
provenance header. See `THIRD_PARTY_NOTICES.md` for the XD attribution and
`CITATION.cff` for the method citation.

## 15. Comparison plan (predeclared)

Three shared ground-truth tasks (maintainer decisions, 2026-08-30), each run for
**both** convMMD and XDGMM on identical data/noise/seed/split:

- **Task A — Gaussian truth (XD home turf).** A known \(K\)-component
  full-covariance GMM latent truth (reuse the `phase4_recovery` regime), additive
  full-covariance Gaussian noise. Expectation: XD wins or ties.
- **Task B — non-Gaussian truth (convMMD turf).** A non-Gaussian latent truth
  (Two Moons / Circles), additive Gaussian noise. Expectation: convMMD wins under
  model misspecification.
- **Task C — Gaussian truth under MAR missingness (this revision's new capability).**
  The Task A latent truth and additive full-covariance Gaussian noise, plus a
  predeclared per-coordinate MAR `observed_mask` (mixed patterns including at least
  one fully-observed group and one \(M=0\) group). Both methods recover the density
  and denoise from identical masked data — convMMD via the grouped projected path
  (§16), XDGMM via its general grouped missing-coordinate path. The predeclared
  metrics are evaluated on the recovered full-\(D\) latent density (Wasserstein) and
  against the true full-\(D\) latent signals including imputed missing coordinates
  (denoising MSE). Each method's endpoints are gated against its own oracle before
  comparison. Expectation: parity (both do exact marginalization on Gaussian truth);
  the task demonstrates the new capability and is **not** a performance claim.
- **Task D — Gaussian truth under known selection (MNAR), convMMD vs pygmmis**
  (added in `0.3.0-draft.1`; contract §17). A Gaussian latent truth under a known
  Gaussian completeness \(\Omega\) on a **noise-free coordinate** so that
  \(\Omega(\tilde x)\equiv\Omega(z)\) (Design B of
  `development/convmmd_pygmmis_comparison_plan.md`): convMMD runs its \(\Omega(z)\)
  analytic path (§17.5), **pygmmis** (Melchior & Goulding 2018) runs its native
  \(\Omega(\tilde x)\) path — **both correctly specified** on identical data from the
  **same** sklearn warm start, both fitting the true (de-biased) population, both
  scored with the **same** selection-aware GMM-posterior denoiser (§17.6). Each
  endpoint's scoring machinery is gated against the independent NumPy oracle before
  comparison. **Honest expectation:** parity on denoising, and a density-recovery gap
  in convMMD's favor — because Gaussian truth + Gaussian selection is convMMD's
  **exact analytic** case whereas pygmmis uses approximate Monte-Carlo imputation (the
  mirror of Task A, where XD is exact and wins). It is a **capability/fairness
  demonstration, not a performance claim**. convMMD now implements the full
  \(\Omega(\tilde x)\) family — homoscedastic (§17.12), heteroscedastic (§17.13), and
  composed with the §16 projection (§17.14) — so the head-to-head extends across
  Designs A (genuine \(\Omega(\tilde x)\)), A-het (heteroscedastic), and the MAR-composed
  case (het \(\Omega(\tilde x)\) + missing coordinates), where convMMD marginalizes the
  missing part exactly and pygmmis can only inflate it. `performance_claim` stays `none`.

Predeclared metrics (all tasks, both methods):

1. recovered-density Wasserstein-1 / sliced-Wasserstein to the known latent truth
   (deconvolution quality);
2. denoised posterior-mean MSE \(\tfrac1N\sum_i\lVert\hat z_i-z_i\rVert^2\)
   against the true latent signals (denoising quality);
3. held-out mean log-likelihood of the recovered density on latent-truth samples;
   and
4. calibration of the denoising posterior (secondary).

Runtime is a **secondary** observation only; `performance_claim` remains `none`.
Every repeated measurement retains median plus min/max/quartiles; the task is
**not** changed to flatter either method after numbers are seen. Each method's
endpoints are gated against its own oracle before any cross-method comparison.

## 16. Per-coordinate missing-at-random (MAR) data via projection

Added in contract version `0.2.0-draft.1` as a backward-compatible input form. Each
observation MAY be observed in only a subset of its coordinates; the missing
coordinates are **exactly marginalized** through a per-observation projection
\(P_i\) (not the noise-inflation shortcut). Fully-observed inputs reduce to §§3–7
exactly. This section is normative for the missing-data path; §§1–15 are otherwise
unchanged.

### 16.1 Symbols and shapes

For observation \(i\):

| Quantity | Symbol | Shape | Meaning |
|---|---:|---:|---|
| observed mask | mask | `(N, D)` bool | `True` where a coordinate is observed |
| observed count | \(M_i\) | scalar | \(=\sum_d \text{mask}[i,d]\), \(0\le M_i\le D\) |
| projection | \(P_i\) | `(M_i, D)` | ascending row-subset of \(I_D\) for observed coordinates \(C_i\) |
| observed sub-vector | \(\tilde x_i\) | `(M_i,)` | \(=P_i x_i\); observed coordinates in ascending order |
| observed noise | \(S_i\) | `(M_i, M_i)` | principal block \(S_i[C_i,C_i]\) of the full noise |
| projected convolved cov | \(B_k^{(i)}\) | `(M_i, M_i)` | \(=P_i\Sigma_k P_i^\top + S_i\) |

\(N_{\mathrm{inf}}=\#\{i:M_i>0\}\) is the number of informative rows. Inputs are
supplied at full width — observations `(N, D)` (finite at **every** entry, including
masked positions), a boolean `observed_mask` `(N, D)`, and full-\(D\) measurement
covariances `(N, D, D)` (or an explicit shared form). The observed subspace is
formed by host-side grouping (§16.7). Missingness is expressed **only** through the
mask; a large covariance value is **not** treated as missing.

### 16.2 Generative model

\(\tilde x_i = P_i z_i + \varepsilon_i\), \(z_i\sim q_\theta\),
\(\varepsilon_i\sim\mathcal N(0,S_i)\). The projected noise-convolved model for
observation \(i\) is \(\sum_k\pi_k\,\mathcal N(P_i\mu_k, B_k^{(i)})\) on
\(\mathbb R^{M_i}\).

### 16.3 Projected convMMD objective (normative)

With the **same** dimension-generic \(G(\delta,\Omega;\gamma)\) of §4, now on
\(\mathbb R^{M_i}\):

\[
\ell_i(\gamma)=\sum_{k,k'}\pi_k\pi_{k'}\,G\big(P_i(\mu_k-\mu_{k'}),\,B_k^{(i)}+B_{k'}^{(i)};\gamma\big)
-2\sum_k\pi_k\,G\big(\tilde x_i-P_i\mu_k,\,B_k^{(i)};\gamma\big).
\]

The **analytic masked loss is normalized by the informative rows** (maintainer
decision, 2026-08-30):

\[
\boxed{\;\mathcal L^{\mathrm{an}}_{\mathrm{mask}}(\theta)=\frac{1}{G\,N_{\mathrm{inf}}}\sum_{g=1}^{G}\sum_{i:\,M_i>0}\ell_i(\gamma_g),\;}
\]

or, with per-observation weights \(w_i\ge0\),
\(\mathcal L^{\mathrm{an}}_{\mathrm{mask}}=\big(\sum_g\sum_{i:M_i>0}w_i\,\ell_i(\gamma_g)\big)\big/\big(G\sum_{i:M_i>0}w_i\big)\).
When every \(M_i=D\) (fully observed) this equals the §4 loss exactly.

**\(M_i=0\) semantics.** A fully-missing row contributes **exactly 0** to both the
numerator and the denominator (it is excluded from \(N_{\mathrm{inf}}\)); its
fixed-\(M\) leaf is not evaluated. Adding or removing \(M_i=0\) rows MUST change no
loss value, gradient, accepted parameter, or status. The naive per-row value from
the formula is \(-1\) per scale — the dropped RBF data self-kernel evaluated in
\(\mathbb R^0\) is \(1\), so \(\ell_i=1-2=-1\); excluding the row from **both**
numerator and denominator (rather than summing its \(-1\)) is what delivers this
invariance, mirroring XD `XD-GEN-M0-001`. A collection with \(N_{\mathrm{inf}}=0\)
has loss defined to be exactly \(0\); **fitting** such a collection MUST fail with
`no_informative_weight`. Equivalently, a collection whose informative rows all
carry zero sample weight (zero informative weight) has loss exactly \(0\) and
likewise fails to fit with `no_informative_weight`.

### 16.4 Projected empirical-Bayes denoiser (normative)

Full-\(D\) posterior mean, imputing missing coordinates from the prior:

- \(r_{ik}\propto\pi_k\,\mathcal N(\tilde x_i;P_i\mu_k,B_k^{(i)})\), \(\sum_k r_{ik}=1\);
- \(m_{ik}=\mu_k+\Sigma_k P_i^\top (B_k^{(i)})^{-1}(\tilde x_i-P_i\mu_k)\in\mathbb R^{D}\);
- \(\hat z_i=\sum_k r_{ik}\,m_{ik}\).

For \(M_i=0\), \(\hat z_i=\sum_k\pi_k\mu_k\) (prior mean) and \(r_{ik}=\pi_k\). This
is identical in form to the XD general posterior mean.

### 16.5 Projected Monte-Carlo loss

Reparameterized draws \(z_k^{(m)}=\mu_k+L_k\zeta^{(m)}\),
\(\varepsilon^{(m)}=L_{S_i}\eta^{(m)}\), projected model draw
\(\tilde x_{k,i}^{(m)}=P_i z_k^{(m)}+\varepsilon^{(m)}\) (mean \(P_i\mu_k\),
covariance \(B_k^{(i)}\)); kernels evaluated in \(\mathbb R^{M_i}\). One explicit
PRNG key; its \(M\to\infty\) limit is \(\mathcal L^{\mathrm{an}}_{\mathrm{mask}}\).

### 16.6 Masked bandwidth protocol (predeclared)

A **single global** scalar bandwidth set (maintainer decision, 2026-08-30):

\[
\gamma_g=b_{\mathrm{mask}}\times10^{s_g},\quad s_g\in\operatorname{linspace}(-2,2,9),\ G=9,
\]

where \(b_{\mathrm{mask}}\) is the median, over informative pairs \((i,j),\,i<j\)
that share at least one observed coordinate, of the Euclidean distance computed on
their shared coordinates \(C_i\cap C_j\). For fully-observed data this equals the §5
median pairwise distance **exactly**. If no pair shares an observed coordinate, the
heuristic raises (mirroring §5's \(n<2\) failure). \(\Gamma\) remains an **input**
to the loss operators; callers MAY supply any positive \(\Gamma\).

### 16.7 Public masked operations, grouping, inputs, outputs

| Operation | Inputs | Output | Notes |
|---|---|---|---|
| `convmmd_loss_analytic_masked` | canonical params, \(X\), `observed_mask`, \(S\), \(\Gamma\), (opt.) \(w\) | scalar | grouped; informative-normalized; PRNG-free |
| `convmmd_loss_mc_masked` | + key, \(M\) | scalar | explicit key; static \(M\) |
| `convmmd_denoise_masked` | canonical params, \(X\), `observed_mask`, \(S\) | \(\hat z\) `(N,D)` | full-\(D\), original row order |
| `convmmd_posterior_components_masked` | as above | \(r\) `(N,K)`, \(m\) `(N,K,D)` | responsibilities + full-\(D\) component means |
| `median_bandwidths_masked` | \(X\), `observed_mask` | \(\Gamma\) `(G,)` | §16.6 host heuristic |

Grouping is deterministic and host-side, reusing the identity-projection
coordinate-selection adapter: groups are emitted in ascending lexicographic
boolean-tuple order (`False < True`), relative row order is preserved, observed
coordinates are selected in ascending order, and the exact projection rows and
principal noise blocks are formed; outputs are restored to the original row order.
The supported masked input forms are the **identity projection** (coordinate
selection, `P == D`) with **per-item full** or an explicit **shared** (full,
diagonal, or isotropic) measurement covariance. Per-item isotropic/diagonal
*masked* noise is **excluded** for this revision (mirrors the XD mask adapter).

### 16.8 Numerical robustness and status (masked path)

- `observed_mask` MUST be boolean with shape `(N, D)` matching the observations; a
  non-boolean or mismatched mask is rejected at the eager boundary.
- Every entry of the observations and measurement covariances MUST be finite,
  **including masked positions**; any NaN/Inf fails actionably.
- Each selected principal noise block is re-validated PSD **at its own scale** after
  slicing (a residual negligible at full width can be material in one block).
- \(B_k^{(i)}\) is symmetric PD for coordinate-selection \(P_i\), PD \(\Sigma_k\),
  and PSD \(S_i\); a degenerate/singular projected covariance surfaces through the
  documented status family (Phase 3) or as a visible `NaN`, never as a finite
  success.
- \(M_i=0\) is handled per §16.3/§16.4; an all-\(M=0\) fit fails with
  `no_informative_weight`.

### 16.9 JAX contract (masked path)

- The per-group fixed-\(M\) leaf (projected analytic/MC loss, projected denoiser and
  posterior components) MUST be `jit`/`grad`/`vmap`-compatible, device-agnostic,
  correct at float32 and float64, and (MC) take one explicit PRNG key.
- Mask grouping, validation, restoration, and the masked bandwidth heuristic are
  **host-only** and carry no JIT/autodiff guarantee (mirrors §10 and XD
  `XD-GEN-JIT-001`).
- For a **fixed** group structure the grouped loss is a single differentiable
  function of the parameters (a host loop over fixed-\(M\) leaves with the group
  arrays as closed-over constants), so `value_and_grad` and `jit` apply; there is
  **no** whole-operation guarantee over re-grouping inside a traced call.

### 16.10 Tolerances and validation gates (masked path)

The §13 gates apply to the masked operators against an independent NumPy oracle that
implements §§16.3–16.6 clean-room: f64 analytic parity to `rtol 5e-8`, `atol 5e-10`;
a declared f32 profile (`rtol ~1e-4`, `atol ~1e-5`, recorded before acceptance); and
the MC estimator held only to the \(M^{-1/2}\) convergence rate. Fully-observed
masked inputs MUST reproduce the §4/§7 results. All gates run warning-as-error in
the pinned lane.

### 16.11 Scope: missing-not-at-random (MNAR) selection

> **Superseded by §17** for the known-selection cases: selection on the true value
> \(\Omega(z)\) (as of `0.3.0-draft.1`) and on the observed value \(\Omega(\tilde x)\) —
> homoscedastic (§17.12), heteroscedastic (§17.13), and composed with the §16 projection
> (§17.14, as of `0.5.0-draft.1`) — are all now normatively implemented. The paragraph
> below is retained for the `0.2.0-draft.1` MAR-only revision; only an **unknown**
> \(\Omega\) remains genuinely out of scope.

The `0.2.0-draft.1` revision covers **MAR only**. A known selection function /
completeness \(\Omega(\cdot)\in[0,1]\) (MNAR) is **not** implemented in that revision,
but is a **planned future revision, not a fundamental limitation**: because convMMD's
MC path is a full forward simulator, a known, simulable, differentiable \(\Omega\) can
be incorporated by a differentiable self-normalized-importance-sampling estimator that
never needs the effective volume \(Z_\theta\) explicitly, and the analytic closed form
additionally survives for Gaussian-family \(\Omega\). This is a capability convMMD
supports more naturally than analytic XD (which needs \(Z_\theta\) in closed form).
Only an **unknown** selection function — or one that cannot be evaluated/simulated —
is genuinely out of scope, for XD and convMMD alike. The mathematics and validation
strategy are recorded in the development design note
`development/convmmd_mnar_design_note.md` (development material, not normative);
§17 makes the \(\Omega(z)\) case normative.

## 17. Known selection function (MNAR) via a simulable completeness \(\Omega\)

Added in contract version `0.3.0-draft.1` as a backward-compatible input form. A
known **completeness / detection probability** \(\Omega(\cdot)\in[0,1]\) retains a
whole object with a probability that depends on its value — **missing-not-at-random
(MNAR)** truncation, flux-limited detection, or survey incompleteness. Selection
**composes** with the §16 per-coordinate projection: an object MAY be both
value-selected by \(\Omega\) and observed in only a coordinate subset via \(P_i\).
This section is normative for the selection path; §§1–16 are otherwise unchanged, and
\(\Omega\equiv 1\) reduces every operation here to its §16 (or, fully observed, §§3–7)
counterpart exactly.

**Scope of this revision.** Selection on the **true value** \(\Omega(z)\) is
normative. Two co-equal operators are provided, mirroring §§4/16: a **general-\(\Omega\)
self-normalized-importance-sampling (SNIS) Monte-Carlo loss** (the default, for any
simulable differentiable \(\Omega\)) and a **Gaussian-window analytic loss** (the
exact closed-form oracle and supported fast path). Selection on the **observed value**
\(\Omega(\tilde x)\) is additionally implemented under a **homoscedastic-noise,
fully-observed restriction** (§17.12): the Gaussian window acts on the noise-convolved
model and the denoiser reduces to the §16.4 MAR posterior. Revision `0.4.0-draft.1`
lifts the homoscedastic restriction to **heteroscedastic** per-object noise **exactly**,
via a per-object transform on each detected object's own known \(S_i\) (§17.13;
machine-eps analytic for a Gaussian window, per-object SNIS for a general \(\Omega\), fully
observed, no noise model). Revision `0.5.0-draft.1` lifts the **last** restriction:
heteroscedastic \(\Omega(\tilde x)\) now **composes with the §16 projection** — selection on
the observed value acting when the object is observed in only a coordinate subset (§17.14;
the window is marginalized onto each observed subspace, machine-eps analytic for a Gaussian
window, per-object projected SNIS for a general \(\Omega\), no noise model). An **unknown**
\(\Omega\) remains out of scope for all methods (§17.11).

### 17.1 Symbols and shapes

| Quantity | Symbol | Shape | Meaning |
|---|---:|---:|---|
| completeness | \(\Omega\) | callable / window | known detection probability in \([0,1]\) |
| selection convention | — | `on_z` \| `on_observed` | argument of \(\Omega\): true value \(z\) or observed \(\tilde x\) |
| Gaussian-window location | \(a\) | `(D,)` | peak of a Gaussian \(\Omega(z)\) |
| Gaussian-window precision | \(\Psi^{-1}\) | `(D, D)` | symmetric PSD; \(\Psi^{-1}=0\) is "no selection" |
| effective volume | \(Z_\theta\) | scalar | \(=\int\Omega(z)\,q_\theta(z)\,dz\), the "eaten" mass (a diagnostic) |
| SNIS weight | \(\omega\) | scalar | \(=\Omega(\cdot)\) at a draw |
| per-observation normalizer | \(\widehat Z_i\) | scalar | SNIS estimate of \(Z_\theta\) |
| effective sample size | ESS | scalar | \(=(\sum\omega)^2/\sum\omega^2\) (a diagnostic) |
| selected canonical params | \((\pi'_k,\mu'_k,\Sigma'_k)\) | as \((\pi,\mu,\Sigma)\) | latent Gaussian mixture of \(\Omega(z)q_\theta(z)/Z_\theta\) |

### 17.2 Generative model with selection

Draw \(z\sim q_\theta\), **retain** with probability \(\Omega(z)\) (convention
`on_z`), then measure \(\tilde x_i=P_i z+\varepsilon_i\),
\(\varepsilon_i\sim\mathcal N(0,S_i)\). The density of the **detected** population in
latent space is

\[
p_{\mathrm{obs}}(z)\;=\;\frac{\Omega(z)\,q_\theta(z)}{Z_\theta},\qquad
Z_\theta=\int\Omega(z)\,q_\theta(z)\,dz,
\]

and the detected **observed** model at observation \(i\) is the noise-convolved,
projected image \(\int\mathcal N(\tilde x;P_i z,S_i)\,p_{\mathrm{obs}}(z)\,dz\) on
\(\mathbb R^{M_i}\). The fitted prior \(q_\theta\) targets the **true population**
(deconvolve *and* de-bias): the parameters \(\theta=(\pi,\mu,\Sigma)\) are the
pre-selection population, while the loss scores the post-selection model
\(p_{\mathrm{obs}}(\theta)\) against the selected data. The recovered \(q_\theta\) is
what the denoiser (§17.6) consumes.

### 17.3 Selection API (typed spec)

\(\Omega\) is supplied as a typed **selection spec** carrying a declared `convention`
(`on_z` this revision):

- **`GaussianSelection(location a, precision`** \(\Psi^{-1}\)**)** — a Gaussian
  window \(\Omega(z)=\exp\!\big(-\tfrac12(z-a)^\top\Psi^{-1}(z-a)\big)\) (peak \(1\) at
  \(z=a\)). Drives the §17.5 analytic path. \(\Psi^{-1}\) is symmetric PSD; \(\Psi^{-1}=0\)
  is exactly "no selection". A **sum of such windows** models Gaussian-mixture
  completeness and MAY be supported by the same identity (§17.5); a single window is the
  minimum required form.
- **`CallableSelection(omega)`** — any simulable callable \(\Omega:\mathbb R^{D}\to[0,1]\),
  differentiable for the SNIS gradient (a finite-difference fallback otherwise). Drives
  the §17.4 SNIS path.

Hard cuts / half-spaces (\(\Omega=\mathbf 1\{x_1>t\}\), survey footprints) yield
Gaussian-CDF (`erf`) terms with no clean general closed form; they are handled by the
SNIS path (§17.4), **not** the analytic path, in this revision. A `CallableSelection`
that is non-differentiable in a way that defeats both the SNIS gradient and a
finite-difference fallback is out of scope (§17.11).

### 17.4 SNIS Monte-Carlo objective (default, general \(\Omega\))

Hard accept/reject is non-differentiable, so the general-\(\Omega\) estimator uses
**self-normalized importance sampling**: draw reparameterized samples from the
**un-selected** model and weight by \(\omega=\Omega(z)\). Per component \(k\), draw
\(z_k^{(m)}=\mu_k+L_k\zeta^{(m)}\) (\(\Sigma_k=L_kL_k^\top\)); set
\(\omega_k^{(m)}=\Omega\!\big(z_k^{(m)}\big)\) (convention `on_z`); project and add
noise, \(\tilde x_{k,i}^{(m)}=P_i z_k^{(m)}+L_{S_i}\eta^{(m)}\) (mean \(P_i\mu_k\),
covariance \(B_k^{(i)}\)). With \(k_\gamma\) the RBF kernel of §4 on \(\mathbb R^{M_i}\):

\[
\widehat Z_i=\sum_k\pi_k\,\frac1M\sum_m\omega_k^{(m)}\;\approx\;Z_\theta,
\]
\[
\mathrm{cross}_i(\gamma)=\frac1{\widehat Z_i}\sum_k\pi_k\,\frac1M\sum_m\omega_k^{(m)}\,k_\gamma\!\big(\tilde x_{k,i}^{(m)},\tilde x_i\big),
\]
\[
\mathrm{self}_i(\gamma)=\frac1{\widehat Z_i^{\,2}}\sum_{k,k'}\pi_k\pi_{k'}\,\frac1{M^2}\sum_{m,m'}\omega_k^{(m)}\omega_{k'}^{(m')}\,k_\gamma\!\big(\tilde x_{k,i}^{(m)},\tilde x_{k',i}^{(m')}\big),
\]
\[
\ell_i(\gamma)=\mathrm{self}_i(\gamma)-2\,\mathrm{cross}_i(\gamma),
\]

averaged over scales and informative observations exactly as in §16.3 (the same
informative-weight normalization; \(M_i=0\) rows contribute nothing). The self term is
a **full double sum** over \((m,m')\): the SNIS ratio normalizer does **not** factor
across a paired diagonal, so the base MC paired-diagonal shortcut (§4) does **not**
apply here. An implementation MAY draw **two independent sets** and form the self term
as the cross-set double sum with the product normalizer
\(\widehat Z_i^{(1)}\widehat Z_i^{(2)}\) (and the cross term with \(\widehat Z_i^{(1)}\)),
mirroring §4's paired-independent-copies discipline; both variants share the
\(M\to\infty\) limit (the packaged implementation uses the two-set form). The
estimator is fully differentiable in \(\theta\) through the
reparameterized draws and through \(\omega=\Omega(z(\theta))\), requiring \(\Omega\)
differentiable (finite-difference fallback otherwise). One explicit PRNG key; static
\(M\); **MUST NOT** read a global key.

**Honesty (biased at finite \(M\)).** Unlike the base MC loss (§4), which is an
**unbiased** estimator of its analytic form, the selected loss is a **ratio (SNIS)
estimator: biased at finite \(M\), consistent as \(M\to\infty\).** The §17.5
Gaussian-\(\Omega\) analytic form is the exact reference. No machine-eps parity is
claimed for a general \(\Omega\) (§17.10).

### 17.5 Gaussian-window analytic sub-case (\(\Omega(z)\); oracle and fast path)

For a Gaussian window \(\Omega(z)=\exp\!\big(-\tfrac12(z-a)^\top\Psi^{-1}(z-a)\big)\)
the Gaussian-product identity gives
\(\Omega(z)\,\mathcal N(z;\mu_k,\Sigma_k)=\tilde w_k\,\mathcal N(z;\mu'_k,\Sigma'_k)\)
with

\[
\Sigma'_k=(\Sigma_k^{-1}+\Psi^{-1})^{-1},\qquad
\mu'_k=\Sigma'_k(\Sigma_k^{-1}\mu_k+\Psi^{-1}a),\qquad
\tilde w_k=(2\pi)^{D/2}|\Psi|^{1/2}\,\mathcal N(\mu_k;a,\Sigma_k+\Psi),
\]

so \(\Omega(z)q_\theta(z)/Z_\theta\) is **again a Gaussian mixture**
\(\sum_k\pi'_k\,\mathcal N(\mu'_k,\Sigma'_k)\) with

\[
\pi'_k=\frac{\pi_k\,\tilde w_k}{\sum_{k'}\pi_{k'}\tilde w_{k'}}
      =\frac{\pi_k\,\mathcal N(\mu_k;a,\Sigma_k+\Psi)}{\sum_{k'}\pi_{k'}\,\mathcal N(\mu_{k'};a,\Sigma_{k'}+\Psi)} .
\]

The \((2\pi)^{D/2}|\Psi|^{1/2}\) factor **cancels** in \(\pi'_k\). The **selected
analytic loss, denoiser, and posterior components are the §16.3/§16.4 projected
operators evaluated verbatim on \((\pi'_k,\mu'_k,\Sigma'_k)\)** — selection re-weights,
shifts, and shrinks the latent components **before** projection and noise convolution:

\[
\ell_i(\gamma)=\sum_{k,k'}\pi'_k\pi'_{k'}\,G\big(P_i(\mu'_k-\mu'_{k'}),\,B_k'^{(i)}+B_{k'}'^{(i)};\gamma\big)
-2\sum_k\pi'_k\,G\big(\tilde x_i-P_i\mu'_k,\,B_k'^{(i)};\gamma\big),\quad B_k'^{(i)}=P_i\Sigma'_kP_i^\top+S_i.
\]

**Numerically robust transform.** Implementations **SHOULD** parametrize the window by
its **precision** \(\Psi^{-1}\) and compute the log-weight in the equivalent
\(|\Psi|\)-free form obtained by completing the square (the
\((2\pi)^{D/2}|\Psi|^{1/2}\) factors cancel, so this **equals** \(\log\tilde w_k\)
exactly and is finite at \(\Psi^{-1}=0\)):

\[
\log\tilde w_k=\tfrac12\big(\log|\Sigma'_k|-\log|\Sigma_k|\big)-\tfrac12 C_k,\quad
C_k=\mu_k^\top\Sigma_k^{-1}\mu_k+a^\top\Psi^{-1}a-b_k^\top\Sigma'_k b_k,\quad
b_k=\Sigma_k^{-1}\mu_k+\Psi^{-1}a,
\]

used directly for both \(\pi'_k=\operatorname{softmax}_k(\log\pi_k+\log\tilde w_k)\)
and the \(Z_\theta\) diagnostic (§17.7). **Reduction:** at \(\Psi^{-1}=0\),
\(\Sigma'_k=\Sigma_k\), \(\mu'_k=\mu_k\), \(\log\tilde w_k=0\), \(\pi'_k=\pi_k\), so
every §17 operation equals its §16 counterpart exactly. Gradients flow through the
transform, so fitting \(\theta\) recovers the de-biased true population. The
gradient's finite-difference agreement is held to a documented looser tolerance
(§17.10).

### 17.6 Selection-aware denoiser

Condition on a **specific detected** object measured at \(\tilde x_i\):
\(p(z\mid\tilde x_i,\text{det})\propto q_\theta(z)\,\mathcal N(\tilde x_i;P_i z,S_i)\,P(\text{det}\mid z)\).

- **`on_observed` (\(\Omega(\tilde x)\)):** \(P(\text{det}\mid z,\tilde x_i)=\Omega(\tilde x_i)\)
  is constant in \(z\) and **cancels** — the per-object posterior is the **unchanged
  §16.4 MAR posterior**. (Implemented under §17.12's homoscedastic, fully-observed
  restriction; the denoiser is the base empirical-Bayes posterior mean, no selection
  spec needed.)
- **`on_z` (\(\Omega(z)\)):** \(P(\text{det}\mid z)=\Omega(z)\) **tilts** the posterior,
  \(p(z\mid\tilde x_i,\text{det})\propto\Omega(z)\,q_\theta(z)\,\mathcal N(\tilde x_i;P_i z,S_i)\).
  For a Gaussian window this is the §16.4 projected GMM posterior of the **transformed
  prior** \((\pi'_k,\mu'_k,\Sigma'_k)\) (§17.5) — closed-form responsibilities, full-\(D\)
  component means, and posterior mean; this is the implemented, oracle-gated denoiser.
  For a general \(\Omega(z)\) the tilted posterior mean is estimated by SNIS: draw from
  the §16.4 MAR posterior (the closed-form projected GMM posterior of the un-selected
  prior — a Gaussian mixture, samplable in closed form), weight each draw by
  \(\Omega(z)\), and take the self-normalized weighted mean; \(M_i=0\) rows use the
  prior as the proposal (the selected prior mean). This is an **evaluation-time**
  estimator (consistent as \(M\to\infty\); the categorical component choice makes it
  non-differentiable, as for any sampling denoiser) held to a **statistical gate**: for
  a Gaussian \(\Omega\) it converges to the closed-form selected denoiser above.

Either way the denoiser uses the recovered **true** \(q_\theta\).

### 17.7 Effective-volume and ESS diagnostics

- **Effective volume** \(Z_\theta=\int\Omega(z)q_\theta(z)\,dz\) (the eaten mass). The
  loss and denoiser **never** need \(Z_\theta\) (the analytic path uses the
  normalized \(\pi'_k\); the SNIS path is self-normalized). \(Z_\theta\) is a
  **diagnostic only**: from SNIS it is \(\widehat Z_i\); from the analytic path it is
  \(Z_\theta=\sum_k\pi_k\tilde w_k=\sum_k\pi_k\exp(\log\tilde w_k)\) using the same
  \(|\Psi|\)-free \(\log\tilde w_k\) of §17.5, which is well-conditioned everywhere —
  including the \(\Omega\to 1\) limit, where \(\log\tilde w_k\to 0\) and
  \(Z_\theta\to 1\) exactly.
- **ESS** \(=(\sum\omega)^2/\sum\omega^2\) for the SNIS path measures importance-weight
  degeneracy; it collapses when \(\Omega\) is very selective (few model draws land in
  the detected region), inflating the ratio estimator's variance and bias.
  Implementations **SHOULD** `log` the ESS as a fit diagnostic.

### 17.8 Numerical robustness and status (selection path)

- A `GaussianSelection` precision \(\Psi^{-1}\) MUST be symmetric PSD (\(\Psi^{-1}=0\)
  allowed); a `CallableSelection` MUST return values in \([0,1]\). Non-finite window
  parameters or \(\Omega\) outputs fail actionably at the eager boundary.
- \(\Sigma'_k=(\Sigma_k^{-1}+\Psi^{-1})^{-1}\) is symmetric PD for PD \(\Sigma_k\) and
  PSD \(\Psi^{-1}\); \(B_k'^{(i)}\) inherits §16.8's PD guarantees. A degenerate
  transformed or projected covariance surfaces through the documented status family or
  as a visible `NaN`, never as a finite success.
- The SNIS estimator MUST be finite where the model places positive mass in the
  detected region. The supported Gaussian window has \(\Omega(z)>0\) everywhere, so
  \(\widehat Z_i>0\) strictly and the loss is finite. A fully collapsed
  \(\widehat Z_i\to 0\) (e.g. a hard-indicator \(\Omega\) that no finite-\(M\) draw
  satisfies) surfaces as a **visible `NaN`**, never a finite success (§9); the ESS
  diagnostic (§17.7) is the early-warning signal, and fit control rolls back from a
  non-finite iterate.

### 17.9 JAX contract (selection path)

- The per-group fixed-\(M\) selected leaves (analytic loss on transformed params, SNIS
  loss, selection-aware denoiser and posterior components) MUST be
  `jit`/`grad`/`vmap`-compatible, device-agnostic, correct at float32 and float64, and
  (SNIS) take one explicit PRNG key.
- The Gaussian-window **parameter transform** \((\pi,\mu,\Sigma)\mapsto(\pi',\mu',\Sigma')\)
  is a pure, `grad`-able function applied before the loss, so gradients flow to
  \(\theta\); it MUST NOT read a PRNG key.
- Selection-spec validation, host-side grouping, and the ESS/\(Z_\theta\) diagnostics
  carry the §16.9 host-only status (no JIT/autodiff guarantee).

### 17.10 Tolerances and validation gates (selection path)

- **Gaussian-\(\Omega(z)\) analytic parity (machine-eps gate).** The analytic selected
  loss, denoiser, and posterior components in float64 MUST agree with an independent
  NumPy oracle (the §17.5 transform composed with the §16 clean-room oracle) to
  `rtol 5e-8`, `atol 5e-10`. The declared float32 profile is `rtol 2e-4`, `atol 2e-5`
  (recorded before acceptance): the Gaussian-window transform's two matrix inversions
  widen the base masked profile, with the transformed **means** dominating at
  \(\approx1.6\times10^{-4}\) relative (the selected loss itself is \(\approx3\times10^{-7}\)).
  Gradients (through the transform and grouped loss) agree with central differences to
  a documented looser tolerance. The \(\Psi^{-1}=0\) reduction MUST reproduce the §16
  masked results exactly.
- **General-\(\Omega\) SNIS gate (statistical only).** The SNIS loss is **not** held to
  machine-eps parity. (a) For a Gaussian \(\Omega\) it MUST converge to the §17.5
  analytic value at the Monte-Carlo rate (\(\propto M^{-1/2}\), fixed-key discipline,
  every observation preserved). (b) For a non-Gaussian \(\Omega\) it is compared to a
  high-\(M\) reference for its convergence trend and its ESS diagnostic; **no
  machine-eps parity is claimed for a general \(\Omega\).**
- All gates run warning-as-error in the pinned development lane.

### 17.11 Scope, honesty, and relationship to XD / pygmmis

- **XD (exact EM)** needs the effective volume \(Z_\theta=\int\Omega\,\tilde p_\theta\)
  in its normalized marginal likelihood, analytic only for special \(\Omega\); convMMD's
  SNIS path avoids \(Z_\theta\) entirely (samples are self-normalized), so a known,
  simulable, differentiable \(\Omega\) is a **natural** fit — arguably more so than
  analytic XD, and aligned with the reference method's SBI thesis (arXiv:2606.21907).
  This does **not** demote XD: where XD is exact (Gaussian truth, no selection) it
  remains preferable. **pygmmis** handles known \(\Omega\) by imputing the unobserved
  complement inside EM — a genuine, distinct contribution.
- **\(\Omega(\tilde x)\) (selection on the observed value): homoscedastic fully-observed
  is normative (§17.12); heteroscedastic fully-observed is normative and **exact**
  (§17.13); the MAR-composed case is normative (§17.14).** With homoscedastic \(S\) and no
  projection, the window acts on the noise-convolved model and the effective volume is
  global (\(\theta\)-only); this case is normative (§17.12), with the denoiser reducing to
  the §16.4 MAR posterior. With **heteroscedastic** \(S_i\) the per-object effective volume
  \(Z_{\theta,i}\) has no *population* closed form, but the **per-object** selected observed
  density is still an exact Gaussian mixture (the §17.12 transform on \(\Sigma_k+S_i\)), so
  §17.13's loss is the machine-eps-analytic sum of per-object one-sample discrepancies for
  a Gaussian window (per-object SNIS for a general \(\Omega\)) using each detected object's
  **known** \(S_i\) — **no noise model**. This is where convMMD and pygmmis diverge: pygmmis
  must impute the *undetected* complement, so it needs a `covar_callback` model for unseen
  objects' noise; convMMD matches only detected conditional densities and needs no such
  model. The denoiser is still the §7 MAR posterior at each detected object's own \(S_i\).
  Composing \(\Omega(\tilde x)\) with the §16 projection requires the window on each
  observed subspace; this is now **normative** in `0.5.0-draft.1` (§17.14).
- **Out of scope (genuinely, for all methods):** an **unknown** \(\Omega\) (recovering
  selection jointly with \(q_\theta\) is ill-posed without a reference/complete sample or
  an identifiability-constrained parametric model), and any \(\Omega\) that cannot be
  evaluated/simulated or is non-differentiable in a way that defeats both the SNIS
  gradient and a finite-difference fallback.
- **Non-claims.** Every §17 capability-matrix row is **Pending**; `performance_claim`
  remains `none`. No machine-eps claim is made for a general \(\Omega\); the SNIS
  estimator is biased at finite \(M\), consistent as \(M\to\infty\). No GPU/TPU or
  cross-platform claim follows beyond recorded evidence.

### 17.12 Selection on the observed value \(\Omega(\tilde x)\) (homoscedastic, fully observed)

For **homoscedastic** noise \(S\) (shared by every object) and **fully observed** data
(\(P_i=I\)), selection on the observed value is normative. The window acts on the
noise-convolved model \(\tilde p_\theta(\tilde x)=\sum_k\pi_k\,\mathcal N(\tilde x;\mu_k,\Sigma_k+S)\),
so the **same** Gaussian-product identity of §17.5 applies to the **inflated**
covariances \(B_k=\Sigma_k+S\):

\[
B''_k=(B_k^{-1}+\Psi^{-1})^{-1},\quad \nu''_k=B''_k(B_k^{-1}\mu_k+\Psi^{-1}a),\quad
\pi''_k=\operatorname{softmax}_k(\log\pi_k+\log\tilde w_k),
\]

with \(\log\tilde w_k\) the \(|\Psi|\)-free form of §17.5 evaluated on \(B_k\). The
selected observed model \(\sum_k\pi''_k\,\mathcal N(\nu''_k,B''_k)\) is a Gaussian
mixture **in the observed space**, so the analytic selected loss is the **base §4 loss
on \((\pi''_k,\nu''_k,B''_k)\) with zero measurement noise** (model and data are both in
the observed space):

\[
\ell_i(\gamma)=\sum_{k,k'}\pi''_k\pi''_{k'}\,G(\nu''_k-\nu''_{k'},B''_k+B''_{k'};\gamma)
-2\sum_k\pi''_k\,G(\tilde x_i-\nu''_k,B''_k;\gamma).
\]

**Reduction:** at \(\Psi^{-1}=0\) this equals the base §4 loss with noise \(S\).
**Denoiser:** \(\Omega(\tilde x_i)\) is constant in \(z\) and cancels, so the
selection-aware denoiser is the **unchanged §7 empirical-Bayes posterior mean**
\(\hat z_i=\sum_k r_{ik}(\mu_k+\Sigma_k(\Sigma_k+S)^{-1}(\tilde x_i-\mu_k))\), needing no
selection spec (§17.6). **General \(\Omega(\tilde x)\)** uses the SNIS Monte-Carlo loss
of §17.4 with the weight evaluated on the **noisy** draw \(\tilde x=z+\varepsilon\)
(\(\varepsilon\sim\mathcal N(0,S)\)) instead of the latent \(z\); same honesty
statement (biased at finite \(M\), consistent as \(M\to\infty\)). **Gates:** the
Gaussian-\(\Omega(\tilde x)\) analytic loss/transform to the f64 machine-eps profile of
§17.10 against the clean-room oracle; the SNIS \(\Omega(\tilde x)\) loss to the
statistical (convergence) gate. The **heteroscedastic** \(\Omega(\tilde x)\) case is
generalized in §17.13; the **MAR-composed** case is generalized in §17.14.

**Contrast with pygmmis** (Melchior & Goulding 2018, arXiv:1611.05806): pygmmis
handles per-coordinate missing by inflating the missing feature's covariance (an
approximation), whereas §16 does the **exact** marginalization via \(P_i\); and
pygmmis handles a known selection function by imputing the unobserved complement
inside EM.

### 17.13 Heteroscedastic selection on the observed value \(\Omega(\tilde x)\) (fully observed)

Added in `0.4.0-draft.1`. This lifts §17.12's **homoscedastic** restriction to
**per-object** measurement noise \(S_i\), fully observed (\(P_i=I\)). The essential fact
is that a *detected* object's own \(S_i\) is **known** (it is the reported measurement
uncertainty), so the noise-convolved model **for that object** is
\(\tilde p_\theta(\tilde x\mid S_i)=\sum_k\pi_k\,\mathcal N(\tilde x;\mu_k,\Sigma_k+S_i)\)
and the §17.12 Gaussian-product identity applies to the **inflated, per-object**
covariances \(B_k^{(i)}=\Sigma_k+S_i\).

**Per-object transform.** For each detected object \(i\), the §17.5/§17.12 transform on
\(B_k^{(i)}\) gives
\[
B''_{k,i}=(B_k^{(i)-1}+\Psi^{-1})^{-1},\quad
\nu''_{k,i}=B''_{k,i}(B_k^{(i)-1}\mu_k+\Psi^{-1}a),\quad
\pi''_{k,i}=\operatorname{softmax}_k(\log\pi_k+\log\tilde w_{k,i}),
\]
with \(\log\tilde w_{k,i}\) the \(|\Psi|\)-free log-weight of §17.5 evaluated on
\(B_k^{(i)}\). Then \(\Omega(\tilde x)\,\mathcal N(\tilde x;\mu_k,B_k^{(i)})
=\tilde w_{k,i}\,\mathcal N(\tilde x;\nu''_{k,i},B''_{k,i})\), so object \(i\)'s **selected
observed density** is the exact Gaussian mixture \(\sum_k\pi''_{k,i}\,\mathcal
N(\nu''_{k,i},B''_{k,i})\) (a proper density; \(\sum_k\pi_k\tilde w_{k,i}=Z_{\theta,i}\)).

**Analytic loss (Gaussian window; machine-eps).** The loss is the mean over detected
objects of the **base §4 one-sample discrepancy** of \(\tilde x_i\) against object \(i\)'s
own selected mixture, with **zero** residual noise (the noise is baked into \(B''_{k,i}\)):
\[
\ell=\frac1N\sum_i\ell_i,\qquad
\ell_i(\gamma)=\sum_{k,k'}\pi''_{k,i}\pi''_{k',i}\,G(\nu''_{k,i}-\nu''_{k',i},\,B''_{k,i}+B''_{k',i};\gamma)
-2\sum_k\pi''_{k,i}\,G(\tilde x_i-\nu''_{k,i},\,B''_{k,i};\gamma),
\]
averaged over scales as in §4. Unlike §17.12, the mixture parameters are **per object**
(they depend on \(S_i\)), so the leaf carries per-object component parameters rather than
sharing one transform across the batch. **Reductions:** with all \(S_i=S\) equal the
per-object parameters collapse and \(\ell\) equals the §17.12 analytic loss exactly; at
\(\Psi^{-1}=0\), \(\pi''_{k,i}=\pi_k\), \(\nu''_{k,i}=\mu_k\), \(B''_{k,i}=\Sigma_k+S_i\),
and \(\ell\) equals the **base §4 loss with per-item noise \(S_i\)** exactly. This estimator
is a **consistent, exact** de-biaser: each term is conditioned on its own \(S_i\), so
minimizing \(\ell\) recovers the true pre-selection \(\theta\) (a Gaussian window is
positive everywhere, so selected-density equality implies latent equality — identifiable).

**General \(\Omega(\tilde x)\) (per-object SNIS; statistical gate).** For a non-Gaussian
window the per-object one-sample discrepancy is estimated by SNIS using the **known**
\(S_i\): draw \(z_k^{(m)}=\mu_k+L_k\zeta^{(m)}\), \(\tilde x_{k,i}^{(m)}=z_k^{(m)}+L_{S_i}\eta^{(m)}\),
weight \(\omega_{k,i}^{(m)}=\Omega(\tilde x_{k,i}^{(m)})\), with the per-object normalizer
\(\widehat Z_i=\sum_k\pi_k\frac1M\sum_m\omega_{k,i}^{(m)}\) and the §17.4 self-normalized
cross/self terms (full cross-set double sum, two independent sets). **Biased at finite
\(M\), consistent as \(M\to\infty\);** its expectation limit is the analytic \(\ell_i\)
above for a Gaussian \(\Omega\). One explicit PRNG key; static \(M\); **MUST NOT** read a
global key. No measurement-noise **model** is required — only the detected object's own
\(S_i\).

**Denoiser (unchanged).** \(\Omega(\tilde x_i)\) is constant in \(z\) and cancels, so the
per-object posterior is the **§7 empirical-Bayes posterior mean with per-object \(S_i\)**
(§17.6, `on_observed`); no selection spec and no noise model are needed.

**Effective volume / ESS (per-object diagnostic).** \(Z_{\theta,i}=\sum_k\pi_k\tilde
w_{k,i}\) (analytic) or the SNIS \(\widehat Z_i\); the per-object ESS \((\sum\omega)^2/\sum\omega^2\)
tracks importance-weight degeneracy. Both are **diagnostics only**; the loss uses the
normalized \(\pi''_{k,i}\) and the SNIS path is self-normalized, so neither needs
\(Z_{\theta,i}\). Implementations **SHOULD** `log` the per-object ESS.

**No noise model; relationship to pygmmis.** convMMD matches each detected object's
conditional selected density and **never imputes the undetected complement**, so it needs
**no model of the noise of undetected objects**. pygmmis's EM imputes that complement and
therefore requires a `covar_callback`. The assumption-dependence of heteroscedastic MNAR
de-biasing is thus a property of the **imputation** approach, not of the problem; convMMD
sidesteps it (the mirror of §17.11's point that convMMD's SNIS avoids the \(Z_\theta\) that
analytic XD needs). Where a *population* effective volume across the unobserved noise law is
wanted as a science output, that — and only that — requires an assumed noise model, and it
enters no loss or denoiser here.

**Gates.** The **Gaussian-\(\Omega(\tilde x)\) per-object analytic** loss/transform in
float64 MUST agree with an independent NumPy oracle (the per-object §17.12 transform
composed with the §4 clean-room one-sample discrepancy) to the §17.10 machine-eps profile;
the all-\(S_i\)-equal reduction MUST reproduce §17.12 and the \(\Psi^{-1}=0\) reduction the
base per-item-noise §4 loss, exactly. The **general-\(\Omega(\tilde x)\) per-object SNIS**
loss is held to the statistical (convergence) gate of §17.10 only (converges to the
per-object analytic value at the Monte-Carlo rate; ESS logged); **no machine-eps parity is
claimed for a general \(\Omega\).**

**JAX contract.** The per-object analytic leaf (per-object transform → one-sample
discrepancy) and the per-object SNIS leaf MUST be `jit`/`grad`/`vmap`-compatible over the
observation batch, device-agnostic, correct at float32/float64; the per-object transform is
a pure, `grad`-able function that MUST NOT read a PRNG key; the SNIS leaf takes one explicit
key. Non-finite \(S_i\), \(\Sigma_k\), or \(\Omega\) outputs fail actionably at the eager
boundary; a degenerate \(B''_{k,i}\) surfaces as a visible `NaN`, never a finite success.

**MAR-composed (normative as of `0.5.0-draft.1`, §17.14).** Composing heteroscedastic
\(\Omega(\tilde x)\) with the §16 projection is now normative: the same per-object identity
applies to the **projected** inflated covariance \(P_i\Sigma_kP_i^\top+S_i\), with the window
**marginalized onto each observed subspace** \(\mathbb R^{M_i}\) (for a Gaussian window this
integrates the unobserved coordinates out in the window's covariance form — the observed
precision is \((P_i\Psi P_i^\top)^{-1}\), **not** the \(\Psi^{-1}\) submatrix). See §17.14 for
the marginalized window, the per-object projected transform and loss, and both reductions
(\(P_i=I\to\)§17.13; \(\Psi^{-1}=0\to\) the base §16 masked loss).

### 17.14 Heteroscedastic \(\Omega(\tilde x)\) composed with the §16 projection (MAR)

Added in `0.5.0-draft.1`. This lifts §17.13's **fully-observed** restriction: selection on
the observed value now composes with per-coordinate missingness. Object \(i\) is observed in
a coordinate subset \(C_i\) (its ascending projection \(P_i\), an \(M_i\times D\) row-subset
of \(I_D\), with \(U_i=\{1,\dots,D\}\setminus C_i\) unobserved), carries its own known
observed-space noise \(S_i\) (an \(M_i\times M_i\) block), and its detection depends on the
observed value \(\tilde x_i=P_i z+\varepsilon_i\in\mathbb R^{M_i}\). The mechanics are the
§17.13 per-object transform run **inside each observed subspace**, with one genuine subtlety:
the window must be **marginalized** onto \(\mathbb R^{M_i}\).

**Marginalized window (the crux).** Detection can depend only on what was measured, so the
completeness on the observed subspace is the window carried onto \(C_i\), integrating out
\(U_i\). Because the Gaussian window is (up to a constant) the density
\(\mathcal N(\cdot;a,\Psi)\) and Gaussian marginals take **covariance submatrices**, the
observed-subspace window has location and precision
\[
a_i=P_i a,\qquad \boxed{\ \Psi_i^{-1}=\big(P_i\Psi P_i^\top\big)^{-1}\ }\qquad(\textbf{not }P_i\Psi^{-1}P_i^\top).
\]
One MUST materialize the window **covariance** \(\Psi=(\Psi^{-1})^{-1}\), take its principal
submatrix \(\Psi_{C_iC_i}=P_i\Psi P_i^\top\), and only then invert. In general
\((P_i\Psi P_i^\top)^{-1}\ne P_i\Psi^{-1}P_i^\top\) (they differ by a Schur complement), so the
§17.13 fully-observed path (which uses \(\Psi^{-1}\) directly) and the §16 MAR path (which
never touches \(\Psi\)) do not compose automatically. The window inversion is **host-side**,
computed once per fit from the spec (which stays frozen at the precision \(\Psi^{-1}\), §17.3):
the special case \(\Psi^{-1}=0\) (no selection) MUST be routed **without** inverting a singular
\(\Psi^{-1}\) — the marginal of "no selection" is "no selection", i.e. \(\Psi_i^{-1}=0\). A
rank-deficient \(\Psi^{-1}\) (selection only along some directions, so \(\Psi\) does not exist)
is a documented edge case that MUST fail actionably at the eager boundary, not silently.

**Marginal-completeness semantics (normative choice).** The Lebesgue-marginal window
\(\Psi_i=P_i\Psi P_i^\top\) is the **normative** meaning of \(\Omega(\tilde x)\) under partial
observation for a Gaussian window: detection depends on the measured coordinates through the
marginalized window, and this reduces correctly at both boundaries. Equivalently, if a survey
specifies its completeness **directly on the observed subspace** as a per-subspace window whose
precision is \((P_i\Psi P_i^\top)^{-1}\), no marginalization is needed — the two coincide. A
rigorous **conditional-expectation** completeness \(\int\Omega(\tilde x)\,p(\tilde x_{U_i}\mid
\tilde x_{C_i})\,d\tilde x_{U_i}\) is model-dependent (it needs a law for the unobserved
coordinates, precisely the undetected-population model convMMD avoids) and is **out of scope**.

**Per-object projected transform.** With the observed-subspace window \((a_i,\Psi_i^{-1})\)
and the **projected, inflated** component covariance \(B_k^{(i)}=P_i\Sigma_kP_i^\top+S_i\), the
§17.5 Gaussian-product identity applied inside \(\mathbb R^{M_i}\) gives
\[
C_{k,i}=\big((B_k^{(i)})^{-1}+\Psi_i^{-1}\big)^{-1},\quad
\nu_{k,i}=C_{k,i}\big((B_k^{(i)})^{-1}P_i\mu_k+\Psi_i^{-1}a_i\big),\quad
\pi_{k,i}=\operatorname{softmax}_k\!\big(\log\pi_k+\log\tilde w_{k,i}\big),
\]
with \(\log\tilde w_{k,i}\) the \(|\Psi_i|\)-free log-weight of §17.5 evaluated on
\(B_k^{(i)}\) with window \((a_i,\Psi_i^{-1})\). Then \(\Omega_i(\tilde x)\,\mathcal
N(\tilde x;P_i\mu_k,B_k^{(i)})=\tilde w_{k,i}\,\mathcal N(\tilde x;\nu_{k,i},C_{k,i})\), so
object \(i\)'s **selected observed density** is the exact Gaussian mixture
\(\sum_k\pi_{k,i}\,\mathcal N(\nu_{k,i},C_{k,i})\) on \(\mathbb R^{M_i}\).

**Analytic loss (Gaussian window; machine-eps).** Grouped by mask pattern as in §16.7 (all
rows in a group share \(P_i\), hence one marginalized window), the loss is the
informative-weight-normalized mean over detected objects of the base §4 one-sample discrepancy
of \(\tilde x_i\) against its own selected mixture, with **zero** residual noise (baked into
\(C_{k,i}\)):
\[
\ell_i(\gamma)=\sum_{k,k'}\pi_{k,i}\pi_{k',i}\,G(\nu_{k,i}-\nu_{k',i},\,C_{k,i}+C_{k',i};\gamma)
-2\sum_k\pi_{k,i}\,G(\tilde x_i-\nu_{k,i},\,C_{k,i};\gamma),
\]
averaged over scales as in §4 and normalized by the informative weight as in §16.3 (an
all-\(M_i=0\) collection has loss exactly \(0\); fitting rejects it as
`no_informative_weight`). **Reductions:** at \(P_i=I\) the marginalized window is the full
window and \(B_k^{(i)}=\Sigma_k+S_i\), so \(\ell\) equals the §17.13 heteroscedastic loss
exactly; at \(\Psi^{-1}=0\), \(\Psi_i^{-1}=0\Rightarrow C_{k,i}=B_k^{(i)}\),
\(\nu_{k,i}=P_i\mu_k\), \(\pi_{k,i}=\pi_k\), so \(\ell\) equals the **base §16 masked (MAR)
loss** exactly. Both MUST hold to the §17.10 machine-eps profile.

**Consistency.** Proposition-of-§17.5 logic runs in each observed subspace: the population
loss term for object \(i\) is, up to a constant, an MMD between \(P^{\det}_\theta(\cdot\mid
P_i,S_i)\) and \(P^{\det}_{\theta_\star}(\cdot\mid P_i,S_i)\) on \(\mathbb R^{M_i}\); noise
convolution is injective on \(\mathbb R^{M_i}\) and \(\Omega_i>0\) for a Gaussian window, so
matching forces \(P_i(q_\theta*\mathcal N(0,\cdot))=P_i(q_{\theta_\star}*\cdot)\) on that
subspace. Full recovery of \(q_\theta\) then rests on the GMM's parametric coupling across the
union of observed subspaces — the **same** MAR identifiability caveat as §16 (a coordinate
observed in no object is pinned only by the model form). No selection normalization is ever
estimated.

**General \(\Omega(\tilde x)\) (per-object projected SNIS; statistical gate).** For a
non-Gaussian window the per-object one-sample discrepancy is estimated by SNIS using the
**known** \(S_i\): draw \(z_k^{(m)}=\mu_k+L_k\zeta^{(m)}\), project and add the object's own
noise \(\tilde x_{k,i}^{(m)}=P_i z_k^{(m)}+L_{S_i}\eta^{(m)}\in\mathbb R^{M_i}\), weight
\(\omega_{k,i}^{(m)}=\Omega_i(\tilde x_{k,i}^{(m)})\) by the completeness **evaluated on the
observed subspace**, and take the §17.4 self-normalized cross/self terms with the per-object
normalizer \(\widehat Z_i=\sum_k\pi_k\frac1M\sum_m\omega_{k,i}^{(m)}\) (full cross-set double
sum, two independent sets). The general \(\Omega\) MUST be supplied as acting on the observed
subspace (the per-subspace convention above); a canonical marginalization of a general
\(\Omega\) over unmeasured coordinates is model-dependent and out of scope. For a Gaussian
window supplied as its own marginal the SNIS limit is the analytic \(\ell_i\) above.
**Biased at finite \(M\), consistent as \(M\to\infty\);** one explicit PRNG key (split across
objects/groups), static \(M\), **MUST NOT** read a global key. No noise model.

**Denoiser (unchanged).** \(\Omega(\tilde x_i)\) is constant in \(z\) and cancels, so the
per-object posterior is the **§16.4 projected empirical-Bayes posterior** at the object's own
\(S_i\) (full-\(D\) output, original row order via the group restoration of §16.7); it needs no
selection spec and no noise model. This is the shipped masked denoiser applied verbatim.

**Effective volume / ESS (per-object diagnostic).** \(Z_{\theta,i}=\sum_k\pi_k\tilde w_{k,i}\)
(analytic, in the observed subspace) or the SNIS \(\widehat Z_i\); the per-object ESS
\((\sum u)^2/\sum u^2\) with \(u_{k,m}=\pi_k\omega_{k,i}^{(m)}\) tracks importance-weight
degeneracy. Both are **diagnostics only** — the loss uses the normalized \(\pi_{k,i}\) and the
SNIS path is self-normalized. Implementations **SHOULD** `log` the per-object ESS.

**No noise model; relationship to pygmmis.** convMMD does the **missing** part by **exact
marginalization** (\(P_i\)) and the **selection** part by matching each detected object's
observed-subspace conditional density, so it needs **no model of undetected objects' noise**.
pygmmis composes missing with selection only via **covariance inflation** (masking missing
features as `NaN` and inflating that coordinate's variance — an approximation); its **exact**
projection route (`R`) raises `NotImplementedError` when a selection callback is also set (its
EM cannot generate imputation samples under a non-trivial projection), and its imputation of
the undetected complement requires a `covar_callback` noise model. This composition is thus a
case where convMMD is both **more exact** on the missing part and **assumption-lighter** on the
selection part. This is a capability statement; **no performance claim** follows.

**Gates.** The **Gaussian-\(\Omega(\tilde x)\)+MAR per-object projected analytic** loss/
transform in float64 MUST agree with an independent NumPy oracle (the marginalized window ∘ the
§17.5 transform on the projected inflated \(B_k^{(i)}\) ∘ the §4 clean-room one-sample
discrepancy) to the §17.10 machine-eps profile (`rtol 5e-8`/`atol 5e-10`; float32 declared
profile); the marginalized-window identity MUST be checked directly on a \(D\ge3\) object with a
genuinely off-diagonal \(\Psi\) against a brute-force marginalization; the \(P_i=I\) reduction
MUST reproduce §17.13 and the \(\Psi^{-1}=0\) reduction the base §16 masked loss, **exactly**.
The **general-\(\Omega\) per-object projected SNIS** loss is held to the statistical
(convergence) gate of §17.10 only (converges to the per-object analytic value at the
Monte-Carlo rate; ESS logged); **no machine-eps parity is claimed for a general \(\Omega\).**

**JAX contract.** The per-object projected analytic leaf (marginalized window → per-object
projected transform → one-sample discrepancy, grouped by mask pattern) and the per-object
projected SNIS leaf MUST be `jit`/`grad`/`vmap`-compatible over each group's observation batch,
device-agnostic, correct at float32/float64; the marginalized window and the per-object
transform are pure and `grad`-able and MUST NOT read a PRNG key (the fixed window carries no
gradient; gradients flow to \((\pi,\mu,\Sigma)\) through \(B_k^{(i)}\) and \(P_i\mu_k\)); the
SNIS leaf takes one explicit key. Mask grouping, window inversion, and row restoration are
host-only and outside the JIT/autodiff contract. Non-finite \(S_i\), \(\Sigma_k\), \(\Psi\), or
\(\Omega\) outputs fail actionably at the eager boundary; a degenerate \(C_{k,i}\) surfaces as a
visible `NaN`, never a finite success.
