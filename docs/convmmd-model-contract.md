# convMMD density-deconvolution model contract

- Contract ID: `xdgmm-jax.convmmd` (historical identifier retained after the
  package rename to DeconvGMM; do not rename)
- Contract version: `0.2.0-draft.1`
- Status: normative design target for the convMMD development effort (next beta)
- Method: convolutional Maximum Mean Discrepancy (convMMD), a likelihood-free
  / simulation-based density deconvolution and empirical-Bayes denoiser
- Reference: Vashistha, Sarkar, Farahi, "Nonparametric Deconvolution and
  Denoising using Simulation Based Inference", arXiv:2606.21907 (maintainer's own
  work; see the provenance note in §14)
- Applies to: [`xdgmm-jax.convmmd` capability
  matrix](convmmd-capability-matrix.md)
- Last updated: 2026-08-30

Revision `0.2.0-draft.1` adds per-coordinate **missing-at-random (MAR)** data as a
backward-compatible input form (new §16); §§1–15 are unchanged except the comparison
plan (§15), which gains a masked task.

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
exactly. **Missing-not-at-random (MNAR)** selection via a known completeness
\(\Omega\) is **not** implemented in this revision but is a *planned future
revision, not a fundamental limitation* (see §16.11 and the development design note
`development/convmmd_mnar_design_note.md`); only an **unknown** selection function
is genuinely out of scope.

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

Predeclared metrics (both tasks, both methods):

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

This revision covers **MAR only**. A known selection function / completeness
\(\Omega(\cdot)\in[0,1]\) (MNAR) is **not** implemented here, but is a **planned
future revision, not a fundamental limitation**: because convMMD's MC path is a full
forward simulator, a known, simulable, differentiable \(\Omega\) can be incorporated
by a differentiable self-normalized-importance-sampling estimator that never needs
the effective volume \(Z_\theta\) explicitly, and the analytic closed form
additionally survives for Gaussian-family \(\Omega\). This is a capability convMMD
supports more naturally than analytic XD (which needs \(Z_\theta\) in closed form).
Only an **unknown** selection function — or one that cannot be evaluated/simulated —
is genuinely out of scope, for XD and convMMD alike. The mathematics and validation
strategy are recorded in the development design note
`development/convmmd_mnar_design_note.md` (development material, not normative).

**Contrast with pygmmis** (Melchior & Goulding 2018, arXiv:1611.05806): pygmmis
handles per-coordinate missing by inflating the missing feature's covariance (an
approximation), whereas §16 does the **exact** marginalization via \(P_i\); and
pygmmis handles a known selection function by imputing the unobserved complement
inside EM.
