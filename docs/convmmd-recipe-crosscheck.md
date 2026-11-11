# convMMD robust fit recipe: background and cross-check against J1

- Applies to: [`xdgmm-jax.convmmd` contract `0.6.0-draft.1`](convmmd-model-contract.md) §9 and §18,
  and the `CMMD-FIT-005`–`008` and `CMMD-RECIPE-*` rows of the
  [capability matrix](convmmd-capability-matrix.md)
- Date: 2026-11-11
- `performance_claim: none`. Every row this evidence supports is **Pending**.

`deconvgmm.convmmd.fit_convmmd_recipe` implements the fit recipe that the maintainer's
convMMD robustness campaign adopted (October 2026). This note records the campaign
findings the recipe rests on, and a per-dataset cross-check of the implementation
against the campaign's recorded fits. The campaign's harness and raw results are the
maintainer's own unpublished evidence. The cross-check's results are published
alongside the scripts that produced them (`scripts/convmmd_j1_crosscheck/`).

## 1. Findings the recipe rests on

Each of these was measured on the campaign's grid of situations (clean, outliers,
heteroscedastic-broad, heteroscedastic-thin, thin with outliers), with D ∈ {2, 5, 10}
and 20–30 seeds per cell. Each is evidence for a package decision, not a released
claim.

| Finding | What it showed | What the package does |
|---|---|---|
| **B1** | The two-point loss-change test reported `CONVERGED` at points far from stationary | `converged` is the gradient certificate `max\|grad\| ≤ grad_tol` (§9) |
| **B2** | Constant-rate Adam at learning rate 1e-2 to 3e-2 reaches a certified optimum faster than the cosine schedule | `schedule="constant"`; the recipe uses 3e-2 with no weight decay |
| **G2** | A fixed `max\|grad\| ≤ 1e-6` keeps optimization error well below statistical error from N = 300 to 50,000 | `grad_tol = 1e-6` |
| **I1/I2** | A greedy k-means++ start recovers as well as costlier starts. An always-on witness rarely helps | K + 1 greedy start (§18.1); repair only when triggered (§18.6) |
| **G3/H2/L1** | Saddles are found reliably by an exact check for P ≤ 80 and Lanczos m = 80 above. Escaping very shallow saddles (above −1e-6) on fresh fits does net harm | The curvature check and tolerances of §18.4 |
| **G1/H1** | Of the repair triggers, the raw peak (F), the standardized leave-one-out peak (Z) and the health check (H) together gave the best worst-cell recovery. No variant fully qualified in H1; J1 added the floor and the keep rule | The HU trigger (§18.6) |
| **J1** | `kplus1+floor:HU Kkz` fixes the outlier cells and is safe on clean and heteroscedastic data. Controls without the HU trigger fail to fix outliers | The recipe as a whole (§18) |

## 2. Cross-check design

All 390 of J1's datasets for the adopted start (17 cells, K = 3) were regenerated draw
for draw from J1's seeds. Each was fitted two ways:

- **j1start:** the recipe given J1's own K + 1 centres, bandwidths and witness
  candidates, with the exact curvature check at every P. This tests the
  implementation step by step against J1's recorded decisions.
- **recipe:** the recipe as shipped, with `key = PRNGKey(dataset)`. Its start,
  bandwidth subsample and candidates come from JAX keys rather than J1's NumPy
  streams.

Each fit was compared with J1's recorded base fit and its `HU Kkz` result. There were
780 fits and no errors. The run used JAX 0.10.0 and NumPy 2.4.6, **not** the
qualified JAX 0.6.2 / NumPy 1.26.4 stack.

## 3. j1start: the implementation reproduces J1

| Compared with J1 | Match |
|---|---|
| Base-fit loss (< 1e-9) | 390 / 390 |
| Saddle escapes | 390 / 390 |
| Base-fit steps | 389 / 390 |
| Repair moves attempted and kept | 390 / 390 |
| Final loss (< 1e-9) | 390 / 390 |
| Recovered | 387 / 390, the same datasets |
| Health flag | 80 / 80, the same datasets |

The one step-count difference is `outliers_d5_n500|1`: 1800 steps against 1400, after
one escape each. Both reach the same minimum; the loss is 7.8e-11 lower here. Repairs
lifted base-fit recovery from 382 to 387, with 5 kept moves (4 outlier, 1 clean).

## 4. recipe: as shipped

| Group | Base fit recovered | Final recovered | J1 final |
|---|---|---|---|
| Outliers (incl. thin outliers), 170 | 167 | **169** | 167 |
| Clean, 140 | 138 | 139 | 140 |
| Heteroscedastic (broad and thin), 80 | 79 | 79 | 80 |
| **All, 390** | 384 | **387** | 387 |

- **Recovery** equals J1's overall. Outlier recovery is up 2, and clean and
  heteroscedastic are each down 1.
- **Saddles.** The base fits made 7 escapes, and the repair refits 44 more. One base
  fit, `clean_d2_n2000|20`, used all 3 escapes and stopped at a flagged saddle
  (λ = −4.8e-7). Its repair was the same move J1 made (z = 8.4), and it reached J1's
  final loss to 5e-10. No final fit is at a saddle.
- **Health flag.** It fired on 83 fits against J1's 80. Where the recipe attempted a
  repair that J1 did not, the move was refused, with one exception:
  `outliers_d5_n500|11`. There a kept move (z = 4.8) brought a worse start to within
  1.0e-5 of J1's final loss, and both fits recovered.
- **All six recovery differences trace to the start.**
  - Gains: `outliers_d10_n500|4`, `outliers_d5_n2000|28` and `outliers_d5_n500|7`.
    The JAX start landed on the right fit where J1's NumPy start did not.
  - Losses: in each, the base fit landed above J1's. The repair found a move whose
    loss drop equals (to two digits) that gap, but the Kkz rule refused it.
    - `clean_d2_n500|20`: gap and drop 5.1e-4, z = 2.99 < 3.
    - `outliers_d2_n500|20`: 4.1e-4, z = 2.42 < 3.
    - `hetero_thin_d2_n2000|8`: 2.7e-3, z = 14.5, refused because κ fell. The true
      mixture has a deliberately thin component. On the same dataset J1 started at the
      right fit, and its attempted move went the other way (z = −14.5).
- **N = 10,000 losses are not comparable** one to one. The bandwidths come from a
  different 2000-row subsample, so the objective differs (|Δloss| up to 3.6e-3). All
  30 fits recovered.
- **Lanczos (D = 10, P = 197)** found shallow negative curvature on two fits:
  λ = −4.3e-7 (`outliers_d10_n500|4`) and −2.7e-7 (`clean_d10_n500|9`). Both are within
  the −1e-6 tolerance, so both were treated as minima, as finding H2 recommends. Both
  recovered.
- **Time.** The fits took 540 s (j1start) and 689 s (recipe) on average, against J1's
  recorded 226 s on these datasets. The run shared its cores with other work, which
  explains part of the gap but probably not all of it. There is no clean timing
  comparison, so **no cost claim** is made.

## 5. Open questions

1. **The κ part of Kkz blocks repairs toward thin truths.** It refuses any move that
   makes a component thinner, so a fit that starts on the wrong side of a deliberately
   thin truth cannot be repaired (`hetero_thin_d2_n2000|8`). J1 chose this rule for
   clean-data safety. Revisiting it is a recipe decision, not an implementation bug.
2. **The z ≥ 3 cut refused two correct repairs** (z = 2.99 and 2.42). Lowering it
   trades against J1's clean-data safety evidence.
3. **Cost.** A timing run on dedicated cores is needed before any performance
   statement.
4. **Scope.** The evidence covers J1's grid only: K = 3, D ≤ 10, N ≤ 10,000, Gaussian
   noise.

## 6. Reproducing it

The scripts are in `scripts/convmmd_j1_crosscheck/`. Only `crosscheck.py` needs JAX.

- **`build_reference.py`** turns J1's per-dataset records into the reference file and
  work queue. The records are unpublished, so this step needs the maintainer's copy.
- **`run.sh`** launches workers, one per core.
- **`crosscheck.py`** fits the queue, or single datasets with `--only 'cell|dataset'`.
- **`summarize.py`** prints the tables above from a results file and the reference
  file. It reads the stored `.gz` directly.

The 780 fits are stored as `scripts/convmmd_j1_crosscheck/results_20261111.jsonl.gz`:
- sha256 of the `.gz`: `f8172fe508cee0fd8fce20f132da678ccda8505568699b374a40146500e26c45`;
- sha256 uncompressed: `58b59260999bd3a98d2bf782a792f94bc79bdc9883c04d37cfbab1f2297e8fe2`.

A rerun with the same stack reproduces stored fits exactly, apart from wall time.
That was checked on two datasets in both modes.
