# Examples

Two small, self-contained, generated-data examples for **DeconvGMM**. Run them
from an environment with `deconvgmm` installed:

```bash
python examples/xd_quickstart.py       # Extreme Deconvolution (exact EM)
python examples/convmmd_quickstart.py  # convMMD (likelihood-free)
```

- `xd_quickstart.py` constructs an explicit parameter state, builds diagonal
  measurement covariances, runs an identity-projection XD fit, scores the
  observations, and returns marginalized latent posterior means.
- `convmmd_quickstart.py` builds observations with known per-observation noise,
  fits a latent Gaussian mixture with the robust fit recipe
  (`fit_convmmd_recipe`), and denoises with the fitted prior. It then fits the same
  data from a supplied start with `fit_analytic`, once with the defaults (which
  stop before the gradient certificate and report `MAX_ITER`) and once with a
  constant learning rate that reaches it. It enables float64, which the recipe
  requires.

They intentionally do not imply a stable API, tuned hyperparameters,
global-optimum recovery, or any performance guarantee (`performance_claim: none`).
The recipe's automatic start is a heuristic, not a guarantee of the global
optimum. A fair, worked comparison of both methods is in
[`../notebooks/convmmd_xdgmm_comparison.ipynb`](../notebooks/convmmd_xdgmm_comparison.ipynb).
