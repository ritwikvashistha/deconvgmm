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
- `convmmd_quickstart.py` builds noisy observations, computes the predeclared
  bandwidth set, fits a latent Gaussian mixture with the analytic convMMD loss,
  and denoises with the fitted prior.

They intentionally do not imply a stable API, automatic initialization, tuned
hyperparameters, global-optimum recovery, or any performance guarantee
(`performance_claim: none`). A fair, worked comparison of both methods is in
[`../notebooks/convmmd_xdgmm_comparison.ipynb`](../notebooks/convmmd_xdgmm_comparison.ipynb).
