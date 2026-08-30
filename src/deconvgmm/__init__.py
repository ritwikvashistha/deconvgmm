"""Public facade for DeconvGMM: Gaussian-mixture density deconvolution in JAX.

DeconvGMM recovers latent densities and denoises individual signals under known,
heterogeneous measurement noise. It offers two co-equal estimators behind a
shared API:

* **Extreme Deconvolution (XD)** — exact expectation-maximization, exposed as
  ``identity`` (identity projection) and ``general`` (general linear projection);
* **convMMD** — a likelihood-free / simulation-based objective, exposed as
  ``convmmd``.

Both share the same full-covariance Gaussian-mixture model and the same
empirical-Bayes denoiser. See ``artifacts`` for serialization and ``experimental``
for helpers.
"""

from . import artifacts, convmmd, experimental, general, identity
from ._impl.general_validation import NoInformativeWeightError
from ._impl.identity_xd import Params
from ._impl.validation import PrecisionError, ValidationError
from ._impl.version import __version__


__all__ = [
    "NoInformativeWeightError",
    "Params",
    "PrecisionError",
    "ValidationError",
    "__version__",
    "artifacts",
    "convmmd",
    "experimental",
    "general",
    "identity",
]
