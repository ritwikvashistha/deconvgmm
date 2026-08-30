# Pinned astroML identity-XD parity fixture

- Matrix row: `XD-IP-REF-001`
- Fixture: `tests/fixtures/astroml_identity_ref_001.npz`
- Fixture metadata: `tests/fixtures/astroml_identity_ref_001.metadata.json`
- Generated and hardened: 2026-08-25
- Status: reproducible development evidence; not a public-release record

## What this fixture proves

`XD-IP-REF-001` is deliberately **composite evidence**:

1. Pinned astroML endpoints supply component log-joints, per-observation and
   total observed likelihood, and weights, means, and covariances after one
   `_EMstep`.
2. The repository's independently structured NumPy oracle supplies component
   log densities, responsibilities, conditional latent moments, sufficient
   statistics, and one-step parameters.
3. Fixture generation aborts unless the two sources agree on their shared
   endpoints: component log-joints, score samples, total likelihood, and all
   updated parameters under `rtol=2e-8, atol=2e-10`.

Consequently, the stored responsibilities, conditional moments, and sufficient
statistics are independent-oracle intermediate evidence. They are **not direct
astroML intermediate outputs**. The astroML endpoint comparison and the oracle
intermediate comparison complement each other; this record does not overclaim
full external intermediate parity.

## Reference artifact and import custody

The numeric reference comes from the official PyPI wheel
`astroML-1.0.2.post1-py3-none-any.whl`, SHA-256
`e87b2bda2526e678e62954d5230351fe389039390bd0c99a25e6c41a95f863f3`.
The imported module reports `1.0.2`; the root-restricted installed distribution
reports `1.0.2.post1`.

The generator now requires both the wheel and an isolated extraction root. It:

- verifies the wheel filename, archive structure, and whole-wheel digest;
- verifies hashes for `astroML/__init__.py`, `xdeconv.py`, the license,
  `METADATA`, `WHEEL`, and the wheel's original `RECORD`;
- compares installed `METADATA`, `WHEEL`, package initialization, `xdeconv.py`,
  and the license byte-for-byte with members of the verified wheel;
- resolves distribution metadata only with
  `importlib.metadata.distributions(path=[reference_root])`, so a host astroML
  installation cannot satisfy the version check; and
- aborts unless the actual `astroML` and `xdeconv` import files are exactly
  beneath the verified extraction root.

The installed `RECORD` hash is recorded separately because pip legitimately
rewrites that file to add installer, request, and bytecode entries. The wheel's
original `RECORD` is still bound by its member hash and by the whole-wheel hash.

Actual generation paths are recorded in the JSON metadata. For this fixture the
verified root and imported source were:

```text
/private/tmp/xdgmm_astro_ref
/private/tmp/xdgmm_astro_ref/astroML/__init__.py
/private/tmp/xdgmm_astro_ref/astroML/density_estimation/xdeconv.py
/private/tmp/xdgmm_astro_ref/astroML-1.0.2.post1.dist-info
```

The wheel's `xdeconv.py` SHA-256 is
`06a533b339065967294929e19c4e1359981642fee39f5f5ad81eda1a62f3d315`.
That file is byte-identical to the audited `v1.0.2` and 2026-08-25 `main`
versions recorded in `../../THIRD_PARTY_NOTICES.md`. Its last modifying commit is
`9c92eda800ec5447e7d76e04026a57ceabf3fb0f`; release tag `v1.0.2` points to
`ef1ae3c0d3beaf7176849c7b796acbcfa1425855`.

astroML's license member has SHA-256
`829eccd5a3dc1dafa02fdfe6b810ff7a8d7c0dc97630eb3658d3cb8900e55384`,
matches the SPDX BSD-2-Clause form, and records `Copyright (c) 2012-2013,
Jacob Vanderplas`. The wheel metadata calls the license “BSD 3-Clause,” but the
actual two-condition license text is authoritative for this record. No astroML
source is copied into the development implementation or ordinary tests; the
repository stores generated numeric values and provenance.

Astropy 5.3.4 and PyERFA 2.0.1.1 were present in the isolated reference root.
Their official wheel hashes and root-restricted distribution paths are recorded
and verified, but they were not imported by this parity operation; the metadata
states this distinction explicitly.

## Generation method

The generator uses `PCG64(20260825)` once to create a stored, well-conditioned
`N=25,K=3,D=2` identity-projection problem. It assigns literal initial
`alpha`, `mu`, and `V` arrays to `astroML.density_estimation.XDGMM`, then calls:

- `XDGMM.logprob_a`;
- `XDGMM.logL`; and
- exactly one `XDGMM._EMstep`.

The loop-oriented NumPy oracle then independently evaluates the full E-step,
conditional moments, sufficient statistics, and centered M-step. Ordinary CI
loads the stored results; it never imports astroML or regenerates the fixture.

The generation environment was:

```text
Python 3.10.11 / CPython / Clang 14.0.6
NumPy 1.26.4
SciPy 1.12.0
scikit-learn 1.1.3
threadpoolctl 3.1.0
platform macOS 26.5.1, Darwin 25.5.0, arm64/armv8, little-endian
sysconfig platform macosx-11.0-arm64
NumPy BLAS OpenBLAS ILP64 0.3.23.dev
SciPy BLAS/LAPACK OpenBLAS 0.3.21.dev
astroML distribution 1.0.2.post1 / module 1.0.2
Astropy 5.3.4 (installed in reference root; not imported here)
PyERFA 2.0.1.1 (installed in reference root; not imported here)
```

The JSON also records actual Python/module/library paths, platform fields,
NumPy and SciPy compiler/build configuration, runtime BLAS library paths,
threading layers, architecture, and observed thread counts.

## Deterministic archive and code bindings

The fixture is a deterministic NPY-1.0/NPZ archive with sorted, uncompressed
`ZIP_STORED` members, fixed 1980 timestamps, fixed Unix file mode, and no ZIP
comments or extra fields. Avoiding compression removes zlib-version variance.
Its SHA-256 is:

```text
bed40b6420a73d817c1bffc24349aef5edc3d6c2afeed4813992763c8736be79
```

During hardening, all 23 uncompressed `.npy` member payloads were verified
byte-for-byte identical to the earlier compressed fixture; scientific values
did not change. The metadata records a separate SHA-256 for every NPY payload,
plus canonical manifest SHA-256
`4e30848c662b30a1c74da23adcd911a2ede79e5d3daea07e6b7dea21bc1a3ce4`,
so values remain verifiable even if a future container representation changes.

The evidence record binds the exact local code used:

```text
generator  scripts/generate_astroml_parity_fixture.py
           d0e56676da7ec9b50b2c44860933aa14f78623ee384cc711ad4b3bc37827cde0
helper     scripts/deterministic_npz.py
           26fbc5b542cebf966f6ee2e47fb5f3639510e1c8c32fa22cd29dc20612d51d52
oracle     tests/reference/identity_xd.py
           7cc4656ff8ded4607e6c150392d098bfe2325042e60460c3ada7770973be5f28
```

Ordinary reference tests recompute these three hashes, the archive hash, and
all per-array payload hashes from repository files.

## Reproduction

Extract the exact reference wheels into one isolated directory, retain the
downloaded wheel archives, and run:

```text
/path/to/python scripts/generate_astroml_parity_fixture.py \
  --astroml-root /path/to/isolated/reference-root \
  --astroml-wheel /path/to/astroML-1.0.2.post1-py3-none-any.whl \
  --astropy-wheel /path/to/astropy-5.3.4-cp310-cp310-macosx_11_0_arm64.whl \
  --pyerfa-wheel /path/to/pyerfa-2.0.1.1-cp39-abi3-macosx_11_0_arm64.whl
```

The generator refuses a wrong wheel digest, mismatched extracted identity file,
host-resolved astroML distribution, or astroML import outside the supplied root.
A regenerated archive must match both the archive digest and every NPY payload
digest before it may replace this fixture.

## Limitations

This is identity-projection, float64, one-step parity only. It does not validate
astroML initialization or convergence behavior, direct external intermediate
state, general projection matrices, missing coordinates, float32, GPU execution,
final-fit label alignment, or performance. It complements rather than replaces
the independent analytic and NumPy tests.
