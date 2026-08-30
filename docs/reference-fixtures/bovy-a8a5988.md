# Pinned Bovy general-XD reference-fixture scaffold

- Matrix row: `XD-GEN-REF-BOVY-001`
- Pinned upstream commit:
  `a8a5988d2ab3ceeecbe7f0c23e0554d8a3a4222c`
- Planned fixture: `tests/fixtures/bovy_general_ref_001.npz`
- Planned metadata: `tests/fixtures/bovy_general_ref_001.metadata.json`
- Scaffold date: 2026-08-26
- Authoritative numeric fixture: **Pending**

This record documents the verifier, isolated build recipe, output schema, and
remaining execution gate for one general-projection parity fixture. No numeric
archive in this repository is currently claimed to contain Bovy results. This
scaffold does not yet establish `XD-GEN-REF-BOVY-001`, public general-projection
conformance, performance, or platform support.

## Source custody

The selected upstream source is the official
[`jobovy/extreme-deconvolution`](https://github.com/jobovy/extreme-deconvolution)
commit above, whose Git tree is
`d8cd1cb3dbe024d872992e461ba1290f22f722a8`. The official GitHub archive and
the codeload archive inspected on 2026-08-26 were byte-identical:

```text
archive size     536082 bytes
archive SHA-256  c1882ea6be58c4f08a9d66c539504f1a1c4bc892fa2a5adad7abe47fcaf165fc
archive root     extreme-deconvolution-a8a5988d2ab3ceeecbe7f0c23e0554d8a3a4222c
regular files    66
```

The tracked-file manifest SHA-256 is
`7111a901ba02094f16f3fb44f748da4b831ab9fa5c4e6dea75ce336b624081b8`.
Its byte definition is exact:

1. For every regular tracked file, remove the archive's single root prefix.
2. Form one UTF-8 line as lowercase 64-character SHA-256, two ASCII spaces,
   repository-relative POSIX path, and one LF.
3. Sort the complete lines bytewise under the C locale. This is full-record
   sorting, so the digest leads the sort; it is not path sorting.
4. Concatenate the 66 lines (5,674 bytes) and SHA-256 the result.

This reproduces the audit command:

```text
git ls-files -z | xargs -0 shasum -a 256 | LC_ALL=C sort | shasum -a 256
```

The generator also verifies these critical archive members independently:

| Member | SHA-256 |
|---|---|
| `LICENSE` | `e52808797a9bd901b30bbd0a42d2189090f9390803fa8102c1afbb9919f3c18e` |
| `Makefile` | `7b027b483c67d685f40efbc822b429d00bf6ab83aa8035ffea0013f11b77c4f3` |
| `py/extreme_deconvolution.py` | `0340e31ab4d3fd2652cbf847c61e6c36888add630a5c151fefd0d226bcb07a49` |
| `py/extreme_deconvolution_TEMPLATE.py` | `18a972372f9822960ed39890dc078944079dc18489f9b44ed9d99666733ecc2b` |
| `src/proj_EM.c` | `b3428152ab4555655a66820be9dc8da20876a03b90de09945a5ead1ca2c4bfac` |
| `src/proj_EM_step.c` | `53b37ca9a1baca8908e1f64c25ce8ef622a380bdd93f608a8274d1d6d4fe9d10` |
| `src/proj_gauss_mixtures.c` | `354456bf061168b3e5f54d770c5ca17a9790612b2d46fe2ff593535c0c5b1e99` |
| `src/proj_gauss_mixtures_IDL.c` | `4dabb1f9c75334939c96068e9749fa245fe40d275c220257939796de116f675d` |

`scripts/generate_bovy_parity_fixture.py` requires the archive as a local
argument. It contains no network client. Before extraction it verifies the
whole archive, exact single root, regular-file count, manifest, and critical
members. It rejects absolute paths, `.`/`..`, extra roots, duplicate names,
links, devices, and other special members. Extraction writes regular files
manually into a new temporary directory, and every extracted tracked file is
then compared with its archive size and digest. The archive is checked again to
close a replace-after-verification window.

The source archive, extracted source, generated wrapper, shared library, and
container image are not committed or distributed. The existing
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) carries the complete
conservative BSD-3-Clause notice. These records make no endorsement claim; no
endorsement by Jo Bovy, David W. Hogg, Sam Roweis, or other upstream authors is
implied.

## Planned scientific evidence

The deterministic ordinary workload has:

```text
N=19, K=3, D=4, M=2, float64
PCG64 seed 20260825
unit observation weights (the Bovy weight argument is omitted)
per-row dense, full-row-rank, nonorthogonal R
heterogeneous correlated full S with both off-diagonal signs
max kappa_2(T_ik) <= 1e4
```

It uses literal initial weights, means, and positive-definite covariances, and
no fixed parameters, conjugate prior regularization, or split-and-merge. The
planned direct Bovy arrays are initial and candidate per-row scores, total and
mean observed likelihoods, and weights, means, and covariances after exactly one
EM update. The independent NumPy oracle supplies effective covariances,
residuals, gains, component densities/joints, responsibilities, conditional
moments, sufficient statistics, updated parameters, and candidate scores.

Fixture generation aborts before writing unless the two references agree at
every shared endpoint with `rtol=5e-8`, `atol=5e-10`, and maximum absolute log
error `<=5e-8`. The two evidence sources remain explicitly separated in the
metadata. Oracle intermediates will not be labeled as direct Bovy output.

The exact output schema requires finite NumPy float64 arrays and rejects
missing, extra, wrong-shaped, or wrong-dtype members. The eventual archive will
use the repository's deterministic NPY-1.0/NPZ writer with sorted uncompressed
members and per-array payload hashes. Both artifacts are first written as
private sibling temporary files. Metadata is atomically published first and the
numeric archive last through no-clobber filesystem links; a failed second
publication rolls metadata back and removes both staging files. No placeholder
archive or metadata file is created by this scaffold.

## Upstream one-step semantics

The pinned Bovy control behavior matters for an honest comparison:

- `tol` must be positive. With `tol=0`, the upstream loop starts with zero
  difference and performs no update.
- `tol=1e-12, maxiter=1` performs exactly one EM update.
- That updating call returns the average likelihood evaluated before the
  update.
- Candidate likelihood is obtained by a separate `likeonly=True` call on
  copied updated parameters.
- Per-row direct scores are obtained through one-row `likeonly=True` calls.

The generator cross-checks the returned pre-update mean as well as the
separately obtained row scores, total, and candidate endpoints.

## Pinned CPU build

The authoritative recipe is Linux/amd64 and requires no GPU:

```text
base tag       debian:bookworm-20260824-slim
amd64 digest   sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867
index digest   sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171
APT snapshot   20260824T000000Z
GCC            4:12.2.0-3
Make           4.3-4.1
GSL            2.7.1+dfsg-5+deb12u1
Python         3.11.2-1+b1
NumPy          1:1.24.2-1+deb12u1
```

The upstream library is compiled with `make -j1`, GCC, `-O2`, OpenMP, and no
fast-math. `OMP_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`, BLAS-related thread counts
are one, `LC_ALL=C.UTF-8`, and `TZ=UTC`. The runtime container has networking
disabled, a read-only root, no added capabilities, and a writable temporary
filesystem only for extraction/build. It records the resolved Debian package
manifest, compiler/tool output, flags, built-library digest, dynamic-library
dependencies, generated-wrapper operation, import paths, and final container
image ID.

The image build uses only the two-file `scripts/reference/bovy/` scaffold as its
context; library source, tests, fixtures, and unrelated repository contents are
not sent to a local or remote container daemon. The repository and externally
supplied upstream archive are mounted read-only only for the offline generation
run. Runtime privilege reduction uses Docker's explicit
`--security-opt no-new-privileges=true` form.

GSL is GPL-licensed. It and the linked reference library remain transient
inside the evidence run; neither is placed in a source distribution, wheel, or
fixture archive.

## Invocation

With Docker or a compatible container engine available, run from any directory:

```text
scripts/reference/bovy/generate.sh \
  /path/to/extreme-deconvolution-a8a5988d2ab3ceeecbe7f0c23e0554d8a3a4222c.tar.gz \
  /path/to/empty/output-directory
```

Set `XDGMM_CONTAINER_ENGINE=podman` to select Podman. The image build may access
only the pinned Debian snapshot. The generation run itself uses `--network
none`, mounts the Bovy archive and this repository read-only, and writes only
the planned numeric fixture and metadata into the output mount.

For a source-custody check that performs no extraction, build, or output write:

```text
python scripts/generate_bovy_parity_fixture.py \
  --source-archive /path/to/pinned-source.tar.gz \
  --verify-only
```

## Scaffold verification evidence

The scaffold was developed tests-first on 2026-08-26:

```text
red:    collection failed because the generator module did not exist
red:    17 passed, 2 failed while the container and custody document were absent
green:  20 passed after the extracted-tree substitution guard was added
red:    19 passed, 3 failed before paired publication and minimal build context
green:  22 passed after atomic paired publication and context isolation
red:    22 passed, 1 failed before interrupted-staging cleanup was centralized
green:  23 passed with interrupted writes leaving no staged or final artifact
```

The completed verifier then accepted the locally held official archive and
reproduced its whole-archive SHA-256, 66-file count, and manifest SHA-256 shown
above. This is source-custody evidence only; it is not numeric parity evidence.

## Remaining gate

The pinned source archive has been inspected locally, but the current macOS
host has no Docker/Podman runner for the selected Linux/amd64 environment. The
next step is one execution on a container-capable CPU runner, followed by an
audit of the generated metadata, deterministic archive, direct-versus-oracle
scope, and ordinary production parity test. Until that is done, the capability
row remains Pending. GPU access is neither needed nor relevant to this fixture.
