# Numerical artifact serialization contract

- Format ID: `xdgmm-jax.numeric-artifact`
- Format version: `0.1.0-draft.1`
- Status: exact draft contract implemented in the `0.1.0b1` private beta;
  CPU-qualified, cross-device GPU evidence pending
- Scope: parameters and host fit results for the identity and grouped-general
  model contracts
- Last updated: 2026-08-28

This document defines a safe, versioned transport format for numerical model
state. The private implementation is exposed as `xdgmm_jax.artifacts` in
`xdgmm-jax==0.1.0b1`, but the format's compatibility is controlled by its own
exact IDs and versions. This contract does not select a project license, author
list, or future public-release policy.

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## 1. Goals and non-goals

The format MUST:

- preserve the mathematical contract and record version used to interpret an
  artifact;
- preserve float32 or float64 numerical values without silent conversion;
- fail closed on unknown versions, fields, array kinds, or inconsistent fit
  diagnostics;
- be deterministic for identical canonical inputs;
- be readable without executing code or enabling pickle; and
- support an explicit host-to-device load boundary.

The format does not preserve a JAX backend, device identity, sharding, compiled
program, PRNG implementation, or cross-version bitwise computation guarantee.
It is not an authenticity mechanism. A trusted external digest or signature is
required when provenance against malicious replacement matters.

## 2. Container

One artifact is an uncompressed ZIP container with exactly these members:

```text
manifest.json
arrays/<logical-array-name>.npy
...
```

Requirements:

1. `manifest.json` is UTF-8 canonical JSON: sorted object keys, compact
   separators, no byte-order mark, and one trailing newline.
2. Numerical members use NumPy NPY format 1.0, C order, and `allow_pickle=False`.
3. Floating values are canonical little-endian `<f4` or `<f8`. Boolean status
   arrays use the NPY boolean dtype.
4. ZIP compression is `ZIP_STORED`. Members are written in deterministic order:
   `manifest.json` first, then array paths in lexicographic order.
5. Every member uses the timestamp `1980-01-01T00:00:00`, no extra field or
   comment, and regular-file permissions `0644`.
6. Directory entries, duplicate members, encrypted members, absolute paths,
   `..`, backslashes, and unlisted extra members are forbidden.
7. A writer MUST create a sibling temporary file, flush and synchronize it, and
   atomically replace the destination only when overwrite was explicitly
   requested. The default MUST NOT overwrite an existing artifact.

No public filename extension is selected by this draft.

## 3. Record and contract identifiers

This format version recognizes exactly these draft record IDs:

| Record | ID | Version |
|---|---|---|
| Parameters | `xdgmm-jax.parameters` | `0.1.0-draft.1` |
| Identity fit result | `xdgmm-jax.identity-fit-result` | `0.1.0-draft.1` |
| Grouped fit result | `xdgmm-jax.grouped-general-fit-result` | `0.1.0-draft.1` |

Parameter records MUST also carry either the identity contract
`xdgmm-jax.identity-xd` or the general contract `xdgmm-jax.general-xd`, with
its exact supported version. A generic parameter loader returns a tagged
parameter artifact; it MUST NOT erase the contract and return a bare parameter
tuple whose semantics are ambiguous.

`package_version` is informational and comes from one central version source.
The private-beta writer uses the exact package version `0.1.0b1`. A reader MUST
NOT use `package_version` to override format, record, or contract
compatibility.

## 4. Manifest

Every manifest has exactly these top-level fields for its record version:

```json
{
  "arrays": {},
  "artifact_kind": "parameters",
  "contract_id": "xdgmm-jax.identity-xd",
  "contract_version": "0.1.0-draft.1",
  "format_id": "xdgmm-jax.numeric-artifact",
  "format_version": "0.1.0-draft.1",
  "model": {
    "dtype": "float64",
    "latent_dimension": 2,
    "n_components": 3
  },
  "package_version": "0.1.0b1",
  "record_id": "xdgmm-jax.parameters",
  "record_version": "0.1.0-draft.1"
}
```

Unknown, missing, or duplicate fields fail validation. JSON `NaN`, `Infinity`,
and `-Infinity` are forbidden. Floating diagnostics, including nonfinite failure
sentinels, live in NPY members rather than JSON.

Each `arrays` entry maps one logical name to exactly:

```json
{
  "data_nbytes": 24,
  "dtype": "float64",
  "member_nbytes": 152,
  "path": "arrays/parameters.weights.npy",
  "sha256": "<digest of the complete NPY member>",
  "shape": [3]
}
```

The logical name, path, dtype, rank, shape, raw data byte count, complete member
byte count, and SHA-256 MUST agree with the actual NPY payload. Paths and logical
names are fixed by the record schema, not chosen by input data.

## 5. Parameter records

A parameter record contains exactly:

- `parameters.weights`, shape `(K,)`;
- `parameters.means`, shape `(K,D)`; and
- `parameters.covariances`, shape `(K,D,D)`.

All three arrays use the manifest computation dtype. Weights are finite,
strictly positive, and normalized under the applicable contract tolerance.
Means are finite. Covariances are finite, symmetric within the contract
tolerance, and positive definite under the selected-dtype factorization policy.
A loader validates without normalizing weights, clipping eigenvalues, adding
jitter, or otherwise repairing numerical state.

## 6. Fit-result records

A fit-result record includes the final parameter arrays, a complete copy of the
user-supplied initial parameter arrays, and a `fit` object. This draft supports
only `initialization.kind == "user_supplied"`. Library-generated initialization
requires a later record version that defines the initializer ID/version,
restart count, selected restart, and PRNG provenance.

The `fit` object contains host metadata:

- `mode`: `converged` or `fixed_steps`;
- the semantic terminal `status` string;
- `iteration_limit` and accepted `n_iter`;
- `converged`, `objective_valid`, `attempted_iteration`,
  `attempted_objective_valid`, `numerical_failure`, and `collapsed`;
- `objective_semantics` from the closed set in Section 7;
- `ridge_application == "post_em_latent_covariance"`; and
- `initialization == {"kind": "user_supplied"}`.

Converged-mode records additionally contain the logical arrays `fit.tol` and
`fit.decrease_tol`; fixed-step records do not. Integer enum values used by an
in-memory implementation MUST be converted to the record's semantic strings.

Every fit record contains these computation-dtype NPY members:

- `fit.objective`, scalar;
- `fit.history`, shape `(H,)`;
- `fit.attempted_objective`, scalar;
- `fit.factor_jitter`, scalar; and
- `fit.covariance_ridge`, scalar.

It also contains `fit.collapsed_components`, boolean shape `(K,)`.

A grouped-general result additionally records `n_samples` and `n_groups` in its
model metadata and contains:

- `fit.informative_weight`, computation-dtype scalar;
- `fit.group_numerical_failure`, boolean shape `(G,)`; and
- `fit.failed_pairs`, boolean shape `(N,K)`.

The grouped record stores its terminal failure-stage string. Fields that cannot
be reconstructed from the fitted parameters alone MUST be retained in the
in-memory fit result before serialization is implemented.

### 6.1 Fit-state invariants

- A valid accepted history is finite, has length `n_iter + 1`, and its final
  element is bitwise equal to `fit.objective`.
- An invalid initial objective has `objective_valid == false`, an empty history,
  `n_iter == 0`, unchanged final/initial parameters, and a nonfinite objective.
- `fit.attempted_objective` is finite exactly when it is marked valid.
- Scalar jitter, ridge, and converged-mode tolerances are finite and
  nonnegative.
- Fixed-step mode never records `converged == true`.
- A successful/converged/max-iteration result has no failure or collapse masks.
- A numerical-failure result has diagnostics consistent with its terminal stage
  and does not present a failed candidate as accepted.
- A component-collapse result has at least one collapsed component and exact
  rollback to the last accepted state.
- Grouped mask shapes agree with the manifest's `N`, `G`, and `K`.
- A contract ID or version cannot be relabeled during load or migration.

## 7. Objective semantics

The manifest uses exactly one of:

- `identity_exact_observed_mean`;
- `identity_fixed_jitter_effective_observed_mean`;
- `general_exact_informative_weighted_observed_mean`; or
- `general_fixed_jitter_effective_informative_weighted_observed_mean`.

The fixed-jitter forms apply when `factor_jitter` is nonzero. Covariance ridge is
not part of the observed objective; `ridge_application` records where it enters
the update.

## 8. Reader validation order and limits

A reader MUST validate before constructing JAX arrays:

1. ordinary-file/container size, ZIP metadata, member count, member names,
   compression, duplication, encryption, declared sizes, and allowed paths;
2. bounded UTF-8 JSON with duplicate-key and nonstandard-constant rejection;
3. exact format, record, and contract ID/version;
4. each NPY member hash, bounded header, descriptor, dtype, order, rank, and
   shape;
5. cross-field model and fit-state invariants; and
6. parameter numerical domains without repair.

Development defaults are:

| Limit | Default |
|---|---:|
| Manifest bytes | 256 KiB |
| NPY header bytes | 16 KiB |
| Members | 32 |
| One uncompressed member | 64 MiB |
| Total uncompressed bytes | 128 MiB |

Larger artifacts require explicit caller-supplied limits. A reader reads members
directly; it MUST NOT extract archive paths to the filesystem. It rejects object,
structured, string, complex, integer parameter, big-endian, Fortran-order, and
unsupported-rank arrays. It never calls pickle, `eval`, or an import named by
the artifact.

## 9. Device boundary

Saving synchronizes arrays and transfers them with `jax.device_get`. A writer
MUST reject a non-fully-addressable multi-host or sharded array rather than
silently writing one process's shard.

Loading validates host arrays first and then places them on the current default
JAX device or one explicit single device. A float64 artifact loaded while JAX
x64 is disabled fails with an actionable precision error before conversion.
CPU-to-GPU and GPU-to-CPU round trips preserve dtype and numerical values; they
do not promise to preserve device identity, sharding, or subsequent computation
bits.

## 10. Version compatibility and migration

Draft `0.x` formats and records support only explicitly listed exact versions.
For a future stable `1.x` format:

- a newer reader supports only older minor versions with an implemented handler;
- an older reader rejects a newer minor version;
- adding a field requires a new record or format minor version;
- a major version requires an explicit migration; and
- migration first decodes and validates the old artifact, then constructs the
  new schema. Changing only labels is not migration.

Unknown contract versions always fail, even when the container/record version is
otherwise recognized.

## 11. Required evidence and private-beta status

The tests-first matrix covers:

1. float32/float64 identity and general-tagged parameter round trips;
2. successful, decrease, collapse, and numerical-failure fit records;
3. nonzero jitter/ridge, stopping controls, weighted objective metadata, and
   exact initialization custody;
4. deterministic container bytes and ZIP/NPY metadata;
5. every unknown/missing/extra/duplicate JSON or ZIP field/member case;
6. path traversal, compression, encryption, object/pickle, structured, complex,
   endian, order, oversize, hash, shape, dtype, and byte-count sentinels;
7. invalid parameter domains and inconsistent fit-state diagnostics;
8. atomic-write and overwrite behavior;
9. float64 rejection with x64 disabled and CPU placement; and
10. a scheduled CPU-to-GPU-to-CPU float32/float64 round trip once controlled GPU
    hardware exists.

The identity and grouped-general fit-result schemas have been aligned with
Section 6. They retain factor jitter, covariance ridge, iteration limit,
converged-mode tolerances, initialization provenance, and the empty-history
invalid-initial-objective rule. General result metadata uses the central
contract registry, and the private implementation uses one central package
version source. A serializer still MUST NOT infer absent fields.

The Python 3.10.11/JAX-JAXlib 0.6.2 CPU serialization/prerequisite lane passes
280/280 tests warning-strict. The supported-Python private-package lane passes
17/17 warning-strict tests on Python 3.12, JAX/JAXlib 0.6.2, NumPy 1.26.4, and
macOS arm64 CPU; it exercises isolated wheel installation and verifies that a
written manifest records `package_version == "0.1.0b1"`. The scheduled
float32/float64 CPU-to-GPU-to-CPU round trip remains unrun, so the device-boundary
requirements are not a GPU compatibility claim.

The documentation-inclusive integrated repository command passes 1,198 tests
warning-strict in 671.48 seconds (0:11:11), with the package fixture directed to
the supported Python 3.12 builder. This does not replace the still-unrun
cross-device gate.
