"""Cross-check development.convmmd_recipe.fit_convmmd_recipe against J1's records.

Regenerates each of J1's 390 ``kplus1_floor`` datasets draw for draw
(``convmmd_j1_guard.make_dataset`` / ``convmmd_g_common.make_data``), then fits
it two ways:

  j1start  the recipe with J1's exact K + 1 greedy centres (numpy rng stream 21),
           J1's bandwidths (the median heuristic on J1's 2000-row subsample,
           stream 3, when N > 2000), J1's witness candidates (stream 99) and the
           exact curvature check at every P, so each step of the recipe is
           compared with J1's per dataset;
  recipe   the recipe as shipped: jax greedy seeding, the default bandwidth
           subsample and candidates, and Lanczos above P = 80, with
           ``key = PRNGKey(dataset)``. Its losses are not comparable one to one
           with J1's when N > 2000 (different bandwidth rows); recovery is.

Units are taken from a shared queue (file lock), one JSON line per fit. Build the
reference and queue with ``build_reference.py``, launch workers with ``run.sh``,
and read the results with ``summarize.py``.

usage: python scripts/convmmd_j1_crosscheck/crosscheck.py --ref WORK/j1_ref.json --out WORK/results.jsonl
           (--queue WORK/queue.json --counter WORK/counter.txt | --only cell|dataset ... [--modes ...])
"""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "development"))

from development import convmmd_recipe as r  # noqa: E402
from development.convmmd import median_bandwidths  # noqa: E402
from development.convmmd_fit import ConvMMDFitStatus  # noqa: E402
from test_convmmd_recipe import _recovered, _truth_d  # noqa: E402

SEED = 2026102601
SITUATIONS = ("clean", "outliers", "hetero_broad", "hetero_thin", "thin_outliers")
K = 3


def make_truth(situation, dim, dataset):
    w, mu, cov = (np.array(a, float) for a in _truth_d(np.random.default_rng([SEED, dim, dataset]), K, dim))
    if situation == "hetero_broad":
        cov[0] = 6.0 * cov[0]
    elif situation in ("hetero_thin", "thin_outliers"):
        values, vectors = np.linalg.eigh(cov[0])
        values[0] = 0.005
        cov[0] = (vectors * values) @ vectors.T
        cov[0] = 0.5 * (cov[0] + cov[0].T)
    return w, mu, cov


def outlier_points(rng, truth, count):
    _, means, covs = truth
    smax = float(np.sqrt(max(np.linalg.eigvalsh(c).max() for c in covs)))
    out = []
    while len(out) < count:
        direction = rng.standard_normal(means.shape[1])
        point = means[rng.integers(len(means))] + rng.uniform(8, 15) * smax * direction / np.linalg.norm(direction)
        if np.min(np.linalg.norm(means - point, axis=1)) >= 8 * smax:
            out.append(point)
    return np.asarray(out).reshape(count, means.shape[1])


def make_data(rng, truth, n, outliers):
    weights, means, covs = truth
    labels = rng.choice(len(weights), size=n, p=weights)
    z = means[labels] + np.einsum("nij,nj->ni", np.linalg.cholesky(covs)[labels], rng.standard_normal((n, means.shape[1])))
    dim = z.shape[1]
    variances = rng.uniform(0.05, 0.2, size=(n, dim))
    e = np.sqrt(variances) * rng.standard_normal((n, dim))
    x = z + e
    if outliers:
        rows = rng.choice(n, size=int(round(outliers * n)), replace=False)
        x[rows] = outlier_points(rng, truth, len(rows)) + e[rows]
    s = np.zeros((n, dim, dim))
    s[:, np.arange(dim), np.arange(dim)] = variances
    return x, s


def greedy_np(rng, x, k, trials):
    """convmmd_exp_i2_init.greedy_kmeanspp, draw for draw."""

    def total(d2, i):
        return float(np.minimum(d2, ((x - x[i]) ** 2).sum(axis=1)).sum())

    d2 = np.full(len(x), np.inf)
    centers = []
    for j in range(k):
        cand = rng.integers(len(x), size=trials) if j == 0 else rng.choice(len(x), size=trials, p=d2 / d2.sum())
        best = min(cand, key=lambda i: total(d2, i))
        centers.append(x[best])
        d2 = np.minimum(d2, ((x - x[best]) ** 2).sum(axis=1))
    return np.asarray(centers)


def full_median_distance(x, chunk=500):
    parts = []
    for i in range(0, len(x), chunk):
        block = np.sqrt(((x[i:i + chunk, None, :] - x[None, :, :]) ** 2).sum(-1))
        rows = np.arange(i, min(i + chunk, len(x)))
        parts.append(block[np.arange(len(rows))[:, None] < (np.arange(len(x))[None, :] - i)].astype(np.float64))
    return float(np.median(np.concatenate(parts)))


def dataset(ref):
    sit, dim, n, r_ = ref["situation"], ref["dim"], ref["n"], ref["dataset"]
    truth = make_truth(sit, dim, r_)

    def rng_of(stream):
        return np.random.default_rng([SEED, 1 + SITUATIONS.index(sit), dim, n, r_, stream])

    x, s = make_data(rng_of(1), truth, n, 0.05 if sit in ("outliers", "thin_outliers") else 0.0)
    rows = np.arange(n) if n <= 2000 else np.sort(rng_of(3).choice(n, size=2000, replace=False))
    bw = np.asarray(median_bandwidths(jnp.asarray(x[rows])))
    centres = greedy_np(rng_of(21), x, K + 1, 10)
    return truth, x, s, bw, centres


def truth_kappa(truth):
    return float(min(np.linalg.eigvalsh(c).min() / np.linalg.eigvalsh(c).max() for c in truth[2]))


def j1_candidates(ref):
    n = ref["n"]
    if n <= 2000:
        return np.arange(n)
    rng = np.random.default_rng([SEED, 99, ref["dim"], n, ref["dataset"]])
    return np.sort(rng.choice(n, size=2000, replace=False))


def _f(a):
    return None if a is None else float(a)


def fit(mode, ref, truth, x, s, bw, centres):
    """j1start: J1's start, bandwidths, candidates and exact curvature everywhere; recipe: as shipped."""
    originals = (r.greedy_kmeanspp, r.witness_candidates)
    kwargs = {}
    if mode == "j1start":
        cand = jnp.asarray(j1_candidates(ref))
        r.greedy_kmeanspp = lambda key, x_, k, trials=10: jnp.asarray(centres)
        r.witness_candidates = lambda key, n, size=2000: cand
        kwargs = {"bandwidths": jnp.asarray(bw), "exact_limit": 10**6}
    try:
        t0 = time.time()
        res = r.fit_convmmd_recipe(jnp.asarray(x), jnp.asarray(s), K, key=jax.random.PRNGKey(ref["dataset"]), **kwargs)
        loss = float(res.loss)
        seconds = time.time() - t0
    finally:
        r.greedy_kmeanspp, r.witness_candidates = originals
    params = tuple(np.asarray(a) for a in res.parameters)
    b = res.base
    moves = []
    for m in res.repairs:
        moves.append({"raw_peak": float(m.witness.raw_peak), "raw_row": int(m.witness.raw_row), "t": float(m.witness.t),
                      "F": bool(m.fired_F), "Z": bool(m.fired_Z), "H": bool(m.fired_H), "attempted": bool(m.attempted),
                      "accepted": bool(m.accepted), "kept": bool(m.kept), "z": float(m.z), "kappa_before": float(m.kappa_before),
                      "kappa_after": float(m.kappa_after), "loss_drop": float(m.loss_drop),
                      "fit_loss": _f(m.fit.loss) if m.fit is not None else None,
                      "fit_steps": int(m.fit.n_iter) if m.fit is not None else None,
                      "fit_escapes": int(m.fit.escapes) if m.fit is not None else None})
    base_params = tuple(np.asarray(a) for a in b.parameters)
    return {"loss": loss, "n_iter": int(res.n_iter), "status": ConvMMDFitStatus(int(res.status)).name,
            "converged": bool(res.converged), "max_abs_grad": float(res.max_abs_grad), "floor": float(res.covariance_floor),
            "b": float(res.health.b), "kappa": float(res.health.kappa), "flag": bool(res.collapsed),
            "saddle": bool(res.saddle), "escapes": int(res.escapes), "min_eigenvalue": float(res.min_eigenvalue),
            "init_status": int(res.initialization_status), "recovered": bool(_recovered(params, truth)[0]),
            "weights": params[0].tolist(), "means": params[1].tolist(), "covariances": params[2].tolist(),
            "total_steps": int(res.total_steps), "seconds": seconds, "bw_used_median": float(res.bandwidths[len(res.bandwidths) // 2]),
            "base": {"loss": float(b.loss), "n_iter": int(b.n_iter), "escapes": int(b.escapes), "saddle": bool(b.saddle),
                     "min_eigenvalue": float(b.min_eigenvalue), "converged": bool(b.converged),
                     "recovered": bool(_recovered(base_params, truth)[0]), "checks": int(b.curvature_checks)},
            "moves": moves}


def next_unit(queue_path, counter_path):
    with open(counter_path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        text = fh.read().strip()
        i = int(text) if text else 0
        fh.seek(0)
        fh.truncate()
        fh.write(str(i + 1))
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)
    units = json.load(open(queue_path))
    return units[i] if i < len(units) else None


def run_unit(unit, refs, out):
    ref = refs[unit["key"]]
    truth, x, s, bw, centres = dataset(ref)
    record = {"key": unit["key"], "mode": unit["mode"], "cell": ref["cell"], "dataset": ref["dataset"],
              "truth_kappa": truth_kappa(truth), "bw_median": float(bw[len(bw) // 2])}
    if unit["mode"] == "j1start" and ref["n"] > 2000:
        record["full_median_distance"] = full_median_distance(x)
    try:
        record.update(fit(unit["mode"], ref, truth, x, s, bw, centres))
    except Exception as exc:  # report and continue
        record["error"] = repr(exc)
    with open(out, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(record) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--queue")
    ap.add_argument("--counter")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--modes", nargs="*", default=["j1start", "recipe"])
    a = ap.parse_args()
    refs = json.load(open(a.ref))
    if a.only:
        for key in a.only:
            for mode in a.modes:
                run_unit({"key": key, "mode": mode}, refs, a.out)
        return
    while True:
        unit = next_unit(a.queue, a.counter)
        if unit is None:
            break
        run_unit(unit, refs, a.out)


if __name__ == "__main__":
    main()
