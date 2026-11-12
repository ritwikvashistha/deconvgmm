"""Build the J1 reference file and work queue for the convMMD recipe cross-check.

Reads J1's per-dataset records for the adopted start (K + 1 start, covariance
floor on; ``j1_<cell>_r<dataset>_kplus1_floor.json``, 390 datasets) and writes
into ``--out``:

  j1_ref.json  per dataset: the regeneration parameters, J1's saddle-aware base
               fit and its 'none' row (no repair), its ``HU Kkz`` result (the
               adopted recipe) and its repair chain, move by move;
  queue.json   one unit per (mode, dataset), modes ``j1start`` then ``recipe``,
               each slowest-first by J1's HU-Kkz time so the long fits start early.

This script only reads JSON; it needs no JAX. J1's records are the
maintainer's unpublished campaign output and are not in this repository, so
outside the maintainer's checkout ``--records`` must point at a copy of them.
The cross-check's own results are published as ``results_20261111.jsonl.gz``
next to this script; ``summarize.py`` reads them, given the reference file.

usage: python scripts/convmmd_j1_crosscheck/build_reference.py --out WORK_DIR [--records DIR]
"""

import argparse
import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORDS = REPO / "recovered-artifacts" / "convmmd_b1" / "results" / "convmmd_g" / "j_20261025" / "j1"
MODES = ("j1start", "recipe")


def chain_move(move):
    fit = move.get("fit") or {}
    stats = move["stats"]
    return {"raw_peak": stats["raw_peak"], "raw_row": stats["raw_row"], "t": stats["t"],
            "F": move["fired_F"], "Z": move["fired_Z"], "H": move["fired_H"], "attempted": move["attempted"],
            "accepted": move.get("accepted"), "z": move.get("z"), "kappa_before": move.get("kappa_before"),
            "kappa_after": move.get("kappa_after"), "loss_drop": move.get("loss_drop"),
            "fit_loss": fit.get("loss"), "fit_steps": fit.get("steps"), "fit_escapes": fit.get("escapes")}


def reference(records):
    out = {}
    for path in sorted(glob.glob(os.path.join(records, "j1_*_kplus1_floor.json"))):
        with open(path) as fh:
            d = json.load(fh)
        p = d["task"]["params"]
        rows = {(r["trigger"], r["keep"]): r for r in d["rows"]}
        none, hu = rows[("none", "-")], rows[("HU", "Kkz")]
        out[f"{p['cell']}|{p['dataset']}"] = {
            "cell": p["cell"], "situation": p["situation"], "dim": p["dim"], "n": p["n"], "dataset": p["dataset"],
            "floor_value": none["floor_value"], "truth_kappa": none["truth_kappa"],
            "base": {k: d["base"][k] for k in ("steps", "loss", "max_abs_grad", "status", "escapes", "min_eigenvalue", "seconds")},
            "init_status": none["init_status"], "recovered": none["recovered"], "health": none["health"],
            "collapsed": none["collapsed"], "weights": none["weights"], "means": none["means"],
            "covariances": none["covariances"],
            "hukkz": {k: hu[k] for k in ("recovered", "loss", "accepted", "attempted", "total_steps", "status", "health",
                                         "collapsed", "seconds_per_fit", "weights", "means", "covariances")},
            "chain": [chain_move(m) for m in d["chain"]],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(RECORDS), help="directory of J1's j1_*_kplus1_floor.json records")
    ap.add_argument("--out", required=True, help="work directory for j1_ref.json and queue.json")
    a = ap.parse_args()
    ref = reference(a.records)
    if not ref:
        raise SystemExit(f"no j1_*_kplus1_floor.json records under {a.records}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "j1_ref.json"), "w") as fh:
        json.dump(ref, fh)
    slowest = sorted(ref, key=lambda k: -ref[k]["hukkz"]["seconds_per_fit"])
    units = [{"key": k, "mode": mode} for mode in MODES for k in slowest]
    with open(os.path.join(a.out, "queue.json"), "w") as fh:
        json.dump(units, fh)
    total = sum(r["hukkz"]["seconds_per_fit"] for r in ref.values())
    print(f"{len(ref)} datasets, {len(units)} units; J1 HU-Kkz core-seconds {total:.0f}")


if __name__ == "__main__":
    main()
