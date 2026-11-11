"""Summarize cross-check fits against J1's HU-Kkz results and base fits.

Prints, per mode:

* a per-cell table: how many fits match J1's base loss (to 1e-9), base step
  count and escapes, repair attempts and kept moves, and final loss (bucketed
  < 1e-9, < 1e-6, >= 1e-6); recovered fits against J1's; health flags against
  J1's; mean seconds per fit against J1's;
* totals, and recovery by situation group (clean / hetero / outliers, where
  outliers includes thin_outliers) with the base-fit recovery before repair;
* with ``-v``, every disagreement (final loss off by >= 1e-6, recovery,
  attempted or kept count differs, or an unescaped saddle), move by move next to
  J1's chain.

This script only reads JSON; it needs no JAX.

usage: python scripts/convmmd_j1_crosscheck/summarize.py WORK/j1_ref.json WORK/results.jsonl[.gz] [-v]
"""

import collections
import gzip
import json
import sys

COLUMNS = ("n", "base=", "steps=", "esc=", "att=", "kept=", "fin<1e-9", "fin<1e-6", "fin>=1e-6", "rec", "J1rec",
           "rec!=", "flag", "J1flag")


def group(situation):
    return "outliers" if "outliers" in situation else ("hetero" if "hetero" in situation else "clean")


def main():
    args = [a for a in sys.argv[1:] if a != "-v"]
    verbose = "-v" in sys.argv[1:]
    with open(args[0]) as fh:
        ref = json.load(fh)
    with (gzip.open(args[1], "rt") if args[1].endswith(".gz") else open(args[1])) as fh:
        fits = [json.loads(line) for line in fh if line.strip()]
    errors = [f for f in fits if "error" in f]
    fits = [f for f in fits if "error" not in f]
    print(f"fits {len(fits)}  errors {len(errors)}")
    for f in errors[:5]:
        print("  ERROR", f["key"], f["mode"], f["error"][:200])

    by_cell = collections.defaultdict(list)
    for f in fits:
        by_cell[(f["mode"], ref[f["key"]]["cell"])].append(f)

    print(f"{'mode':7s} {'cell':24s} " + " ".join(f"{c:>8s}" for c in COLUMNS + ('sec', 'J1sec')))
    totals = collections.defaultdict(collections.Counter)
    groups = collections.defaultdict(collections.Counter)
    odd = []
    for (mode, cell), cell_fits in sorted(by_cell.items()):
        c = collections.Counter()
        sec = j1sec = 0.0
        for f in cell_fits:
            r = ref[f["key"]]
            hu, base = r["hukkz"], r["base"]
            att = sum(m["attempted"] for m in f["moves"])
            kept = sum(m["kept"] for m in f["moves"])
            dfin = abs(f["loss"] - hu["loss"])
            c["n"] += 1
            c["base="] += abs(f["base"]["loss"] - base["loss"]) < 1e-9
            c["steps="] += f["base"]["n_iter"] == base["steps"]
            c["esc="] += f["base"]["escapes"] == base["escapes"]
            c["att="] += att == hu["attempted"]
            c["kept="] += kept == hu["accepted"]
            c["fin<1e-9"] += dfin < 1e-9
            c["fin<1e-6"] += 1e-9 <= dfin < 1e-6
            c["fin>=1e-6"] += dfin >= 1e-6
            c["rec"] += f["recovered"]
            c["J1rec"] += hu["recovered"]
            c["rec!="] += f["recovered"] != hu["recovered"]
            c["flag"] += f["flag"]
            c["J1flag"] += hu["health"]["fired"]
            sec += f["seconds"]
            j1sec += hu["seconds_per_fit"]
            g = groups[(mode, group(r["situation"]))]
            g["n"] += 1
            g["base rec"] += f["base"]["recovered"]
            g["rec"] += f["recovered"]
            g["J1rec"] += hu["recovered"]
            g["kept"] += kept
            g["saddle"] += f["saddle"]
            if (dfin >= 1e-6 or f["recovered"] != hu["recovered"] or att != hu["attempted"]
                    or kept != hu["accepted"] or f["saddle"]):
                odd.append((mode, f, r, att, kept))
        for name, value in c.items():
            totals[mode][name] += value
        totals[mode]["sec"] += sec
        totals[mode]["J1sec"] += j1sec
        print(f"{mode:7s} {cell:24s} " + " ".join(f"{c[k]:8d}" for k in COLUMNS)
              + f" {sec / len(cell_fits):8.0f} {j1sec / len(cell_fits):8.0f}")
    for mode, t in totals.items():
        print(mode, "TOTAL", {k: (round(v) if isinstance(v, float) else v) for k, v in t.items()})

    print("\nrecovery by group (n, base fit recovered, final recovered, J1 final recovered, kept moves, unescaped saddles):")
    for (mode, name), g in sorted(groups.items()):
        print(f"  {mode:7s} {name:8s} n {g['n']:3d}  base {g['base rec']:3d}  final {g['rec']:3d}  J1 {g['J1rec']:3d}  "
              f"kept {g['kept']:2d}  saddle {g['saddle']}")

    flips = [(mode, f["key"], f["recovered"]) for mode, f, r, _, _ in odd if f["recovered"] != r["hukkz"]["recovered"]]
    print("\nrecovery differs from J1 (mode, dataset, recovered here):")
    for flip in sorted(flips):
        print("  ", *flip)

    if not verbose:
        print(f"\n{len(odd)} disagreements; -v lists them")
        return
    print("\ndisagreements (|final dloss| >= 1e-6, recovery, attempted or kept count differs, or saddle):")
    for mode, f, r, att, kept in sorted(odd, key=lambda o: (o[0], o[1]["key"])):
        hu = r["hukkz"]
        print(f"  {mode:7s} {f['key']:28s} base d={f['base']['loss'] - r['base']['loss']:+.1e} "
              f"steps {f['base']['n_iter']}/{r['base']['steps']} esc {f['base']['escapes']}/{r['base']['escapes']} | "
              f"att {att}/{hu['attempted']} kept {kept}/{hu['accepted']} | final d={f['loss'] - hu['loss']:+.1e} "
              f"rec {f['recovered']}/{hu['recovered']} flag {f['flag']}/{hu['health']['fired']} "
              f"saddle={f['saddle']} lam={f['min_eigenvalue']:+.1e}")
        for i, m in enumerate(f["moves"]):
            j = r["chain"][i] if i < len(r["chain"]) else None
            jt = (f" | J1 F{int(j['F'])}Z{int(j['Z'])}H{int(j['H'])} raw={j['raw_peak']:.2e} row={j['raw_row']} "
                  f"z={j['z']} acc={j['accepted']}" if j else "")
            print(f"      move{i}: F{int(m['F'])}Z{int(m['Z'])}H{int(m['H'])} raw={m['raw_peak']:.2e} row={m['raw_row']} "
                  f"t={m['t']:.2f} att={m['attempted']} acc={m['accepted']} kept={m['kept']} z={m['z']:.2f} "
                  f"drop={m['loss_drop']:.1e}{jt}")


if __name__ == "__main__":
    main()
