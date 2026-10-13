#!/usr/bin/env python3
"""Attribution-guided pruning (Section 7, "Acting on an attribution").

For every operator i of every configuration, compares the within-run improvement of
the full coalition, v(N) = f0 - f_N, with that of the coalition in which operator i is
replaced by its Definition-2 neutral, v(N \\ {i}), at the same budget and over the same
common-seeded runs that the exact-coalition Shapley values use. The quantity reported is
the fraction of v(N) lost by removing the operator,

    loss_i = (v(N) - v(N \\ {i})) / v(N)   (in %; negative = removing it helps).

No new run is needed: v(N \\ {i}) is one of the 2^n coalition values already stored.
Reads reproduce/data/{exact_shapley_results,shade_shapley_results,jade_shapley_results}.json
and writes reproduce/data/pruning_gain.json.
Usage: python experiments/pruning_gain.py
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reproduce" / "data"
FILES = ["exact_shapley_results.json", "shade_shapley_results.json", "jade_shapley_results.json"]


def losses():
    out = {}
    for fname in FILES:
        d = json.load(open(DATA / fname))
        for alg in d:
            for fn in d[alg]:
                for dim in d[alg][fn]:
                    e = d[alg][fn][dim]
                    ops = e["operators"]
                    co = {}
                    for k, v in e["coalitions"].items():
                        idx = tuple(sorted(int(x) for x in k.strip("()").split(",") if x.strip()))
                        co[idx] = np.asarray(v, float)
                    full = tuple(range(len(ops)))
                    f0 = co[()]
                    vN = (f0 - co[full]).mean()
                    for i, op in enumerate(ops):
                        S = tuple(j for j in full if j != i)
                        vS = (f0 - co[S]).mean()
                        loss = 100.0 * (vN - vS) / vN
                        out.setdefault(alg, {}).setdefault(op, []).append(
                            {"function": fn, "dimension": int(dim), "loss_pct": float(loss)})
    return out


def summarize(out):
    rows = []
    for alg in out:
        for op, cells in out[alg].items():
            g = np.array([c["loss_pct"] for c in cells])
            worst = max(cells, key=lambda c: c["loss_pct"])
            rows.append({
                "algorithm": alg, "operator": op, "n": len(g),
                "median_loss_pct": float(np.median(g)),
                "n_loss_le_1pct": int((g <= 1).sum()),
                "n_loss_gt_5pct": int((g > 5).sum()),
                "n_gain": int((g < 0).sum()),
                "max_loss_pct": float(g.max()),
                "max_loss_at": f"{worst['function']} D={worst['dimension']}",
                "cells_gt_5pct": [f"{c['function']} D={c['dimension']} ({c['loss_pct']:.1f})"
                                   for c in sorted(cells, key=lambda c: -c["loss_pct"]) if c["loss_pct"] > 5],
            })
    return rows


if __name__ == "__main__":
    out = losses()
    rows = summarize(out)
    json.dump({"cells": out, "summary": rows}, open(DATA / "pruning_gain.json", "w"), indent=1)
    print(f"{'alg':6s} {'operator':34s} {'median':>7s} {'<=1%':>5s} {'>5%':>4s} {'gain':>5s} {'max':>7s}  at")
    for r in rows:
        print(f"{r['algorithm']:6s} {r['operator']:34s} {r['median_loss_pct']:+7.1f} {r['n_loss_le_1pct']:5d} "
              f"{r['n_loss_gt_5pct']:4d} {r['n_gain']:5d} {r['max_loss_pct']:+7.1f}  {r['max_loss_at']}")
    for r in rows:
        if r["cells_gt_5pct"] and r["n_loss_gt_5pct"] <= 8:
            print(f"  {r['algorithm']} {r['operator']} >5%: {', '.join(r['cells_gt_5pct'])}")
