#!/usr/bin/env python3
"""Emit the data the GitHub Pages viewer reads.

The published dataset is private and its arrays are byte blobs, so a browser
cannot read it: the page needs its own copy. This writes one JSON per task
holding all ten episodes, plus an index and a thumbnail file the task grid
paints from.

Two encodings keep that affordable. Grids are written as a full first state and
then per-step changed cells, because most operations touch a handful of them.
Selections are written as coordinate lists rather than full masks, because a
selection is usually a small object in a large grid. Storing both in full made
them 78% of the payload and put ten episodes per task at 58 MB; this brings the
same content in at roughly a third of that.

    python viz/build_web.py --root /hdd_data/yunho/ARC_best10_r4/whole
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(g) -> list[str]:
    return ["".join(str(int(c)) for c in row) for row in g]


def crop(grid, h, w) -> np.ndarray:
    return np.asarray(grid, dtype=int)[:h, :w]


def cells(mask) -> list[list[int]]:
    rr, cc = np.nonzero(np.asarray(mask, dtype=bool))
    return [[int(r), int(c)] for r, c in zip(rr, cc)]


def build_episode(path: Path) -> dict:
    d = json.loads(path.read_text())
    ops = d["operation_name"]
    n = len(ops)

    states, prev = [], None
    for i in range(n + 1):
        h, w = d["grid_dim"][i]
        g = crop(d["grid"][i], h, w)
        if prev is not None and prev.shape == g.shape:
            rr, cc = np.nonzero(prev != g)
            states.append({"h": int(h), "w": int(w),
                           "d": [[int(r), int(c), int(g[r, c])] for r, c in zip(rr, cc)]})
        else:
            states.append({"h": int(h), "w": int(w), "g": rows(g)})
        prev = g

    sel = []
    for i in range(n):
        h, w = d["grid_dim"][i]
        sel.append(cells(crop(d["selection_mask"][i], h, w)))
    after = []
    for i in range(1, n + 1):
        h, w = d["grid_dim"][i]
        m = crop(d["selected"][i], h, w).astype(bool)
        after.append(cells(m) if m.any() else None)

    ex = []
    for i in range(len(d["ex_in"])):
        ih, iw = d["ex_in_grid_dim"][i]
        oh, ow = d["ex_out_grid_dim"][i]
        ex.append({"in": {"h": int(ih), "w": int(iw), "g": rows(crop(d["ex_in"][i], ih, iw))},
                   "out": {"h": int(oh), "w": int(ow), "g": rows(crop(d["ex_out"][i], oh, ow))}})

    return {"id": d.get("desc", {}).get("id", path.stem), "ops": ops,
            "states": states, "sel": sel, "after": after, "ex": ex}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hdd_data/yunho/ARC_best10_r4/whole")
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--per_task", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.json"):
        stale.unlink()

    index, thumbs, total = [], {}, 0
    for folder in sorted(Path(args.root).iterdir()):
        if not folder.is_dir():
            continue
        task = folder.name.split(".")[1]
        eps = [build_episode(f) for f in sorted(folder.glob("*.json"))[:args.per_task]]
        p = out / f"{task}.json"
        p.write_text(json.dumps(eps, separators=(",", ":")))
        total += p.stat().st_size
        index.append({"task": task, "eps": len(eps), "n": len(eps[0]["ops"]),
                      "ops": eps[0]["ops"]})
        # the grid painted on the task card: the first demonstration pair, which
        # states the rule without needing the episode file
        thumbs[task] = eps[0]["ex"][0]

    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    (out / "thumbs.json").write_text(json.dumps(thumbs, separators=(",", ":")))
    print(f"{len(index)} tasks x {args.per_task} episodes -> {out}"
          f"  ({total / 1e6:.0f} MB, index {(out/'index.json').stat().st_size/1e3:.0f} kB,"
          f" thumbs {(out/'thumbs.json').stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
