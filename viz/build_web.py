#!/usr/bin/env python3
"""Emit the data the GitHub Pages viewer reads.

The published dataset is private and its arrays are byte blobs, so a browser
cannot read it: the page needs its own copy. This writes one small JSON per task
plus an index, so opening the page costs a few kB and picking a task costs one
more request — rather than shipping a single multi-megabyte bundle.

Grids are written as strings of digits, one per row, cropped to their real
extent. That is about as compact as JSON gets without base64, and it stays
readable if anyone opens the file.

    python viz/build_web.py --root /hdd_data/yunho/ARC_best10_r4/whole
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(grid, h, w) -> list[str]:
    g = np.asarray(grid, dtype=int)[:h, :w]
    return ["".join(str(int(c)) for c in row) for row in g]


def mask_rows(mask, h, w) -> list[str]:
    m = np.asarray(mask, dtype=int)[:h, :w]
    return ["".join("1" if c else "0" for c in row) for row in m]


def build_task(path: Path) -> dict:
    d = json.loads(path.read_text())
    ops = d["operation_name"]
    n = len(ops)
    states = []
    for i in range(n + 1):
        h, w = d["grid_dim"][i]
        states.append({"h": int(h), "w": int(w), "g": rows(d["grid"][i], h, w)})
    sels = []
    for i in range(n):
        h, w = d["grid_dim"][i]
        sels.append(mask_rows(d["selection_mask"][i], h, w))
    # what the environment holds selected after the step: for object ops this is
    # the object at its new place, which is what should stay outlined
    after = []
    for i in range(1, n + 1):
        h, w = d["grid_dim"][i]
        m = np.asarray(d["selected"][i], dtype=bool)[:h, :w]
        after.append(mask_rows(m, h, w) if m.any() else None)

    ex = []
    for i in range(len(d["ex_in"])):
        ih, iw = d["ex_in_grid_dim"][i]
        oh, ow = d["ex_out_grid_dim"][i]
        ex.append({"in": {"h": int(ih), "w": int(iw), "g": rows(d["ex_in"][i], ih, iw)},
                   "out": {"h": int(oh), "w": int(ow), "g": rows(d["ex_out"][i], oh, ow)}})
    return {"id": d.get("desc", {}).get("id", path.stem), "ops": ops,
            "states": states, "sel": sels, "after": after, "ex": ex}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hdd_data/yunho/ARC_best10_r4/whole")
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--per_task", type=int, default=1,
                    help="trajectories kept per task (the page shows one at a time)")
    args = ap.parse_args()

    out = Path(args.out)
    (out).mkdir(parents=True, exist_ok=True)
    index = []
    total = 0
    for folder in sorted(Path(args.root).iterdir()):
        if not folder.is_dir():
            continue
        task = folder.name.split(".")[1]
        files = sorted(folder.glob("*.json"))[:args.per_task]
        episodes = [build_task(f) for f in files]
        p = out / f"{task}.json"
        p.write_text(json.dumps(episodes, separators=(",", ":")))
        total += p.stat().st_size
        index.append({"task": task, "n": len(episodes[0]["ops"]),
                      "ops": episodes[0]["ops"]})
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"{len(index)} tasks -> {out}  ({total / 1e6:.1f} MB, "
          f"index {(out / 'index.json').stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
