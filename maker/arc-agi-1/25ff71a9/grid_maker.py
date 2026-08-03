"""
ARC Task: 25ff71a9 (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

SOLAR_ROOT = Path(__file__).resolve().parents[3]
if str(SOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_ROOT))

REARC_ROOT = SOLAR_ROOT / "re-arc"
_rs = str(REARC_ROOT)
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from maker.base_grid_maker import BaseGridMaker

import importlib
for _m in ["utils", "dsl", "generators"]:
    if _m in sys.modules:
        del sys.modules[_m]
from utils import *  # noqa: F401,F403  (unifint, choice, sample, etc.)
from dsl import *    # noqa: F401,F403

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
def sample_colors() -> dict:
    # Force bgc=0 so ARCLE object mode (nonzero = object) cleanly captures only fgc cells.
    cols = [c for c in range(1, 10)]
    bgc = 0
    fgc = random.choice(cols)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if a >= b:
            return a
        r = random.uniform(lb, ub)
        return int(round(a + r * (b - a)))

    h = max(2, unifint(diff_lb, diff_ub, (2, max_h)))
    w = max(2, unifint(diff_lb, diff_ub, (2, max_w)))
    nc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 2 - 1)))

    bounds_set = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(list(bounds_set))
    shp = {start}

    def nbrs(p):
        r, c = p
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)}

    for _ in range(nc - 1):
        cands = set()
        for p in shp:
            cands |= nbrs(p)
        cands = (cands & bounds_set) - shp
        if not cands:
            break
        shp.add(random.choice(list(cands)))

    min_r = min(p[0] for p in shp)
    min_c = min(p[1] for p in shp)
    shp_n = {(p[0] - min_r, p[1] - min_c) for p in shp}
    oh = max(p[0] for p in shp_n) + 1
    ow = max(p[1] for p in shp_n) + 1

    # Keep object off the last row so MoveD stays fully in bounds:
    #   loci + oh - 1 <= h - 2  →  loci <= h - oh - 1
    if h - oh - 1 < 0:
        # Object too tall — clip bottom rows so it fits in h-1 rows
        shp_n = {p for p in shp_n if p[0] < h - 1}
        if not shp_n:
            shp_n = {(0, 0)}
        oh = max(p[0] for p in shp_n) + 1
        ow = max(p[1] for p in shp_n) + 1
    max_loci = max(0, h - oh - 1)
    max_locj = max(0, w - ow)
    loci = random.randint(0, max_loci)
    locj = random.randint(0, max_locj)

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for r, c in shp_n:
        gi[r + loci][c + locj] = fgc
        go[r + loci + 1][c + locj] = fgc

    return {"input": gi, "output": go}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # Background color = most common cell in I
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    # Find bbox of non-bgc cells (the single object to move down)
    mask = (I != bgc)
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())

        # MoveD once — object shifts down 1 row.
        # Generator guarantees r1 <= hi - 2, so this stays in bounds.
        ops.append(21)
        sels.append([r0, c0, r1 - r0, c1 - c0])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
    return ops, sels


# ── GridMaker ─────────────────────────────────────────────────────────────────

class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[
        List[NDArray], List[NDArray],
        List[NDArray], List[NDArray],
        Dict[str, Any],
    ]]:
        num_samples  = kwargs.get("num_samples", 1)
        num_examples = kwargs.get("num_examples", 3)
        max_h, max_w = kwargs.get("max_grid_dim", [30, 30])
        dataset = []

        for _sn in range(num_samples):
            pr_in:  List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in:  List[NDArray] = []
            ex_out: List[NDArray] = []
            ops:  List[int]       = []
            sels: List[List[int]] = []

            # sample color roles once per episode → consistent across all instances
            colors = sample_colors()

            j = 0
            while j < num_examples + 1:
                ok = False
                for _ in range(10):
                    try:
                        r = generate(
                            random.uniform(0.2, 0.5),
                            random.uniform(0.5, 0.8),
                            max_h, max_w,
                            **colors,
                        )
                        I = np.array(r["input"],  dtype=np.uint8)
                        O = np.array(r["output"], dtype=np.uint8)
                        # enforce max_grid_dim — skip oversized grids
                        if I.shape[0] > max_h or I.shape[1] > max_w:
                            continue
                        if O.shape[0] > max_h or O.shape[1] > max_w:
                            continue
                        ok = True
                        break
                    except (IndexError, ValueError, KeyError):
                        continue
                if not ok:
                    j += 1
                    continue
                if j == num_examples:
                    pr_in.append(I)
                    pr_out.append(O)
                    ops, sels = derive_operations(I, O)
                else:
                    ex_in.append(I)
                    ex_out.append(O)
                j += 1

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"25ff71a9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
