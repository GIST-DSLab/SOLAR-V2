"""
ARC Task: bdad9b1f (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
import random


def sample_colors() -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if a >= b:
            return a
        return round(a + (b - a) * random.uniform(lb, ub))

    cols = [c for c in range(10) if c != 4]
    if bgc not in cols:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]

    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    numh = unifint(diff_lb, diff_ub, (1, max(1, h // 2 - 1)))
    numw = unifint(diff_lb, diff_ub, (1, max(1, w // 2 - 1)))

    h_cand = list(range(2, h - 1))
    w_cand = list(range(2, w - 1))
    numh = max(0, min(numh, len(h_cand)))
    numw = max(0, min(numw, len(w_cand)))
    hlocs = random.sample(h_cand, numh) if numh > 0 else []
    wlocs = random.sample(w_cand, numw) if numw > 0 else []

    numcols = unifint(diff_lb, diff_ub, (2, min(8, len(remcols))))
    numcols = max(2, min(numcols, len(remcols)))
    ccols = random.sample(remcols, numcols)

    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)

    fc = -1
    for ii in sorted(hlocs):
        avail = [c for c in ccols if c != fc] or list(ccols)
        col = random.choice(avail)
        fc = col
        objw = random.randint(2, ii)
        gi[ii, 0:objw] = col
        go[ii, 0:w] = col

    fc = -1
    for jj in sorted(wlocs):
        avail = [c for c in ccols if c != fc] or list(ccols)
        col = random.choice(avail)
        fc = col
        objh = random.randint(2, jj)
        gi[0:objh, jj] = col
        go[0:h, jj] = col

    for ii in hlocs:
        for jj in wlocs:
            go[ii, jj] = 4

    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k=k)
    go = np.rot90(go, k=k)

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    visited = np.zeros_like(I, dtype=bool)
    hlines = []
    vlines = []

    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or I[r, c] == bgc:
                continue
            color = int(I[r, c])
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                if 0 <= y < hi and 0 <= x < wi and not visited[y, x] and I[y, x] == color:
                    visited[y, x] = True
                    cells.append((y, x))
                    stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
            rows_set = set(y for y, _ in cells)
            cols_set = set(x for _, x in cells)
            if len(rows_set) == 1 and len(cols_set) >= 2:
                hlines.append((cells[0][0], color))
            elif len(cols_set) == 1 and len(rows_set) >= 2:
                vlines.append((cells[0][1], color))

    ops = []
    sels = []

    for r, c in hlines:
        ops.append(int(c))
        sels.append([r, 0, 0, wi - 1])

    for j, c in vlines:
        ops.append(int(c))
        sels.append([0, j, hi - 1, 0])

    for r, _ in hlines:
        for j, _ in vlines:
            ops.append(4)
            sels.append([r, j, 0, 0])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                "id":         f"bdad9b1f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
