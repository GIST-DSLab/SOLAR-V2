"""
ARC Task: b6afb2da (RE-ARC) — LLM-generated grid_maker
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
import random
import numpy as np


def sample_colors() -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 4)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 4)]

    h_lo = max(4, min(10, max_h))
    h_hi = min(30, max_h)
    w_lo = max(4, min(10, max_w))
    w_hi = min(30, max_w)
    if h_lo > h_hi:
        h_lo = h_hi
    if w_lo > w_hi:
        w_lo = w_hi
    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)

    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    num = random.randint(1, 9)
    indss = set((i, j) for i in range(h) for j in range(w))

    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if not remcols or not indss:
            break
        oh_hi = min(7, h - 1)
        ow_hi = min(7, w - 1)
        if oh_hi < 3 or ow_hi < 3:
            break
        oh = random.randint(3, oh_hi)
        ow = random.randint(3, ow_hi)
        subs = [(i, j) for (i, j) in indss if i < h - oh and j < w - ow]
        if not subs:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        bd = set((loci + di, locj + dj) for di in range(oh) for dj in range(ow))
        col = random.choice(remcols)
        if bd.issubset(indss):
            remcols.remove(col)
            for (i, j) in bd:
                gi[i][j] = col
                go[i][j] = 2
            for j in range(locj, locj + ow):
                go[loci][j] = 4
                go[loci + oh - 1][j] = 4
            for i in range(loci, loci + oh):
                go[i][locj] = 4
                go[i][locj + ow - 1] = 4
            go[loci][locj] = 1
            go[loci][locj + ow - 1] = 1
            go[loci + oh - 1][locj] = 1
            go[loci + oh - 1][locj + ow - 1] = 1
            succ += 1
            indss -= bd
        tr += 1

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    same_mask = (I == O)
    if same_mask.any():
        bgc = int(I[same_mask].flat[0])
    else:
        vals, counts = np.unique(I, return_counts=True)
        bgc = int(vals[np.argmax(counts)])

    visited = np.zeros_like(I, dtype=bool)
    rects = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not visited[r, c]:
                color = int(I[r, c])
                stack = [(r, c)]
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    if rr < 0 or rr >= hi or cc < 0 or cc >= wi:
                        continue
                    if visited[rr, cc]:
                        continue
                    if int(I[rr, cc]) != color:
                        continue
                    visited[rr, cc] = True
                    cells.append((rr, cc))
                    stack.extend([(rr + 1, cc), (rr - 1, cc),
                                  (rr, cc + 1), (rr, cc - 1)])
                if cells:
                    rs = [x[0] for x in cells]
                    cs = [x[1] for x in cells]
                    r0, c0 = min(rs), min(cs)
                    h0 = max(rs) - r0 + 1
                    w0 = max(cs) - c0 + 1
                    rects.append((r0, c0, h0, w0))

    ops, sels = [], []
    for (r, c, h, w) in rects:
        ops.append(2); sels.append([r, c, h - 1, w - 1])
        ops.append(4); sels.append([r, c, 0, w - 1])
        if h > 1:
            ops.append(4); sels.append([r + h - 1, c, 0, w - 1])
        if h > 1:
            ops.append(4); sels.append([r, c, h - 1, 0])
        if w > 1:
            ops.append(4); sels.append([r, c + w - 1, h - 1, 0])
        ops.append(1); sels.append([r, c, 0, 0])
        if w > 1:
            ops.append(1); sels.append([r, c + w - 1, 0, 0])
        if h > 1:
            ops.append(1); sels.append([r + h - 1, c, 0, 0])
        if h > 1 and w > 1:
            ops.append(1); sels.append([r + h - 1, c + w - 1, 0, 0])

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
                "id":         f"b6afb2da-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
