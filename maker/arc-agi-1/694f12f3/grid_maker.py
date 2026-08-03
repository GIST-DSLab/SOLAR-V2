"""
ARC Task: 694f12f3 (RE-ARC) — LLM-generated grid_maker
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
    import random
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc) -> dict:
    import numpy as np
    import random

    max_dim = min(max_h, max_w)
    ub = min(30, max_dim)
    if ub < 9:
        ub = 9
    lb = 9

    h = random.randint(lb, ub)
    w = random.randint(lb, ub)

    seploc = random.randint(4, h - 5)
    bigh = random.randint(4, seploc)
    bigw = random.randint(3, w - 1)
    bigloci = random.randint(0, seploc - bigh)
    biglocj = random.randint(0, w - bigw)

    smallmaxh = h - seploc - 1
    smallmaxw = w - 1

    cands = []
    bigsize = bigh * bigw
    for a in range(3, smallmaxh + 1):
        for b in range(3, smallmaxw + 1):
            if a * b < bigsize:
                cands.append((a, b))

    if not cands:
        smallh, smallw = 3, 3
    else:
        cands.sort(key=lambda ab: ab[0] * ab[1])
        idx = random.randint(0, len(cands) - 1)
        smallh, smallw = cands[idx]

    smallloci = random.randint(seploc + 1, h - smallh)
    smalllocj = random.randint(0, w - smallw)

    gi = np.full((h, w), bgc, dtype=int)
    gi[bigloci:bigloci + bigh, biglocj:biglocj + bigw] = sqc
    gi[smallloci:smallloci + smallh, smalllocj:smalllocj + smallw] = sqc

    go = gi.copy()
    go[bigloci + 1:bigloci + bigh - 1, biglocj + 1:biglocj + bigw - 1] = 2
    go[smallloci + 1:smallloci + smallh - 1, smalllocj + 1:smalllocj + smallw - 1] = 1

    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k=k)
    go = np.rot90(go, k=k)

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[counts.argmax()])

    visited = np.zeros_like(I, dtype=bool)
    rects = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not visited[r, c]:
                color = int(I[r, c])
                stack = [(r, c)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= hi or x < 0 or x >= wi:
                        continue
                    if visited[y, x] or I[y, x] != color:
                        continue
                    visited[y, x] = True
                    cells.append((y, x))
                    stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
                ys = [p[0] for p in cells]
                xs = [p[1] for p in cells]
                rmin, rmax = min(ys), max(ys)
                cmin, cmax = min(xs), max(xs)
                oh = rmax - rmin + 1
                ow = cmax - cmin + 1
                if len(cells) == oh * ow:
                    rects.append((len(cells), rmin, cmin, oh, ow))

    rects.sort(key=lambda t: t[0])
    ops, sels = [], []

    if len(rects) >= 1:
        smallest = rects[0]
        largest = rects[-1]
        _, srmin, scmin, sh, sw = smallest
        if sh >= 3 and sw >= 3:
            ops.append(1)
            sels.append([srmin + 1, scmin + 1, sh - 3, sw - 3])
        _, lrmin, lcmin, lh, lw = largest
        if lh >= 3 and lw >= 3:
            ops.append(2)
            sels.append([lrmin + 1, lcmin + 1, lh - 3, lw - 3])

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
                "id":         f"694f12f3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
