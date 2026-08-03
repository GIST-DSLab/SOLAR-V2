"""
ARC Task: cdecee7f (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    numc = unifint(diff_lb, diff_ub, (1, min(9, w)))
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    inds = interval(0, w, 1)
    locs = sample(inds, numc)
    locs = order(locs, identity)
    gi = canvas(bgc, (h, w))
    go = []
    for j in locs:
        iloc = randint(0, h - 1)
        col = choice(ccols)
        gi = fill(gi, col, {(iloc, j)})
        go.append(col)
    go = go + [bgc] * (9 - len(go))
    go = tuple(go)
    go = tuple([go[:3], go[3:6][::-1], go[6:]])
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    from collections import Counter
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = most common color in input
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # collect non-bg cells and sort by column (leftmost)
    cells = []
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                cells.append((c, r, v))
    cells.sort()  # (col, row, color) → column-first

    colors = [t[2] for t in cells]
    padded = (colors + [bgc] * 9)[:9]

    # intermediate 3x3 with row 1 in NATURAL order (pre-flip)
    intermediate = [
        [padded[0], padded[1], padded[2]],
        [padded[3], padded[4], padded[5]],
        [padded[6], padded[7], padded[8]],
    ]

    ops, sels = [], []
    # 1) shrink canvas to 3x3
    ops.append(33); sels.append([0, 0, 2, 2])
    # 2) fill with bgc (clears any stale residue)
    ops.append(bgc); sels.append([0, 0, 2, 2])
    # 3) paint every non-bg cell of the intermediate
    for r in range(3):
        for c in range(3):
            v = intermediate[r][c]
            if v != bgc:
                ops.append(int(v))
                sels.append([r, c, 0, 0])
    # 4) FlipH on row 1 → reverses [c3,c4,c5] into [c5,c4,c3]
    ops.append(26); sels.append([1, 0, 0, 2])
    # 5) submit
    ops.append(34); sels.append([0, 0, 2, 2])
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
                "id":         f"cdecee7f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
