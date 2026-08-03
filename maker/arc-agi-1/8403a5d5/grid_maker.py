"""
ARC Task: 8403a5d5 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 5]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    loccinv = unifint(diff_lb, diff_ub, (1, w - 1))
    locc = w - loccinv
    c = canvas(bgc, (h, w))
    idx = (h - 1, locc)
    gi = fill(c, fgc, {idx})
    go = canvas(bgc, (h, w))
    for j in range(locc, w, 2):
        go = fill(go, fgc, connect((0, j), (h - 1, j)))
    for j in range(locc + 1, w, 4):
        go = fill(go, 5, {(0, j)})
    for j in range(locc + 3, w, 4):
        go = fill(go, 5, {(h - 1, j)})
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # bgc = majority (top-left is bg since single fg cell sits at (hi-1, locc))
    bgc = int(I[0, 0])
    fg_cells = np.argwhere(I != bgc)
    r_fg, c_fg = int(fg_cells[0][0]), int(fg_cells[0][1])
    fgc = int(I[r_fg, c_fg])
    locc = c_fg

    ops, sels = [], []

    # Paint vertical fgc columns at locc, locc+2, ...
    for j in range(locc, wi, 2):
        ops.append(fgc)
        sels.append([0, j, hi - 1, 0])

    # Top-row 5s at locc+1, locc+5, ...
    for j in range(locc + 1, wi, 4):
        ops.append(5)
        sels.append([0, j, 0, 0])

    # Bottom-row 5s at locc+3, locc+7, ...
    for j in range(locc + 3, wi, 4):
        ops.append(5)
        sels.append([hi - 1, j, 0, 0])

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
                "id":         f"8403a5d5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
