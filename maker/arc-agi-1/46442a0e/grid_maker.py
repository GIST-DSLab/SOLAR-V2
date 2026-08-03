"""
ARC Task: 46442a0e (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    def unifint(lb, ub, bounds):
        return random.randint(bounds[0], bounds[1])

    # output is 2h x 2h (w=h forced). h_upper limited by canvas.
    h_upper = min(15, max_h // 2, max_w // 2)
    if h_upper < 1:
        h_upper = 1
    h = unifint(diff_lb, diff_ub, (1, h_upper))
    w = h

    cols = list(range(10))
    gi = np.full((h, w), bgc, dtype=int)
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    numc = min(numc, len(remcols))
    colsch = random.sample(remcols, numc) if numc > 0 else []
    inds = [(i, j) for i in range(h) for j in range(w)]
    for col in colsch:
        max_num = max(1, len(inds) // max(1, numc))
        num = unifint(diff_lb, diff_ub, (1, max_num))
        num = min(num, len(inds))
        if num <= 0:
            continue
        chos = random.sample(inds, num)
        for (r, c) in chos:
            gi[r, c] = col
        chos_set = set(chos)
        inds = [ix for ix in inds if ix not in chos_set]

    go = np.zeros((2 * h, 2 * w), dtype=int)
    go[0:h, 0:w] = gi
    go[0:h, w:2 * w] = np.rot90(gi, k=3)   # CW  = DSL rot90
    go[h:2 * h, 0:w] = np.rot90(gi, k=1)   # CCW = DSL rot270
    go[h:2 * h, w:2 * w] = np.rot90(gi, k=2)  # 180

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape  # 2h x 2w with hi==wi

    ops, sels = [], []

    # 1. Expand canvas to 2h x 2w (I stays intact at top-left, rest zero)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    # 2. Copy input to clipboard
    ops.append(28); sels.append([0, 0, hi - 1, wi - 1])

    # 3. Paste I into top-right quadrant
    ops.append(30); sels.append([0, wi, 0, 0])
    # 4. Rotate CW on top-right (square selection -> position math correct)
    ops.append(25); sels.append([0, wi, hi - 1, wi - 1])

    # 5. Paste I into bottom-left quadrant
    ops.append(30); sels.append([hi, 0, 0, 0])
    # 6. Rotate CCW on bottom-left
    ops.append(24); sels.append([hi, 0, hi - 1, wi - 1])

    # 7. Paste I into bottom-right quadrant
    ops.append(30); sels.append([hi, wi, 0, 0])
    # 8. FlipH on bottom-right
    ops.append(26); sels.append([hi, wi, hi - 1, wi - 1])
    # 9. FlipV on bottom-right (FlipH + FlipV = rot180)
    ops.append(27); sels.append([hi, wi, hi - 1, wi - 1])

    # 10. Submit
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])

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
                "id":         f"46442a0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
