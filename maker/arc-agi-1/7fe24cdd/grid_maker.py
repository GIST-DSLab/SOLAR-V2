"""
ARC Task: 7fe24cdd (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import *


def sample_colors() -> dict:
    # Force bgc=0 so paste (which is transparent to 0-cells) preserves I's
    # pattern exactly at destination — non-zero cells overwrite, bgc=0 stays 0.
    return {"bgc": 0}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0) -> dict:
    cols = list(range(10))
    # output is 2h × 2h → h ≤ min(max_h, max_w) // 2
    max_side = max(1, min(max_h // 2, max_w // 2, 15))
    h = unifint(diff_lb, diff_ub, (1, max_side))
    w = h
    gi = canvas(bgc, (h, w))
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc) if numc > 0 else ()
    inds = totuple(asindices(gi))
    for col in colsch:
        upper = max(1, len(inds) // max(1, numc))
        num = unifint(diff_lb, diff_ub, (1, upper))
        num = min(num, len(inds))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go1 = hconcat(gi, rot90(gi))
    go2 = hconcat(rot270(gi), rot180(gi))
    go = vconcat(go1, go2)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h = I.shape[0]  # generator forces w = h → square
    ops, sels = [], []

    # 1. CopyI: clipboard = I (non-zero cells)
    ops.append(28); sels.append([0, 0, h - 1, h - 1])

    # 2. ResizeGrid to 2h × 2h. Top-left preserves I (bgc=0 → all cells match).
    ops.append(33); sels.append([0, 0, 2 * h - 1, 2 * h - 1])

    # 3. Paste I at (0, h) → top-right initially = I
    ops.append(30); sels.append([0, h, 0, 0])
    # 4. Rotate CW (rot90 = op25) on square h×h region → top-right = rot90(I)
    ops.append(25); sels.append([0, h, h - 1, h - 1])

    # 5. Paste I at (h, 0) → bottom-left initially = I
    ops.append(30); sels.append([h, 0, 0, 0])
    # 6. Rotate CCW (rot270 = op24) → bottom-left = rot270(I)
    ops.append(24); sels.append([h, 0, h - 1, h - 1])

    # 7. Paste I at (h, h) → bottom-right initially = I
    ops.append(30); sels.append([h, h, 0, 0])
    # 8. rot180 = FlipH + FlipV → bottom-right = rot180(I)
    ops.append(26); sels.append([h, h, h - 1, h - 1])
    ops.append(27); sels.append([h, h, h - 1, h - 1])

    # 9. Submit
    ops.append(34); sels.append([0, 0, 2 * h - 1, 2 * h - 1])

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
                "id":         f"7fe24cdd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
