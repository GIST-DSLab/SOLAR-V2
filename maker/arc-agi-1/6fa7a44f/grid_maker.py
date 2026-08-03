"""
ARC Task: 6fa7a44f (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}

def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = list(range(10))
    h_max = max(1, min(15, max_h // 2))
    w_max = max(1, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (1, h_max))
    w = unifint(diff_lb, diff_ub, (1, w_max))
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    if numc > 0:
        colsch = sample(remcols, numc)
        inds = totuple(asindices(gi))
        for col in colsch:
            num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
            chos = sample(inds, num)
            gi = fill(gi, col, chos)
            inds = difference(inds, chos)
    go = vconcat(gi, hmirror(gi))
    return {'input': gi, 'output': go}

def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # 1. Expand canvas to (2*hi, wi) — top half keeps I, bottom half zeros
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    # 2. Copy I from input to clipboard
    ops.append(28); sels.append([0, 0, hi - 1, wi - 1])
    # 3. Paste I into the bottom half starting at (hi, 0)
    ops.append(30); sels.append([hi, 0, 0, 0])
    # 4. FlipV bottom half (hmirror = flipud = op27)
    ops.append(27); sels.append([hi, 0, hi - 1, wo - 1])
    # 5. Submit
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
                "id":         f"6fa7a44f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
