"""
ARC Task: e9afcf9a (RE-ARC) — LLM-generated grid_maker
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
    # Task rule is color-agnostic (depends only on row/column positions).
    # No fixed roles needed across episode; generate() will sample per-instance.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if a > b:
            a, b = b, a
        diff = random.uniform(lb, ub)
        val = round(a + diff * (b - a))
        return max(a, min(b, val))

    h = unifint(diff_lb, diff_ub, (2, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    # Exclude 0 so ARCLE's object-mode FlipV correctly flips every cell in a column.
    cols = list(range(1, 10))
    numc = unifint(diff_lb, diff_ub, (1, min(9, h)))
    colss = random.sample(cols, numc)
    rr = tuple(random.choice(colss) for _ in range(h))
    rr2 = rr[::-1]
    gi_cols = [rr for _ in range(w)]
    go_cols = [rr if k % 2 == 0 else rr2 for k in range(w)]
    # dmirror = transpose: (w, h) -> (h, w)
    gi = tuple(tuple(gi_cols[c][r] for c in range(w)) for r in range(h))
    go = tuple(tuple(go_cols[c][r] for c in range(w)) for r in range(h))
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops = []
    sels = []
    # For each odd column, flip it vertically (up<->down).
    for c in range(1, wi, 2):
        ops.append(27)                         # FlipV = flipud on the single-column selection
        sels.append([0, c, hi - 1, 0])         # rows 0..hi-1, single column c
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
                "id":         f"e9afcf9a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
