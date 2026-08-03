"""
ARC Task: 178fcbfb (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c not in (1, 2, 3)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    def unifint(lb, ub, rng):
        lo, hi = rng
        if hi < lo:
            return lo
        d = random.uniform(lb, ub)
        return int(round(lo + (hi - lo) * d))

    h_max = min(30, max_h)
    w_max = min(30, max_w)
    h = unifint(diff_lb, diff_ub, (3, h_max))
    w = unifint(diff_lb, diff_ub, (3, w_max))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    inds = [(i, j) for i in range(h) for j in range(w)]
    iforb = set()
    jforb = set()

    for col in (2, 1, 3):
        if col == 2:
            bnd = unifint(diff_lb, diff_ub, (1, w))
        else:
            bnd = unifint(diff_lb, diff_ub, (1, max(1, h // 2)))
        for _ in range(bnd):
            if col == 2:
                candidates = [ij for ij in inds if ij[1] not in jforb]
            else:
                candidates = [ij for ij in inds if ij[0] not in iforb]
            if not candidates:
                break
            ij = random.choice(candidates)
            if col == 2:
                jforb.add(ij[1])
            else:
                iforb.add(ij[0])
            gi[ij[0]][ij[1]] = col
            if col == 2:
                for r in range(h):
                    go[r][ij[1]] = col
            else:
                for c in range(w):
                    go[ij[0]][c] = col
            inds.remove(ij)

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    two_cols = sorted({int(c) for r, c in zip(*np.where(I == 2))})
    three_rows = sorted({int(r) for r, c in zip(*np.where(I == 3))})
    one_rows = sorted({int(r) for r, c in zip(*np.where(I == 1))})

    for c in two_cols:
        ops.append(2)
        sels.append([0, c, hi - 1, 0])
    for r in three_rows:
        ops.append(3)
        sels.append([r, 0, 0, wi - 1])
    for r in one_rows:
        ops.append(1)
        sels.append([r, 0, 0, wi - 1])

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
                "id":         f"178fcbfb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
