"""
ARC Task: d4a91cb9 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c not in (2, 4, 8)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    gloci = unifint(diff_lb, diff_ub, (1, h - 1))
    glocj = unifint(diff_lb, diff_ub, (1, w - 1))
    gloc = (gloci, glocj)
    g = canvas(bgc, (h, w))
    g = fill(g, 8, {gloc})
    g = rot180(g)
    glocinv = center(ofcolor(g, 8))
    glocinvi, glocinvj = glocinv
    rloci = unifint(diff_lb, diff_ub, (glocinvi + 1, h - 1))
    rlocj = unifint(diff_lb, diff_ub, (glocinvj + 1, w - 1))
    rlocinv = (rloci, rlocj)
    g = fill(g, 2, {rlocinv})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(g)
    a, b = center(ofcolor(gi, 2))
    c, d = center(ofcolor(gi, 8))
    go = fill(gi, 4, connect((a, b), (a, d)))
    go = fill(go, 4, connect((a, d), (c, d)))
    go = fill(go, 2, {(a, b)})
    go = fill(go, 8, {(c, d)})
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    pos8 = np.argwhere(I == 8)[0]
    pos2 = np.argwhere(I == 2)[0]
    r8, c8 = int(pos8[0]), int(pos8[1])
    r2, c2 = int(pos2[0]), int(pos2[1])

    ops, sels = [], []

    # Vertical strip on column c8: from just past 8 to corner (r2, c8) inclusive
    if r8 < r2:
        v_start, v_end = r8 + 1, r2
    else:
        v_start, v_end = r2, r8 - 1
    if v_start <= v_end:
        ops.append(4)
        sels.append([v_start, c8, v_end - v_start, 0])

    # Horizontal strip on row r2: from just past 2 to corner (r2, c8) inclusive
    if c2 < c8:
        h_start, h_end = c2 + 1, c8
    else:
        h_start, h_end = c8, c2 - 1
    if h_start <= h_end:
        ops.append(4)
        sels.append([r2, h_start, 0, h_end - h_start])

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
                "id":         f"d4a91cb9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
