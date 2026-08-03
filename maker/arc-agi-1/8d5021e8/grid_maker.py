"""
ARC Task: 8d5021e8 (RE-ARC) — LLM-generated grid_maker
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
    def unifint(dl, du, rng):
        a, b = rng
        lo = a + int((b - a) * dl)
        hi = a + int((b - a) * du)
        lo = max(a, min(lo, b))
        hi = max(a, min(hi, b))
        if lo > hi:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    cols = list(range(10))
    hmax = max(1, min(10, max_h // 3))
    wmax = max(1, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (1, hmax))
    w = unifint(diff_lb, diff_ub, (1, wmax))

    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in cols if c != bgc]
    numc_cap = min(9, h * w, len(remcols))
    numc = unifint(diff_lb, diff_ub, (0, numc_cap))
    colsch = random.sample(remcols, numc) if numc > 0 else []
    all_inds = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(all_inds)
    idx = 0
    for col in colsch:
        remaining = len(all_inds) - idx
        if remaining <= 0:
            break
        max_num = max(1, remaining // max(1, numc))
        num = unifint(diff_lb, diff_ub, (1, max_num))
        num = min(num, remaining)
        for k in range(num):
            r, c = all_inds[idx + k]
            gi[r][c] = col
        idx += num

    I = np.array(gi, dtype=int)
    x0 = np.fliplr(I)
    x1 = np.hstack([x0, I])
    x2 = np.flipud(x1)
    x3 = np.vstack([x1, x2])
    x4 = np.vstack([x3, x1])
    x5 = np.flipud(x4)
    O = x5.tolist()

    return {"input": [row[:] for row in gi], "output": O}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # 1. CopyI: capture I to clipboard (nonzero cells).
    ops.append(28); sels.append([0, 0, hi - 1, wi - 1])
    # 2. ResizeGrid to (3h, 2w). I preserved at top-left.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    # 3. Paste I at (0, wi): put I on right half of top band.
    ops.append(30); sels.append([0, wi, 0, 0])
    # 4. FlipH on left half of top band -> vmirror(I).
    #    Top band [0:hi, 0:2wi] = [vmirror(I) | I] = x1.
    ops.append(26); sels.append([0, 0, hi - 1, wi - 1])
    # 5. CopyO: capture x1 (top band) to clipboard.
    ops.append(29); sels.append([0, 0, hi - 1, wo - 1])
    # 6. Paste x1 at (hi, 0): middle band.
    ops.append(30); sels.append([hi, 0, 0, 0])
    # 7. Paste x1 at (2h, 0): bottom band.
    ops.append(30); sels.append([2 * hi, 0, 0, 0])
    # 8. FlipV on top band -> flipud(x1) = x2.
    ops.append(27); sels.append([0, 0, hi - 1, wo - 1])
    # 9. FlipV on bottom band -> flipud(x1) = x2.
    ops.append(27); sels.append([2 * hi, 0, hi - 1, wo - 1])
    # 10. Submit.
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
                "id":         f"8d5021e8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
