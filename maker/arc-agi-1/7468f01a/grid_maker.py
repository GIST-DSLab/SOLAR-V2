"""
ARC Task: 7468f01a (RE-ARC) — LLM-generated grid_maker
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
from random import randint, choice


def sample_colors() -> dict:
    bgc = random.choice(list(range(10)))
    # Force sgc, fgc to be non-zero so CropGrid preserves them
    nonbg_nonzero = [c for c in range(1, 10) if c != bgc]
    sgc, fgc = random.sample(nonbg_nonzero, 2)
    return {"bgc": bgc, "sgc": sgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sgc, fgc) -> dict:
    def unifint(lb, ub, bounds):
        lo, hi = bounds
        upper = int(hi - lb * (hi - lo))
        lower = int(lo + (1 - ub) * (hi - lo))
        if lower > upper:
            return lower
        return randint(lower, upper)

    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    oh = unifint(diff_lb, diff_ub, (2, max(2, int(h * 2 / 3))))
    ow = unifint(diff_lb, diff_ub, (2, max(2, int(w * 2 / 3))))

    # Build output grid: sgc base + connected fgc shape from (0,0)
    go = [[sgc for _ in range(ow)] for _ in range(oh)]
    shp = {(0, 0)}
    nc = unifint(diff_lb, diff_ub, (0, max(1, (oh * ow) // 2)))
    for _ in range(nc):
        cand = set()
        for (r, c) in shp:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc2 = r + dr, c + dc
                if 0 <= nr < oh and 0 <= nc2 < ow and (nr, nc2) not in shp:
                    cand.add((nr, nc2))
        if not cand:
            break
        shp.add(choice(list(cand)))
    for (r, c) in shp:
        go[r][c] = fgc

    # Input: bgc canvas with vmirror(go) placed at random location
    gi = [[bgc for _ in range(w)] for _ in range(h)]
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    for r in range(oh):
        for c in range(ow):
            gi[loci + r][locj + c] = go[r][ow - 1 - c]

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # bgc = majority color of I (object area < 4/9 of total, so bgc always majority)
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    # Bounding box of non-bgc cells
    mask = (I != bgc)
    rows_idx = np.where(np.any(mask, axis=1))[0]
    cols_idx = np.where(np.any(mask, axis=0))[0]
    r0, r1 = int(rows_idx[0]), int(rows_idx[-1])
    c0, c1 = int(cols_idx[0]), int(cols_idx[-1])
    oh = r1 - r0 + 1
    ow = c1 - c0 + 1

    ops = []
    sels = []

    # 1. FlipH (vmirror) the object region in place
    ops.append(26)
    sels.append([r0, c0, oh - 1, ow - 1])

    # 2. CropGrid to the object bbox — canvas becomes oh x ow
    ops.append(33)
    sels.append([r0, c0, oh - 1, ow - 1])

    # 3. Submit
    ops.append(34)
    sels.append([0, 0, oh - 1, ow - 1])

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
                "id":         f"7468f01a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
