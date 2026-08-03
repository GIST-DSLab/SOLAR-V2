"""
ARC Task: 8f2ea7aa (RE-ARC) — LLM-generated grid_maker
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

def sample_colors():
    return {"bgc": random.choice(list(range(10)))}

def generate(diff_lb, diff_ub, max_h, max_w, bgc):
    import math
    max_side = min(max_h, max_w)
    max_d = int(math.isqrt(max_side))
    max_d = min(5, max(2, max_d))
    d = random.randint(2, max_d)
    d2 = d * d
    colopts = list(range(10))
    remcols = [c for c in colopts if c != bgc]
    mp = d2 // 2
    dev = random.randint(0, mp)
    devs = random.choice([1, -1])
    num = mp + devs * dev
    num = max(min(num, d2), 1)
    inds = [(i, j) for i in range(d) for j in range(d)]
    locs = set(random.sample(inds, num))
    while True:
        rows_set = {r for (r, c) in locs}
        cols_set = {c for (r, c) in locs}
        if (max(rows_set) - min(rows_set) + 1 == d and
            max(cols_set) - min(cols_set) + 1 == d):
            break
        remaining = [ij for ij in inds if ij not in locs]
        if not remaining:
            break
        locs.add(random.choice(remaining))
    ncols = random.randint(1, min(9, len(remcols)))
    cols_used = random.sample(remcols, ncols)
    pattern = [[bgc for _ in range(d)] for _ in range(d)]
    for (i, j) in locs:
        pattern[i][j] = random.choice(cols_used)
    plc_r = random.choice(list(range(0, d2, d)))
    plc_c = random.choice(list(range(0, d2, d)))
    gi = [[bgc for _ in range(d2)] for _ in range(d2)]
    for i in range(d):
        for j in range(d):
            gi[plc_r + i][plc_c + j] = pattern[i][j]
    go = [[bgc for _ in range(d2)] for _ in range(d2)]
    for (i, j) in locs:
        for ii in range(d):
            for jj in range(d):
                go[i*d + ii][j*d + jj] = pattern[ii][jj]
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[counts.argmax()])
    nz = np.argwhere(I != bgc)
    r0 = int(nz[:, 0].min())
    r1 = int(nz[:, 0].max())
    c0 = int(nz[:, 1].min())
    c1 = int(nz[:, 1].max())
    d = int(max(r1 - r0 + 1, c1 - c0 + 1))
    pattern = I[r0:r0+d, c0:c0+d]
    ops = []
    sels = []
    # 1. Copy pattern from input
    ops.append(28)
    sels.append([r0, c0, d-1, d-1])
    # 2. Fill entire output canvas with bgc
    ops.append(bgc)
    sels.append([0, 0, ho-1, wo-1])
    # 3. Paste pattern at (i*d, j*d) for each non-bgc cell (i,j) in pattern
    non_bgc_cells = [(i, j) for i in range(d) for j in range(d)
                     if int(pattern[i, j]) != bgc]
    for (i, j) in non_bgc_cells:
        ops.append(30)
        sels.append([i*d, j*d, 0, 0])
    # 4. Paste doesn't write 0s. If bgc != 0 and pattern has 0-cells,
    #    explicitly paint 0 at each replicated 0-cell location.
    if bgc != 0:
        zero_cells = [(ii, jj) for ii in range(d) for jj in range(d)
                      if int(pattern[ii, jj]) == 0]
        for (i, j) in non_bgc_cells:
            for (ii, jj) in zero_cells:
                ops.append(0)
                sels.append([i*d + ii, j*d + jj, 0, 0])
    # 5. Submit
    ops.append(34)
    sels.append([0, 0, ho-1, wo-1])
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
                "id":         f"8f2ea7aa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
