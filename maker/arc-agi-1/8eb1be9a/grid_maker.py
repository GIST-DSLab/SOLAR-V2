"""
ARC Task: 8eb1be9a (RE-ARC) — LLM-generated grid_maker
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
    return {"bgc": 0}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if a >= b:
            return a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    oh_bound = max(2, h // 3)
    oh = unifint(diff_lb, diff_ub, (2, oh_bound))
    ow = unifint(diff_lb, diff_ub, (2, w))

    remcols = [c for c in cols if c != bgc]
    ncols = unifint(diff_lb, diff_ub, (1, min(9, len(remcols))))
    ccols = random.sample(remcols, ncols)

    all_cells = [(i, j) for i in range(oh) for j in range(ow)]
    ncells_max = max(2, (oh * ow) * 2 // 3)
    ncells_max = min(ncells_max, len(all_cells))
    ncells = unifint(diff_lb, diff_ub, (2, ncells_max))
    ncells = min(ncells, len(all_cells))
    obj_cells = random.sample(all_cells, ncells)

    min_i = min(c[0] for c in obj_cells)
    min_j = min(c[1] for c in obj_cells)
    obj_cells = [(i - min_i, j - min_j) for i, j in obj_cells]

    oh_actual = max(c[0] for c in obj_cells) + 1
    ow_actual = max(c[1] for c in obj_cells) + 1

    obj = [(random.choice(ccols), (i, j)) for i, j in obj_cells]

    loci = random.randint(0, h - oh_actual)
    locj = random.randint(0, w - ow_actual)

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    for c, (i, j) in obj:
        gi[loci + i][locj + j] = c

    go = [row[:] for row in gi]
    K = h // oh_actual + 2
    for k in range(-K, K + 1):
        for c, (i, j) in obj:
            r = loci + i + k * oh_actual
            if 0 <= r < h:
                go[r][locj + j] = c

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    bgc = int(np.bincount(I.flatten()).argmax())

    non_bg_mask = I != bgc
    rows_with_obj = np.where(non_bg_mask.any(axis=1))[0]
    loci = int(rows_with_obj.min())
    loci_end = int(rows_with_obj.max()) + 1
    oh = loci_end - loci

    start = loci % oh

    ops, sels = [], []

    ops.append(28)
    sels.append([loci, 0, oh - 1, w - 1])

    r_paste = start
    while r_paste < h:
        if r_paste != loci:
            ops.append(30)
            sels.append([r_paste, 0, 0, 0])
        r_paste += oh

    if start > 0:
        ops.append(28)
        sels.append([loci + oh - start, 0, start - 1, w - 1])
        ops.append(30)
        sels.append([0, 0, 0, 0])

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])

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
                "id":         f"8eb1be9a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
