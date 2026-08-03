"""
ARC Task: 928ad970 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter


def sample_colors() -> dict:
    cols = list(range(10))
    bgc, linc, dotc = random.sample(cols, 3)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc) -> dict:
    h_lo = 10 if max_h >= 10 else max_h
    w_lo = 10 if max_w >= 10 else max_w
    h = random.randint(h_lo, max_h)
    w = random.randint(w_lo, max_w)
    ih_lo = 9 if h >= 9 else h
    iw_lo = 9 if w >= 9 else w
    ih = random.randint(ih_lo, h)
    iw = random.randint(iw_lo, w)
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)

    dot1 = (random.randint(loci + 1, loci + ih - 2), locj)                 # left edge
    dot2 = (loci, random.randint(locj + 1, locj + iw - 2))                 # top edge
    dot3 = (loci + ih - 1, random.randint(locj + 1, locj + iw - 2))        # bottom edge
    dot4 = (random.randint(loci + 1, loci + ih - 2), locj + iw - 1)        # right edge

    row_choices = list(range(loci + 2, loci + ih - 2))
    while True:
        a, b = sorted(random.sample(row_choices, 2))
        if a + 1 != b:
            break
    col_choices = list(range(locj + 2, locj + iw - 2))
    while True:
        c, d = sorted(random.sample(col_choices, 2))
        if c + 1 != d:
            break

    gi = [[bgc] * w for _ in range(h)]
    for r, cc in [dot1, dot2, dot3, dot4]:
        gi[r][cc] = dotc
    for cc in range(c, d + 1):
        gi[a][cc] = linc
        gi[b][cc] = linc
    for r in range(a, b + 1):
        gi[r][c] = linc
        gi[r][d] = linc

    go = [row[:] for row in gi]
    r0, r1 = loci + 1, loci + ih - 2
    c0, c1 = locj + 1, locj + iw - 2
    for cc in range(c0, c1 + 1):
        go[r0][cc] = linc
        go[r1][cc] = linc
    for r in range(r0, r1 + 1):
        go[r][c0] = linc
        go[r][c1] = linc

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    flat = I.flatten().tolist()
    cnt = Counter(flat)
    bgc = cnt.most_common(1)[0][0]
    dotc = min(cnt, key=cnt.get)
    all_colors = set(flat)
    linc = next(c for c in all_colors if c != bgc and c != dotc)

    dot_positions = np.argwhere(I == dotc)
    min_r = int(dot_positions[:, 0].min())
    max_r = int(dot_positions[:, 0].max())
    min_c = int(dot_positions[:, 1].min())
    max_c = int(dot_positions[:, 1].max())

    r0, r1 = min_r + 1, max_r - 1
    c0, c1 = min_c + 1, max_c - 1

    ops, sels = [], []
    color_op = int(linc)
    # top row of inbox frame
    ops.append(color_op); sels.append([r0, c0, 0, c1 - c0])
    # bottom row
    ops.append(color_op); sels.append([r1, c0, 0, c1 - c0])
    # left column
    ops.append(color_op); sels.append([r0, c0, r1 - r0, 0])
    # right column
    ops.append(color_op); sels.append([r0, c1, r1 - r0, 0])

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                "id":         f"928ad970-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
