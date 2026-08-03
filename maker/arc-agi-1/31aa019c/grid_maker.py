"""
ARC Task: 31aa019c (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (5, max_h))
        w = unifint(diff_lb, diff_ub, (5, max_w))
        remcols = remove(bgc, cols)
        canv = canvas(bgc, (h, w))
        inds = totuple(asindices(canv))
        mp = (h * w) // 2 - 1
        ncols_ub = min(9, mp // 2 - 1)
        if ncols_ub < 2:
            continue
        ncols = unifint(diff_lb, diff_ub, (2, ncols_ub))
        chcols = sample(remcols, ncols)
        trgcol = chcols[0]
        chcols = chcols[1:]
        dic = {c: set() for c in chcols}
        nnoise = unifint(diff_lb, diff_ub, (2 * (ncols - 1), mp))
        locc = choice(inds)
        inds = remove(locc, inds)
        noise = sample(inds, nnoise)
        for c in chcols:
            ij = choice(inds)
            dic[c].add(ij)
            inds = remove(ij, inds)
        for c in chcols:
            ij = choice(inds)
            dic[c].add(ij)
            inds = remove(ij, inds)
        for ij in noise:
            c = choice(chcols)
            dic[c].add(ij)
            inds = remove(ij, inds)
        gi = fill(canv, trgcol, {locc})
        for c, ss in dic.items():
            gi = fill(gi, c, ss)
        gi = fill(gi, trgcol, {locc})
        if len(sfilter(palette(gi), lambda c: colorcount(gi, c) == 1)) == 1:
            break
    lc = leastcolor(gi)
    locc_set = ofcolor(gi, lc)
    go = fill(canv, lc, locc_set)
    go = fill(go, 2, neighbors(first(locc_set)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    trg_color = min(cnt.keys(), key=lambda k: (cnt[k], k))
    positions = np.argwhere(I == trg_color)
    r, c = int(positions[0][0]), int(positions[0][1])
    bgc = max(cnt.keys(), key=lambda k: cnt[k])

    ops, sels = [], []

    # 1. Fill entire grid with bgc
    ops.append(int(bgc))
    sels.append([0, 0, hi - 1, wi - 1])

    # 2. Paint 3x3 (clipped) around (r,c) with color 2
    r0 = max(0, r - 1)
    r1 = min(hi - 1, r + 1)
    c0 = max(0, c - 1)
    c1 = min(wi - 1, c + 1)
    ops.append(2)
    sels.append([r0, c0, r1 - r0, c1 - c0])

    # 3. Paint center (r,c) with target color
    ops.append(int(trg_color))
    sels.append([r, c, 0, 0])

    # 4. Submit
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
                "id":         f"31aa019c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
