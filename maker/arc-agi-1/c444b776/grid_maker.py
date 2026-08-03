"""
ARC Task: c444b776 (RE-ARC) — LLM-generated grid_maker
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
    linc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc) -> dict:
    def unifint(lb, ub, rng):
        lo, hi = rng
        if lo >= hi:
            return lo
        val = random.uniform(lb, ub) * (hi - lo) + lo
        return int(round(val))

    cols = list(range(10))

    nh = unifint(diff_lb, diff_ub, (1, 3))
    nw_lo = 2 if nh == 1 else 1
    nw = unifint(diff_lb, diff_ub, (nw_lo, 3))

    h_hi = min(9, (max_h - nh + 1) // nh)
    w_hi = min(9, (max_w - nw + 1) // nw)

    while h_hi < 2 and nh > 1:
        nh -= 1
        h_hi = min(9, (max_h - nh + 1) // nh)
    while w_hi < 2 and nw > 1:
        nw -= 1
        w_hi = min(9, (max_w - nw + 1) // nw)
    if nh == 1 and nw == 1:
        nw = 2
        w_hi = min(9, (max_w - 1) // 2)
    h_hi = max(2, h_hi)
    w_hi = max(2, w_hi)

    h = unifint(diff_lb, diff_ub, (2, h_hi))
    w = unifint(diff_lb, diff_ub, (2, w_hi))

    remcols = [x for x in cols if x != bgc and x != linc]
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)

    gi = np.full((fullh, fullw), linc, dtype=int)

    llocs = []
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.append((a, b))

    for (a, b) in llocs:
        gi[a:a + h, b:b + w] = bgc

    numcol = unifint(diff_lb, diff_ub, (1, min(8, len(remcols))))
    numcol = max(1, min(numcol, len(remcols)))
    ccols = random.sample(remcols, numcol)
    max_cels = max(1, (h * w) // 2)
    numcels = unifint(diff_lb, diff_ub, (1, max_cels))
    all_cells = [(i, j) for i in range(h) for j in range(w)]
    cels = random.sample(all_cells, numcels)
    obj = [(random.choice(ccols), ij) for ij in cels]

    srcloc = random.choice(llocs)
    sa, sb = srcloc
    for (col, (ci, cj)) in obj:
        gi[sa + ci, sb + cj] = col

    go = gi.copy()
    for (a, b) in llocs:
        if (a, b) == srcloc:
            continue
        for (col, (ci, cj)) in obj:
            go[a + ci, b + cj] = col

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # bgc = majority color
    cnt = Counter(I.flatten().tolist())
    bgc = max(cnt, key=cnt.get)

    # linc = color of any row/col that is uniform and not bgc
    linc = None
    for r in range(hi):
        vals = set(I[r, :].tolist())
        if len(vals) == 1 and int(I[r, 0]) != bgc:
            linc = int(I[r, 0])
            break
    if linc is None:
        for c in range(wi):
            vals = set(I[:, c].tolist())
            if len(vals) == 1 and int(I[0, c]) != bgc:
                linc = int(I[0, c])
                break

    if linc is not None:
        div_rows = [r for r in range(hi) if all(int(I[r, k]) == linc for k in range(wi))]
        div_cols = [c for c in range(wi) if all(int(I[k, c]) == linc for k in range(hi))]
    else:
        div_rows, div_cols = [], []

    row_bands = []
    prev = 0
    for r in div_rows:
        if r > prev:
            row_bands.append((prev, r - 1))
        prev = r + 1
    if prev < hi:
        row_bands.append((prev, hi - 1))

    col_bands = []
    prev = 0
    for c in div_cols:
        if c > prev:
            col_bands.append((prev, c - 1))
        prev = c + 1
    if prev < wi:
        col_bands.append((prev, wi - 1))

    # Find source subgrid (contains non-bgc pixels)
    src = None
    for (r0, r1) in row_bands:
        for (c0, c1) in col_bands:
            sub = I[r0:r1 + 1, c0:c1 + 1]
            if np.any(sub != bgc):
                src = (r0, r1, c0, c1)
                break
        if src is not None:
            break

    if src is None:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    sr0, sr1, sc0, sc1 = src
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1

    # CopyI the source region
    ops.append(28)
    sels.append([sr0, sc0, sh - 1, sw - 1])

    # 0-valued source cells (paste won't write 0; need explicit Color0)
    zero_cells = []
    if bgc != 0:
        for dr in range(sh):
            for dc in range(sw):
                if int(I[sr0 + dr, sc0 + dc]) == 0:
                    zero_cells.append((dr, dc))

    # Paste + fix zero cells at each non-source subgrid
    for (r0, r1) in row_bands:
        for (c0, c1) in col_bands:
            if (r0, r1, c0, c1) == src:
                continue
            ops.append(30)
            sels.append([r0, c0, 0, 0])
            for (dr, dc) in zero_cells:
                ops.append(0)
                sels.append([r0 + dr, c0 + dc, 0, 0])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                "id":         f"c444b776-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
