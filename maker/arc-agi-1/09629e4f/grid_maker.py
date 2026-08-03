"""
ARC Task: 09629e4f (RE-ARC) — LLM-generated grid_maker
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
    barcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "barcol": barcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol) -> dict:
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc and c != barcol]

    def max_k(max_val):
        k = 5
        while k >= 2 and k * k + k - 1 > max_val:
            k -= 1
        return max(2, k)

    h_upper = min(5, max_k(max_h))
    w_upper = min(5, max_k(max_w))
    h = random.randint(2, h_upper)
    w = random.randint(2, w_upper)
    nrows, ncolumns = h, w

    ncols_upper = min(7, h * w - 2, len(remcols))
    ncols = random.randint(2, max(2, ncols_upper))

    fullh = h * nrows + nrows - 1
    fullw = w * ncolumns + ncolumns - 1

    gi = np.full((fullh, fullw), barcol, dtype=int)

    locs = [(i * (h + 1), j * (w + 1)) for i in range(nrows) for j in range(ncolumns)]
    trgloc = random.choice(locs)
    remlocs = [l for l in locs if l != trgloc]

    colssf = random.sample(remcols, ncols)
    dropped = random.choice(colssf)
    colsss = [c for c in colssf if c != dropped]

    inds = [(i, j) for i in range(h) for j in range(w)]
    trgssf = random.sample(inds, ncols - 1)

    # target cell bg
    for (i, j) in inds:
        gi[trgloc[0] + i, trgloc[1] + j] = bgc
    for (i, j), cl in zip(trgssf, colsss):
        gi[trgloc[0] + i, trgloc[1] + j] = cl

    # non-target cells
    for rl in remlocs:
        for (i, j) in inds:
            gi[rl[0] + i, rl[1] + j] = bgc
        trgss = random.sample(inds, ncols)
        for (i, j), cl in zip(trgss, colssf):
            gi[rl[0] + i, rl[1] + j] = cl

    # output
    go = np.full((fullh, fullw), bgc, dtype=int)
    go[gi == barcol] = barcol
    for (i, j), cl in zip(trgssf, colsss):
        r0 = i * (h + 1)
        c0 = j * (w + 1)
        go[r0:r0 + h, c0:c0 + w] = cl

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # barcol = uniform-row color
    barcol = None
    for r in range(hi):
        if len(set(I[r].tolist())) == 1:
            barcol = int(I[r, 0])
            break
    if barcol is None:
        for c in range(wi):
            if len(set(I[:, c].tolist())) == 1:
                barcol = int(I[0, c])
                break

    bar_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1 and I[r, 0] == barcol]
    bar_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1 and I[0, c] == barcol]

    h = bar_rows[0] if bar_rows else hi
    w = bar_cols[0] if bar_cols else wi

    nrows = (hi + 1) // (h + 1)
    ncolumns = (wi + 1) // (w + 1)

    non_bar_vals = I[I != barcol].tolist()
    bgc = int(Counter(non_bar_vals).most_common(1)[0][0])

    # locate target cell: fewest distinct non-bgc colors
    best = None
    best_count = None
    for i in range(nrows):
        for j in range(ncolumns):
            r0 = i * (h + 1)
            c0 = j * (w + 1)
            block = I[r0:r0 + h, c0:c0 + w]
            distinct = set(block.flatten().tolist()) - {bgc}
            cnt = len(distinct)
            if best is None or cnt < best_count:
                best = (i, j, r0, c0, block.copy())
                best_count = cnt

    _, _, _, _, tblock = best

    ops, sels = [], []

    # paint every sub-cell with bgc
    for i in range(nrows):
        for j in range(ncolumns):
            r0 = i * (h + 1)
            c0 = j * (w + 1)
            ops.append(bgc)
            sels.append([r0, c0, h - 1, w - 1])

    # paint colored sub-cells according to target block pattern
    for i in range(h):
        for j in range(w):
            cl = int(tblock[i, j])
            if cl != bgc:
                r0 = i * (h + 1)
                c0 = j * (w + 1)
                ops.append(cl)
                sels.append([r0, c0, h - 1, w - 1])

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
                "id":         f"09629e4f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
