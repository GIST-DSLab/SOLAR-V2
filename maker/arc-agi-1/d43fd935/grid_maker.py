"""
ARC Task: d43fd935 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors() -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    ccol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "ccol": ccol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccol) -> dict:
    cols = interval(0, 10, 1)
    h_lo = min(10, max_h)
    w_lo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (h_lo, max_h))
    w = unifint(diff_lb, diff_ub, (w_lo, max_w))
    boxh = unifint(diff_lb, diff_ub, (2, max(2, h // 2)))
    boxw = unifint(diff_lb, diff_ub, (2, max(2, w // 2)))
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    remcols = remove(bgc, cols)
    remcols = remove(ccol, remcols)
    ndcols = unifint(diff_lb, diff_ub, (1, min(8, len(remcols))))
    dcols = sample(remcols, ndcols)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ccol, bd)
    reminds = totuple(asindices(gi) - bd)
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = sample(reminds, nnoise)
    truenoise = sfilter(noise, lambda ij: (ij[0] < loci or ij[0] > loci + boxh - 1) and (ij[1] < locj or ij[1] > locj + boxw - 1))
    rem = difference(noise, truenoise)
    top = sfilter(rem, lambda ij: ij[0] < loci)
    bottom = sfilter(rem, lambda ij: ij[0] > loci + boxh - 1)
    left = sfilter(rem, lambda ij: ij[1] < locj)
    right = sfilter(rem, lambda ij: ij[1] > locj + boxw - 1)
    truenoiseobj = {(choice(dcols), ij) for ij in truenoise}
    gi = paint(gi, truenoiseobj)
    go = tuple(e for e in gi)
    for jj in apply(last, top):
        col = choice(dcols)
        mf = matcher(last, jj)
        subs = sfilter(top, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((valmin(subs, first), jj), (loci - 1, jj)))
    for jj in apply(last, bottom):
        col = choice(dcols)
        mf = matcher(last, jj)
        subs = sfilter(bottom, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((valmax(subs, first), jj), (loci + boxh, jj)))
    for ii in apply(first, left):
        col = choice(dcols)
        mf = matcher(first, ii)
        subs = sfilter(left, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((ii, valmin(subs, last)), (ii, locj - 1)))
    for ii in apply(first, right):
        col = choice(dcols)
        mf = matcher(first, ii)
        subs = sfilter(right, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((ii, valmax(subs, last)), (ii, locj + boxw)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops = []
    sels = []

    bgc = int(np.bincount(I.flatten()).argmax())

    visited = np.zeros_like(I, dtype=bool)
    box_r, box_c, box_h, box_w = -1, -1, -1, -1
    max_size = 0
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or int(I[r, c]) == bgc:
                continue
            color = int(I[r, c])
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= hi or x < 0 or x >= wi:
                    continue
                if visited[y, x] or int(I[y, x]) != color:
                    continue
                visited[y, x] = True
                cells.append((y, x))
                stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
            if len(cells) > 0:
                rs = [y for y, x in cells]
                cs = [x for y, x in cells]
                min_r, max_r_ = min(rs), max(rs)
                min_c, max_c_ = min(cs), max(cs)
                bh = max_r_ - min_r + 1
                bw = max_c_ - min_c + 1
                if bh * bw == len(cells) and bh >= 2 and bw >= 2 and len(cells) > max_size:
                    max_size = len(cells)
                    box_r, box_c, box_h, box_w = min_r, min_c, bh, bw

    if box_r >= 0:
        for j in range(box_c, box_c + box_w):
            top_rows = [r for r in range(box_r) if int(I[r, j]) != bgc]
            if top_rows:
                color = int(I[top_rows[0], j])
                topmost = min(top_rows)
                ops.append(color)
                sels.append([topmost, j, box_r - 1 - topmost, 0])

        for j in range(box_c, box_c + box_w):
            bot_rows = [r for r in range(box_r + box_h, hi) if int(I[r, j]) != bgc]
            if bot_rows:
                color = int(I[bot_rows[0], j])
                botmost = max(bot_rows)
                ops.append(color)
                sels.append([box_r + box_h, j, botmost - (box_r + box_h), 0])

        for i in range(box_r, box_r + box_h):
            left_cols = [c for c in range(box_c) if int(I[i, c]) != bgc]
            if left_cols:
                color = int(I[i, left_cols[0]])
                leftmost = min(left_cols)
                ops.append(color)
                sels.append([i, leftmost, 0, box_c - 1 - leftmost])

        for i in range(box_r, box_r + box_h):
            right_cols = [c for c in range(box_c + box_w, wi) if int(I[i, c]) != bgc]
            if right_cols:
                color = int(I[i, right_cols[0]])
                rightmost = max(right_cols)
                ops.append(color)
                sels.append([i, box_c + box_w, 0, rightmost - (box_c + box_w)])

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
                "id":         f"d43fd935-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
