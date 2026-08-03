"""
ARC Task: 543a7ed5 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    from random import randint
    cols = difference(interval(0, 10, 1), (3, 4))
    h_lo = min(10, max_h)
    w_lo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (h_lo, max_h))
    w = unifint(diff_lb, diff_ub, (w_lo, max_w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = randint(4, 8)
        ow = randint(4, 8)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(ccols)
        if bd.issubset(indss):
            bdibd = backdrop(frozenset({(loci + 1, locj + 1), (loci + oh - 2, locj + ow - 2)}))
            go = fill(go, col, bdibd)
            go = fill(go, 3, box(bd))
            gi = fill(gi, col, bdibd)
            if oh > 5 and ow > 5 and randint(1, 10) != 1:
                ulci, ulcj = ulcorner(bdibd)
                lrci, lrcj = lrcorner(bdibd)
                aa = randint(ulci + 1, lrci - 1)
                aa = randint(ulci + 1, aa)
                bb = randint(ulcj + 1, lrcj - 1)
                bb = randint(ulcj + 1, bb)
                cc = randint(aa, lrci - 1)
                dd = randint(bb, lrcj - 1)
                cc = randint(cc, lrci - 1)
                dd = randint(dd, lrcj - 1)
                ins = backdrop({(aa, bb), (cc, dd)})
                go = fill(go, 4, ins)
                gi = fill(gi, bgc, ins)
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    def flood_label(mask):
        labels = np.zeros(mask.shape, dtype=int)
        lab = 0
        H, W = mask.shape
        for i in range(H):
            for j in range(W):
                if mask[i, j] and labels[i, j] == 0:
                    lab += 1
                    stack = [(i, j)]
                    while stack:
                        r, c = stack.pop()
                        if 0 <= r < H and 0 <= c < W and mask[r, c] and labels[r, c] == 0:
                            labels[r, c] = lab
                            stack.append((r + 1, c))
                            stack.append((r - 1, c))
                            stack.append((r, c + 1))
                            stack.append((r, c - 1))
        return labels, lab

    ops = []
    sels = []

    non_bg = sorted(set(int(x) for x in I.flatten()) - {bgc})

    for c in non_bg:
        mask = (I == c)
        labeled, num = flood_label(mask)
        for k in range(1, num + 1):
            cells = np.argwhere(labeled == k)
            rmin = int(cells[:, 0].min())
            rmax = int(cells[:, 0].max())
            cmin = int(cells[:, 1].min())
            cmax = int(cells[:, 1].max())

            r0 = rmin - 1
            r1 = rmax + 1
            c0 = cmin - 1
            c1 = cmax + 1

            if r0 >= 0:
                sc0, sc1 = max(0, c0), min(wi - 1, c1)
                if sc0 <= sc1:
                    ops.append(3)
                    sels.append([r0, sc0, 0, sc1 - sc0])
            if r1 < hi:
                sc0, sc1 = max(0, c0), min(wi - 1, c1)
                if sc0 <= sc1:
                    ops.append(3)
                    sels.append([r1, sc0, 0, sc1 - sc0])
            if c0 >= 0:
                sr0, sr1 = max(0, r0), min(hi - 1, r1)
                if sr0 <= sr1:
                    ops.append(3)
                    sels.append([sr0, c0, sr1 - sr0, 0])
            if c1 < wi:
                sr0, sr1 = max(0, r0), min(hi - 1, r1)
                if sr0 <= sr1:
                    ops.append(3)
                    sels.append([sr0, c1, sr1 - sr0, 0])

            delta = []
            for r in range(rmin, rmax + 1):
                for cc in range(cmin, cmax + 1):
                    if I[r, cc] == bgc:
                        delta.append((r, cc))
            if delta:
                drs = [d[0] for d in delta]
                dcs = [d[1] for d in delta]
                drmin, drmax = min(drs), max(drs)
                dcmin, dcmax = min(dcs), max(dcs)
                rect_area = (drmax - drmin + 1) * (dcmax - dcmin + 1)
                if len(delta) == rect_area:
                    ops.append(4)
                    sels.append([drmin, dcmin, drmax - drmin, dcmax - dcmin])
                else:
                    for r, cc in delta:
                        ops.append(4)
                        sels.append([r, cc, 0, 0])

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
                "id":         f"543a7ed5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
