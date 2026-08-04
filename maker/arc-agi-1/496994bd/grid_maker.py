"""
ARC Task: 496994bd (RE-ARC) — LLM-generated grid_maker
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
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w_ub = max(3, min(14, (max_w - 1) // 2))
    w = unifint(diff_lb, diff_ub, (3, w_ub))
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)
    obj = {
        (choice(remcols), choice(totuple(sfilter(bx, lambda ij: ij[0] < h // 2)))),
        (choice(remcols), choice(totuple(sfilter(bx, lambda ij: ij[0] > h // 2))))
    }
    for kk in range(nc - 2):
        dns = mapply(neighbors, toindices(obj))
        cand = totuple(bx & dns)
        if not cand:
            break
        ch = choice(cand)
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gix = paint(canv, obj)
    gix = apply(rbind(order, matcher(identity, bgc)), gix)
    flag = choice((True, False))
    gi = hconcat(gix, canv if flag else hconcat(canvas(bgc, (h, 1)), canv))
    go = hconcat(gix, vmirror(gix) if flag else hconcat(canvas(bgc, (h, 1)), vmirror(gix)))
    if choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    if choice((True, False)):
        gi = hmirror(gi)
        go = hmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])

    vm = np.fliplr(I)
    v_out = I.copy()
    m = vm != bgc
    v_out[m] = vm[m]

    hm = np.flipud(I)
    h_out = I.copy()
    m = hm != bgc
    h_out[m] = hm[m]

    ops, sels = [], []
    non_bgc = I != bgc

    if np.array_equal(v_out, O) and non_bgc.any():
        cols_with_pattern = np.any(non_bgc, axis=0)
        pat_cmin = int(np.argmax(cols_with_pattern))
        pat_cmax = wi - 1 - int(np.argmax(cols_with_pattern[::-1]))
        ops.append(26); sels.append([0, 0, hi - 1, wi - 1])
        ops.append(28); sels.append([0, pat_cmin, hi - 1, pat_cmax - pat_cmin])
        ops.append(30); sels.append([0, pat_cmin, 0, 0])
    elif np.array_equal(h_out, O) and non_bgc.any():
        rows_with_pattern = np.any(non_bgc, axis=1)
        pat_rmin = int(np.argmax(rows_with_pattern))
        pat_rmax = hi - 1 - int(np.argmax(rows_with_pattern[::-1]))
        ops.append(27); sels.append([0, 0, hi - 1, wi - 1])
        ops.append(28); sels.append([pat_rmin, 0, pat_rmax - pat_rmin, wi - 1])
        ops.append(30); sels.append([pat_rmin, 0, 0, 0])
    else:
        diffs = np.argwhere(O != I)
        for r, c in diffs:
            ops.append(int(O[r, c]))
            sels.append([int(r), int(c), 0, 0])

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
                "id":         f"496994bd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
