"""
ARC Task: 952a094c (RE-ARC) — LLM-generated grid_maker
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
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    ih = unifint(diff_lb, diff_ub, (4, h - 2))
    iw = unifint(diff_lb, diff_ub, (4, w - 2))
    loci = randint(1, h - ih - 1)
    locj = randint(1, w - iw - 1)
    sp = (loci, locj)
    ep = (loci + ih - 1, locj + iw - 1)
    bx = box(frozenset({sp, ep}))
    remaining = [c for c in cols if c != bgc and c != fgc]
    a, b, c, d = sample(remaining, 4)
    canv = canvas(bgc, (h, w))
    canvv = fill(canv, fgc, bx)
    gi = tuple(e for e in canvv)
    go = tuple(e for e in canvv)
    gi = fill(gi, a, {(loci + 1, locj + 1)})
    go = fill(go, a, {(loci + ih, locj + iw)})
    gi = fill(gi, b, {(loci + 1, locj + iw - 2)})
    go = fill(go, b, {(loci + ih, locj - 1)})
    gi = fill(gi, c, {(loci + ih - 2, locj + 1)})
    go = fill(go, c, {(loci - 1, locj + iw)})
    gi = fill(gi, d, {(loci + ih - 2, locj + iw - 2)})
    go = fill(go, d, {(loci - 1, locj - 1)})
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Identify bgc (most common) and fgc (box color = 2nd most common)
    ctr = Counter(I.flatten().tolist())
    ranked = ctr.most_common()
    bgc = ranked[0][0]
    fgc = ranked[1][0]

    # Bounding box of fgc frame
    rows, cols = np.where(I == fgc)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    # 4 inner corner cells and their diagonal outer targets
    pairs = [
        ((r0 + 1, c0 + 1), (r1 + 1, c1 + 1)),  # inner TL -> outer BR
        ((r0 + 1, c1 - 1), (r1 + 1, c0 - 1)),  # inner TR -> outer BL
        ((r1 - 1, c0 + 1), (r0 - 1, c1 + 1)),  # inner BL -> outer TR
        ((r1 - 1, c1 - 1), (r0 - 1, c0 - 1)),  # inner BR -> outer TL
    ]

    # Erase each inner cell by painting bgc
    for (ir, ic), _ in pairs:
        col = int(I[ir, ic])
        if col != bgc:
            ops.append(int(bgc))
            sels.append([ir, ic, 0, 0])

    # Paint each color at its outer diagonal target
    for (ir, ic), (orr, occ) in pairs:
        col = int(I[ir, ic])
        if 0 <= orr < ho and 0 <= occ < wo:
            ops.append(col)
            sels.append([orr, occ, 0, 0])

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
                "id":         f"952a094c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
