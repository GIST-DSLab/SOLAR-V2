"""
ARC Task: 7c008303 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    fgc = random.choice([c for c in cols if c not in (bgc, linc)])
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, fgc):
    cols = interval(0, 10, 1)
    # Input dims after rotation may be (2h+3, 2w+3) or (2w+3, 2h+3).
    # Bound both by min(max_h, max_w).
    max_dim = min(max_h, max_w)
    max_half = max(2, (max_dim - 3) // 2)
    h_upper = min(13, max_half)
    w_upper = min(13, max_half)
    h = unifint(diff_lb, diff_ub, (2, h_upper))
    w = unifint(diff_lb, diff_ub, (2, w_upper))
    h = h * 2
    w = w * 2
    remcols = [c for c in cols if c not in (bgc, linc, fgc)]
    n_frem = unifint(diff_lb, diff_ub, (1, min(4, len(remcols))))
    fremcols = sample(remcols, n_frem)
    qc = [choice(fremcols) for j in range(4)]
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = choice((ncd, h * w - ncd))
    nc = min(max(0, nc), h * w)
    cels = sample(inds, nc)
    go = fill(c, fgc, cels)
    gi = canvas(bgc, (h + 3, w + 3))
    gi = paint(gi, shift(asobject(go), (3, 3)))
    gi = fill(gi, linc, connect((2, 0), (2, w + 2)))
    gi = fill(gi, linc, connect((0, 2), (h + 2, 2)))
    gi = fill(gi, qc[0], {(0, 0)})
    gi = fill(gi, qc[1], {(0, 1)})
    gi = fill(gi, qc[2], {(1, 0)})
    gi = fill(gi, qc[3], {(1, 1)})
    A = lefthalf(tophalf(go))
    B = righthalf(tophalf(go))
    C = lefthalf(bottomhalf(go))
    D = righthalf(bottomhalf(go))
    A2 = replace(A, fgc, qc[0])
    B2 = replace(B, fgc, qc[1])
    C2 = replace(C, fgc, qc[2])
    D2 = replace(D, fgc, qc[3])
    go = vconcat(hconcat(A2, B2), hconcat(C2, D2))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # Detect main-pattern origin by locating the linc row/col.
    # Line row is row 2 OR row hi-3; line col is col 2 OR col wi-3.
    # A "line" row/col has all cells equal (single-color linc).
    if hi > 3 and len(set(I[2].tolist())) == 1:
        main_r0 = 3
    else:
        main_r0 = 0
    if wi > 3 and len(set(I[:, 2].tolist())) == 1:
        main_c0 = 3
    else:
        main_c0 = 0

    ops, sels = [], []

    # 1. CropGrid to the main-pattern region (size ho x wo).
    ops.append(33)
    sels.append([main_r0, main_c0, main_r0 + ho - 1, main_c0 + wo - 1])

    # 2. Fill entire cropped canvas with the most common color in O
    #    (background after recoloring). This handles transparent-crop bgc loss
    #    and gives us a clean base to paint recolored cells on.
    vals, counts = np.unique(O, return_counts=True)
    fill_color = int(vals[np.argmax(counts)])
    ops.append(fill_color)
    sels.append([0, 0, ho - 1, wo - 1])

    # 3. Paint each output cell that differs from fill_color.
    #    (These are the fgc cells recolored by their quadrant's qc.)
    for r in range(ho):
        for c in range(wo):
            v = int(O[r, c])
            if v != fill_color:
                ops.append(v)
                sels.append([r, c, 0, 0])

    # 4. Submit.
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
                "id":         f"7c008303-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
