"""
ARC Task: db3e9e38 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors():
    import random
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc):
    import random
    import numpy as np

    def unifint(lb, ub, rng):
        vmin, vmax = rng
        if vmax < vmin:
            vmax = vmin
        d = random.randint(round(lb * (vmax - vmin)), round(ub * (vmax - vmin)))
        return vmin + d

    max_h = max(3, max_h)
    max_w = max(3, max_w)

    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    barth = unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    barth = max(1, min(barth, w - 2))
    loci = unifint(diff_lb, diff_ub, (1, max(1, h - 2)))
    locj = random.randint(1, w - barth - 1)

    gi = [[bgc] * w for _ in range(h)]
    for r in range(0, loci + 1):
        for c in range(locj, locj + barth):
            gi[r][c] = fgc

    go = [[bgc] * w for _ in range(h)]
    for k in range(32):
        for sign in (+1, -1):
            dr = -2 * k
            dc = sign * 2 * k * barth
            for r in range(0, loci + 1):
                for c in range(locj, locj + barth):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        go[nr][nc] = fgc
            dr = -(2 * k + 1)
            dc = sign * (2 * k + 1) * barth
            for r in range(0, loci + 1):
                for c in range(locj, locj + barth):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        go[nr][nc] = 8

    rotf = random.choice(['identity', 'rot90', 'rot180', 'rot270'])
    gi_arr = np.array(gi)
    go_arr = np.array(go)
    if rotf == 'rot90':
        gi_arr = np.rot90(gi_arr, k=3)
        go_arr = np.rot90(go_arr, k=3)
    elif rotf == 'rot180':
        gi_arr = np.rot90(gi_arr, k=2)
        go_arr = np.rot90(go_arr, k=2)
    elif rotf == 'rot270':
        gi_arr = np.rot90(gi_arr, k=1)
        go_arr = np.rot90(go_arr, k=1)

    return {"input": gi_arr.tolist(), "output": go_arr.tolist()}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    unique, counts = np.unique(I, return_counts=True)
    bgc = int(unique[np.argmax(counts)])

    mask = I != bgc
    if not mask.any():
        return [34], [[0, 0, hi - 1, wi - 1]]

    rs, cs = np.where(mask)
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    fgc = int(I[r0, c0])

    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if r0 == 0:
        edge = 'top'
    elif r1 == hi - 1:
        edge = 'bottom'
    elif c0 == 0:
        edge = 'left'
    elif c1 == wi - 1:
        edge = 'right'
    else:
        edge = 'bottom'

    if edge in ('top', 'bottom'):
        barth = width
        dr_unit = -1 if edge == 'top' else +1
        dc_unit = 0
        perp_r = 0
        perp_c = barth
    else:
        barth = height
        dr_unit = 0
        dc_unit = -1 if edge == 'left' else +1
        perp_r = barth
        perp_c = 0

    ops = []
    sels = []
    max_k = max(hi, wi) + 4

    for k in range(1, max_k):
        for sign in (+1, -1):
            dr = k * dr_unit + sign * k * perp_r
            dc = k * dc_unit + sign * k * perp_c
            nr0 = r0 + dr
            nr1 = r1 + dr
            nc0 = c0 + dc
            nc1 = c1 + dc
            a = max(0, nr0)
            b = min(hi - 1, nr1)
            c = max(0, nc0)
            d = min(wi - 1, nc1)
            if a > b or c > d:
                continue
            color = 8 if (k % 2 == 1) else fgc
            ops.append(color)
            sels.append([a, c, b - a, d - c])

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
                "id":         f"db3e9e38-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
