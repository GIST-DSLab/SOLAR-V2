"""
ARC Task: b527c5c6 (RE-ARC) — LLM-generated grid_maker
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


def generate(diff_lb, diff_ub, max_h, max_w, bgc):
    def unifint(lb, ub, bnds):
        lo, hi = bnds
        return int(round(lo + random.uniform(lb, ub) * (hi - lo)))

    max_h = max(max_h, 8)
    max_w = max(max_w, 8)

    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
    remcols = [c for c in cols if c != bgc]
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ncols = max(2, min(ncols, len(remcols)))
    ccols = random.sample(remcols, ncols)

    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)

    inds = {(r, c) for r in range(h) for c in range(w)}

    noccs = unifint(diff_lb, diff_ub, (1, 10))
    tr = 0
    succ = 0
    maxtr = 10 * noccs

    while succ < noccs and tr < maxtr:
        tr += 1
        max_dim = min(h, w) // 2 - 1
        if max_dim < 3:
            break
        d1_hi = random.randint(3, max_dim)
        d1 = random.randint(3, d1_hi)
        max_d2 = min(h, w) - 1
        if d1 * 2 + 1 > max_d2:
            continue
        d2_hi = random.randint(d1 * 2 + 1, max_d2)
        d2 = random.randint(d1 * 2 + 1, d2_hi)
        oh, ow = random.sample([d1, d2], 2)

        cands = [(r, c) for (r, c) in inds if 1 <= r <= h - oh - 1 and 1 <= c <= w - ow - 1]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        bd = {(r, c) for r in range(loci, loci + oh) for c in range(locj, locj + ow)}

        if ow < oh:
            lrflag = True
            r_lo = loci + ow - 1
            r_hi = loci + oh - ow
            dcands1 = {(rr, locj) for rr in range(r_lo, r_hi + 1)}
            dcands2 = {(rr, locj + ow - 1) for rr in range(r_lo, r_hi + 1)}
        else:
            lrflag = False
            c_lo = locj + oh - 1
            c_hi = locj + ow - oh
            dcands1 = {(loci, cc) for cc in range(c_lo, c_hi + 1)}
            dcands2 = {(loci + oh - 1, cc) for cc in range(c_lo, c_hi + 1)}

        dcands = dcands1 | dcands2
        if not dcands:
            continue
        loc = random.choice(list(dcands))
        sgnflag = -1 if loc in dcands1 else 1
        direc = (sgnflag * (0 if lrflag else 1), sgnflag * (0 if not lrflag else 1))

        ln_list = []
        rr, cc = loc
        while 0 <= rr < h and 0 <= cc < w:
            ln_list.append((rr, cc))
            rr += direc[0]
            cc += direc[1]
        ln = set(ln_list)
        if not ln_list:
            continue

        s_val = min(oh, ow) - 1
        line_rs = [x[0] for x in ln_list]
        line_cs = [x[1] for x in ln_list]
        lr0, lr1 = min(line_rs), max(line_rs)
        lc0, lc1 = min(line_cs), max(line_cs)

        shell = set()
        for k in range(1, s_val + 1):
            rr0, rr1 = lr0 - k, lr1 + k
            cc0, cc1 = lc0 - k, lc1 + k
            for c_ in range(cc0, cc1 + 1):
                shell.add((rr0, c_))
                shell.add((rr1, c_))
            for r_ in range(rr0, rr1 + 1):
                shell.add((r_, cc0))
                shell.add((r_, cc1))

        if len(ccols) < 2:
            continue
        sqc, dotc = random.sample(ccols, 2)

        giobj_cells = [(sqc, cell) for cell in (bd - {loc})] + [(dotc, loc)]
        combined = (bd | shell) - ln
        goobj_cells = [(sqc, cell) for cell in combined] + [(dotc, cell) for cell in ln]
        goobj_cells = [(col, (r_, c_)) for (col, (r_, c_)) in goobj_cells if 0 <= r_ < h and 0 <= c_ < w]
        goobji = {(r_, c_) for (_, (r_, c_)) in goobj_cells}

        if goobji.issubset(inds):
            succ += 1
            body_neighbors = set()
            for (br, bc) in bd:
                for (nr, nc) in [(br - 1, bc), (br + 1, bc), (br, bc - 1), (br, bc + 1)]:
                    if 0 <= nr < h and 0 <= nc < w:
                        body_neighbors.add((nr, nc))
            inds = (inds - goobji) - body_neighbors
            for col, (r_, c_) in giobj_cells:
                gi[r_, c_] = col
            for col, (r_, c_) in goobj_cells:
                go[r_, c_] = col

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I_arr = np.asarray(I, dtype=int)
    O_arr = np.asarray(O, dtype=int)
    hi, wi = I_arr.shape

    bgc = int(Counter(I_arr.flatten().tolist()).most_common(1)[0][0])

    ops = []
    sels = []

    visited = np.zeros((hi, wi), dtype=bool)

    for sr in range(hi):
        for sc in range(wi):
            if I_arr[sr, sc] == bgc or visited[sr, sc]:
                continue

            stack = [(sr, sc)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cr >= hi or cc < 0 or cc >= wi:
                    continue
                if visited[cr, cc] or I_arr[cr, cc] == bgc:
                    continue
                visited[cr, cc] = True
                cells.append((cr, cc))
                stack.append((cr + 1, cc))
                stack.append((cr - 1, cc))
                stack.append((cr, cc + 1))
                stack.append((cr, cc - 1))

            if len(cells) < 2:
                continue

            rs = [c[0] for c in cells]
            cs = [c[1] for c in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            oh = r1 - r0 + 1
            ow = c1 - c0 + 1

            colors_list = [int(I_arr[cr, cc]) for cr, cc in cells]
            counter = Counter(colors_list)
            sqc = counter.most_common(1)[0][0]
            dot_cells = [(cr, cc) for cr, cc in cells if int(I_arr[cr, cc]) != sqc]
            if len(dot_cells) != 1:
                continue
            dr, dc = dot_cells[0]
            dotc = int(I_arr[dr, dc])

            s_val = min(oh, ow)
            if ow >= oh:
                direc = (-1, 0) if dr == r0 else (1, 0)
            else:
                direc = (0, -1) if dc == c0 else (0, 1)

            line_cells = []
            cr, cc = dr, dc
            while 0 <= cr < hi and 0 <= cc < wi:
                line_cells.append((cr, cc))
                cr += direc[0]
                cc += direc[1]

            if not line_cells:
                continue

            lrs = [c[0] for c in line_cells]
            lcs = [c[1] for c in line_cells]
            lr0, lr1 = min(lrs), max(lrs)
            lc0, lc1 = min(lcs), max(lcs)

            shell_r0 = max(0, lr0 - (s_val - 1))
            shell_r1 = min(hi - 1, lr1 + (s_val - 1))
            shell_c0 = max(0, lc0 - (s_val - 1))
            shell_c1 = min(wi - 1, lc1 + (s_val - 1))

            ops.append(int(sqc))
            sels.append([shell_r0, shell_c0, shell_r1 - shell_r0, shell_c1 - shell_c0])

            ops.append(int(dotc))
            sels.append([lr0, lc0, lr1 - lr0, lc1 - lc0])

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
                "id":         f"b527c5c6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
