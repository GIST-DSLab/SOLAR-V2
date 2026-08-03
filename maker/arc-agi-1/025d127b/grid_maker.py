"""
ARC Task: 025d127b (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
import random


def sample_colors():
    return {"bgc": 0}


def generate(diff_lb, diff_ub, max_h, max_w, bgc):
    def unifint(lb, ub, bounds):
        a, b = bounds
        low = round(a * (1 - lb) + b * lb)
        high = round(a * (1 - ub) + b * ub)
        if low > high:
            low, high = high, low
        low = max(a, low)
        high = min(b, high)
        if low > high:
            return low
        return random.randint(low, high)

    def connect(a, b):
        if a == b:
            return {a}
        dr, dc = b[0] - a[0], b[1] - a[1]
        if dr == 0:
            step = 1 if dc > 0 else -1
            return {(a[0], a[1] + i * step) for i in range(abs(dc) + 1)}
        if dc == 0:
            step = 1 if dr > 0 else -1
            return {(a[0] + i * step, a[1]) for i in range(abs(dr) + 1)}
        if abs(dr) == abs(dc):
            sr = 1 if dr > 0 else -1
            sc = 1 if dc > 0 else -1
            return {(a[0] + i * sr, a[1] + i * sc) for i in range(abs(dr) + 1)}
        return set()

    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, min(9, len(remcols))))
    ccols = random.sample(remcols, numcols)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    used = set()

    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(3, 6)
        ow = random.randint(3, 6)
        if h - oh < 0 or w - ow < 0:
            continue
        cands = [(i, j) for i in range(h - oh + 1) for j in range(w - ow + 1)]
        if not cands:
            continue
        r0, c0 = random.choice(cands)

        topl = connect((0, 0), (0, ow - 1))
        leftl = connect((1, 0), (oh - 2, oh - 3))
        rightl = connect((1, ow), (oh - 2, ow + oh - 3))
        botl = connect((oh - 1, oh - 2), (oh - 1, oh - 3 + ow))

        inobj = topl | leftl | rightl | botl
        rm = max(c for _, c in inobj)

        topl_shifted = {(r, c + 1) for r, c in topl}
        leftl_shifted = {(r, c + 1) for r, c in leftl}
        diag2 = connect((1, ow + 1), (oh - 3, ow + oh - 3))
        outobj_raw = topl_shifted | botl | leftl_shifted | diag2 | {(oh - 2, ow + oh - 3)}
        outobj = {(r, c) for r, c in outobj_raw if c <= rm}

        fullobj = inobj | outobj
        fullobj_g = {(r + r0, c + c0) for r, c in fullobj}
        inobj_g = {(r + r0, c + c0) for r, c in inobj}
        outobj_g = {(r + r0, c + c0) for r, c in outobj}

        if not all(0 <= r < h and 0 <= c < w for r, c in fullobj_g):
            continue
        if fullobj_g & used:
            continue

        col = random.choice(ccols)
        for r, c in inobj_g:
            gi[r][c] = col
        for r, c in outobj_g:
            go[r][c] = col

        for r, c in fullobj_g:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    used.add((r + dr, c + dc))
        succ += 1

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    unique, counts = np.unique(I, return_counts=True)
    bgc = int(unique[np.argmax(counts)])

    visited = np.zeros_like(I, dtype=bool)

    def flood_8(sr, sc, color):
        cells = []
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= hi or c < 0 or c >= wi:
                continue
            if visited[r, c]:
                continue
            if int(I[r, c]) != color:
                continue
            visited[r, c] = True
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    stack.append((r + dr, c + dc))
        return cells

    def flood_4_within(cells_set, sr, sc):
        result = []
        stack = [(sr, sc)]
        local = set()
        while stack:
            r, c = stack.pop()
            if (r, c) in local:
                continue
            if (r, c) not in cells_set:
                continue
            local.add((r, c))
            result.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                stack.append((r + dr, c + dc))
        return result, local

    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or int(I[r, c]) == bgc:
                continue
            color = int(I[r, c])
            cells = flood_8(r, c, color)
            cells_set = set(cells)
            sub_visited_local = set()
            sub_parts = []
            for cell in cells:
                if cell in sub_visited_local:
                    continue
                sub_cells, sub_vis = flood_4_within(cells_set, cell[0], cell[1])
                sub_visited_local |= sub_vis
                sub_parts.append(sub_cells)

            if not sub_parts:
                continue
            max_right = max(max(cc for _, cc in sp) for sp in sub_parts)
            keep_idx = next(i for i, sp in enumerate(sub_parts)
                            if max(cc for _, cc in sp) == max_right)

            for i, sp in enumerate(sub_parts):
                if i == keep_idx:
                    continue
                rows = [rr for rr, _ in sp]
                cols_sp = [cc for _, cc in sp]
                rr0, rr1 = min(rows), max(rows)
                cc0, cc1 = min(cols_sp), max(cols_sp)
                ops.append(22)  # MoveR
                sels.append([rr0, cc0, rr1 - rr0, cc1 - cc0])

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
                "id":         f"025d127b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
