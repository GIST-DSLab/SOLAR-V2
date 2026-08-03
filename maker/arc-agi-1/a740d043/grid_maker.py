"""
ARC Task: a740d043 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    import random as _r
    cols = [c for c in range(1, 10) if c != bgc]

    h = _r.randint(2, max(2, max_h))
    w = _r.randint(2, max(2, max_w))
    max_cells = (h - 1) * (w - 1)
    if max_cells < 1:
        max_cells = 1
    ncd = _r.randint(1, max(1, max_cells // 2))
    nc = min(max(1, ncd), max(1, max_cells - 1))

    numc = _r.randint(1, len(cols))
    remcols = _r.sample(cols, numc)

    grid = [[bgc] * w for _ in range(h)]

    bounds = set((i, j) for i in range(h - 1) for j in range(w - 1))
    if not bounds:
        bounds = {(0, 0)}
    ch = _r.choice(list(bounds))
    shp = {ch}
    bounds.discard(ch)

    def neigh(cell):
        i, j = cell
        return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

    for _ in range(nc):
        adj = set()
        for cell in shp:
            adj |= neigh(cell)
        cand = (bounds - shp) & adj
        if not cand:
            break
        pick = _r.choice(list(cand))
        shp.add(pick)

    minr = min(i for i, _ in shp)
    minc = min(j for _, j in shp)
    shp = {(i - minr, j - minc) for i, j in shp}
    oh = max(i for i, _ in shp) + 1
    ow = max(j for _, j in shp) + 1

    loci = _r.randint(0, h - oh)
    locj = _r.randint(0, w - ow)

    for (i, j) in shp:
        grid[loci + i][locj + j] = _r.choice(remcols)

    non_bg = [(i, j) for i in range(h) for j in range(w) if grid[i][j] != bgc]
    if not non_bg:
        grid[loci][locj] = remcols[0]
        non_bg = [(loci, locj)]

    r0 = min(i for i, _ in non_bg)
    r1 = max(i for i, _ in non_bg)
    c0 = min(j for _, j in non_bg)
    c1 = max(j for _, j in non_bg)

    go = []
    for i in range(r0, r1 + 1):
        row = []
        for j in range(c0, c1 + 1):
            v = grid[i][j]
            row.append(0 if v == bgc else v)
        go.append(row)

    return {"input": grid, "output": go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    mask = (I != bgc)
    rows = np.where(mask.any(axis=1))[0]
    cols_idx = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols_idx.min()), int(cols_idx.max())

    ops, sels = [], []

    # Crop grid to bounding box of non-bgc cells
    ops.append(33)
    sels.append([r0, c0, r1 - r0, c1 - c0])

    # After crop, canvas is (ho x wo). bgc cells (nonzero) survived the transparent copy.
    # FloodFill each connected bgc region with 0.
    cropped = I[r0:r1 + 1, c0:c1 + 1]
    visited = np.zeros_like(cropped, dtype=bool)
    for r in range(ho):
        for c in range(wo):
            if cropped[r, c] == bgc and not visited[r, c]:
                stack = [(r, c)]
                seed = (r, c)
                while stack:
                    y, x = stack.pop()
                    if 0 <= y < ho and 0 <= x < wo and cropped[y, x] == bgc and not visited[y, x]:
                        visited[y, x] = True
                        stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
                ops.append(10)  # FloodFill0
                sels.append([seed[0], seed[1], 0, 0])

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
                "id":         f"a740d043-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
