"""
ARC Task: a79310a0 (RE-ARC) — LLM-generated grid_maker
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
    # Fix bgc=0 so Move ops treat only fgc cells as the "object" (nonzero).
    # If bgc were nonzero, MoveD would translate the entire bbox including bg cells.
    bgc = 0
    fgc = random.choice([c for c in range(10) if c not in (0, 2)])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    for _attempt in range(50):
        h = random.randint(3, max(3, max_h))
        w = random.randint(2, max(2, max_w))
        nc_max = max(1, (h * w) // 2 - 1)
        nc = random.randint(1, nc_max)

        start = (random.randint(0, h - 1), random.randint(0, w - 1))
        shp = {start}
        for _ in range(nc - 1):
            candidates = set()
            for (r, c) in shp:
                for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc_ = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc_ < w and (nr, nc_) not in shp:
                        candidates.add((nr, nc_))
            if not candidates:
                break
            shp.add(random.choice(list(candidates)))

        rs = [r for r, c in shp]
        cs = [c for r, c in shp]
        min_r = min(rs)
        min_c = min(cs)
        shp_norm = {(r - min_r, c - min_c) for (r, c) in shp}

        oh = max(r for r, c in shp_norm) + 1
        ow = max(c for r, c in shp_norm) + 1

        if oh >= h or ow > w:
            continue

        loci = random.randint(0, h - oh - 1)
        locj = random.randint(0, w - ow)

        gi = [[bgc] * w for _ in range(h)]
        for (r, c) in shp_norm:
            gi[loci + r][locj + c] = fgc

        go = [[bgc] * w for _ in range(h)]
        for (r, c) in shp_norm:
            go[loci + r + 1][locj + c] = 2

        return {"input": gi, "output": go}

    # Fallback
    h, w = 3, 2
    gi = [[bgc] * w for _ in range(h)]
    gi[0][0] = fgc
    go = [[bgc] * w for _ in range(h)]
    go[1][0] = 2
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    fgc_positions = np.argwhere(I != bgc)

    if len(fgc_positions) == 0:
        return [34], [[0, 0, hi - 1, wi - 1]]

    r0 = int(fgc_positions[:, 0].min())
    r1 = int(fgc_positions[:, 0].max())
    c0 = int(fgc_positions[:, 1].min())
    c1 = int(fgc_positions[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    ops = []
    sels = []

    # 1. Move the fgc object down by 1 (bgc=0 → only fgc cells are the "object")
    ops.append(21)  # MoveD
    sels.append([r0, c0, h - 1, w - 1])

    # 2. FloodFill from any moved fgc cell to recolor connected region to 2
    seed_r = int(fgc_positions[0][0]) + 1
    seed_c = int(fgc_positions[0][1])
    ops.append(12)  # FloodFill<2>
    sels.append([seed_r, seed_c, 0, 0])

    # 3. Submit
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
                "id":         f"a79310a0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
