"""
ARC Task: ded97339 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc) -> dict:
    def unifint(lb, ub, rng):
        lo, hi_ = rng
        if hi_ < lo:
            hi_ = lo
        r = random.random() * (ub - lb) + lb
        return int(round(lo + (hi_ - lo) * r))

    h = unifint(diff_lb, diff_ub, (5, min(30, max_h)))
    w = unifint(diff_lb, diff_ub, (5, min(30, max_w)))
    if h < 5:
        h = 5
    if w < 5:
        w = 5

    gi = [[bgc] * w for _ in range(h)]
    max_dots = max(2, (h * w) // 9)
    ndots = unifint(diff_lb, diff_ub, (2, max_dots))
    if ndots < 2:
        ndots = 2

    inds = set((i, j) for i in range(h) for j in range(w))
    dots = set()

    if random.choice((True, False)):
        idxi = random.randint(0, h - 1)
        locj1 = random.randint(0, w - 3)
        locj2 = random.randint(locj1 + 2, w - 1)
        dots.add((idxi, locj1))
        dots.add((idxi, locj2))
    else:
        idxj = random.randint(0, w - 1)
        loci1 = random.randint(0, h - 3)
        loci2 = random.randint(loci1 + 2, h - 1)
        dots.add((loci1, idxj))
        dots.add((loci2, idxj))

    def neighbors(loc):
        i, j = loc
        return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

    for _ in range(ndots - 2):
        if not inds:
            break
        loc = random.choice(tuple(inds))
        dots.add(loc)
        inds = (inds - {loc}) - neighbors(loc)

    for (i, j) in dots:
        gi[i][j] = linc

    go = [row[:] for row in gi]
    for ii in range(h):
        cs = [jj for jj in range(w) if gi[ii][jj] == linc]
        if len(cs) >= 2:
            for jj in range(min(cs), max(cs) + 1):
                go[ii][jj] = linc
    for jj in range(w):
        rs = [ii for ii in range(h) if gi[ii][jj] == linc]
        if len(rs) >= 2:
            for ii in range(min(rs), max(rs) + 1):
                go[ii][jj] = linc

    gi_out = tuple(tuple(r) for r in gi)
    go_out = tuple(tuple(r) for r in go)
    return {"input": gi_out, "output": go_out}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])
    linc_candidates = [int(v) for v in vals if int(v) != bgc]

    ops, sels = [], []

    if linc_candidates:
        linc = linc_candidates[0]

        for ii in range(hi):
            cs = [jj for jj in range(wi) if I[ii, jj] == linc]
            if len(cs) >= 2:
                a, b = min(cs), max(cs)
                ops.append(int(linc))
                sels.append([int(ii), int(a), 0, int(b - a)])

        for jj in range(wi):
            rs = [ii for ii in range(hi) if I[ii, jj] == linc]
            if len(rs) >= 2:
                a, b = min(rs), max(rs)
                ops.append(int(linc))
                sels.append([int(a), int(jj), int(b - a), 0])

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
                "id":         f"ded97339-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
