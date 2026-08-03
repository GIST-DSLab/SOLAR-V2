"""
ARC Task: 321b1fc6 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors():
    cols = list(range(10))
    bgc = random.choice(cols)
    dmyc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dmyc": dmyc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dmyc):
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
    objh = unifint(diff_lb, diff_ub, (2, 5))
    objw = unifint(diff_lb, diff_ub, (2, 5))
    bounds = asindices(canvas(0, (objh, objw)))
    shp = {choice(totuple(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for j in range(nc):
        cand = totuple((bounds - shp) & mapply(dneighbors, shp))
        if len(cand) == 0:
            break
        ij = choice(cand)
        shp.add(ij)
    shp = normalize(shp)
    remcols = remove(bgc, cols)
    remcols = remove(dmyc, remcols)
    oh, ow = shape(shp)
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    shpp = shift(shp, (loci, locj))
    numco = unifint(diff_lb, diff_ub, (2, min(8, len(remcols))))
    colll = sample(remcols, numco)
    shppc = frozenset({(choice(colll), ij) for ij in shpp})
    while numcolors(shppc) == 1:
        shppc = frozenset({(choice(colll), ij) for ij in shpp})
    shppcn = normalize(shppc)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shppc)
    go = tuple(e for e in gi)
    ub = ((h * w) / (oh * ow)) // 2
    ub = max(1, ub)
    numlocs = unifint(diff_lb, diff_ub, (1, int(ub)))
    cnt = 0
    fails = 0
    maxfails = 5 * numlocs
    idns = (asindices(gi) - shpp) - mapply(dneighbors, shpp)
    idns = sfilter(idns, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
    while cnt < numlocs and fails < maxfails:
        if len(idns) == 0:
            break
        loc = choice(totuple(idns))
        plcd = shift(shppcn, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(idns):
            go = paint(go, plcd)
            gi = fill(gi, dmyc, plcdi)
            cnt += 1
            idns = (idns - plcdi) - mapply(dneighbors, plcdi)
        else:
            fails += 1
    go = fill(go, bgc, shpp)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    flat = I.flatten().tolist()
    bgc = int(Counter(flat).most_common(1)[0][0])

    visited = np.zeros_like(I, dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not visited[r, c]:
                stack = [(r, c)]
                cells = []
                colors = set()
                while stack:
                    rr, cc = stack.pop()
                    if rr < 0 or rr >= hi or cc < 0 or cc >= wi:
                        continue
                    if visited[rr, cc] or I[rr, cc] == bgc:
                        continue
                    visited[rr, cc] = True
                    cells.append((rr, cc))
                    colors.add(int(I[rr, cc]))
                    stack.append((rr + 1, cc))
                    stack.append((rr - 1, cc))
                    stack.append((rr, cc + 1))
                    stack.append((rr, cc - 1))
                objs.append((cells, colors))

    ops = []
    sels = []

    if len(objs) == 0:
        ops.append(34)
        sels.append([0, 0, int(ho - 1), int(wo - 1)])
        return ops, sels

    t_idx = max(range(len(objs)), key=lambda i: len(objs[i][1]))
    t_cells, _ = objs[t_idx]
    min_r = min(r for r, _ in t_cells)
    max_r = max(r for r, _ in t_cells)
    min_c = min(c for _, c in t_cells)
    max_c = max(c for _, c in t_cells)
    th = max_r - min_r + 1
    tw = max_c - min_c + 1

    d_uls = []
    for i, (cells, _) in enumerate(objs):
        if i == t_idx:
            continue
        br = min(rc[0] for rc in cells)
        bc = min(rc[1] for rc in cells)
        d_uls.append((br, bc))

    ops.append(28)
    sels.append([int(min_r), int(min_c), int(th - 1), int(tw - 1)])

    for (br, bc) in d_uls:
        ops.append(30)
        sels.append([int(br), int(bc), 0, 0])

    ops.append(int(bgc))
    sels.append([int(min_r), int(min_c), int(th - 1), int(tw - 1)])

    ops.append(34)
    sels.append([0, 0, int(ho - 1), int(wo - 1)])
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
                "id":         f"321b1fc6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
