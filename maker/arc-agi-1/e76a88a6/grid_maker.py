"""
ARC Task: e76a88a6 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    dmyc = random.choice(remcols)
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
        ij = choice(totuple((bounds - shp) & mapply(dneighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    remcols = remove(bgc, cols)
    remcols = remove(dmyc, remcols)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
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
    numlocs = unifint(diff_lb, diff_ub, (1, ub))
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
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import deque
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    visited = np.zeros((hi, wi), dtype=bool)
    components = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                comp = []
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] != bgc:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                components.append(comp)

    ops = []
    sels = []

    if len(components) >= 1:
        def num_colors(comp):
            return len(set(int(I[y, x]) for y, x in comp))

        template_idx = max(range(len(components)), key=lambda i: num_colors(components[i]))
        template = components[template_idx]

        tr_min = min(y for y, x in template)
        tr_max = max(y for y, x in template)
        tc_min = min(x for y, x in template)
        tc_max = max(x for y, x in template)
        th = tr_max - tr_min + 1
        tw = tc_max - tc_min + 1

        ops.append(28)
        sels.append([tr_min, tc_min, th - 1, tw - 1])

        for i, comp in enumerate(components):
            if i == template_idx:
                continue
            dr_min = min(y for y, x in comp)
            dc_min = min(x for y, x in comp)
            ops.append(30)
            sels.append([dr_min, dc_min, 0, 0])

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
                "id":         f"e76a88a6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
