"""
ARC Task: 4938f0c2 (RE-ARC) — LLM-generated grid_maker
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
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc):
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    oh = unifint(diff_lb, diff_ub, (2, max(2, (h - 3) // 2)))
    ow = unifint(diff_lb, diff_ub, (2, max(2, (w - 3) // 2)))
    remcols = remove(bgc, cols)
    cc = choice(remcols)
    remcols = remove(cc, remcols)
    objc = choice(remcols)
    sg = canvas(bgc, (oh, ow))
    locc = (oh - 1, ow - 1)
    sg = fill(sg, cc, {locc})
    reminds = totuple(remove(locc, asindices(sg)))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
    cells = sample(reminds, ncells)
    while ncells == 4 and shape(cells) == (2, 2):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
        cells = sample(reminds, ncells)
    sg = fill(sg, objc, cells)
    G1 = sg
    G2 = vmirror(sg)
    G3 = hmirror(sg)
    G4 = vmirror(hmirror(sg))
    vbar = canvas(bgc, (oh, 1))
    hbar = canvas(bgc, (1, ow))
    cp = canvas(cc, (1, 1))
    topg = hconcat(hconcat(G1, vbar), G2)
    botg = hconcat(hconcat(G3, vbar), G4)
    ggm = hconcat(hconcat(hbar, cp), hbar)
    GG = vconcat(vconcat(topg, ggm), botg)
    gg = asobject(GG)
    canv = canvas(bgc, (h, w))
    loci = randint(0, h - 2 * oh - 1)
    locj = randint(0, w - 2 * ow - 1)
    loc = (loci, locj)
    go = paint(canv, shift(gg, loc))
    gi = paint(canv, shift(asobject(sg), loc))
    gi = fill(gi, cc, ofcolor(go, cc))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    ccpi, ccpj = center(ofcolor(gi, cc))
    gi = gi[:ccpi] + gi[ccpi+1:]
    gi = tuple(r[:ccpj] + r[ccpj + 1:] for r in gi)
    go = go[:ccpi] + go[ccpi+1:]
    go = tuple(r[:ccpj] + r[ccpj + 1:] for r in go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    cc = None
    marker_r, marker_c = 0, 0
    for r in range(hi - 1):
        for c in range(wi - 1):
            v = int(I[r, c])
            if v != bgc and int(I[r, c+1]) == v and int(I[r+1, c]) == v and int(I[r+1, c+1]) == v:
                if int((I == v).sum()) == 4:
                    cc = v
                    marker_r, marker_c = r, c
                    break
        if cc is not None:
            break

    if cc is None:
        return [34], [[0, 0, hi-1, wi-1]]

    objc = None
    for v in vals:
        vi = int(v)
        if vi != bgc and vi != cc:
            objc = vi
            break

    if objc is None:
        return [34], [[0, 0, hi-1, wi-1]]

    objc_cells = np.argwhere(I == objc)
    if len(objc_cells) == 0:
        return [34], [[0, 0, hi-1, wi-1]]

    pr0 = int(objc_cells[:, 0].min())
    pr1 = int(objc_cells[:, 0].max())
    pc0 = int(objc_cells[:, 1].min())
    pc1 = int(objc_cells[:, 1].max())
    ph = pr1 - pr0 + 1
    pw = pc1 - pc0 + 1

    R = marker_r
    C = marker_c

    vm_pc0 = 2 * C + 1 - pc1
    hm_pr0 = 2 * R + 1 - pr1
    r180_pr0 = hm_pr0
    r180_pc0 = vm_pc0

    ops = []
    sels = []

    # 1. Copy source pattern bbox from INPUT
    ops.append(28)
    sels.append([pr0, pc0, ph - 1, pw - 1])

    # 2. vmirror: paste at (pr0, vm_pc0) then FlipH in place
    ops.append(30)
    sels.append([pr0, vm_pc0, 0, 0])
    ops.append(26)
    sels.append([pr0, vm_pc0, ph - 1, pw - 1])

    # 3. hmirror: paste at (hm_pr0, pc0) then FlipV in place
    ops.append(30)
    sels.append([hm_pr0, pc0, 0, 0])
    ops.append(27)
    sels.append([hm_pr0, pc0, ph - 1, pw - 1])

    # 4. rot180: paste at (r180_pr0, r180_pc0) then FlipH + FlipV
    ops.append(30)
    sels.append([r180_pr0, r180_pc0, 0, 0])
    ops.append(26)
    sels.append([r180_pr0, r180_pc0, ph - 1, pw - 1])
    ops.append(27)
    sels.append([r180_pr0, r180_pc0, ph - 1, pw - 1])

    # 5. Submit
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
                "id":         f"4938f0c2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
