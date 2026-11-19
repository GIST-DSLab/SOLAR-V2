"""
ARC Task: c8cbb738 (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

import inspect

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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    return {"bgc": random.choice(list(range(10)))}


def _placement(norm_cells, gh, gw):
    """Verifier's placement: maximise ring hits (weighted), tie-break by closeness of
    the object's centre to the frame centre."""
    ring = {(i, j) for i in range(gh) for j in range(gw) if i in (0, gh - 1) or j in (0, gw - 1)}
    M = 2 * max(gh, gw)
    oh = max(r for r, _ in norm_cells) + 1
    ow = max(c for _, c in norm_cells) + 1
    cr, cc = (gh - 1) // 2 if False else gh // 2, gw // 2
    best, bestloc, nbest = None, None, 0
    for i in range(gh):
        for j in range(gw):
            sh = {(r + i, c + j) for r, c in norm_cells}
            sc = M * len(sh & ring) - (abs(i + oh // 2 - cr) + abs(j + ow // 2 - cc))
            if best is None or sc > best:
                best, bestloc, nbest = sc, (i, j), 1
            elif sc == best:
                nbest += 1
    return bestloc, nbest


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = list(range(10))
    gh = unifint(diff_lb, diff_ub, (3, min(10, max_h // 2)))
    gw = unifint(diff_lb, diff_ub, (3, min(10, max_w // 2)))
    h = unifint(diff_lb, diff_ub, (gh * 2, max_h))
    w = unifint(diff_lb, diff_ub, (gw * 2, max_w))
    remcols = [c for c in cols if c != bgc]
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (gh, gw))
    goinds = asindices(go)
    ring = box(goinds)
    crns = corners(ring)
    remring = ring - crns
    nrr = len(remring)
    sc = ccols[0]
    go = fill(go, sc, crns)
    loci = randint(0, h - gh)
    locj = randint(0, w - gw)
    gi = fill(gi, sc, shift(crns, (loci, locj)))
    ccols = ccols[1:]
    bL = connect((0, 0), (gh - 1, 0))
    bR = connect((0, gw - 1), (gh - 1, gw - 1))
    bT = connect((0, 0), (0, gw - 1))
    bB = connect((gh - 1, 0), (gh - 1, gw - 1))
    validpairs = [(bL, bT), (bL, bB), (bR, bT), (bR, bB)]
    for c in ccols:
        if len(remring) < 3:
            break
        obj = set(sample(totuple(remring), unifint(diff_lb, diff_ub, (3, max(3, min(len(remring), nrr // len(ccols)))))))
        flag = False
        for b1, b2 in validpairs:
            if len(obj & b1) > 0 and len(obj & b2) > 0:
                flag = True
                break
        if flag:
            # placement must be the unique one the rule recovers
            objn = normalize(obj)
            loc, nb = _placement(list(objn), gh, gw)
            if nb != 1 or shift(objn, loc) != frozenset(obj):
                continue
            oh, ow = shape(obj)
            locs = ofcolor(gi, bgc)
            cands = sfilter(locs, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) > 0:
                cands2 = sfilter(cands, lambda ij: shift(objn, ij).issubset(locs))
                if len(cands2) > 0:
                    loc = choice(totuple(cands2))
                    gi = fill(gi, c, shift(objn, loc))
                    go = fill(go, c, obj)
                    remring -= obj
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])
    objs = {int(v): [(int(r), int(c)) for r, c in zip(*np.where(I == v))] for v in vals if int(v) != bgc}

    # frame = colour whose 4 cells sit on the corners of the largest rectangle
    frame, best = None, -1
    for col, cells in objs.items():
        if len(cells) != 4:
            continue
        rs = sorted({r for r, _ in cells}); cs = sorted({c for _, c in cells})
        if len(rs) == 2 and len(cs) == 2 and set(cells) == {(r, c) for r in rs for c in cs}:
            area = (rs[1] - rs[0] + 1) * (cs[1] - cs[0] + 1)
            if area > best:
                best, frame = area, (col, rs[0], cs[0], rs[1] - rs[0] + 1, cs[1] - cs[0] + 1)
    fcol, R0, C0, gh, gw = frame
    corners = {(0, 0), (0, gw - 1), (gh - 1, 0), (gh - 1, gw - 1)}

    # destination (frame coords) of every other colour object
    dest = {}
    for col, cells in objs.items():
        if col == fcol:
            continue
        r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
        norm = [(r - r0, c - c0) for r, c in cells]
        (i, j), _ = _placement(norm, gh, gw)
        dest[col] = (i, j, [(r + i, c + j) for r, c in norm], r0, c0)

    ops, sels = [], []
    # 1. crop the canvas to the frame window (full rectangle, background included)
    ops.append(33); sels.append([R0, C0, gh - 1, gw - 1])
    win = I[R0:R0 + gh, C0:C0 + gw]
    # objects that already sit exactly at their destination inside the window stay
    done = {col for col, (i, j, tc, r0, c0) in dest.items()
            if all(win[r, c] == col for r, c in tc)
            and all(R0 <= r < R0 + gh and C0 <= c < C0 + gw and (r - R0, c - C0) in set(tc) for r, c in objs[col])}
    # 2. clear stray object cells that fell inside the window
    stray = [(r, c) for r in range(gh) for c in range(gw)
             if win[r, c] != bgc and (r, c) not in corners and int(win[r, c]) not in done]
    if stray:
        ops.append(bgc); sels.append(sel_of(stray))
    # 3. copy each object from the input and paste it onto its spot on the frame ring
    for col in sorted(dest, key=lambda k: (dest[k][0], dest[k][1])):
        if col in done:
            continue
        i, j, tc, r0, c0 = dest[col]
        if col == 0:
            ops.append(0); sels.append(sel_of(tc))   # 0 cannot be copied: paint its landing cells
        else:
            ops.append(28); sels.append(sel_of(objs[col]))
            ops.append(30); sels.append([i, j, 0, 0])
    ops.append(34); sels.append([0, 0, gh - 1, gw - 1])
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
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    raise ValueError(
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
                        )
                    if any(not isinstance(entry, dict) for entry in instance_plan):
                        raise ValueError("every instance_plan entry must be a kwargs dict")
                    if instance_plan[-1] not in instance_plan[:-1]:
                        raise ValueError(
                            "instance_plan test variant must appear among the examples"
                        )

                try:
                    j = 0
                    while j < num_examples + 1:
                        ok = False
                        for _ in range(10):
                            try:
                                call_kwargs = dict(colors)
                                if instance_plan is not None:
                                    call_kwargs.update(instance_plan[j])
                                elif category_plan is not None:
                                    call_kwargs["category"] = category_plan[j]
                                r = generate(
                                    random.uniform(0.2, 0.5),
                                    random.uniform(0.5, 0.8),
                                    max_h, max_w,
                                    **call_kwargs,
                                )
                                I = np.array(r["input"],  dtype=np.uint8)
                                O = np.array(r["output"], dtype=np.uint8)
                                if I.shape[0] > max_h or I.shape[1] > max_w:
                                    continue
                                if O.shape[0] > max_h or O.shape[1] > max_w:
                                    continue
                                ok = True
                                break
                            except (IndexError, ValueError, KeyError):
                                continue
                        if not ok:
                            raise RuntimeError(
                                f"Failed to generate instance {j} after 10 attempts "
                                f"for task c8cbb738"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task c8cbb738 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c8cbb738-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
