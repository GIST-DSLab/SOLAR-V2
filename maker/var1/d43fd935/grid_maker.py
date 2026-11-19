"""
ARC Task: d43fd935 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    cols = list(range(10))
    bgc = random.choice(cols)
    ccol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "ccol": ccol}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, ccol: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    boxh = unifint(diff_lb, diff_ub, (2, max(2, h // 2)))
    boxw = unifint(diff_lb, diff_ub, (2, max(2, w // 2)))
    boxh = min(boxh, h)
    boxw = min(boxw, w)
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    remcols = remove(bgc, cols)
    remcols = remove(ccol, remcols)
    ndcols = unifint(diff_lb, diff_ub, (1, 8))
    dcols = sample(remcols, ndcols)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ccol, bd)
    reminds = totuple(asindices(gi) - bd)
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = sample(reminds, nnoise)
    truenoise = sfilter(noise, lambda ij: (ij[0] < loci or ij[0] > loci + boxh - 1) and (ij[1] < locj or ij[1] > locj + boxw - 1))
    rem = difference(noise, truenoise)
    top = sfilter(rem, lambda ij: ij[0] < loci)
    bottom = sfilter(rem, lambda ij: ij[0] > loci + boxh - 1)
    left = sfilter(rem, lambda ij: ij[1] < locj)
    right = sfilter(rem, lambda ij: ij[1] > locj + boxw - 1)
    truenoiseobj = {(choice(dcols), ij) for ij in truenoise}
    gi = paint(gi, truenoiseobj)
    go = tuple(e for e in gi)
    for jj in apply(last, top):
        col = choice(dcols)
        mf = matcher(last, jj)
        subs = sfilter(top, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((valmin(subs, first), jj), (loci - 1, jj)))
    for jj in apply(last, bottom):
        col = choice(dcols)
        mf = matcher(last, jj)
        subs = sfilter(bottom, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((valmax(subs, first), jj), (loci + boxh, jj)))
    for ii in apply(first, left):
        col = choice(dcols)
        mf = matcher(first, ii)
        subs = sfilter(left, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((ii, valmin(subs, last)), (ii, locj - 1)))
    for ii in apply(first, right):
        col = choice(dcols)
        mf = matcher(first, ii)
        subs = sfilter(right, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((ii, valmax(subs, last)), (ii, locj + boxw)))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule: one solid rectangle (the block) sits in a field of scattered dots.
    Every dot that shares a column with the block (above/below it) or a row with
    the block (left/right of it) is drawn out into a ray: the ray runs from the
    outermost such dot in that line all the way up to the block's edge, in the
    dot's own colour.  Nothing else changes.

    The route (which side of the block first, which way to sweep it, whether to
    draw ray by ray or gather a colour's rays into one stroke) is decided from
    the input alone -- block position/size/colour, grid size, ray count -- so a
    given input always yields the same route, and different inputs differ.
    """
    import numpy as np
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in cells]}

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = int(np.bincount(I.reshape(-1), minlength=10).argmax())

    # ---- locate the block: biggest solid (>=2x2) single-colour rectangle ----
    visited = np.zeros((hi, wi), dtype=bool)
    box_r = box_c = box_h = box_w = -1
    best = 0
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or int(I[r, c]) == bgc:
                continue
            color = int(I[r, c])
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= hi or x < 0 or x >= wi:
                    continue
                if visited[y, x] or int(I[y, x]) != color:
                    continue
                visited[y, x] = True
                cells.append((y, x))
                stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
            if not cells:
                continue
            rs = [y for y, _ in cells]
            cs = [x for _, x in cells]
            bh = max(rs) - min(rs) + 1
            bw = max(cs) - min(cs) + 1
            if bh * bw == len(cells) and bh >= 2 and bw >= 2 and len(cells) > best:
                best = len(cells)
                box_r, box_c, box_h, box_w = min(rs), min(cs), bh, bw

    if box_r < 0:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    ccol = int(I[box_r, box_c])
    br, bc = box_r + box_h, box_c + box_w      # first row/col past the block

    # ---- collect the rays (side 0=top, 1=right, 2=bottom, 3=left) ----
    rays = []
    for j in range(box_c, bc):                                  # top
        rs = [r for r in range(box_r) if int(I[r, j]) != bgc]
        if rs:
            t = min(rs)
            rays.append({'side': 0, 'pos': j, 'color': int(I[t, j]),
                         'cells': [(r, j) for r in range(t, box_r)]})
    for i in range(box_r, br):                                  # right
        cs = [c for c in range(bc, wi) if int(I[i, c]) != bgc]
        if cs:
            t = max(cs)
            rays.append({'side': 1, 'pos': i, 'color': int(I[i, t]),
                         'cells': [(i, c) for c in range(t, bc - 1, -1)]})
    for j in range(box_c, bc):                                  # bottom
        rs = [r for r in range(br, hi) if int(I[r, j]) != bgc]
        if rs:
            t = max(rs)
            rays.append({'side': 2, 'pos': j, 'color': int(I[t, j]),
                         'cells': [(r, j) for r in range(t, br - 1, -1)]})
    for i in range(box_r, br):                                  # left
        cs = [c for c in range(box_c) if int(I[i, c]) != bgc]
        if cs:
            t = min(cs)
            rays.append({'side': 3, 'pos': i, 'color': int(I[i, t]),
                         'cells': [(i, c) for c in range(t, box_c)]})

    # a dot already touching the block draws no ray -- nothing would change
    rays = [ry for ry in rays
            if any(int(I[r, c]) != ry['color'] for (r, c) in ry['cells'])]

    if not rays:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # ---- pick the route from the input ----
    k = (box_r * 13 + box_c * 7 + box_h * 5 + box_w * 3
         + ccol * 11 + bgc * 2 + len(rays) * 17 + hi * 2 + wi * 3)
    mode = k % 4            # ray-by-ray / longest-first / colour strokes / side strokes
    start = (k // 4) % 4    # which side of the block to start on
    cw = (k // 16) % 2      # go round the block clockwise or anticlockwise
    rev = (k // 32) % 2     # sweep each side forwards or backwards

    side_seq = [(start + (i if cw else -i)) % 4 for i in range(4)]

    by_side = {s: [ry for ry in rays if ry['side'] == s] for s in range(4)}
    for s in range(4):
        if mode == 1:
            by_side[s].sort(key=lambda ry: (len(ry['cells']), ry['pos']),
                            reverse=bool(rev))
        else:
            by_side[s].sort(key=lambda ry: ry['pos'], reverse=bool(rev))

    def emit(color, cells):
        ops.append(int(color))
        sels.append(sel_of(cells))

    if mode in (0, 1):
        # draw one ray at a time, side by side
        for s in side_seq:
            for ry in by_side[s]:
                emit(ry['color'], ry['cells'])
    elif mode == 3:
        # finish one side at a time, but pull that side's same-coloured rays
        # into a single stroke
        for s in side_seq:
            order, groups = [], {}
            for ry in by_side[s]:
                if ry['color'] not in groups:
                    groups[ry['color']] = []
                    order.append(ry['color'])
                groups[ry['color']].extend(ry['cells'])
            for col in order:
                emit(col, groups[col])
    else:
        # work colour by colour: every ray of one colour, wherever it is, at once
        order, groups = [], {}
        for s in side_seq:
            for ry in by_side[s]:
                if ry['color'] not in groups:
                    groups[ry['color']] = []
                    order.append(ry['color'])
                groups[ry['color']].extend(ry['cells'])
        for col in order:
            emit(col, groups[col])

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
            # Episode-level retry: if 10 attempts at some instance all fail, that's
            # transient (bad luck with the generator's randomness) — retry the WHOLE
            # episode from scratch (fresh colors/instance plan) up to 5 times, rather
            # than silently continuing with a partial episode (fewer examples than
            # requested, or a missing test instance with operations=[]/selections=[]
            # quietly appended as if it were a normal sample).
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                # sample color roles once per episode → consistent across all instances
                # sample_colors() may optionally accept num_examples (to pre-plan
                # per-instance categories) — call it either way for compatibility
                # with grid_makers generated before this parameter existed.
                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                # Plans are consumed by INDEX, not mutated: retries for instance j
                # must receive the same variant. category_plan is retained as a
                # backwards-compatible single-key form; new makers use kwargs dict entries.
                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    # A wrong plan length is a deterministic bug in sample_colors(),
                    # not bad luck — retrying the episode won't fix it. Fail loudly
                    # instead of clamping the index and silently reusing an entry.
                    raise ValueError(
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task d43fd935"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d43fd935"
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
                            raise RuntimeError(
                                f"Failed to generate instance {j} after 10 attempts "
                                f"for task d43fd935"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break  # episode complete
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task d43fd935 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d43fd935-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
