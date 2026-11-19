"""
ARC Task: 543a7ed5 (RE-ARC) — LLM-generated grid_maker
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
    # generator draws from cols = 0..9 minus {3, 4}; 3 (frame) and 4 (hole) are
    # hardcoded by the rule, so they are never a sampled role.
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 7)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, ccols=None) -> dict:
    cols = difference(interval(0, 10, 1), (3, 4))
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        remcols = remove(bgc, cols)
        ccols = sample(remcols, unifint(diff_lb, diff_ub, (1, 7)))
    ccols = list(ccols)

    hlo = min(10, max_h)
    wlo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = randint(4, 8)
        ow = randint(4, 8)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(ccols)
        if bd.issubset(indss):
            bdibd = backdrop(frozenset({(loci + 1, locj + 1), (loci + oh - 2, locj + ow - 2)}))
            go = fill(go, col, bdibd)
            go = fill(go, 3, box(bd))
            gi = fill(gi, col, bdibd)
            if oh > 5 and ow > 5 and randint(1, 10) != 1:
                ulci, ulcj = ulcorner(bdibd)
                lrci, lrcj = lrcorner(bdibd)
                aa = randint(ulci + 1, lrci - 1)
                aa = randint(ulci + 1, aa)
                bb = randint(ulcj + 1, lrcj - 1)
                bb = randint(ulcj + 1, bb)
                cc = randint(aa, lrci - 1)
                dd = randint(bb, lrcj - 1)
                cc = randint(cc, lrci - 1)
                dd = randint(dd, lrcj - 1)
                ins = backdrop({(aa, bb), (cc, dd)})
                go = fill(go, 4, ins)
                gi = fill(gi, bgc, ins)
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


# ── derive_operations ─────────────────────────────────────────────────────────
# Rule (read off the examples, never off O): every solid block of one colour gets
# a colour-3 frame drawn on the ring of cells one step outside its bounding box,
# and every background cell trapped inside that bounding box becomes colour 4.
#
# The ROUTE through that rule is chosen from the input itself (background colour,
# number of blocks, where the first block sits, how much ink is on the grid), the
# way two people would each pick their own way round the same puzzle: which block
# to start with and in which direction to sweep, whether the frame is one outline,
# a pen-stroke round the four sides, two corner strokes, or a filled patch with the
# block redrawn on top, whether a trapped hole is flood-filled or painted, and
# whether an identical twin block is redrawn or stamped from the first one.
# The same input always yields the same route; O is never consulted.

def derive_operations(I, O):
    def _components(I, bgc):
        """Connected (4-neighbour) single-colour blocks of non-background colour."""
        h, w = I.shape
        seen = np.zeros((h, w), dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if I[r, c] == bgc or seen[r, c]:
                    continue
                col = int(I[r, c])
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                cs = set(cells)
                r0 = min(p[0] for p in cells); r1 = max(p[0] for p in cells)
                c0 = min(p[1] for p in cells); c1 = max(p[1] for p in cells)
                hole = sorted({(rr, cc) for rr in range(r0, r1 + 1)
                               for cc in range(c0, c1 + 1)} - cs)
                comps.append({
                    "col": col, "cells": cs, "hole": hole,
                    "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                    "area": len(cs),
                })
        return comps


    def _ring(comp, h, w):
        """The outbox: the ring of cells one step outside the block's bounding box."""
        R0, R1 = comp["r0"] - 1, comp["r1"] + 1
        C0, C1 = comp["c0"] - 1, comp["c1"] + 1
        cells = []
        for cc in range(C0, C1 + 1):
            cells.append((R0, cc)); cells.append((R1, cc))
        for rr in range(R0 + 1, R1):
            cells.append((rr, C0)); cells.append((rr, C1))
        return sorted({(r, c) for r, c in cells if 0 <= r < h and 0 <= c < w})


    def _sel(cells):
        """bbox selection when `cells` is exactly a full rectangle, else a cell mask."""
        cells = sorted(set(cells))
        r0 = min(p[0] for p in cells); r1 = max(p[0] for p in cells)
        c0 = min(p[1] for p in cells); c1 = max(p[1] for p in cells)
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
            return [r0, c0, r1 - r0, c1 - c0]
        return sel_of(cells)


    def _flood(I, start, h, w):
        """Cells reachable from `start` over equal colour, 4-neighbour."""
        col = I[start]
        reg = {start}
        st = [start]
        while st:
            rr, cc = st.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reg and I[nr, nc] == col:
                    reg.add((nr, nc))
                    st.append((nr, nc))
        return reg

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # Every block is inset by at least one cell from every edge (the generator
    # keeps room for the frame), so the first row and column are pure background.
    border = [int(v) for v in I[0, :]] + [int(v) for v in I[:, 0]]
    bgc = max(set(border), key=border.count)

    comps = _components(I, bgc)
    if not comps:
        return [34], [[0, 0, h - 1, w - 1]]

    # ── pick this instance's route out of the input alone ─────────────────────
    n = len(comps)
    ink = sum(cp["area"] for cp in comps)
    first = min(comps, key=lambda cp: (cp["r0"], cp["c0"]))
    fr, fc = first["r0"], first["c0"]

    order_mode = (bgc + n + fr + fc + ink) % 5
    frame_mode = (bgc * 3 + n + w) % 4
    hole_mode = (h + ink) % 2
    stamp_twins = ((fc + n) % 2 == 0)

    if order_mode == 0:      # reading order, top-left block first
        comps.sort(key=lambda cp: (cp["r0"], cp["c0"]))
    elif order_mode == 1:    # column by column, left to right
        comps.sort(key=lambda cp: (cp["c0"], cp["r0"]))
    elif order_mode == 2:    # biggest block first
        comps.sort(key=lambda cp: (-cp["area"], cp["r0"], cp["c0"]))
    elif order_mode == 3:    # backwards, bottom-right block first
        comps.sort(key=lambda cp: (-cp["r1"], -cp["c1"]))
    else:                    # one colour at a time
        comps.sort(key=lambda cp: (cp["col"], cp["r0"], cp["c0"]))

    ops, sels = [], []
    clip = None          # what the clipboard currently holds
    stamped = []         # blocks already finished, available to copy from

    for comp in comps:
        r0, r1, c0, c1 = comp["r0"], comp["r1"], comp["c0"], comp["c1"]
        col = comp["col"]
        hole = comp["hole"]
        R0, R1, C0, C1 = r0 - 1, r1 + 1, c0 - 1, c1 + 1

        # ── stamp an identical twin instead of redrawing it ───────────────────
        # Only when nothing carried holds colour 0: Paste treats 0 as "nothing
        # here", so a block painted 0 would not travel.
        twin = None
        if stamp_twins and col != 0:
            off = {(rr - r0, cc - c0) for rr, cc in hole}
            for src in stamped:
                if (src["col"] == col and src["r1"] - src["r0"] == r1 - r0
                        and src["c1"] - src["c0"] == c1 - c0
                        and src["off"] == off
                        and src["R0"] >= 0 and src["C0"] >= 0
                        and src["R1"] < h and src["C1"] < w
                        and R0 >= 0 and C0 >= 0 and R1 < h and C1 < w):
                    twin = src
                    break
        if twin is not None:
            box_sel = [twin["R0"], twin["C0"],
                       twin["R1"] - twin["R0"], twin["C1"] - twin["C0"]]
            if clip != tuple(box_sel):
                # full rectangle: the finished twin, frame and all
                ops.append(29); sels.append(list(box_sel))
                clip = tuple(box_sel)
            ops.append(30); sels.append([R0, C0, 0, 0])
            stamped.append({"col": col, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                            "R0": R0, "R1": R1, "C0": C0, "C1": C1, "off": off})
            continue

        ring = _ring(comp, h, w)
        hole_done = False

        if frame_mode == 0:
            # one stroke: the whole outline at once
            if ring:
                ops.append(3); sels.append(_sel(ring))
        elif frame_mode == 1:
            # round the four sides, clockwise from the top-left corner
            top = [(R0, cc) for cc in range(C0, C1 + 1)]
            right = [(rr, C1) for rr in range(R0 + 1, R1 + 1)]
            bottom = [(R1, cc) for cc in range(C0, C1)]
            left = [(rr, C0) for rr in range(R0 + 1, R1)]
            for seg in (top, right, bottom, left):
                seg = [(r, c) for r, c in seg if 0 <= r < h and 0 <= c < w]
                if seg:
                    ops.append(3); sels.append(_sel(seg))
        elif frame_mode == 2:
            # two corner strokes: top-left L, then bottom-right L
            l1 = [(R0, cc) for cc in range(C0, C1 + 1)] + \
                 [(rr, C0) for rr in range(R0 + 1, R1 + 1)]
            l2 = [(rr, C1) for rr in range(R0 + 1, R1 + 1)] + \
                 [(R1, cc) for cc in range(C0 + 1, C1)]
            for seg in (l1, l2):
                seg = [(r, c) for r, c in seg if 0 <= r < h and 0 <= c < w]
                if seg:
                    ops.append(3); sels.append(_sel(seg))
        else:
            # lay the frame colour over the whole patch, then draw the block back
            # on top of it (an explicit Color op, so a block painted 0 survives)
            patch = [(rr, cc) for rr in range(R0, R1 + 1) for cc in range(C0, C1 + 1)
                     if 0 <= rr < h and 0 <= cc < w]
            if patch:
                ops.append(3); sels.append(_sel(patch))          # full rectangle
                ops.append(col)
                sels.append([r0, c0, r1 - r0, c1 - c0])          # full rectangle
                if hole:
                    ops.append(4); sels.append(_sel(hole))
                hole_done = True

        if hole and not hole_done:
            enclosed = all(I[rr, cc] == bgc for rr, cc in hole)
            if enclosed:
                seed = hole[len(hole) // 2]
                enclosed = _flood(I, seed, h, w) == set(hole)
            if enclosed and hole_mode == 0:
                ops.append(14); sels.append([seed[0], seed[1], 0, 0])
            else:
                ops.append(4); sels.append(_sel(hole))

        stamped.append({"col": col, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                        "R0": R0, "R1": R1, "C0": C0, "C1": C1,
                        "off": {(rr - r0, cc - c0) for rr, cc in hole}})

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 543a7ed5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 543a7ed5"
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
                                f"for task 543a7ed5"
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
                    f"Failed to build a complete episode for task 543a7ed5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"543a7ed5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
