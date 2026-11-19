"""
ARC Task: 444801d8 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """Episode-level palette: the background and the pool the box/dot colours come from.

    The rule itself is structural (the dot's own colour floods its box), but keeping
    one palette for the whole episode keeps every instance visually coherent.
    `instance_plan` only fixes the orientation of each instance's FIRST box so that
    both "gap on top / line above" and "gap on bottom / line below" are demonstrated.
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(2, 9)
    ccols = random.sample(remcols, numcols)

    VARIANTS = [{"first_orient": 0}, {"first_orient": 1}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, ccols=None, first_orient=None, instance_plan=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = sample(remcols, 2)
    ccols = list(ccols)

    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))

    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(4, 6)
        ow = 5
        bx = box({(1, 0), (oh - 1, 4)}) - {(1, 2)}
        fullobj = backdrop({(0, 0), (oh - 1, 4)})
        cands = backdrop(bx) - bx
        dot = choice(totuple(cands))
        dcol, bxcol = sample(ccols, 2)
        inobj = recolor(bxcol, bx) | recolor(dcol, {dot})
        outobj = recolor(bxcol, bx) | recolor(dcol, fullobj - bx)
        if succ == 0 and first_orient is not None:
            mirror = bool(first_orient)
        else:
            mirror = choice((True, False))
        if mirror:
            inobj = shift(hmirror(inobj), UP)
            outobj = hmirror(outobj)
        locs = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(locs) == 0:
            continue
        loc = choice(totuple(locs))
        outplcd = shift(outobj, loc)
        outplcdi = toindices(outplcd)
        if outplcdi.issubset(inds):
            succ += 1
            inplcd = shift(inobj, loc)
            inds = (inds - outplcdi) - outbox(inplcd)
            gi = paint(gi, inplcd)
            go = paint(go, outplcd)
    if succ == 0:
        raise ValueError('444801d8: no object could be placed')
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule, read entirely off I:
      every multi-cell object is a 5-wide box outline (3-5 rows tall) whose top or
      bottom edge is broken by a single gap in its middle column, with one dot of a
      second colour sitting in its hollow.  For each box the dot's colour floods that
      box's hollow (interior + the gap cell) and spills one 5-wide line onto the row
      just outside the gap.

    The ROUTE is derived from the input too, so two different puzzles get attacked the
    way two different people would attack them, while the rule performed is identical:
      * which box first / which way to sweep  -> anchor position + box count
      * how each box is painted               -> background colour + count + anchor col
          style 0: trace straight onto the grid, working in reading order through the
                   gap (line first when it is above, hollow first when it is below)
          style 1: one stroke for the whole plume (hollow + line are one region of one
                   colour, joined through the gap)
          style 2: block the whole frame in with the fill colour, then re-draw the
                   frame on top of it
    Nothing here ever looks at O, so a different answer grid yields the same ops.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # ---- components of a candidate background ---------------------------------
    def components(bg):
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or I[r, c] == bg:
                    continue
                col = int(I[r, c])
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                out.append((col, cells))
        return out

    # ---- try to read the whole picture as gapped boxes + their dots ------------
    def parse(bg):
        boxes, dots = [], []
        for col, cells in components(bg):
            if len(cells) == 1:
                dots.append((col, cells[0]))
                continue
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if c1 - c0 != 4 or not (3 <= r1 - r0 + 1 <= 5):
                return None
            frame = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                     if r in (r0, r1) or c in (c0, c1)}
            cellset = set(cells)
            missing = frame - cellset
            if len(missing) != 1 or (cellset - frame):
                return None
            gr, gc = missing.pop()
            if gc != c0 + 2 or gr not in (r0, r1):
                return None
            line_r = r0 - 1 if gr == r0 else r1 + 1
            if not (0 <= line_r < h):
                return None
            boxes.append({
                "bxcol": col,
                "cells": sorted(cellset),
                "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                "line_r": line_r,
                "hollow": sorted((r, c) for r in range(r0, r1 + 1)
                                 for c in range(c0, c1 + 1) if (r, c) not in cellset),
                "dcol": None,
            })
        if not boxes:
            return None
        for col, (r, c) in dots:                      # each dot belongs to one box
            owner = None
            for b in boxes:
                if b["r0"] <= r <= b["r1"] and b["c0"] <= c <= b["c1"]:
                    owner = b
            if owner is None or owner["dcol"] is not None:
                return None
            owner["dcol"] = col
        if any(b["dcol"] is None for b in boxes):
            return None
        return boxes

    bgc, boxes = None, None
    for cand, _ in Counter(I.flatten().tolist()).most_common():
        parsed = parse(int(cand))
        if parsed:
            bgc, boxes = int(cand), parsed
            break
    if not boxes:
        return [34], [[0, 0, h - 1, w - 1]]

    nb = len(boxes)
    anchor = min(boxes, key=lambda b: (b["r0"], b["c0"]))
    order_mode = (anchor["r0"] + nb) % 4
    style = (bgc + nb + anchor["c0"]) % 3

    if order_mode == 0:                                  # sweep down the rows
        order = sorted(boxes, key=lambda b: (b["r0"], b["c0"]))
    elif order_mode == 1:                                # sweep across the columns
        order = sorted(boxes, key=lambda b: (b["c0"], b["r0"]))
    elif order_mode == 2:                                # start at the far corner
        order = sorted(boxes, key=lambda b: (-b["r1"], -b["c1"]))
    else:                                                # all the 2s, then all the 9s...
        seen_col = {}
        for b in sorted(boxes, key=lambda b: (b["r0"], b["c0"])):
            seen_col.setdefault(b["dcol"], len(seen_col))
        order = sorted(boxes, key=lambda b: (seen_col[b["dcol"]], b["r0"], b["c0"]))

    ops, sels = [], []
    for b in order:
        dcol, r0, r1, c0, c1 = b["dcol"], b["r0"], b["r1"], b["c0"], b["c1"]
        line = [(b["line_r"], c) for c in range(c0, c1 + 1)]
        hollow = b["hollow"]
        if style == 0:
            # two strokes, top one first: through the gap in reading order
            parts = [line, hollow] if b["line_r"] < r0 else [hollow, line]
            for part in parts:
                ops.append(dcol)
                sels.append(sel_of(sorted(part)))
        elif style == 1:
            # hollow and line are one connected region of one colour: one stroke
            ops.append(dcol)
            sels.append(sel_of(sorted(set(hollow) | set(line))))
        else:
            # block the whole 5-wide frame+line in with the fill colour ...
            top = min(r0, b["line_r"])
            bot = max(r1, b["line_r"])
            ops.append(dcol)
            sels.append(sel_of([(r, c) for r in range(top, bot + 1)
                                for c in range(c0, c1 + 1)]))
            # ... then re-draw the box frame on top of it
            ops.append(b["bxcol"])
            sels.append(sel_of(b["cells"]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 444801d8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 444801d8"
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
                                f"for task 444801d8"
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
                    f"Failed to build a complete episode for task 444801d8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"444801d8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
