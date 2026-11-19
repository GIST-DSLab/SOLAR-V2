"""
ARC Task: d4f3cd78 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # the generator samples two distinct colors from 0..9 excluding 8
    # (8 is the fixed "leak" color the rule paints with, never a canvas/box color)
    cols = [c for c in range(10) if c != 8]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    ih = unifint(diff_lb, diff_ub, (3, max(3, h // 3 * 2)))
    iw = unifint(diff_lb, diff_ub, (3, max(3, w // 3 * 2)))
    ih = min(ih, h - 2)
    iw = min(iw, w - 2)
    loci = randint(1, h - ih - 1)
    locj = randint(1, w - iw - 1)
    crns = frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)})
    fullcrns = corners(crns)
    bx = box(crns)
    opts = bx - fullcrns
    c = canvas(bgc, (h, w))
    nholes = unifint(diff_lb, diff_ub, (1, len(opts)))
    holes = sample(totuple(opts), nholes)
    gi = fill(c, fgc, bx - set(holes))
    bib = backdrop(inbox(bx))
    go = fill(gi, 8, bib)
    A, B = ulcorner(bib)
    C, D = lrcorner(bib)
    f1 = lambda idx: 1 if idx > C else (-1 if idx < A else 0)
    f2 = lambda idx: 1 if idx > D else (-1 if idx < B else 0)
    f = lambda d: shoot(d, (f1(d[0]), f2(d[1])))
    res = mapply(f, set(holes))
    go = fill(go, 8, res)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule: a hollow box is drawn on a plain canvas, with some cells missing from its
    walls.  The room inside the box fills with 8, and every gap in a wall lets that 8
    leak straight out of the box to the edge of the grid.

    Everything below is read off I alone (box bounds, which wall cells are missing).
    The ROUTE -- which region is dealt with first, which wall is attacked first, which
    way each wall is swept, and whether neighbouring gaps are treated as one wide
    breach or as separate missing bricks -- is also decided from I (sizes, positions,
    counts), so two instances of this task get solved in visibly different orders while
    performing exactly the same rule.  No op ever repaints a cell that already holds 8:
    the room and the four walls' leaks are pairwise disjoint regions.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # ---- background: the canvas colour the box is drawn on -------------------
    # counted over I and (when offered) every demonstration input, which share the
    # episode's colour scheme.
    counts = Counter(I.flatten().tolist())
    if examples:
        for pair in examples:
            try:
                ex_in = np.asarray(pair[0], dtype=int)
            except Exception:
                continue
            counts.update(ex_in.flatten().tolist())
    bgc = counts.most_common(1)[0][0]

    # ---- the box ------------------------------------------------------------
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    r0 = min(r for r, _ in fg)
    r1 = max(r for r, _ in fg)
    c0 = min(c for _, c in fg)
    c1 = max(c for _, c in fg)

    # gaps: non-corner wall cells that are missing
    top = [c for c in range(c0 + 1, c1) if I[r0, c] == bgc]
    bot = [c for c in range(c0 + 1, c1) if I[r1, c] == bgc]
    lft = [r for r in range(r0 + 1, r1) if I[r, c0] == bgc]
    rgt = [r for r in range(r0 + 1, r1) if I[r, c1] == bgc]

    inner = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]

    # ---- route choices, all read from I -------------------------------------
    # 1) wide breaches: adjacent missing bricks read as ONE gap (one wide beam)
    #    or as separate bricks (one thin ray each) -- decided by the box's corner.
    merge = ((r0 + c0) % 2 == 0)

    # 2) sweep each wall starting from the side of the grid the box sits nearest.
    fwd_horiz = (c0 <= (w - 1 - c1))
    fwd_vert = (r0 <= (h - 1 - r1))

    # 3) walk the walls clockwise or anticlockwise depending on which way the box
    #    leans in the grid.
    ring = ['top', 'right', 'bottom', 'left'] if (c0 + c1) <= (w - 1) \
        else ['top', 'left', 'bottom', 'right']

    walls = {
        'top':    (top, r0 + 1),      # (gaps, how far that wall's leak has to travel)
        'bottom': (bot, h - r1),
        'left':   (lft, c0 + 1),
        'right':  (rgt, w - c1),
    }
    # 4) begin with the busiest wall; ties go to the wall closest to the grid edge.
    start = min(ring, key=lambda s: (-len(walls[s][0]), walls[s][1], ring.index(s)))
    k = ring.index(start)
    order = ring[k:] + ring[:k]

    def groups(vals, forward):
        vs = sorted(vals)
        out = []
        for v in vs:
            if merge and out and v == out[-1][-1] + 1:
                out[-1].append(v)
            else:
                out.append([v])
        if not forward:
            out.reverse()
        return out

    def leak_cells(side, grp):
        a, b = grp[0], grp[-1]
        if side == 'top':        # beam from the gap up to the top edge
            return [(r, c) for r in range(0, r0 + 1) for c in range(a, b + 1)]
        if side == 'bottom':     # down to the bottom edge
            return [(r, c) for r in range(r1, h) for c in range(a, b + 1)]
        if side == 'left':       # out to the left edge
            return [(r, c) for r in range(a, b + 1) for c in range(0, c0 + 1)]
        return [(r, c) for r in range(a, b + 1) for c in range(c1, w)]  # right edge

    leaks = []   # list of cell-lists, in the order this solver deals with them
    for side in order:
        gaps = walls[side][0]
        if not gaps:
            continue
        forward = fwd_horiz if side in ('top', 'bottom') else fwd_vert
        for grp in groups(gaps, forward):
            leaks.append(leak_cells(side, grp))

    ops, sels = [], []

    def paint(cells):
        ops.append(8)
        sels.append(sel_of(cells))

    # 5) do the bigger piece of work first: the room, or the leaks.
    room_first = len(inner) >= sum(len(x) for x in leaks)

    if room_first:
        paint(inner)                      # fill the room (a full rectangle of bgc cells)
        for cells in leaks:
            paint(cells)                  # then let each gap leak outward
    else:
        for cells in leaks:
            paint(cells)                  # trace the leaks out of each gap
        paint(inner)                      # then fill the room they came from

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])     # whole grid, exactly this rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task d4f3cd78"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d4f3cd78"
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
                                f"for task d4f3cd78"
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
                    f"Failed to build a complete episode for task d4f3cd78 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d4f3cd78-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
