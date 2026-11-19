"""
ARC Task: 8403a5d5 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # generator samples bgc, fgc from colours != 5 (5 is the reserved marker colour)
    cols = [c for c in range(10) if c != 5]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    loccinv = unifint(diff_lb, diff_ub, (1, w - 1))
    locc = w - loccinv
    c = canvas(bgc, (h, w))
    idx = (h - 1, locc)
    gi = fill(c, fgc, {idx})
    go = canvas(bgc, (h, w))
    for j in range(locc, w, 2):
        go = fill(go, fgc, connect((0, j), (h - 1, j)))
    for j in range(locc + 1, w, 4):
        go = fill(go, 5, {(0, j)})
    for j in range(locc + 3, w, 4):
        go = fill(go, 5, {(h - 1, j)})
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (read off the examples, not off O):
      one marked cell sits on the bottom row at column `locc`.
      -> raise a full-height stripe of its colour at locc, locc+2, locc+4, ...
      -> drop a 5 on the TOP row in the 1st, 3rd, 5th ... gap  (locc+1, locc+5, ...)
      -> drop a 5 on the BOTTOM row in the 2nd, 4th ... gap    (locc+3, locc+7, ...)

    The route (which end to start from, stripe-by-stripe vs. lay-the-band-and-carve
    vs. draw-one-and-copy-it, markers grouped by row vs. walked gap by gap, markers
    before or after the stripes) is chosen from the input itself, so one instance
    always yields one route and different instances differ the way two solvers do.
    """
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour the canvas is made of (a single foreground cell exists)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    marked = np.argwhere(I != bgc)
    r_fg, c_fg = int(marked[0][0]), int(marked[0][1])
    fgc = int(I[r_fg, c_fg])
    locc = c_fg

    stripe_cols = list(range(locc, wi, 2))
    gap_cols = list(range(locc + 1, wi, 2))
    top_marks = [(0, j) for j in range(locc + 1, wi, 4)]
    bot_marks = [(hi - 1, j) for j in range(locc + 3, wi, 4)]

    ops, sels = [], []

    def col_cells(j):
        return [(r, j) for r in range(hi)]

    # ---- route selection, derived purely from the input grid -------------------
    stripe_mode = (locc + fgc + wi) % 5
    if stripe_mode == 3 and not gap_cols:
        stripe_mode = 2                     # nothing to carve -> plain mask fill
    if stripe_mode == 4 and (fgc == 0 or len(stripe_cols) < 2):
        stripe_mode = 2                     # 0 is invisible to Copy/Paste
    marker_mode = (hi + wi + locc) % 4
    # band-fill would bury markers painted first, so it always goes first
    markers_first = ((hi + fgc) % 2 == 1) and stripe_mode != 3

    # ---- the stripes -----------------------------------------------------------
    def paint_stripes():
        if stripe_mode == 0:                                  # left to right
            for j in stripe_cols:
                ops.append(fgc); sels.append(sel_of(col_cells(j)))
        elif stripe_mode == 1:                                # right to left
            for j in reversed(stripe_cols):
                ops.append(fgc); sels.append(sel_of(col_cells(j)))
        elif stripe_mode == 2:                                # all stripes at once
            cells = [c for j in stripe_cols for c in col_cells(j)]
            ops.append(fgc); sels.append(sel_of(cells))
        elif stripe_mode == 3:                                # lay the band, carve gaps
            band = [(r, j) for j in range(locc, wi) for r in range(hi)]
            ops.append(fgc); sels.append(sel_of(band))
            gaps = [c for j in gap_cols for c in col_cells(j)]
            ops.append(bgc); sels.append(sel_of(gaps))
        else:                                                 # draw one, copy it across
            ops.append(fgc); sels.append(sel_of(col_cells(locc)))
            ops.append(29); sels.append(sel_of(col_cells(locc)))   # CopyO the stripe
            for j in stripe_cols[1:]:
                ops.append(30); sels.append(sel_of([(0, j)]))      # Paste at stripe top

    # ---- the 5 markers ---------------------------------------------------------
    def paint_markers():
        if marker_mode == 0:                                  # top row, then bottom row
            for cell in top_marks:
                ops.append(5); sels.append(sel_of([cell]))
            for cell in bot_marks:
                ops.append(5); sels.append(sel_of([cell]))
        elif marker_mode == 1:                                # bottom row first, backwards
            for cell in reversed(bot_marks):
                ops.append(5); sels.append(sel_of([cell]))
            for cell in reversed(top_marks):
                ops.append(5); sels.append(sel_of([cell]))
        elif marker_mode == 2:                                # each row in one go
            if top_marks:
                ops.append(5); sels.append(sel_of(top_marks))
            if bot_marks:
                ops.append(5); sels.append(sel_of(bot_marks))
        else:                                                 # gap by gap, left to right
            walk = sorted(top_marks + bot_marks, key=lambda rc: rc[1])
            for cell in walk:
                ops.append(5); sels.append(sel_of([cell]))

    if markers_first:
        paint_markers()
        paint_stripes()
    else:
        paint_stripes()
        paint_markers()

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # bbox = the whole grid, exactly as intended
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
                        f"num_examples+1 ({num_examples + 1}) for task 8403a5d5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8403a5d5"
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
                                f"for task 8403a5d5"
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
                    f"Failed to build a complete episode for task 8403a5d5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8403a5d5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
