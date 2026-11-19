"""
ARC Task: feca6190 (RE-ARC) — LLM-generated grid_maker
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
import math
import random
import numpy as np
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. episode-level colours
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """The generator hardcodes bgc = 0 and draws the pixel colours from
    remove(0, cols) -> the picture colours are ALWAYS non-zero here, so nothing
    the route carries (Move / CopyI / Paste, which all treat 0 as 'nothing')
    can ever hold 0.  Fixing the palette ordering keeps the colour roles stable
    across every instance of the episode."""
    palette = [c for c in range(1, 10)]          # 0 is the background, never a pixel
    random.shuffle(palette)
    return {"palette": palette}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo = max(a, int(math.ceil(lo - 1e-9)))
    hi = min(b, int(math.floor(hi + 1e-9)))
    if hi < lo:
        lo, hi = a, b
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, palette=None) -> dict:
    if palette is None:
        palette = random.sample([c for c in range(1, 10)], 9)
    palette = list(palette)

    maxdim = max(2, min(int(max_h), int(max_w)))

    w = _unifint(diff_lb, diff_ub, (2, max(2, min(6, maxdim))))
    nc_ub = max(1, min(w, 5, maxdim // w))
    ncols = _unifint(diff_lb, diff_ub, (1, nc_ub))

    ccols = palette[:ncols]
    locs = random.sample(list(range(w)), ncols)

    n = w * ncols
    gi = [[0] * w]
    go = [[0] * n for _ in range(n)]
    for col, j in zip(ccols, locs):
        gi[0][j] = col
        r, c = n - 1, j                       # shoot((n-1, j), UP_RIGHT)
        while r >= 0 and c < n:
            go[r][c] = col
            r -= 1
            c += 1

    return {
        'input': tuple(tuple(row) for row in gi),
        'output': tuple(tuple(row) for row in go),
    }


# ----------------------------------------------------------------------------
# 3. ARCLE trajectory
# ----------------------------------------------------------------------------
def derive_operations(I, O, examples=None):
    """Rule (read off the demonstrations): the 1 x w line is laid on the bottom
    row of a square canvas of side n = w * (number of pixels), and every pixel
    grows a ray up-and-to-the-right from there.

    The ROUTE is chosen from the input line itself (its colours, its pixel
    positions, its width), so one instance always gives one route, but
    different instances are solved differently:
      * draw the diagonals first, or slide the line to the bottom first;
      * grow the rays pixel by pixel, or stamp the whole line again and again;
      * sweep from the leftmost pixel / lowest stamp, or from the other end;
      * copy the line from the original input, or from the copy on the canvas.
    Every op below is needed in every route; nothing is added for looks."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    bgc = 0                                    # the generator paints the canvas with 0
    js = [c for c in range(wi) if int(I[0, c]) != bgc]
    cols = [int(I[0, j]) for j in js]
    n = wi * len(js)                           # canvas side, measured from I alone

    ops, sels = [], []

    # --- route bits, all read out of the input line -------------------------
    first_c, last_c = cols[0], cols[-1]
    grow_rays = ((first_c + len(js)) % 2 == 0)          # ray by ray, or stamp the line
    move_first = ((last_c + js[0] + wi) % 2 == 0)       # which region gets dealt with first
    reverse = ((sum(js) + last_c) % 2 == 1)             # sweep direction
    copy_from_input = ((first_c + js[-1]) % 2 == 0)     # clipboard source

    kmax = n - 1 - min(js)                     # last stamp/ray step still on the grid

    # --- 1. grow the canvas from 1 x wi to n x n ----------------------------
    # bgc == 0, so ResizeGrid's zero padding already IS the background: no fill.
    # Full-rectangle bbox on purpose (the whole new canvas).
    ops.append(33)
    sels.append([0, 0, n - 1, n - 1])

    # --- the line slides from row 0 to the bottom row (same pixels, new place)
    def emit_move():
        src = [(0, j) for j in js]
        ops.append(21)
        sels.append(sel_of(src))               # first MoveD GRABS the line
        for _ in range(n - 2):
            ops.append(21)
            sels.append(sel_of([]))            # empty -> same object keeps gliding
        # vacated footprint (row 0) is left at 0, which IS the background: no repair

    # --- the diagonals, pixel by pixel --------------------------------------
    def emit_rays():
        order = js[::-1] if reverse else js
        for j in order:
            cells = [(n - 1 - k, j + k) for k in range(1, n - j)]
            if not cells:
                continue                       # this pixel's ray leaves the grid at once
            ops.append(int(I[0, j]))           # Color<c> on the ray's own cells
            sels.append(sel_of(cells))

    # --- the diagonals, by stamping the whole line ---------------------------
    def emit_stamps(line_row):
        if kmax < 1:
            return
        if copy_from_input:
            ops.append(28)                     # CopyI: the original line
            sels.append([0, 0, 0, wi - 1])
        else:
            ops.append(29)                     # CopyO: the line as it lies on the canvas
            sels.append([line_row, 0, 0, wi - 1])
        ks = range(kmax, 0, -1) if reverse else range(1, kmax + 1)
        for k in ks:
            ops.append(30)                     # one step up, one step right, each time
            sels.append([n - 1 - k, k, 0, 0])

    if move_first:
        emit_move()
        if grow_rays:
            emit_rays()
        else:
            emit_stamps(n - 1)                 # the line now lies on the bottom row
    else:
        if grow_rays:
            emit_rays()
        else:
            emit_stamps(0)                     # the line still lies on row 0
        emit_move()

    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task feca6190"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task feca6190"
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
                                f"for task feca6190"
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
                    f"Failed to build a complete episode for task feca6190 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"feca6190-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
