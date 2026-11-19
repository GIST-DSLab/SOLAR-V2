"""
ARC Task: 007bbfb7 (RE-ARC) — LLM-generated grid_maker
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
import math
import numpy as np
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    # The generator samples exactly one random color: fgc = choice(interval(1,10,1)).
    # Background 0 is hardcoded, so it is not part of the palette.
    # No discrete structural variants exist (the rule is the same fractal tiling
    # for every instance), so no instance_plan is needed.
    fgc = random.choice(list(range(1, 10)))
    return {"fgc": fgc}


# ---------------------------------------------------------------- generator
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, fgc: int) -> dict:
    # output is (h**2, w**2), so the h/w upper bounds come from sqrt(max_h/max_w)
    hub = 2
    while hub < 5 and (hub + 1) * (hub + 1) <= max_h:
        hub += 1
    wub = 2
    while wub < 5 and (wub + 1) * (wub + 1) <= max_w:
        wub += 1

    h = unifint(diff_lb, diff_ub, (2, hub))
    w = unifint(diff_lb, diff_ub, (2, wub))
    c = canvas(0, (h, w))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    inds = totuple(asindices(c))
    locs = sample(inds, numc)
    gi = fill(c, fgc, locs)
    go = canvas(0, (h ** 2, w ** 2))
    for loc in locs:
        go = fill(go, fgc, shift(locs, multiply(loc, (h, w))))
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- derivation
def derive_operations(I, O, examples=None):
    """
    Rule (read off I alone): the h*w input is a stamp.  The canvas grows to
    (h*h, w*w) = a grid of h*w blocks, each block h*w cells.  Every block whose
    (r, c) is marked in I receives a copy of the stamp; unmarked blocks stay empty.

    The ROUTE varies with the instance (never with the answer grid):
      * whether to load the stamp into the clipboard and paste it, or to draw
        each copy directly with one Color op on the stamp's exact cells,
      * whether the stamp is grabbed before or after the canvas is grown,
      * in which order the marked blocks are visited (down, across, backwards,
        or outward from the most central mark),
      * when the leftover stamp in the top-left block has to be wiped, whether
        that is done first or last.
    All of it is a pure function of I (its colour, its mark count, its mark
    positions), so one instance always yields one route.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi * hi, wi * wi          # output size is known from I alone

    # the stamp: every marked cell of I, in raster order
    cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != 0]

    ops, sels = [], []

    if not cells:                       # degenerate: nothing to stamp
        ops.append(33); sels.append([0, 0, ho - 1, wo - 1])   # full rectangle
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    fg = int(I[cells[0][0], cells[0][1]])
    n = len(cells)
    r_last, c_last = cells[-1]

    # route choices, derived from the picture itself
    key = fg * 5 + n * 3 + r_last * 7 + c_last * 11
    use_clip = (key % 2 == 0) and fg != 0   # fg==0 could not survive a Paste
    order_mode = (key // 2) % 4
    copy_first = ((key // 8) % 2 == 1)
    clear_first = ((key // 16) % 2 == 1)

    # ResizeGrid keeps the input pattern where it is, i.e. inside block (0,0).
    # If (0,0) is marked, that block is already finished -> do not restamp it.
    # If (0,0) is not marked, that leftover must be wiped.
    need_clear = (I[0, 0] == 0)
    targets = [b for b in cells if b != (0, 0)]

    anchor = min(cells, key=lambda b: (abs(2 * b[0] - (hi - 1)) + abs(2 * b[1] - (wi - 1)),
                                       b[0], b[1]))
    if order_mode == 0:                 # sweep down the rows
        targets.sort(key=lambda b: (b[0], b[1]))
    elif order_mode == 1:               # sweep across the columns
        targets.sort(key=lambda b: (b[1], b[0]))
    elif order_mode == 2:               # sweep backwards from the far corner
        targets.sort(key=lambda b: (-b[0], -b[1]))
    else:                               # outward from the most central mark
        targets.sort(key=lambda b: (abs(b[0] - anchor[0]) + abs(b[1] - anchor[1]),
                                    b[0], b[1]))

    def grow():
        # full rectangle: the whole new canvas, background included
        ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    def grab():
        # full rectangle: the stamp is the entire input grid
        ops.append(28); sels.append([0, 0, hi - 1, wi - 1])

    def wipe():
        # the stamp ResizeGrid left behind in block (0,0), which must be empty
        ops.append(0); sels.append(sel_of(cells))

    take_clip = use_clip and bool(targets)
    if take_clip and copy_first:
        grab(); grow()
    elif take_clip:
        grow(); grab()
    else:
        grow()

    if need_clear and clear_first:
        wipe()

    for (br, bc) in targets:
        if take_clip:
            ops.append(30); sels.append([br * hi, bc * wi, 0, 0])   # paste origin
        else:
            stamp = [(br * hi + dr, bc * wi + dc) for (dr, dc) in cells]
            ops.append(fg); sels.append(sel_of(stamp))

    if need_clear and not clear_first:
        wipe()

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 007bbfb7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 007bbfb7"
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
                                f"for task 007bbfb7"
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
                    f"Failed to build a complete episode for task 007bbfb7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"007bbfb7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
