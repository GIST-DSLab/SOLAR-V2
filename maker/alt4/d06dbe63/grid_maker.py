"""
ARC Task: d06dbe63 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# 1. colours
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """The generator samples exactly two colours: the canvas colour and the dot
    colour (5 is reserved for the drawn staircases, so it is excluded)."""
    cols = [c for c in range(10) if c != 5]
    bgc, dotc = random.sample(cols, 2)
    return {"bgc": bgc, "dotc": dotc}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc) -> dict:
    obj1 = mapply(
        lbind(shift, frozenset({(-1, 0), (-2, 0), (-2, 1), (-2, 2)})),
        {(-k * 2, 2 * k) for k in range(15)},
    )
    obj2 = mapply(
        lbind(shift, frozenset({(1, 0), (2, 0), (2, -1), (2, -2)})),
        {(2 * k, -k * 2) for k in range(15)},
    )
    obj = obj1 | obj2
    objf = lambda ij: shift(obj, ij)
    hub = max(5, max_h)
    wub = max(5, max_w)
    h = unifint(diff_lb, diff_ub, (5, hub))
    w = unifint(diff_lb, diff_ub, (5, wub))
    ndots = unifint(diff_lb, diff_ub, (1, min(h, w)))
    succ = 0
    tr = 0
    maxtr = 4 * ndots
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    while tr < maxtr and succ < ndots:
        tr += 1
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        objx = objf(loc)
        if (objx & fullinds).issubset(inds):
            succ += 1
            inds = (inds - objx) - {loc}
            gi = fill(gi, dotc, {loc})
            go = fill(go, dotc, {loc})
            go = fill(go, 5, objx)
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. trajectory
# ----------------------------------------------------------------------------
# Rule: every dot grows a staircase of 5s -- one flight climbing up-and-right,
# one flight descending down-and-left (the two are 180-symmetric about the dot).
#
# Each flight is literally a chain of identical two-cell RUNGS laid on a 2x2
# lattice, alternating orientation:
#       up-right flight, step k:  vertical rung (r-2k-2, c+2k)..(r-2k-1, c+2k)
#                                 horizontal rung (r-2k-2, c+2k+1)..(+2k+2)
#       down-left flight, step k: vertical rung (r+2k+1, c-2k)..(r+2k+2, c-2k)
#                                 horizontal rung (r+2k+2, c-2k-2)..(c-2k-1)
# Those rungs tile the whole figure exactly (no overlaps), and the figures of
# different dots never overlap either.  So the honest account of the drawing is
# "stamp the repeating rung": paint ONE vertical rung, CopyO it, and Paste that
# stamp at every other vertical rung position (walking outward along each flight
# of each staircase); then do the same for the horizontal rung.  Rungs that hang
# off the edge of the grid cannot be stamped whole -- their surviving cells are
# painted directly.

_VERT, _HORZ = 0, 1


def _flight_rungs(r, c, direction):
    """Rungs of one flight of the staircase at (r, c), ordered outward."""
    out = []
    for k in range(15):
        if direction == 'up':
            vert = [(r - 2 * k - 2, c + 2 * k), (r - 2 * k - 1, c + 2 * k)]
            horz = [(r - 2 * k - 2, c + 2 * k + 1), (r - 2 * k - 2, c + 2 * k + 2)]
        else:
            vert = [(r + 2 * k + 1, c - 2 * k), (r + 2 * k + 2, c - 2 * k)]
            horz = [(r + 2 * k + 2, c - 2 * k - 2), (r + 2 * k + 2, c - 2 * k - 1)]
        out.append((_VERT, vert))
        out.append((_HORZ, horz))
    return out


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = the canvas colour the generator fills before placing dots
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    # every rung of every staircase, grouped dot by dot, flight by flight,
    # ordered outward from the dot
    plan = []
    for (r, c) in dots:
        for direction in ('up', 'down'):
            for orient, cells in _flight_rungs(r, c, direction):
                visible = [(a, b) for (a, b) in cells
                           if 0 <= a < hi and 0 <= b < wi]
                if not visible:
                    continue
                plan.append((orient, cells, visible, len(visible) == len(cells)))

    ops, sels = [], []

    # One pass per rung orientation: the clipboard can only hold one stamp at a
    # time, so all vertical rungs are laid first, then all horizontal ones.
    for orient in (_VERT, _HORZ):
        rungs = [p for p in plan if p[0] == orient]
        n_whole = sum(1 for p in rungs if p[3])
        stamped = False
        for (_, cells, visible, whole) in rungs:
            if whole and not stamped:
                # draw the master rung, then load it into the clipboard
                ops.append(5)
                sels.append(sel_of(cells))          # exactly the rung's 2 cells
                if n_whole > 1:
                    ops.append(29)                  # CopyO: the rung is the stamp
                    sels.append(sel_of(cells))
                stamped = True
            elif whole:
                top_left = min(cells)               # paste origin of this rung
                ops.append(30)
                sels.append(sel_of([top_left]))
            else:
                # rung running off the grid edge: only its surviving cells exist
                ops.append(5)
                sels.append(sel_of(visible))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])             # whole grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task d06dbe63"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d06dbe63"
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
                                f"for task d06dbe63"
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
                    f"Failed to build a complete episode for task d06dbe63 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d06dbe63-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
