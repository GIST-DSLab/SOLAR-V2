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


def sample_colors(num_examples=None) -> dict:
    # 5 is reserved by the rule for the staircase it draws, so it is never a
    # canvas/dot colour.  bgc and dotc are the generator's two sampled roles.
    cols = [c for c in range(10) if c != 5]
    bgc, dotc = random.sample(cols, 2)
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, dotc: int) -> dict:
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
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
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


def derive_operations(I, O, examples=None):
    """Every dot shoots a staircase through itself.  A vertical bar runs through
    the dot, and from its two ends the SAME 3-cell bar repeats over and over,
    alternately horizontal and vertical, climbing up-right and falling down-left.

    So the ladder is built the way it is read: draw one rung, pick it up, and
    stamp that one rung at every further rung of the ladder.  Only the rungs at
    the two ends, which run off the edge of the grid, are painted directly --
    there is no whole rung there to stamp.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # colour the rule draws with — read off the demonstrations, not off O
    fillc = None
    if examples:
        for ein, eout in examples:
            a = np.asarray(ein, dtype=int)
            b = np.asarray(eout, dtype=int)
            extra = set(b.flatten().tolist()) - set(a.flatten().tolist())
            if len(extra) == 1:
                fillc = int(extra.pop())
                break
    if fillc is None:
        fillc = 5                      # hardcoded in the task, not a sampled role

    # the canvas colour the generator fills before dropping the dots
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]

    G = I.copy()
    ops, sels = [], []
    clip = None                        # which rung the clipboard is holding

    def inb(r, c):
        return 0 <= r < h and 0 <= c < w

    def cells_of(bar):
        r0, c0, o = bar
        if o == 'H':
            return [(r0, c0), (r0, c0 + 1), (r0, c0 + 2)]
        return [(r0, c0), (r0 + 1, c0), (r0 + 2, c0)]

    def paint(cells):
        todo = [(r, c) for (r, c) in cells if inb(r, c) and G[r, c] != fillc]
        if not todo:
            return False
        ops.append(fillc)
        sels.append(sel_of(todo))
        for (r, c) in todo:
            G[r, c] = fillc
        return True

    for (dr, dc) in dots:
        # 1. the bar that runs through the dot itself (the dot keeps its colour)
        paint([(dr - 2, dc), (dr - 1, dc), (dr + 1, dc), (dr + 2, dc)])

        # 2. the ladder's rungs, listed outward from the dot in both directions
        hbars, vbars = [], []
        for k in range(15):
            hbars.append((dr - 2 - 2 * k, dc + 2 * k, 'H'))          # up-right ray
            hbars.append((dr + 2 + 2 * k, dc - 2 - 2 * k, 'H'))      # down-left ray
        for k in range(14):
            vbars.append((dr - 4 - 2 * k, dc + 2 + 2 * k, 'V'))
            vbars.append((dr + 2 + 2 * k, dc - 2 - 2 * k, 'V'))
        groups = []
        for group in (hbars, vbars):
            bars = [b for b in group if any(inb(r, c) for (r, c) in cells_of(b))]
            whole = [b for b in bars if all(inb(r, c) for (r, c) in cells_of(b))]
            groups.append((bars, whole))

        # 3. one whole rung is drawn, then that same rung is stamped at every
        #    other one.  The clipboard keeps the last rung picked up, so it is
        #    only picked up again when the other orientation is needed.
        for kind, (bars, whole) in zip(('H', 'V'), groups):
            if not whole:
                continue
            if clip == kind:                          # rung already in hand
                start = 0
            else:
                paint(cells_of(whole[0]))             # draw this rung once ...
                start = 1
                if len(whole) > 1:
                    ops.append(29)                    # ... CopyO: pick it up
                    sels.append(sel_of(cells_of(whole[0])))   # 3 cells in a line = its own rectangle
                    clip = kind
            for b in whole[start:]:
                cells = cells_of(b)
                if any(G[r, c] != fillc for (r, c) in cells):
                    ops.append(30)                    # Paste the same rung again
                    sels.append(sel_of([cells[0]]))
                    for (r, c) in cells:
                        G[r, c] = fillc

        # 4. the rungs at the two ends run off the grid: paint the stub that shows
        for bars, whole in groups:
            for b in bars:
                if b not in whole:
                    paint(cells_of(b))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])                 # full-grid bbox: submit
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
