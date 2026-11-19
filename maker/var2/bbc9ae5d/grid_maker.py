"""
ARC Task: bbc9ae5d (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if a > b:
        a, b = b, a
    lo = int(max(a, np.floor(a + (b - a) * diff_lb)))
    hi = int(min(b, np.ceil(a + (b - a) * diff_ub)))
    if lo > hi:
        lo, hi = hi, lo
    return random.randint(lo, hi)


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # c1 = the leading colour of row 0.  Structural role (it is the field the
    # output is drawn on), so it is fixed for the whole episode.  ccols_pool =
    # the palette the tail pixels draw from; fixing the pool keeps one colour
    # scheme across examples and test.  Either may be 0 — 0 is an ordinary
    # colour of this picture, never "nothing", and the route below handles it.
    cols = list(range(10))
    c1 = random.choice(cols)
    pool = [c for c in cols if c != c1]
    random.shuffle(pool)
    return {"c1": c1, "ccols_pool": pool}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, c1=None, ccols_pool=None) -> dict:
    cols = list(range(10))
    if c1 is None:
        c1 = random.choice(cols)
    remcols = [c for c in cols if c != c1]
    if ccols_pool is None:
        ccols_pool = remcols[:]
        random.shuffle(ccols_pool)
    ccols_pool = [c for c in ccols_pool if c != c1]
    if not ccols_pool:
        ccols_pool = remcols[:]

    # input 1 x w, output (w//2) x w  ->  w <= max_w  and  w//2 <= max_h
    half_ub = max(2, min(15, max_w // 2, max_h))
    half = _unifint(diff_lb, diff_ub, (2, half_ub))
    w = half * 2

    locinv = _unifint(diff_lb, diff_ub, (2, w))
    locj = w - locinv               # last column of the leading run, 0 <= locj <= w-2

    numc = _unifint(diff_lb, diff_ub, (1, min(9, len(ccols_pool))))
    ccols = ccols_pool[:numc]

    row = [c1] * (locj + 1) + [random.choice(ccols) for _ in range(w - locj - 1)]
    gi = (tuple(row),)

    ho = w // 2
    go = tuple(
        tuple(c1 if c <= locj + r else row[c - r] for c in range(w))
        for r in range(ho)
    )
    return {"input": gi, "output": go}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
#
# The rule (read off the examples): the 1 x w strip becomes a (w//2) x w field.
# Every pixel of the strip shoots a diagonal ray down-right from where it sits;
# the leading colour, which owns column 0, also owns everything left of its own
# ray, so it is the field the rest of the picture is drawn on.
#
# There is more than one sane way to draw that, and which one gets used is read
# off the strip itself — the leading colour, how far its run reaches, how many
# runs the tail has — so one instance always takes one route, and the variety
# comes from the instances differing:
#
#   route 0  lay the leading colour over every row the resize added, then draw
#            the rays on top of it
#   route 1  paint the leading colour's own wedge exactly, then draw the rays
#            into the empty part that is left
#   route 2  lay the leading colour down, then slide the tail strip one step
#            down-right per row (copy/paste instead of drawing rays)
#
# and for routes 0/1 the rays are swept per pixel / per run / per colour,
# left to right or right to left.
# ----------------------------------------------------------------------------
def _lead_run(row):
    """Leading colour and the last column of its run.  Tail colours never repeat it."""
    c1 = int(row[0])
    j = 0
    while j + 1 < len(row) and int(row[j + 1]) == c1:
        j += 1
    return c1, j


def _tail_runs(row, locj):
    runs, j, w = [], locj + 1, len(row)
    while j < w:
        c, a = int(row[j]), j
        while j + 1 < w and int(row[j + 1]) == c:
            j += 1
        runs.append((c, a, j))
        j += 1
    return runs


def _ray(j, ho, wo):
    """The part of column j's down-right ray that lies below row 0."""
    return [(r, j + r) for r in range(1, ho) if j + r < wo]


def _ray_groups(row, locj, ho, wo, mode):
    """The rays to draw, grouped and ordered the way this instance asks for."""
    runs = _tail_runs(row, locj)
    if mode in (0, 1):                       # one op per tail pixel
        items = [(int(row[j]), _ray(j, ho, wo)) for j in range(locj + 1, len(row))]
        if mode == 1:
            items = items[::-1]
    elif mode in (2, 3):                     # one op per run (a parallelogram)
        items = [(c, [x for j in range(a, b + 1) for x in _ray(j, ho, wo)])
                 for c, a, b in runs]
        if mode == 3:
            items = items[::-1]
    else:                                    # one op per colour, first seen first
        order, cells = [], {}
        for c, a, b in runs:
            if c not in cells:
                cells[c] = []
                order.append(c)
            for j in range(a, b + 1):
                cells[c].extend(_ray(j, ho, wo))
        items = [(c, cells[c]) for c in order]
    # the last column's ray never reaches below row 0 — nothing to draw for it
    return [(c, cs) for c, cs in items if cs]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    row = [int(v) for v in I[0]]
    wo = len(row)
    ho = wo // 2                              # the examples fix this: half the width

    c1, locj = _lead_run(row)
    runs = _tail_runs(row, locj)

    ops, sels = [], []

    # 1. grow the strip into the field (whole rectangle -> bbox is exact)
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])

    route = (c1 + locj + ho) % 3
    mode = (locj + len(runs)) % 5

    if route in (0, 1):
        # 2. the leading colour first: over every row the resize just added
        #    (route 0), or exactly over the wedge it owns (route 1).  When the
        #    leading colour is 0 the new rows already read 0 — nothing to lay.
        if c1 != 0:
            if route == 0:
                ops.append(c1)
                sels.append([1, 0, ho - 2, wo - 1])   # the added rows, whole
            else:
                wedge = [(r, c) for r in range(1, ho)
                         for c in range(0, min(locj + r, wo - 1) + 1)]
                ops.append(c1)
                sels.append(sel_of(wedge))

        # 3. the tail's rays on top, in this instance's grouping and direction.
        for col, cells in _ray_groups(row, locj, ho, wo, mode):
            if route == 1 and col == 0:
                # nothing was laid outside the wedge, so a 0 ray already reads 0
                continue
            ops.append(col)
            sels.append(sel_of(cells))
    else:
        # 2. same field of the leading colour under everything
        if c1 != 0:
            ops.append(c1)
            sels.append([1, 0, ho - 2, wo - 1])

        # 3. each row is the tail strip pushed one further down-right and cut
        #    off at the right edge: copy the piece that still fits, paste it.
        top_down = (c1 + wo) % 2 == 0
        for k in (range(1, ho) if top_down else range(ho - 1, 0, -1)):
            a, b = locj + 1, wo - 1 - k
            if b < a:
                continue                      # this row is all leading colour
            if all(row[j] == 0 for j in range(a, b + 1)):
                continue                      # a copy of nothing pastes nothing
            ops.append(28)
            sels.append([0, a, 0, b - a])     # the piece of the strip that fits
            ops.append(30)
            sels.append([k, a + k, 0, 0])     # dropped one row, one column right

        # 4. a paste carries no 0s, so the strip's 0 pixels never travelled:
        #    put them down where they land (nothing emitted if there are none).
        zcells = [x for c, a, b in runs if c == 0
                  for j in range(a, b + 1) for x in _ray(j, ho, wo)]
        if c1 != 0 and zcells:
            ops.append(0)
            sels.append(sel_of(zcells))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task bbc9ae5d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bbc9ae5d"
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
                                f"for task bbc9ae5d"
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
                    f"Failed to build a complete episode for task bbc9ae5d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bbc9ae5d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
