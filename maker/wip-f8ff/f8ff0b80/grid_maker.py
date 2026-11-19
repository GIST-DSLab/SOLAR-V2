"""
ARC Task: f8ff0b80 (RE-ARC) — LLM-generated grid_maker
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
import random
from collections import Counter

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------- 1
def sample_colors(num_examples=None) -> dict:
    # The only color the generator samples with a *role* is the background it
    # paints the canvas with. The object colors are arbitrary draws from the
    # remaining colors and are merely carried into the answer column, so the
    # rule (list the object colors, largest object first) does not depend on
    # which colors they are.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ----------------------------------------------------------------------------- 2
def generate(diff_lb, diff_ub, max_h, max_w, bgc, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)

    hmax = min(30, max_h)
    wmax = min(30, max_w)
    hlo = min(10, hmax)
    wlo = min(10, wmax)

    fallback = None
    for _attempt in range(60):
        h = unifint(diff_lb, diff_ub, (hlo, hmax))
        w = unifint(diff_lb, diff_ub, (wlo, wmax))
        # nobjs is additionally capped at h so the answer column always fits
        # inside a column of the input grid
        nobjs = unifint(diff_lb, diff_ub, (1, max(1, min(30, h, (h * w) // 25))))
        gi = canvas(bgc, (h, w))
        numcells = unifint(diff_lb, diff_ub, (nobjs + 1, 36))
        base = asindices(canvas(-1, (6, 6)))
        maxtr = 10
        inds = asindices(gi)
        go = []
        for k in range(nobjs):
            if len(inds) == 0 or numcells < 2:
                break
            lo, hi_ = nobjs - k, numcells - 1
            if lo > hi_:
                break
            numcells = unifint(diff_lb, diff_ub, (lo, hi_))
            if numcells == 0:
                break
            sp = choice(totuple(base))
            shp = {sp}
            reminds = remove(sp, base)
            for kk in range(numcells - 1):
                cands = totuple((reminds - shp) & mapply(neighbors, shp))
                if len(cands) == 0:
                    break
                shp.add(choice(cands))
            shp = normalize(shp)
            rems = sfilter(inds, lambda ij: ij[0] <= h - height(shp) and ij[1] <= w - width(shp))
            if len(rems) == 0:
                break
            validloc = False
            tr = 0
            loc = choice(totuple(rems))
            while not validloc and tr < maxtr:
                loc = choice(totuple(inds))
                validloc = shift(shp, loc).issubset(inds)
                tr += 1
            if validloc:
                plcd = shift(shp, loc)
                col = choice(remcols)
                go.append(col)
                inds = (inds - plcd) - mapply(neighbors, plcd)
                gi = fill(gi, col, plcd)
        if len(go) == 0:
            continue
        if fallback is None:
            fallback = (gi, go)
        # keep instances whose mirror is actually a mirror: at least two objects
        # and a column that changes when it is flipped top<->bottom
        if len(go) >= 2 and list(go) != list(go)[::-1]:
            fallback = (gi, go)
            break
    if fallback is None:
        raise ValueError("no object could be placed")
    gi, go = fallback
    return {'input': gi, 'output': dmirror((tuple(go),))}


# ----------------------------------------------------------------------------- 3
def derive_operations(I, O):
    """
    Rule, read off the input alone: every object (8-connected, single-colored,
    non-background blob) shrinks to one cell of its color; those cells are
    stacked into one column in order of object size, SMALLEST FIRST, and the
    column is then mirrored top<->bottom, which is what puts the largest
    object's color at the top of the answer.

    Trajectory:
      1. one Color op per object, smallest object first, writing its color into
         the next cell of an empty column of the input -- the pre-mirror column
         is built while the objects it is measured from are still on the grid.
      2. CropGrid onto that column -- the canvas becomes the n x 1 answer.
      3. FlipV (op27 = flipud = hmirror) over the whole column -- the mirror.
    Every parameter (n, the colors, their order, which column, the axis) comes
    from I; O is never inspected.
    """
    import numpy as _np
    from collections import Counter as _Counter

    I = _np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = the color the canvas is painted with before the objects are
    # placed; the objects together cover at most 36 cells of a >=10x10 grid, so
    # it is the majority color by a wide margin.
    bgc = _Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- the objects: 8-connected, single-colored, non-background ------------
    seen = _np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                and I[nr, nc] == col:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append((size, (r, c), col))

    ops, sels = [], []
    if not comps:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    comps.sort(key=lambda t: (t[0], t[1]))       # ascending size, raster tie-break
    asc = [t[2] for t in comps]                  # the column before the mirror
    n = len(asc)

    # --- where to build the column: the column of I that needs the most work,
    #     i.e. the one holding the fewest cells that already show the color the
    #     object of that rank would write (a background-only column has none).
    def _conflicts(c):
        return sum(1 for i in range(n)
                   if i < hi and int(I[i, c]) == asc[i])

    ccol = min(range(wi), key=_conflicts)

    if n <= hi:
        # 1. write one cell per object into that column, smallest object first
        for i, col in enumerate(asc):
            if int(I[i, ccol]) != col:
                ops.append(int(col)); sels.append(sel_of([(i, ccol)]))
        # 2. the answer is that column: crop the canvas down to it
        ops.append(33); sels.append([0, ccol, n - 1, 0])
    else:
        # more objects than the grid is tall: make the canvas the n x 1 column
        # first (crop pads with 0 below the grid), then write into it
        ops.append(33); sels.append([0, ccol, n - 1, 0])
        for i, col in enumerate(asc):
            cur = int(I[i, ccol]) if i < hi else 0
            if cur != col:
                ops.append(int(col)); sels.append(sel_of([(i, 0)]))

    # 3. the mirror itself: flip the whole n x 1 column top<->bottom
    #    (bbox == exactly the whole canvas)
    ops.append(27); sels.append([0, 0, n - 1, 0])

    ops.append(34); sels.append([0, 0, n - 1, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8ff0b80"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8ff0b80"
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
                                f"for task f8ff0b80"
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
                    f"Failed to build a complete episode for task f8ff0b80 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8ff0b80-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
