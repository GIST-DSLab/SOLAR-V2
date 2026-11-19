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
    # The generator samples only one color randomly that carries a *role*:
    # the background. Object colors are arbitrary picks from the remaining
    # colors and are simply copied into the output column, so the rule
    # (order objects by size, list their colors) does not depend on them.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ----------------------------------------------------------------------------- 2
def generate(diff_lb, diff_ub, max_h, max_w, bgc, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (min(10, max_h), min(10, max_w)))
    go = []
    for _attempt in range(20):
        h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
        nobjs = unifint(diff_lb, diff_ub, (1, max(1, min(30, (h * w) // 25))))
        gi = canvas(bgc, (h, w))
        numcells = unifint(diff_lb, diff_ub, (nobjs + 1, 36))
        base = asindices(canvas(-1, (6, 6)))
        maxtr = 10
        inds = asindices(gi)
        go = []
        for k in range(nobjs):
            if len(inds) == 0 or numcells < 2:
                break
            lo = nobjs - k
            hi_ = numcells - 1
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
            validloc = False
            rems = sfilter(inds, lambda ij: ij[0] <= h - height(shp) and ij[1] <= w - width(shp))
            if len(rems) == 0:
                break
            loc = choice(totuple(rems))
            tr = 0
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
        if len(go) > 0:
            break
    return {'input': gi, 'output': dmirror((tuple(go),))}


# ----------------------------------------------------------------------------- 3
def derive_operations(I, O):
    """
    Rule: every object (8-connected, single-colored, non-background blob) is
    reduced to one cell of its color; the cells are stacked into a single
    column ordered by object size (ascending), and that column is then
    MIRRORED top<->bottom (hmirror) to give the final answer.

    Trajectory:
      1. CropGrid  -> canvas becomes the n x 1 column the answer lives in.
      2. one Color op per object, written top-to-bottom in ASCENDING size
         order (smallest object first) -- the pre-mirror column.
      3. FlipV (op27 = flipud = hmirror) on that whole column -- the
         reflection the rule is about, performed, not just reflected in the
         final grid.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    n = O.shape[0]

    # background: the color the generator paints the canvas with before
    # placing the (sparse, <=36 cells total) objects -> the majority color.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- find the objects: 8-connected, single colored, non background ------
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
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
                comps.append((size, col))

    comps.sort(key=lambda t: t[0])                      # ascending by size
    asc = [c for _, c in comps]                         # pre-mirror column
    desc_target = [int(O[i, 0]) for i in range(n)]
    if len(asc) != n or list(reversed(asc)) != desc_target:
        # size ties (not produced by this generator, but stay robust)
        asc = list(reversed(desc_target))

    ops, sels = [], []

    # 1. the answer is a single column of n cells: resize the canvas to it.
    #    (bbox == exactly the whole region we want the canvas to become)
    ops.append(33); sels.append([0, 0, n - 1, 0])

    # what the canvas holds after the crop: column 0 of I, zero-padded below
    cur = [int(I[r, 0]) if r < hi else 0 for r in range(n)]

    # 2. write one cell per object, ordered by object size.
    mirror = (asc != list(reversed(asc)))               # identity flip? skip it
    target = asc if mirror else desc_target
    for i, col in enumerate(target):
        if cur[i] != col:                               # already that color -> no-op
            ops.append(int(col)); sels.append(sel_of([(i, 0)]))
            cur[i] = int(col)

    # 3. the reflection itself: flipud of the whole n x 1 column
    if mirror:
        ops.append(27); sels.append([0, 0, n - 1, 0])   # bbox == the entire grid

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
