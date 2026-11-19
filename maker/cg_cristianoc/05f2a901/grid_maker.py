"""
ARC Task: 05f2a901 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of

try:
    from dsl import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass


# Reading compared with the golf entry: both treat the task the same way.
# There is one solid rectangle (the anchor) and one ragged shape (the traveller).
# The traveller slides straight toward the anchor along the axis where they are
# separated and stops one step before touching it. The golf entry gets the
# distance from the gap between the two bounding boxes. Here we step the actual
# cells forward until the next step would overlap the rectangle. This matches
# the generator (shift until overlap, then back off one). It also stays right
# when the shape's leading edge is ragged. The unit of action is the same in
# both: one grab of the whole shape, then a Move chain. So the operations do
# not change.


def sample_colors(num_examples=None) -> dict:
    bgc, fgc, destc = random.sample(range(10), 3)
    return {"bgc": bgc, "fgc": fgc, "destc": destc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, destc=None, **kwargs) -> dict:
    if bgc is None or fgc is None or destc is None:
        bgc, fgc, destc = random.sample(range(10), 3)
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
    objh = unifint(diff_lb, diff_ub, (2, min(w // 2, h // 2)))
    objw = unifint(diff_lb, diff_ub, (objh, w // 2))
    bb = asindices(canvas(-1, (objh, objw)))
    sp = choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncells = unifint(diff_lb, diff_ub, (objh + objw, objh * objw))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    if height(obj) * width(obj) == len(obj):
        obj = remove(choice(totuple(obj)), obj)
    obj = normalize(obj)
    objh, objw = shape(obj)
    loci = unifint(diff_lb, diff_ub, (3, h - objh))
    locj = unifint(diff_lb, diff_ub, (0, w - objw))
    loc = (loci, locj)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    obj = shift(obj, loc)
    gi = fill(gi, fgc, obj)
    sqd = randint(1, min(w, loci - 1))
    locisq = randint(0, loci - sqd - 1)
    locjsq = randint(locj - sqd + 1, locj + objw - 1)
    sq = backdrop({(locisq, locjsq), (locisq + sqd - 1, locjsq + sqd - 1)})
    gi = fill(gi, destc, sq)
    go = fill(go, destc, sq)
    while len(obj & sq) == 0:
        obj = shift(obj, (-1, 0))
    obj = shift(obj, (1, 0))
    go = fill(go, fgc, obj)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Group the cells by colour. There are exactly three roles: background, anchor, traveller.
    by_col = {}
    for r in range(hi):
        for c in range(wi):
            by_col.setdefault(int(I[r, c]), set()).add((r, c))

    solid, ragged = [], []
    for col, cells in by_col.items():
        r0, r1, c0, c1 = _bbox(cells)
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
            solid.append((col, cells))
        else:
            ragged.append((col, cells))

    if not solid or len(ragged) < 2:
        if len(solid) >= 2 and len(ragged) == 1:
            pass
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # The background always has more cells than the ragged shape, because the
    # shape fits inside a box of at most half the grid in each direction.
    ragged.sort(key=lambda t: len(t[1]), reverse=True)
    bgc = ragged[0][0]
    obj_col, obj = ragged[1]
    rect = max(solid, key=lambda t: len(t[1]))[1]

    orr0, orr1, occ0, occ1 = _bbox(obj)
    rr0, rr1, rc0, rc1 = _bbox(rect)
    if orr0 > rr1:
        dr, dc = -1, 0
    elif orr1 < rr0:
        dr, dc = 1, 0
    elif occ0 > rc1:
        dr, dc = 0, -1
    elif occ1 < rc0:
        dr, dc = 0, 1
    else:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # Slide the shape forward until the next step would run into the rectangle.
    cur = set(obj)
    steps = 0
    while steps <= 60:
        nxt = {(r + dr, c + dc) for r, c in cur}
        if nxt & rect:
            break
        if not all(0 <= r < hi and 0 <= c < wi for r, c in nxt):
            break
        cur = nxt
        steps += 1

    if steps > 0:
        if obj_col != 0:
            mv = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
            ops.append(mv); sels.append(sel_of(sorted(obj)))      # grab the shape
            for _ in range(steps - 1):
                ops.append(mv); sels.append(sel_of([]))           # keep sliding the same grab
            hole = sorted(set(obj) - cur)                         # vacated original footprint
            if bgc != 0 and hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))
        else:
            # A colour-0 traveller cannot be carried by Move (0 is transparent),
            # so lift it off its footprint and paint it at the landing position.
            ops.append(int(bgc)); sels.append(sel_of(sorted(obj)))
            ops.append(0); sels.append(sel_of(sorted(cur)))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])           # submit the whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task 05f2a901"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 05f2a901"
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
                                f"for task 05f2a901"
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
                    f"Failed to build a complete episode for task 05f2a901 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"05f2a901-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
