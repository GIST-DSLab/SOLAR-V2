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

try:
    from dsl import *
except Exception:
    pass
try:
    from utils import *
except Exception:
    pass

from maker.sel_helpers import sel_of


# Reading of the task: one solid rectangle stays put, and one ragged shape slides
# toward it until they touch. The code-golf entry sorts whole rows (or columns) so
# that the background lines between the two objects move to the far side of the
# shape. That is the same rule. The shape's row band holds nothing except the
# shape and background, so moving that band is the same as moving the shape.
# Both readings treat the ragged shape as one unit that slides in one direction
# until it touches the rectangle. So the plan is kept: grab the shape once, take
# unit Moves until it touches, then repair only the cells it left behind.


def sample_colors(num_examples=None) -> dict:
    bgc, fgc, destc = random.sample(list(range(10)), 3)
    return {"bgc": bgc, "fgc": fgc, "destc": destc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, destc=None, **kw) -> dict:
    if bgc is None or fgc is None or destc is None:
        bgc, fgc, destc = random.sample(list(range(10)), 3)
    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
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


def _cells_of(G, col):
    return {(int(r), int(c)) for r, c in zip(*np.where(G == col))}


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _is_filled(cells):
    r0, r1, c0, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_translate(a, b):
    if len(a) != len(b) or a == b or not a:
        return False
    ar, _, ac, _ = _bbox(a)
    br, _, bc, _ = _bbox(b)
    return {(r - ar, c - ac) for r, c in a} == {(r - br, c - bc) for r, c in b}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    colors = sorted(set(I.flatten().tolist()))
    cells = {c: _cells_of(I, c) for c in colors}

    # Colour roles are fixed for the whole episode. In the demonstrations, the
    # shape is the colour whose cells come back as a pure translation.
    shape_col = None
    if examples:
        votes = Counter()
        for xi, xo in examples:
            xi = np.asarray(xi, dtype=int)
            xo = np.asarray(xo, dtype=int)
            if xi.shape != xo.shape:
                continue
            for c in set(xi.flatten().tolist()):
                a = _cells_of(xi, c)
                if not _is_filled(a) and _is_translate(a, _cells_of(xo, c)):
                    votes[c] += 1
        for c, _ in votes.most_common():
            if c in cells and not _is_filled(cells[c]):
                shape_col = c
                break

    filled = [c for c in colors if _is_filled(cells[c]) and c != shape_col]
    if not filled:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    anchor_col = max(filled, key=lambda c: len(cells[c]))
    rest = [c for c in colors if c != anchor_col]

    if shape_col is None or shape_col not in rest:
        # Fallback: the background spans the grid, so it has the larger bounding box.
        def area(c):
            r0, r1, c0, c1 = _bbox(cells[c])
            return ((r1 - r0 + 1) * (c1 - c0 + 1), len(cells[c]))
        rest_sorted = sorted(rest, key=area, reverse=True)
        if len(rest_sorted) < 2:
            ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
            return ops, sels
        bgc, shape_col = rest_sorted[0], rest_sorted[1]
    else:
        others = [c for c in rest if c != shape_col]
        if not others:
            ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
            return ops, sels
        bgc = max(others, key=lambda c: len(cells[c]))

    obj = cells[shape_col]
    rect = cells[anchor_col]
    orr0, orr1, occ0, occ1 = _bbox(obj)
    rr0, rr1, rc0, rc1 = _bbox(rect)

    if orr0 > rr1:
        dr, dc = -1, 0          # rectangle above: slide up
    elif orr1 < rr0:
        dr, dc = 1, 0           # rectangle below: slide down
    elif occ0 > rc1:
        dr, dc = 0, -1          # rectangle to the left: slide left
    elif occ1 < rc0:
        dr, dc = 0, 1           # rectangle to the right: slide right
    else:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # Slide until the next step would run into the rectangle.
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
        if shape_col != 0:
            mv = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
            ops.append(mv); sels.append(sel_of(sorted(obj)))      # grab the shape
            for _ in range(steps - 1):
                ops.append(mv); sels.append(sel_of([]))           # keep the same grab
            hole = sorted(set(obj) - cur)                         # vacated footprint only
            if bgc != 0 and hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))
        else:
            # A Move carries no 0 cells, so a colour-0 shape is lifted and put down instead.
            ops.append(int(bgc)); sels.append(sel_of(sorted(obj)))
            ops.append(0); sels.append(sel_of(sorted(cur)))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])           # full grid: submit
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
