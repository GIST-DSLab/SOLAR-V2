"""
ARC Task: 776ffc46 (RE-ARC) — LLM-generated grid_maker
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
from random import randint, choice, sample
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # generator samples 4 distinct roles: background, box outline, inside-marker, outside objects.
    # the rule ("recolor outside objects that match the marker's shape into the marker's color")
    # is color-role dependent, so all four must stay fixed across the episode.
    cols = list(range(10))
    bgc, sqc, inc, outc = sample(cols, 4)
    return {"bgc": bgc, "sqc": sqc, "inc": inc, "outc": outc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, sqc: int, inc: int, outc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    gi = canvas(bgc, (h, w))
    sqh = randint(3, h // 3 + 1)
    sqw = randint(3, w // 3 + 1)
    loci = randint(0, 3)
    locj = randint(0, w - sqw)
    bx = box(frozenset({(loci, locj), (loci + sqh - 1, locj + sqw - 1)}))
    bounds = asindices(canvas(-1, (sqh - 2, sqw - 2)))
    obj = {choice(totuple(bounds))}
    ncells = randint(1, (sqh - 2) * (sqw - 2))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    objp = shift(obj, (loci + 1 + randint(0, sqh - oh - 2), locj + 1 + randint(0, sqw - ow - 2)))
    gi = fill(gi, sqc, bx)
    gi = fill(gi, inc, objp)
    inds = (ofcolor(gi, bgc) - backdrop(bx)) - mapply(neighbors, backdrop(bx))
    cands = sfilter(inds, lambda ij: shift(obj, ij).issubset(inds))
    loc = choice(totuple(cands))
    plcd = shift(obj, loc)
    gi = fill(gi, outc, plcd)
    inds = (inds - plcd) - mapply(neighbors, plcd)
    noccs = unifint(diff_lb, diff_ub, (0, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    fullinds = asindices(gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        if choice((True, False)):
            sqh = randint(3, h // 3 + 1)
            sqw = randint(3, w // 3 + 1)
            bx = box(frozenset({(0, 0), (sqh - 1, sqw - 1)}))
            bounds = asindices(canvas(-1, (sqh - 2, sqw - 2)))
            obj2 = {choice(totuple(bounds))}
            ncells = randint(1, (sqh - 2) * (sqw - 2))
            for k in range(ncells - 1):
                obj2.add(choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
            if normalize(obj2) == obj:
                if len(obj2) < (sqh - 2) * (sqw - 2):
                    obj2.add(choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
                else:
                    continue
            obj2 = normalize(obj2)
            ooh, oow = shape(obj2)
            cands1 = connect((-1, -1), (-1, w - sqw + 1))
            cands2 = connect((h - sqh + 1, -1), (h - sqh + 1, w - sqw + 1))
            cands3 = connect((-1, -1), (h - sqh + 1, -1))
            cands4 = connect((-1, w - sqw + 1), (h - sqh + 1, w - sqw + 1))
            cands = cands1 | cands2 | cands3 | cands4
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            sloci, slocj = loc
            plcdbx = shift(bx, loc)
            if (backdrop(plcdbx) & fullinds).issubset(inds):
                succ += 1
                oloci = randint(sloci + 1, sloci + 1 + randint(0, sqh - ooh - 2))
                olocj = randint(slocj + 1, slocj + 1 + randint(0, sqw - oow - 2))
                gi = fill(gi, sqc, plcdbx)
                gi = fill(gi, inc, shift(obj2, (oloci, olocj)))
                inds = inds - backdrop(outbox(plcdbx))
        else:
            ooh = randint(1, h // 3 - 1)
            oow = randint(1, w // 3 - 1)
            bounds = asindices(canvas(-1, (ooh, oow)))
            obj2 = {choice(totuple(bounds))}
            ncells = randint(1, oow * ooh)
            for k in range(ncells - 1):
                obj2.add(choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
            if normalize(obj2) == obj:
                if len(obj2) < ooh * oow:
                    obj2.add(choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
                else:
                    continue
        if choice((True, False, False)):
            obj2 = obj
        obj2 = normalize(obj2)
        ooh, oow = shape(obj2)
        for kk in range(randint(1, 3)):
            cands = sfilter(inds, lambda ij: ij[0] <= h - ooh and ij[1] <= w - oow)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            plcd = shift(obj2, loc)
            if plcd.issubset(inds):
                succ += 1
                inds = (inds - plcd) - mapply(neighbors, plcd)
                gi = fill(gi, outc, plcd)
    objs = objects(gi, T, F, F)
    objs = colorfilter(objs, outc)
    objs = mfilter(objs, lambda o: equality(normalize(toindices(o)), obj))
    go = fill(gi, inc, objs)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read off I alone):
      - background = dominant color of I
      - find the hollow rectangle outline (component whose cells are exactly the ring of
        its bbox, with a real interior i.e. both bbox sides >= 3); the biggest such ring
        is the 'key' box, its color is the outline color.
      - the single object sitting inside that box gives the KEY COLOR and the KEY SHAPE
        (its cells, normalized to its own bbox).
      - every loose object elsewhere in the grid (color other than the outline color and
        other than the key color) whose normalized shape equals the key shape is
        repainted into the key color; everything else is left untouched.
    Each matched object is a 4-connected single-color region -> one FloodFill per object.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- 4-connected single-color components of the non-background cells ---
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = x + dx, y + dy
                    if 0 <= a < h and 0 <= b < w and not seen[a, b] and I[a, b] == col:
                        seen[a, b] = True
                        stack.append((a, b))
            cells.sort()
            comps.append((col, cells))

    def bbox(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return min(rs), max(rs), min(cs), max(cs)

    def is_ring(cells):
        r0, r1, c0, c1 = bbox(cells)
        ring = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)}
        return set(cells) == ring

    def norm(cells):
        r0, r1, c0, c1 = bbox(cells)
        return frozenset((r - r0, c - c0) for r, c in cells)

    # --- the key box: largest hollow outline ---
    rings = []
    for col, cells in comps:
        r0, r1, c0, c1 = bbox(cells)
        if r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3 and is_ring(cells):
            rings.append((col, cells, (r1 - r0 + 1) * (c1 - c0 + 1)))
    if not rings:
        ops.append(34)
        sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
        return ops, sels
    box_col, box_cells, _ = max(rings, key=lambda t: t[2])
    br0, br1, bc0, bc1 = bbox(box_cells)

    # --- the marker inside that box: its color and its shape are the key ---
    inner = [(r, c) for r in range(br0 + 1, br1)
             for c in range(bc0 + 1, bc1) if I[r, c] != bgc]
    if not inner:
        ops.append(34)
        sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
        return ops, sels
    key_col = int(I[inner[0]])
    key_shape = norm(inner)

    # --- repaint each loose object whose shape matches the key shape ---
    for col, cells in comps:
        if col == box_col or col == key_col:
            continue
        if norm(cells) != key_shape:
            continue
        sr, sc = cells[0]
        ops.append(10 + key_col)
        sels.append([sr, sc, 0, 0])

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 776ffc46"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 776ffc46"
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
                                f"for task 776ffc46"
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
                    f"Failed to build a complete episode for task 776ffc46 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"776ffc46-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
