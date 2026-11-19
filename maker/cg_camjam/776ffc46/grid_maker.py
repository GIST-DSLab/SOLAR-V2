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
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, inc, outc = random.sample(range(10), 4)
    return {"bgc": bgc, "sqc": sqc, "inc": inc, "outc": outc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, inc, outc) -> dict:
    from random import randint, choice

    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))

    def dneighbors(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def neighbors(p):
        r, c = p
        return {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    def mapply_n(f, cells):
        out = set()
        for p in cells:
            out |= f(p)
        return out

    def normalize(cells):
        cells = set(cells)
        r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
        return frozenset((r - r0, c - c0) for r, c in cells)

    def shift(cells, d):
        return frozenset((r + d[0], c + d[1]) for r, c in cells)

    def shape(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    def box(r0, c0, r1, c1):
        return frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                         if r in (r0, r1) or c in (c0, c1))

    def backdrop(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return frozenset((r, c) for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1))

    def outbox(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return box(min(rs) - 1, min(cs) - 1, max(rs) + 1, max(cs) + 1)

    def grow(bounds, n):
        obj = {choice(sorted(bounds))}
        for _ in range(n - 1):
            obj.add(choice(sorted((bounds - obj) & mapply_n(dneighbors, obj))))
        return obj

    def fill(g, col, cells):
        for r, c in cells:
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                g[r, c] = col

    def ofcolor(g, col):
        return frozenset((int(a), int(b)) for a, b in zip(*np.nonzero(g == col)))

    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    gi = np.full((h, w), bgc, dtype=int)
    sqh = randint(3, h // 3 + 1)
    sqw = randint(3, w // 3 + 1)
    loci = randint(0, 3)
    locj = randint(0, w - sqw)
    bx = box(loci, locj, loci + sqh - 1, locj + sqw - 1)
    bounds = {(i, j) for i in range(sqh - 2) for j in range(sqw - 2)}
    obj = grow(bounds, randint(1, (sqh - 2) * (sqw - 2)))
    obj = normalize(obj)
    oh, ow = shape(obj)
    objp = shift(obj, (loci + 1 + randint(0, sqh - oh - 2), locj + 1 + randint(0, sqw - ow - 2)))
    fill(gi, sqc, bx)
    fill(gi, inc, objp)
    fullinds = frozenset((i, j) for i in range(h) for j in range(w))
    bd = backdrop(bx)
    inds = set((ofcolor(gi, bgc) - bd) - mapply_n(neighbors, bd))
    cands = [ij for ij in sorted(inds) if shift(obj, ij).issubset(inds)]
    if not cands:
        raise ValueError("no room")
    loc = choice(cands)
    plcd = shift(obj, loc)
    fill(gi, outc, plcd)
    inds = (inds - plcd) - mapply_n(neighbors, plcd)
    noccs = unifint(diff_lb, diff_ub, (0, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    while tr < maxtr and succ < noccs:
        tr += 1
        if choice((True, False)):
            sqh = randint(3, h // 3 + 1)
            sqw = randint(3, w // 3 + 1)
            bounds = {(i, j) for i in range(sqh - 2) for j in range(sqw - 2)}
            obj2 = grow(bounds, randint(1, (sqh - 2) * (sqw - 2)))
            if normalize(obj2) == obj:
                if len(obj2) < (sqh - 2) * (sqw - 2):
                    obj2.add(choice(sorted((bounds - obj2) & mapply_n(dneighbors, obj2))))
                else:
                    continue
            obj2 = normalize(obj2)
            ooh, oow = shape(obj2)
            cs = set()
            for j in range(-1, w - sqw + 2):
                cs.add((-1, j)); cs.add((h - sqh + 1, j))
            for i in range(-1, h - sqh + 2):
                cs.add((i, -1)); cs.add((i, w - sqw + 1))
            if not cs:
                continue
            sloci, slocj = choice(sorted(cs))
            plcdbx = shift(box(0, 0, sqh - 1, sqw - 1), (sloci, slocj))
            if (backdrop(plcdbx) & fullinds).issubset(inds):
                succ += 1
                oloci = randint(sloci + 1, sloci + 1 + randint(0, sqh - ooh - 2))
                olocj = randint(slocj + 1, slocj + 1 + randint(0, sqw - oow - 2))
                fill(gi, sqc, plcdbx)
                fill(gi, inc, shift(obj2, (oloci, olocj)))
                inds = inds - backdrop(outbox(plcdbx))
        else:
            ooh = randint(1, h // 3 - 1)
            oow = randint(1, w // 3 - 1)
            bounds = {(i, j) for i in range(ooh) for j in range(oow)}
            obj2 = grow(bounds, randint(1, oow * ooh))
            if normalize(obj2) == obj:
                if len(obj2) < ooh * oow:
                    obj2.add(choice(sorted((bounds - obj2) & mapply_n(dneighbors, obj2))))
                else:
                    continue
        if choice((True, False, False)):
            obj2 = obj
        obj2 = normalize(obj2)
        ooh, oow = shape(obj2)
        for _ in range(randint(1, 3)):
            cs = [ij for ij in sorted(inds) if ij[0] <= h - ooh and ij[1] <= w - oow]
            if not cs:
                continue
            plcd = shift(obj2, choice(cs))
            if plcd.issubset(inds):
                succ += 1
                inds = (inds - plcd) - mapply_n(neighbors, plcd)
                fill(gi, outc, plcd)
    go = _apply_rule(gi)
    if go is None:
        raise ValueError("ambiguous key box")
    ref = gi.copy()
    for col, cells in _components(gi, bgc):
        if col == outc and normalize(cells) == obj:
            fill(ref, inc, cells)
    if not np.array_equal(ref, go):
        raise ValueError("rule mismatch")
    return {"input": gi.tolist(), "output": go.tolist()}


def _components(I, bgc):
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                x, y = stack.pop(); cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = x + dx, y + dy
                    if 0 <= a < h and 0 <= b < w and not seen[a, b] and I[a, b] == col:
                        seen[a, b] = True; stack.append((a, b))
            cells.sort()
            comps.append((col, cells))
    return comps


def _bbox(cells):
    rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _norm(cells):
    r0, _, c0, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _analyze(I, bgc):
    comps = _components(I, bgc)
    rings = []
    for col, cells in comps:
        r0, r1, c0, c1 = _bbox(cells)
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            ring = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                    if r in (r0, r1) or c in (c0, c1)}
            if set(cells) == ring:
                rings.append((col, cells, (r1 - r0 + 1) * (c1 - c0 + 1)))
    if not rings:
        return None
    best = max(t[2] for t in rings)
    top = [t for t in rings if t[2] == best]
    if len(top) != 1:
        return None
    box_col, box_cells, _ = top[0]
    br0, br1, bc0, bc1 = _bbox(box_cells)
    inner = [(r, c) for r in range(br0 + 1, br1) for c in range(bc0 + 1, bc1) if I[r, c] != bgc]
    if not inner:
        return None
    key_col = int(I[inner[0]])
    key_shape = _norm(inner)
    targets = [cells for col, cells in comps
               if col != box_col and col != key_col and _norm(cells) == key_shape]
    return inner, key_col, targets


def _apply_rule(I):
    I = np.asarray(I, dtype=int)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    res = _analyze(I, bgc)
    if res is None:
        return None
    _, key_col, targets = res
    O = I.copy()
    for cells in targets:
        for r, c in cells:
            O[r, c] = key_col
    return O


def derive_operations(I, O, examples=None):
    """
    Reading taken from the code-golf entry: the framed box holds a STAMP, and every loose
    object whose silhouette matches it gets that stamp printed over it. Where this differs
    from the maker: the maker carries a COLOUR (one FloodFill per match), while this
    reading carries the box's CONTENTS. So the stamp (the object inside the largest hollow
    frame) is copied once from the input using its exact cell mask (CopyI keeps only the
    masked cells), then pasted at each matching object's bbox origin. The shapes are
    identical, so each paste covers exactly that object's cells.
    Where the two readings agree (how the key box and the matching objects are found)
    nothing changes. If the key colour is 0, Copy/Paste cannot carry it (0 = transparent),
    so in that case each match is flood-filled with 0 instead.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    res = _analyze(I, bgc)
    if res is not None:
        inner, key_col, targets = res
        targets = sorted(targets, key=lambda cells: (_bbox(cells)[0], _bbox(cells)[2]))
        if targets:
            if key_col != 0:
                ops.append(28); sels.append(sel_of(inner))          # copy the stamp from I (exact cells)
                for cells in targets:
                    r0, _, c0, _ = _bbox(cells)
                    ops.append(30); sels.append([r0, c0, 0, 0])      # stamp it onto the matching object
            else:
                for cells in targets:
                    ops.append(10); sels.append([cells[0][0], cells[0][1], 0, 0])
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
