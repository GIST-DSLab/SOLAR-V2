"""
ARC Task: 36fdfd69 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- sample_colors
def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, fgc, objc = random.sample(cols, 3)
    return {"bgc": bgc, "fgc": fgc, "objc": objc}


# ---------------------------------------------------------------- generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, objc=None, **kw) -> dict:
    if bgc is None or fgc is None or objc is None:
        c = sample_colors()
        bgc, fgc, objc = c["bgc"], c["fgc"], c["objc"]

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + d * (b - a)))

    def neighbors8(p):
        r, c = p
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}

    def dneighbors(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def backdrop(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return {(r, c) for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1)}

    def outbox(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs) - 1, max(rs) + 1, min(cs) - 1, max(cs) + 1
        out = set()
        for r in range(r0, r1 + 1):
            out.add((r, c0)); out.add((r, c1))
        for c in range(c0, c1 + 1):
            out.add((r0, c)); out.add((r1, c))
        return out

    h = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_w))))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 30)))
    gi = np.full((h, w), bgc, dtype=int)
    inds = {(r, c) for r in range(h) for c in range(w)}
    namt = random.randint(int(0.35 * h * w), int(0.65 * h * w))
    for (r, c) in random.sample(sorted(inds), namt):
        gi[r, c] = fgc
    go = gi.copy()
    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(2, 7)
        ow = random.randint(2, 7)
        cands = [p for p in inds if p[0] <= h - oh and p[1] <= w - ow]
        if not cands:
            continue
        loci, locj = random.choice(sorted(cands))
        bd = backdrop({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        if not bd.issubset(inds):
            continue
        ncells = random.randint(2, oh * ow - 1)

        def grow():
            obj = {random.choice(sorted(bd))}
            for _ in range(ncells - 1):
                ring = set()
                for p in obj:
                    for q in dneighbors(p):
                        ring |= neighbors8(q)
                opts = sorted((bd - obj) & ring)
                obj.add(random.choice(opts))
            return obj

        obj = grow()
        while len(obj) == len(backdrop(obj)):
            obj = grow()
        bd2 = backdrop(obj)
        gi2 = gi.copy()
        for (r, c) in bd2:
            gi2[r, c] = fgc
        for (r, c) in obj:
            gi2[r, c] = objc
        cnt = Counter(gi2.flatten().tolist())
        if cnt.get(objc, 0) < min(cnt.get(fgc, 0), cnt.get(bgc, 0)):
            succ += 1
            ob1 = outbox(bd2)
            inds = (inds - bd2) - (ob1 | outbox(ob1))
            gi = gi2
            for (r, c) in bd2:
                go[r, c] = 4
            for (r, c) in obj:
                go[r, c] = objc
    return {"input": gi.tolist(), "output": go.tolist()}


# ---------------------------------------------------------------- derive_operations
def derive_operations(I, O, examples=None):
    """Reading taken from the golf entry: the fill is gap-closing that spreads out
    from the object's own cells. First every row gap between two object cells is
    bridged, then the corners those bridges leave are closed until the object's
    footprint is a solid rectangle. So for each object we paint 4 one row line at
    a time (the span between its leftmost and rightmost cell in that row), and
    after that the rest of its bounding box. Every selection is built from I
    (the least colour's cells) and the current grid. O is never read."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    least = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    obj_cells = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] == least}

    # group object cells into objects: one object's cells lie within Chebyshev
    # distance 2 of each other, and generated objects are kept further apart.
    def closure(cells):
        out = set()
        cl = list(cells)
        for (r0, c0) in cl:
            near = [(r, c) for (r, c) in cl if max(abs(r - r0), abs(c - c0)) < 3]
            rs = [p[0] for p in near]
            cs = [p[1] for p in near]
            for r in range(min(rs), max(rs) + 1):
                for c in range(min(cs), max(cs) + 1):
                    out.add((r, c))
        return out

    region = closure(closure(obj_cells)) if obj_cells else set()
    seen, groups = set(), []
    for cell in sorted(region):
        if cell in seen:
            continue
        stack, comp = [cell], []
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = (r + dr, c + dc)
                    if nb in region and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        groups.append(comp)
    groups.sort(key=lambda g: (min(p[0] for p in g), min(p[1] for p in g)))

    grid = I.copy()
    for comp in groups:
        objs = [p for p in comp if p in obj_cells]
        if not objs:
            continue
        r0 = min(p[0] for p in objs); r1 = max(p[0] for p in objs)
        c0 = min(p[1] for p in objs); c1 = max(p[1] for p in objs)

        # phase 1: bridge the gaps between object cells along each row
        for r in range(r0, r1 + 1):
            cols = [c for (rr, c) in objs if rr == r]
            if len(cols) < 2:
                continue
            span = [(r, c) for c in range(min(cols), max(cols) + 1)
                    if (r, c) not in obj_cells and grid[r, c] != 4]
            if span:
                ops.append(4); sels.append(sel_of(span))
                for (a, b) in span:
                    grid[a, b] = 4

        # phase 2: close the corners left over so the box becomes solid
        rest = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if (r, c) not in obj_cells and grid[r, c] != 4]
        if rest:
            ops.append(4); sels.append(sel_of(rest))
            for (a, b) in rest:
                grid[a, b] = 4

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
                        f"num_examples+1 ({num_examples + 1}) for task 36fdfd69"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 36fdfd69"
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
                                f"for task 36fdfd69"
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
                    f"Failed to build a complete episode for task 36fdfd69 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"36fdfd69-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
