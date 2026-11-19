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


def _neighbors8(i, j):
    return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}


def _dneighbors(i, j):
    return {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}


def _box(r0, c0, r1, c1):
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}


def _backdrop(cells):
    rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
    return {(r, c) for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1)}


def _normalize(cells):
    r0 = min(p[0] for p in cells); c0 = min(p[1] for p in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _grow(hb, wb):
    bounds = {(i, j) for i in range(hb) for j in range(wb)}
    obj = {random.choice(sorted(bounds))}
    n = random.randint(1, hb * wb)
    for _ in range(n - 1):
        cand = sorted((bounds - obj) & set().union(*[_dneighbors(*p) for p in obj]))
        obj.add(random.choice(cand))
    return obj, bounds


def _components(I, bgc):
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c]); stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                x, y = stack.pop(); cells.append((x, y))
                for a, b in _dneighbors(x, y):
                    if 0 <= a < h and 0 <= b < w and not seen[a, b] and I[a, b] == col:
                        seen[a, b] = True; stack.append((a, b))
            cells.sort()
            comps.append((col, cells))
    return comps


def _read_rule(I, bgc):
    """key box = largest box-shaped object; key = what sits inside it."""
    comps = _components(I, bgc)
    boxes = []
    for col, cells in comps:
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if set(cells) == _box(r0, c0, r1, c1):
            boxes.append(((r1 - r0 + 1) * (c1 - c0 + 1), col, (r0, r1, c0, c1)))
    if not boxes:
        return comps, None, None, None
    best = max(b[0] for b in boxes)
    _, box_col, (r0, r1, c0, c1) = [b for b in boxes if b[0] == best][0]
    inner = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if I[r, c] != bgc]
    if not inner:
        return comps, box_col, None, None
    key_col = Counter(int(I[p]) for p in inner).most_common(1)[0][0]
    return comps, box_col, key_col, _normalize(inner)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, inc, outc) -> dict:
    def unifint(lo, hi):
        lo2 = lo + int(round(diff_lb * (hi - lo))); hi2 = lo + int(round(diff_ub * (hi - lo)))
        return random.randint(min(lo2, hi2), max(lo2, hi2))

    for _attempt in range(200):
        h = unifint(10, max(10, max_h)); w = unifint(10, max(10, max_w))
        gi = np.full((h, w), bgc, dtype=int)
        fullinds = {(i, j) for i in range(h) for j in range(w)}
        sqh = random.randint(3, h // 3 + 1); sqw = random.randint(3, w // 3 + 1)
        loci = random.randint(0, 3); locj = random.randint(0, w - sqw)
        bx = _box(loci, locj, loci + sqh - 1, locj + sqw - 1)
        obj, _ = _grow(sqh - 2, sqw - 2)
        obj = _normalize(obj)
        oh = max(p[0] for p in obj) + 1; ow = max(p[1] for p in obj) + 1
        dr = loci + 1 + random.randint(0, sqh - oh - 2); dc = locj + 1 + random.randint(0, sqw - ow - 2)
        for p in bx: gi[p] = sqc
        for (r, c) in obj: gi[r + dr, c + dc] = inc
        bd = _backdrop(bx)
        nb = set().union(*[_neighbors8(*p) for p in bd])
        bg_cells = {(i, j) for i in range(h) for j in range(w) if gi[i, j] == bgc}
        inds = (bg_cells - bd) - nb
        cands = [ij for ij in sorted(inds) if {(r + ij[0], c + ij[1]) for r, c in obj} <= inds]
        if not cands:
            continue
        loc = random.choice(cands)
        plcd = {(r + loc[0], c + loc[1]) for r, c in obj}
        for p in plcd: gi[p] = outc
        inds = (inds - plcd) - set().union(*[_neighbors8(*p) for p in plcd])
        noccs = unifint(0, (h * w) // 20)
        succ = 0; tr = 0; maxtr = 5 * noccs
        while tr < maxtr and succ < noccs:
            tr += 1
            if random.choice((True, False)):
                sqh = random.randint(3, h // 3 + 1); sqw = random.randint(3, w // 3 + 1)
                obj2, bounds = _grow(sqh - 2, sqw - 2)
                if _normalize(obj2) == obj:
                    if len(obj2) < (sqh - 2) * (sqw - 2):
                        cand = sorted((bounds - obj2) & set().union(*[_dneighbors(*p) for p in obj2]))
                        obj2.add(random.choice(cand))
                    else:
                        continue
                obj2 = _normalize(obj2)
                ooh = max(p[0] for p in obj2) + 1; oow = max(p[1] for p in obj2) + 1
                cs = set()
                for j in range(-1, w - sqw + 2):
                    cs.add((-1, j)); cs.add((h - sqh + 1, j))
                for i in range(-1, h - sqh + 2):
                    cs.add((i, -1)); cs.add((i, w - sqw + 1))
                if not cs:
                    continue
                sloci, slocj = random.choice(sorted(cs))
                pbx = _box(sloci, slocj, sloci + sqh - 1, slocj + sqw - 1)
                if (_backdrop(pbx) & fullinds) <= inds:
                    succ += 1
                    oloci = random.randint(sloci + 1, sloci + 1 + random.randint(0, sqh - ooh - 2))
                    olocj = random.randint(slocj + 1, slocj + 1 + random.randint(0, sqw - oow - 2))
                    for p in pbx & fullinds: gi[p] = sqc
                    for (r, c) in obj2:
                        if (r + oloci, c + olocj) in fullinds:
                            gi[r + oloci, c + olocj] = inc
                    ob = {(r, c) for r in range(sloci - 1, sloci + sqh + 1)
                          for c in range(slocj - 1, slocj + sqw + 1)}
                    inds = inds - ob
            else:
                ooh = random.randint(1, h // 3 - 1); oow = random.randint(1, w // 3 - 1)
                obj2, bounds = _grow(ooh, oow)
                if _normalize(obj2) == obj:
                    if len(obj2) < ooh * oow:
                        cand = sorted((bounds - obj2) & set().union(*[_dneighbors(*p) for p in obj2]))
                        obj2.add(random.choice(cand))
                    else:
                        continue
            if random.choice((True, False, False)):
                obj2 = obj
            obj2 = _normalize(obj2)
            ooh = max(p[0] for p in obj2) + 1; oow = max(p[1] for p in obj2) + 1
            for _kk in range(random.randint(1, 3)):
                cs = [ij for ij in sorted(inds) if ij[0] <= h - ooh and ij[1] <= w - oow]
                if not cs:
                    continue
                loc = random.choice(cs)
                plcd = {(r + loc[0], c + loc[1]) for r, c in obj2}
                if plcd <= inds:
                    succ += 1
                    inds = (inds - plcd) - set().union(*[_neighbors8(*p) for p in plcd])
                    for p in plcd: gi[p] = outc
        go = gi.copy()
        for col, cells in _components(gi, bgc):
            if col == outc and _normalize(cells) == obj:
                for p in cells: go[p] = inc
        # keep only instances where the rule as read from the input is unambiguous
        if Counter(gi.flatten().tolist()).most_common(1)[0][0] != bgc:
            continue
        comps, box_col, key_col, key_shape = _read_rule(gi, bgc)
        if box_col != sqc or key_col != inc or key_shape != obj:
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    """
    Rule (read off I alone):
      - background = dominant colour of I (the generator paints the canvas with it).
      - key box = the largest box-shaped object (cells == outline of its bbox).
      - the stuff sitting inside that box gives the KEY COLOUR and the KEY SHAPE.
      - every loose object (4-connected, single-colour) whose normalized shape equals
        the key shape takes on the key colour: one FloodFill per object.
    The golf entry reads it the same way (one object at a time, compare its shape with
    the key's shape, recolour on a match), so the unit and the order of work agree.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    comps, box_col, key_col, key_shape = _read_rule(I, bgc)
    if key_col is not None:
        for col, cells in comps:
            if col == key_col or col == box_col:
                continue
            if _normalize(cells) != key_shape:
                continue
            ops.append(10 + key_col)
            sels.append(sel_of([cells[0]]))
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
