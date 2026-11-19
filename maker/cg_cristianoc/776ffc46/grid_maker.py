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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, inc, outc = random.sample(range(10), 4)
    return {"bgc": bgc, "sqc": sqc, "inc": inc, "outc": outc}


def _unifint(lb, ub, bounds):
    a, b = bounds
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _box(r0, c0, r1, c1):
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}


def _backdrop(cells):
    rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
    return {(r, c) for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1)}


def _nb8(cells):
    return {(r + a, c + b) for r, c in cells for a in (-1, 0, 1) for b in (-1, 0, 1)
            if (a, b) != (0, 0)}


def _dnb(cells):
    return {(r + a, c + b) for r, c in cells for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))}


def _norm(cells):
    r0 = min(p[0] for p in cells); c0 = min(p[1] for p in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _shift(cells, d):
    return {(r + d[0], c + d[1]) for r, c in cells}


def _shape(cells):
    rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _grow(bh, bw):
    bounds = {(i, j) for i in range(bh) for j in range(bw)}
    obj = {random.choice(sorted(bounds))}
    n = random.randint(1, bh * bw)
    for _ in range(n - 1):
        obj.add(random.choice(sorted((bounds - obj) & _dnb(obj))))
    return obj, bounds


def _comps(g, bgc):
    h, w = len(g), len(g[0])
    seen = set(); out = []
    for r in range(h):
        for c in range(w):
            if g[r][c] == bgc or (r, c) in seen:
                continue
            col = g[r][c]; st = [(r, c)]; seen.add((r, c)); cells = []
            while st:
                x, y = st.pop(); cells.append((x, y))
                for a, b in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                    if 0 <= a < h and 0 <= b < w and (a, b) not in seen and g[a][b] == col:
                        seen.add((a, b)); st.append((a, b))
            out.append((col, cells))
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, inc, outc) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    full = {(r, c) for r in range(h) for c in range(w)}

    def fill(col, cells):
        for r, c in cells:
            if 0 <= r < h and 0 <= c < w:
                gi[r][c] = col

    ksqh = random.randint(3, h // 3 + 1)
    ksqw = random.randint(3, w // 3 + 1)
    loci = random.randint(0, 3)
    locj = random.randint(0, w - ksqw)
    bx = _box(loci, locj, loci + ksqh - 1, locj + ksqw - 1)
    obj, _ = _grow(ksqh - 2, ksqw - 2)
    obj = _norm(obj)
    oh, ow = _shape(obj)
    objp = _shift(obj, (loci + 1 + random.randint(0, ksqh - oh - 2),
                        locj + 1 + random.randint(0, ksqw - ow - 2)))
    fill(sqc, bx); fill(inc, objp)
    bd = _backdrop(bx)
    inds = {p for p in full if gi[p[0]][p[1]] == bgc} - bd - _nb8(bd)
    cands = [ij for ij in sorted(inds) if _shift(obj, ij) <= inds]
    plcd = _shift(obj, random.choice(cands))
    fill(outc, plcd)
    inds = (inds - plcd) - _nb8(plcd)
    noccs = _unifint(diff_lb, diff_ub, (0, (h * w) // 20))
    succ = tr = 0
    while tr < 5 * noccs and succ < noccs:
        tr += 1
        if random.choice((True, False)):
            sqh = random.randint(3, h // 3 + 1)
            sqw = random.randint(3, w // 3 + 1)
            # learnability: the key frame must stay the strictly largest frame
            if sqh * sqw >= ksqh * ksqw:
                continue
            obj2, bounds = _grow(sqh - 2, sqw - 2)
            if _norm(obj2) == obj:
                if len(obj2) < (sqh - 2) * (sqw - 2):
                    obj2.add(random.choice(sorted((bounds - obj2) & _dnb(obj2))))
                else:
                    continue
            obj2 = _norm(obj2)
            ooh, oow = _shape(obj2)
            cs = set()
            for j in range(-1, w - sqw + 2):
                cs.add((-1, j)); cs.add((h - sqh + 1, j))
            for i in range(-1, h - sqh + 2):
                cs.add((i, -1)); cs.add((i, w - sqw + 1))
            sl, sj = random.choice(sorted(cs))
            pb = _box(sl, sj, sl + sqh - 1, sj + sqw - 1)
            if (_backdrop(pb) & full) <= inds:
                succ += 1
                oi = random.randint(sl + 1, sl + 1 + random.randint(0, sqh - ooh - 2))
                oj = random.randint(sj + 1, sj + 1 + random.randint(0, sqw - oow - 2))
                fill(sqc, pb); fill(inc, _shift(obj2, (oi, oj)))
                inds = inds - _backdrop(_box(sl - 1, sj - 1, sl + sqh, sj + sqw))
        else:
            ooh = random.randint(1, h // 3 - 1)
            oow = random.randint(1, w // 3 - 1)
            obj2, bounds = _grow(ooh, oow)
            if _norm(obj2) == obj:
                if len(obj2) < ooh * oow:
                    obj2.add(random.choice(sorted((bounds - obj2) & _dnb(obj2))))
                else:
                    continue
        if random.choice((True, False, False)):
            obj2 = obj
        obj2 = _norm(obj2)
        ooh, oow = _shape(obj2)
        for _ in range(random.randint(1, 3)):
            cs = [ij for ij in sorted(inds) if ij[0] <= h - ooh and ij[1] <= w - oow]
            if not cs:
                continue
            p = _shift(obj2, random.choice(cs))
            if p <= inds:
                succ += 1
                inds = (inds - p) - _nb8(p)
                fill(outc, p)
    # sanity: background dominant, key frame the strictly largest hollow ring
    cnt = Counter(v for row in gi for v in row)
    if cnt.most_common(1)[0][0] != bgc or cnt.most_common(2)[1][1] == cnt[bgc]:
        raise ValueError("bg not dominant")
    karea = ksqh * ksqw
    for col, cells in _comps(gi, bgc):
        if set(cells) == bx:
            continue
        hh, ww = _shape(cells)
        if hh >= 3 and ww >= 3 and hh * ww >= karea and set(cells) == _box(
                min(p[0] for p in cells), min(p[1] for p in cells),
                max(p[0] for p in cells), max(p[1] for p in cells)):
            raise ValueError("ambiguous key frame")
    go = [row[:] for row in gi]
    for col, cells in _comps(gi, bgc):
        if col == outc and _norm(cells) == obj:
            for r, c in cells:
                go[r][c] = inc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule (from I alone): the largest hollow rectangular frame holds a key object;
    every loose object elsewhere with the key's exact (normalized) shape is
    recoloured to the key's colour; nothing else changes.
    The code-golf reading agrees with the maker on the unit (one 4-connected
    single-colour object, matched by normalized shape) and on the action (recolour
    the whole object), so there is nothing to change: each match is one FloodFill
    seeded inside that object. (The golf entry hard-codes frame colour 5, which is
    incidental; here the frame is found structurally as the largest ring, as in
    the verifier.)
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    comps = _comps(I.tolist(), bgc)

    def is_ring(cells):
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        return set(cells) == _box(min(rs), min(cs), max(rs), max(cs))

    rings = []
    for col, cells in comps:
        hh, ww = _shape(cells)
        if hh >= 3 and ww >= 3 and is_ring(cells):
            rings.append((col, cells, hh * ww))
    if rings:
        box_col, box_cells, _ = max(rings, key=lambda t: t[2])
        br0 = min(p[0] for p in box_cells); br1 = max(p[0] for p in box_cells)
        bc0 = min(p[1] for p in box_cells); bc1 = max(p[1] for p in box_cells)
        inner = [(r, c) for r in range(br0 + 1, br1) for c in range(bc0 + 1, bc1)
                 if I[r, c] != bgc]
        if inner:
            key_col = int(I[inner[0]])
            key_shape = _norm(inner)
            for col, cells in comps:
                if col in (box_col, key_col) or _norm(cells) != key_shape:
                    continue
                ops.append(10 + key_col)
                sels.append(sel_of([min(cells)]))   # single seed cell inside the object
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
