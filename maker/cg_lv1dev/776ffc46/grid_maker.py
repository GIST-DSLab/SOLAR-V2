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


def _unifint(lb, ub, rng):
    a, b = rng
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    return max(a, min(b, int(round(random.uniform(lo, hi)))))


def _dn(p):
    r, c = p
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _nb(p):
    r, c = p
    return {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}


def _mapply(f, s):
    out = set()
    for p in s:
        out |= f(p)
    return out


def _normalize(s):
    r0 = min(p[0] for p in s); c0 = min(p[1] for p in s)
    return frozenset((r - r0, c - c0) for r, c in s)


def _shift(s, d):
    return frozenset((r + d[0], c + d[1]) for r, c in s)


def _shape(s):
    rs = [p[0] for p in s]; cs = [p[1] for p in s]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _backdrop(s):
    rs = [p[0] for p in s]; cs = [p[1] for p in s]
    return frozenset((r, c) for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1))


def _box(s):
    rs = [p[0] for p in s]; cs = [p[1] for p in s]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    return frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                     if r in (r0, r1) or c in (c0, c1))


def _outbox(s):
    rs = [p[0] for p in s]; cs = [p[1] for p in s]
    return _box({(min(rs) - 1, min(cs) - 1), (max(rs) + 1, max(cs) + 1)})


def _connect(a, b):
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        return {(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)}
    if c0 == c1:
        return {(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)}
    return set()


def _grow(bounds, n):
    obj = {random.choice(sorted(bounds))}
    for _ in range(n - 1):
        obj.add(random.choice(sorted((bounds - obj) & _mapply(_dn, obj))))
    return obj


def _fill(g, col, cells):
    h, w = len(g), len(g[0])
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            g[r][c] = col


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


def _read_rule(I, bgc):
    """Return (comps, box_col, key_col, key_shape) or None."""
    comps = _components(I, bgc)
    rings = []
    for col, cells in comps:
        hh, ww = _shape(cells)
        if hh >= 3 and ww >= 3 and set(cells) == set(_box(cells)):
            rings.append((col, cells, hh * ww))
    if not rings:
        return None
    box_col, box_cells, _ = max(rings, key=lambda t: t[2])
    rs = [p[0] for p in box_cells]; cs = [p[1] for p in box_cells]
    br0, br1, bc0, bc1 = min(rs), max(rs), min(cs), max(cs)
    inner = [(r, c) for r in range(br0 + 1, br1) for c in range(bc0 + 1, bc1) if I[r, c] != bgc]
    if not inner:
        return None
    return comps, box_col, int(I[inner[0]]), _normalize(inner)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, inc, outc) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    sqh = random.randint(3, h // 3 + 1)
    sqw = random.randint(3, w // 3 + 1)
    loci = random.randint(0, 3)
    locj = random.randint(0, w - sqw)
    bx = _box({(loci, locj), (loci + sqh - 1, locj + sqw - 1)})
    bounds = {(i, j) for i in range(sqh - 2) for j in range(sqw - 2)}
    obj = _grow(bounds, random.randint(1, (sqh - 2) * (sqw - 2)))
    obj = _normalize(obj)
    oh, ow = _shape(obj)
    objp = _shift(obj, (loci + 1 + random.randint(0, sqh - oh - 2),
                        locj + 1 + random.randint(0, sqw - ow - 2)))
    _fill(gi, sqc, bx)
    _fill(gi, inc, objp)
    fullinds = {(i, j) for i in range(h) for j in range(w)}
    bgcells = {(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc}
    bd = _backdrop(bx)
    inds = (bgcells - bd) - _mapply(_nb, bd)
    cands = [ij for ij in sorted(inds) if _shift(obj, ij) <= inds]
    if not cands:
        raise ValueError("no room for key copy")
    plcd = _shift(obj, random.choice(cands))
    _fill(gi, outc, plcd)
    inds = (inds - plcd) - _mapply(_nb, plcd)
    noccs = _unifint(diff_lb, diff_ub, (0, (h * w) // 20))
    succ = 0; tr = 0; maxtr = 5 * noccs
    while tr < maxtr and succ < noccs:
        tr += 1
        if random.choice((True, False)):
            sqh = random.randint(3, h // 3 + 1)
            sqw = random.randint(3, w // 3 + 1)
            bx = _box({(0, 0), (sqh - 1, sqw - 1)})
            bounds = {(i, j) for i in range(sqh - 2) for j in range(sqw - 2)}
            obj2 = _grow(bounds, random.randint(1, (sqh - 2) * (sqw - 2)))
            if _normalize(obj2) == obj:
                if len(obj2) < (sqh - 2) * (sqw - 2):
                    obj2.add(random.choice(sorted((bounds - obj2) & _mapply(_dn, obj2))))
                else:
                    continue
            obj2 = _normalize(obj2)
            ooh, oow = _shape(obj2)
            cands = (_connect((-1, -1), (-1, w - sqw + 1))
                     | _connect((h - sqh + 1, -1), (h - sqh + 1, w - sqw + 1))
                     | _connect((-1, -1), (h - sqh + 1, -1))
                     | _connect((-1, w - sqw + 1), (h - sqh + 1, w - sqw + 1)))
            if not cands:
                continue
            sloci, slocj = loc = random.choice(sorted(cands))
            plcdbx = _shift(bx, loc)
            if (_backdrop(plcdbx) & fullinds) <= inds:
                succ += 1
                oloci = random.randint(sloci + 1, sloci + 1 + random.randint(0, sqh - ooh - 2))
                olocj = random.randint(slocj + 1, slocj + 1 + random.randint(0, sqw - oow - 2))
                _fill(gi, sqc, plcdbx)
                _fill(gi, inc, _shift(obj2, (oloci, olocj)))
                inds = inds - _backdrop(_outbox(plcdbx))
        else:
            ooh = random.randint(1, h // 3 - 1)
            oow = random.randint(1, w // 3 - 1)
            bounds = {(i, j) for i in range(ooh) for j in range(oow)}
            obj2 = _grow(bounds, random.randint(1, ooh * oow))
            if _normalize(obj2) == obj:
                if len(obj2) < ooh * oow:
                    obj2.add(random.choice(sorted((bounds - obj2) & _mapply(_dn, obj2))))
                else:
                    continue
        if random.choice((True, False, False)):
            obj2 = obj
        obj2 = _normalize(obj2)
        ooh, oow = _shape(obj2)
        for _ in range(random.randint(1, 3)):
            cands = [ij for ij in sorted(inds) if ij[0] <= h - ooh and ij[1] <= w - oow]
            if not cands:
                continue
            plcd = _shift(obj2, random.choice(cands))
            if plcd <= inds:
                succ += 1
                inds = (inds - plcd) - _mapply(_nb, plcd)
                _fill(gi, outc, plcd)
    I = np.array(gi, dtype=int)
    go = I.copy()
    for col, cells in _components(I, bgc):
        if col == outc and _normalize(cells) == obj:
            for r, c in cells:
                go[r, c] = inc
    # The rule as a solver reads it (dominant bg, largest outline = key box) must
    # agree with the generator's own key; otherwise the instance is ambiguous -> retry.
    if bgc != Counter(I.flatten().tolist()).most_common(1)[0][0]:
        raise ValueError("background not dominant")
    rule = _read_rule(I, bgc)
    if rule is None or rule[1] != sqc or rule[2] != inc or rule[3] != obj:
        raise ValueError("key box ambiguous")
    return {"input": I.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule (read off I alone): the largest hollow outline is the key box; the object
    inside it gives the key colour and the key shape. Every loose object (not
    box-coloured, not key-coloured) whose normalised shape equals the key shape is
    recoloured to the key colour; everything else stays.

    The code-golf reading agrees with the maker's: it locates the box, takes the
    occupancy pattern inside it as a template, slides that template over the grid and
    repaints every window whose occupancy matches exactly with the key colour. The unit
    is still "one object whose footprint equals the key footprint, repainted whole", and
    the order and the fixed quantities are the same. (Its extra repaint of the key
    itself in its own colour would be a no-op, so it is not emitted.) So the trajectory
    is unchanged: one FloodFill(key colour) per matched object, seeded on that object.
    """
    I = np.asarray(I, dtype=int)
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        votes = Counter()
        for ei, _eo in examples:
            ei = np.asarray(ei, dtype=int)
            votes[Counter(ei.flatten().tolist()).most_common(1)[0][0]] += 1
        cand = votes.most_common(1)[0][0]
        if cand in set(I.flatten().tolist()):
            bgc = cand
    rule = _read_rule(I, bgc)
    if rule is not None:
        comps, box_col, key_col, key_shape = rule
        for col, cells in comps:
            if col in (box_col, key_col):
                continue
            if _normalize(cells) != key_shape:
                continue
            ops.append(10 + key_col)          # FloodFill whole matched object
            sels.append(sel_of([cells[0]]))   # single seed cell on that object
    ops.append(34)
    sels.append([0, 0, I.shape[0] - 1, I.shape[1] - 1])
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
