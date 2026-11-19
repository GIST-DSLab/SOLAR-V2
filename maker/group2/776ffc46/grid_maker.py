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


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _norm(cells):
    cells = set(cells)
    r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _shift(cells, d):
    return frozenset((r + d[0], c + d[1]) for r, c in cells)


def _shape(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _rect(r0, c0, r1, c1):
    return frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))


def _ring(r0, c0, r1, c1):
    return frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                     if r in (r0, r1) or c in (c0, c1))


def _bb(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _dn(p):
    r, c = p
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _nb(p):
    r, c = p
    return {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}


def _mnb(cells):
    out = set()
    for p in cells:
        out |= _nb(p)
    return out


def _mdn(cells):
    out = set()
    for p in cells:
        out |= _dn(p)
    return out


def _grow(bounds, n):
    obj = {random.choice(sorted(bounds))}
    for _ in range(n - 1):
        obj.add(random.choice(sorted((bounds - obj) & _mdn(obj))))
    return obj


def _connect(a, b):
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        return {(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)}
    if c0 == c1:
        return {(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)}
    return set()


def _components(I, skip=None):
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or (skip is not None and I[r, c] == skip):
                continue
            col = int(I[r, c]); stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                x, y = stack.pop(); cells.append((x, y))
                for a, b in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= a < h and 0 <= b < w and not seen[a, b] and I[a, b] == col:
                        seen[a, b] = True; stack.append((a, b))
            comps.append((col, sorted(cells)))
    return comps


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, inc=2, outc=3) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, min(30, max_h))))
    w = _unifint(diff_lb, diff_ub, (10, max(10, min(30, max_w))))
    gi = [[bgc] * w for _ in range(h)]

    def fill(col, cells):
        for r, c in cells:
            if 0 <= r < h and 0 <= c < w:
                gi[r][c] = col

    full = _rect(0, 0, h - 1, w - 1)
    sqh = random.randint(3, h // 3 + 1)
    sqw = random.randint(3, w // 3 + 1)
    loci = random.randint(0, 3)
    locj = random.randint(0, w - sqw)
    bx = _ring(loci, locj, loci + sqh - 1, locj + sqw - 1)
    bounds = set(_rect(0, 0, sqh - 3, sqw - 3))
    obj = _norm(_grow(bounds, random.randint(1, (sqh - 2) * (sqw - 2))))
    oh, ow = _shape(obj)
    objp = _shift(obj, (loci + 1 + random.randint(0, sqh - oh - 2),
                        locj + 1 + random.randint(0, sqw - ow - 2)))
    fill(sqc, bx)
    fill(inc, objp)
    bd = _rect(loci, locj, loci + sqh - 1, locj + sqw - 1)
    bgcells = {(r, c) for r in range(h) for c in range(w) if gi[r][c] == bgc}
    inds = (bgcells - bd) - _mnb(bd)
    cands = [ij for ij in sorted(inds) if _shift(obj, ij) <= inds]
    loc = random.choice(cands)
    plcd = _shift(obj, loc)
    fill(outc, plcd)
    inds = (inds - plcd) - _mnb(plcd)
    noccs = _unifint(diff_lb, diff_ub, (0, (h * w) // 20))
    succ = 0; tr = 0; maxtr = 5 * noccs
    while tr < maxtr and succ < noccs:
        tr += 1
        if random.choice((True, False)):
            sqh = random.randint(3, h // 3 + 1)
            sqw = random.randint(3, w // 3 + 1)
            bx = _ring(0, 0, sqh - 1, sqw - 1)
            bounds = set(_rect(0, 0, sqh - 3, sqw - 3))
            obj2 = _grow(bounds, random.randint(1, (sqh - 2) * (sqw - 2)))
            if _norm(obj2) == obj:
                if len(obj2) < (sqh - 2) * (sqw - 2):
                    obj2.add(random.choice(sorted((bounds - obj2) & _mdn(obj2))))
                else:
                    continue
            obj2 = _norm(obj2)
            ooh, oow = _shape(obj2)
            cc = (_connect((-1, -1), (-1, w - sqw + 1)) | _connect((h - sqh + 1, -1), (h - sqh + 1, w - sqw + 1))
                  | _connect((-1, -1), (h - sqh + 1, -1)) | _connect((-1, w - sqw + 1), (h - sqh + 1, w - sqw + 1)))
            if not cc:
                continue
            sloci, slocj = random.choice(sorted(cc))
            plcdbx = _shift(bx, (sloci, slocj))
            pbd = _rect(sloci, slocj, sloci + sqh - 1, slocj + sqw - 1)
            if (pbd & full) <= inds:
                succ += 1
                oloci = random.randint(sloci + 1, sloci + 1 + random.randint(0, sqh - ooh - 2))
                olocj = random.randint(slocj + 1, slocj + 1 + random.randint(0, sqw - oow - 2))
                fill(sqc, plcdbx)
                fill(inc, _shift(obj2, (oloci, olocj)))
                inds = inds - _rect(sloci - 1, slocj - 1, sloci + sqh, slocj + sqw)
        else:
            ooh = random.randint(1, h // 3 - 1)
            oow = random.randint(1, w // 3 - 1)
            bounds = set(_rect(0, 0, ooh - 1, oow - 1))
            obj2 = _grow(bounds, random.randint(1, ooh * oow))
            if _norm(obj2) == obj:
                if len(obj2) < ooh * oow:
                    obj2.add(random.choice(sorted((bounds - obj2) & _mdn(obj2))))
                else:
                    continue
        if random.choice((True, False, False)):
            obj2 = obj
        obj2 = _norm(obj2)
        ooh, oow = _shape(obj2)
        for _ in range(random.randint(1, 3)):
            cands = [ij for ij in sorted(inds) if ij[0] <= h - ooh and ij[1] <= w - oow]
            if not cands:
                continue
            plcd = _shift(obj2, random.choice(cands))
            if plcd <= inds:
                succ += 1
                inds = (inds - plcd) - _mnb(plcd)
                fill(outc, plcd)
    go = [row[:] for row in gi]
    for col, cells in _components(np.array(gi)):
        if col == outc and _norm(cells) == obj:
            for r, c in cells:
                go[r][c] = inc
    return {"input": gi, "output": go}


def derive_operations(I, O):
    """
    Rule read from I: the largest hollow rectangular outline is the key box; the object
    inside it gives the key colour and key shape. Every loose object (not box-coloured,
    not key-coloured) with exactly that shape takes the key colour.
    Unit of work: the whole class at once -- all cells of every matching object form one
    selection ("every object shaped like the key"), recoloured by a single Color(key) op.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    comps = _components(I, skip=bgc)

    rings = []
    for col, cells in comps:
        r0, c0, r1, c1 = _bb(cells)
        if r1 - r0 >= 2 and c1 - c0 >= 2 and set(cells) == _ring(r0, c0, r1, c1):
            rings.append(((r1 - r0 + 1) * (c1 - c0 + 1), col, (r0, c0, r1, c1)))
    target = []
    if rings:
        _, box_col, (r0, c0, r1, c1) = max(rings, key=lambda t: t[0])
        inner = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if I[r, c] != bgc]
        if inner:
            key_col = int(I[inner[0]])
            key_shape = _norm(inner)
            for col, cells in comps:
                if col in (box_col, key_col):
                    continue
                if _norm(cells) == key_shape:
                    target.extend(cells)
            if target:
                # one selection: every object congruent to the key, recoloured together
                ops.append(key_col); sels.append(sel_of(sorted(target)))
    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
