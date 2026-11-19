"""
ARC Task: 39a8645d (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _neighbors8(cell):
    r, c = cell
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}


def _neighbors4(cell):
    r, c = cell
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _shape(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)


def _normalize(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    hi_h = max(min(15, max_h), max_h)
    hi_w = max(min(15, max_w), max_w)
    h = _unifint(diff_lb, diff_ub, (min(15, max_h), hi_h))
    w = _unifint(diff_lb, diff_ub, (min(15, max_w), hi_w))
    oh = random.randint(2, 4)
    ow = random.randint(2, 4)
    remcols = [c for c in range(10) if c != bgc]
    nobjs = _unifint(diff_lb, diff_ub, (1, oh + ow))
    nobjs = min(nobjs, len(remcols) - 1)
    ccols = random.sample(remcols, nobjs + 1)
    mxcol = ccols[0]
    rcols = ccols[1:]
    maxnocc = _unifint(diff_lb, diff_ub, (nobjs + 2, max(nobjs + 2, (h * w) // 16)))
    tr = 0
    maxtr = 10 * maxnocc
    succ = 0
    allobjs = []
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    for k in range(nobjs + 1):
        while True:
            ncells = random.randint(oh + ow - 1, oh * ow)
            cobj = {random.choice(sorted(bounds))}
            while _shape(cobj) != (oh, ow) and len(cobj) < ncells:
                nb = set()
                for x in cobj:
                    nb |= _neighbors8(x)
                cand = sorted((bounds - cobj) & nb)
                cobj.add(random.choice(cand))
            if frozenset(cobj) not in allobjs:
                break
        allobjs.append(frozenset(cobj))
    mcobj = _normalize(allobjs[0])
    remobjs = [_normalize(o) for o in allobjs[1:]]
    mxobjcounter = 0
    remobjcounter = {robj: 0 for robj in remobjs}
    gi = np.full((h, w), bgc, dtype=int)
    inds = {(i, j) for i in range(h) for j in range(w)}
    while tr < maxtr and succ < maxnocc:
        tr += 1
        candobjs = [robj for robj, cnt in remobjcounter.items() if cnt + 1 < mxobjcounter]
        if len(candobjs) == 0 or random.randint(0, 100) / 100 > diff_lb:
            obj = mcobj
            col = mxcol
        else:
            obj = random.choice(candobjs)
            col = rcols[remobjs.index(obj)]
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if len(cands) == 0:
            break
        loc = random.choice(sorted(cands))
        plcd = {(r + loc[0], c + loc[1]) for r, c in obj}
        forbidden = set()
        for (r, c) in zip(*np.where(gi == col)):
            forbidden |= _neighbors8((int(r), int(c)))
        if plcd.issubset(inds - forbidden):
            succ += 1
            dn = set()
            for x in plcd:
                dn |= _neighbors4(x)
            inds = (inds - plcd) - dn
            for r, c in plcd:
                gi[r, c] = col
            if obj in remobjcounter:
                remobjcounter[obj] += 1
            else:
                mxobjcounter += 1
    sh = _shape(mcobj)
    go = np.full(sh, bgc, dtype=int)
    for r, c in mcobj:
        go[r, c] = mxcol
    return {"input": gi.tolist(), "output": go.tolist()}


def _components(I, bgc):
    H, W = I.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors8((cr, cc)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((int(col), cells))
    return comps


def derive_operations(I, O=None, examples=None):
    from collections import Counter
    I = np.asarray(I, dtype=int)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    comps = _components(I, bgc)
    keyed = []
    for col, cells in comps:
        keyed.append(((col, _normalize(cells)), cells))
    cnt = Counter(k for k, _ in keyed)
    best_key = cnt.most_common(1)[0][0]
    instances = [cells for k, cells in keyed if k == best_key]

    def foreign_cells(cells):
        r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells); c1 = max(c for _, c in cells)
        cs = set(cells)
        return [(r - r0, c - c0) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if (r, c) not in cs and I[r, c] != bgc], (r0, c0, r1 - r0, c1 - c0)

    # prefer an instance whose bbox contains no foreign content
    choice = min(instances, key=lambda cl: (len(foreign_cells(cl)[0]), min(cl)))
    foreign, (r0, c0, hh, ww) = foreign_cells(choice)

    ops, sels = [], []
    # Crop to the full bounding rectangle of the most common object (whole rect intended)
    ops.append(33); sels.append([r0, c0, hh, ww])
    if foreign:
        ops.append(bgc); sels.append(sel_of(sorted(foreign)))
    ops.append(34); sels.append([0, 0, hh, ww])
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
                        f"num_examples+1 ({num_examples + 1}) for task 39a8645d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 39a8645d"
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
                                f"for task 39a8645d"
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
                    f"Failed to build a complete episode for task 39a8645d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"39a8645d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
