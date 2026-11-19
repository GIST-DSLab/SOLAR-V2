"""
ARC Task: e509e548 (RE-ARC) — LLM-generated grid_maker
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
from itertools import permutations

import numpy as np

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback if helper module is unavailable
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# ----------------------------------------------------------------------------
# 1. episode-level colours
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # the task's answer palette is {1, 6, 2}; the generator therefore never uses
    # those three as input colours -- keep that invariant.
    cols = [c for c in range(10) if c not in (1, 2, 6)]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ncols = random.randint(2, 6)
    ccols = random.sample(rem, ncols)
    return {"bgc": bgc, "ccols": ccols}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 6))

    getL = lambda h, w: connect((0, 0), (h - 1, 0)) | connect((0, 0), (0, w - 1))
    getU = lambda h, w: (connect((0, 0), (0, w - 1))
                         | connect((0, 0), (random.randint(1, h - 1), 0))
                         | connect((0, w - 1), (random.randint(1, h - 1), w - 1)))
    getH = lambda h, w: (connect((0, 0), (0, w - 1))
                         | shift(connect((0, 0), (h - 1, 0))
                                 | connect((h - 1, 0), (h - 1, random.randint(1, w - 1))),
                                 (0, random.randint(1, w - 2))))

    minshp_getter_pairs = [(2, 2, getL), (2, 3, getU), (3, 3, getH)]
    colmapper = {getL: 1, getU: 6, getH: 2}

    hlo = min(10, max_h)
    wlo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))

    if bgc is None:
        bgc = random.choice(list(cols))
    remcols = [c for c in cols if c != bgc]
    if ccols is None:
        ncols = unifint(diff_lb, diff_ub, (1, 6))
        ccols = random.sample(remcols, ncols)
    else:
        ccols = [c for c in ccols if c != bgc] or list(remcols)

    nobjs = unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 10)))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)

    # learnability: make sure all three shape classes (L->1, U->6, H->2) occur
    pending = random.sample(minshp_getter_pairs, 3)

    while succ < nobjs and tr < maxtr:
        tr += 1
        if pending:
            minh, minw, getter = pending[0]
        else:
            minh, minw, getter = random.choice(minshp_getter_pairs)
        oh = random.randint(minh, 6)
        ow = random.randint(minw, 6)
        obj = getter(oh, ow)
        mfs = [identity, dmirror, cmirror, vmirror, hmirror]
        nmfs = random.choice((1, 2))
        for fn in random.sample(mfs, nmfs):
            obj = fn(obj)
            obj = normalize(obj)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = random.choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            if pending and getter is pending[0][2]:
                pending.pop(0)
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            col = random.choice(ccols)
            gi = fill(gi, col, plcd)
            go = fill(go, colmapper[getter], plcd)

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. operations
# ----------------------------------------------------------------------------
_PERMS = list(permutations((1, 6, 2)))


def _background(I, examples):
    """Every shape is repainted (1/6/2 never occur in an input), so any cell that
    is unchanged between an example's input and output is background."""
    cnt = Counter()
    if examples:
        for pair in examples:
            try:
                ei, eo = pair[0], pair[1]
            except Exception:
                continue
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            cnt.update(ei[ei == eo].flatten().tolist())
    if cnt:
        bgc = cnt.most_common(1)[0][0]
        if (I == bgc).any():
            return int(bgc)
    return int(Counter(I.flatten().tolist()).most_common(1)[0][0])


def _objects(I, bgc):
    """4-connected same-colour shapes, each labelled with its answer colour:
       L (size == height + width - 1)            -> 1
       U (every cell lies on the bbox border)    -> 6
       H (some cell strictly inside the bbox)    -> 2"""
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            oh, ow = r1 - r0 + 1, c1 - c0 + 1
            if len(cells) == oh + ow - 1:
                new = 1
            elif all(p[0] in (r0, r1) or p[1] in (c0, c1) for p in cells):
                new = 6
            else:
                new = 2
            objs.append({"cells": sorted(cells), "col": col, "new": new,
                         "r0": r0, "r1": r1, "c0": c0, "c1": c1, "size": len(cells)})
    return objs


def _obj_key(o, k):
    if k == 0:                       # sweep down the rows, left to right
        return (o["r0"], o["c0"])
    if k == 1:                       # sweep up from the bottom
        return (-o["r1"], o["c0"])
    if k == 2:                       # sweep across the columns, left to right
        return (o["c0"], o["r0"])
    if k == 3:                       # sweep across the columns, right to left
        return (-o["c1"], o["r0"])
    return (-o["size"], o["r0"], o["c0"])   # biggest shape first


def _cell_key(p, k):
    if k == 1:
        return (-p[0], p[1])
    if k == 2:
        return (p[1], p[0])
    if k == 3:
        return (-p[1], p[0])
    return (p[0], p[1])


def _ordered(objs, k):
    return sorted(objs, key=lambda o: _obj_key(o, k))


def _seed(o, k):
    # start the fill at the corner of the shape the sweep reaches first
    return min(o["cells"], key=lambda p: _cell_key(p, k))


def _floodfill_safe(objs):
    """True unless two distinct same-coloured shapes touch diagonally."""
    owner = {}
    for i, o in enumerate(objs):
        for p in o["cells"]:
            owner[p] = i
    for i, o in enumerate(objs):
        for (y, x) in o["cells"]:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    j = owner.get((y + dy, x + dx))
                    if j is not None and j != i and objs[j]["col"] == o["col"]:
                        return False
    return True


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = _background(I, examples)
    objs = _objects(I, bgc)

    if not objs:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    nL = sum(1 for o in objs if o["new"] == 1)
    nU = sum(1 for o in objs if o["new"] == 6)
    nH = sum(1 for o in objs if o["new"] == 2)
    first = objs[0]

    # the route is a deterministic function of the input alone: its size, its
    # background colour, how many shapes of each kind it holds and where the
    # first one sits. Different puzzles -> different (but reproducible) route.
    sig = (int(bgc) * 31 + hi * 17 + wi * 13 + len(objs) * 7
           + nL * 5 + nU * 3 + nH * 2 + first["r0"] * 11 + first["c0"] * 23)
    mode = sig % 4
    order_k = (sig // 4) % 5
    perm = _PERMS[(sig // 20) % 6]

    if mode == 0 and nH == 0:
        # nothing would survive the 2-base -> laying it would be pointless
        mode = 2
    if mode == 1 and not _floodfill_safe(objs):
        mode = 3

    if mode == 0:
        # "paint every shape the common colour, then correct the exceptions"
        allcells = [p for o in _ordered(objs, order_k) for p in o["cells"]]
        ops.append(2)
        sels.append(sel_of(allcells))
        for cl in perm:
            if cl == 2:
                continue
            cells = [p for o in _ordered(objs, order_k) if o["new"] == cl for p in o["cells"]]
            if cells:
                ops.append(cl)
                sels.append(sel_of(cells))

    elif mode == 1:
        # "walk the grid and flood each shape as I meet it"
        for o in _ordered(objs, order_k):
            ops.append(10 + o["new"])
            sels.append(sel_of([_seed(o, order_k)]))

    elif mode == 2:
        # "all the Ls at once, then all the Us, then all the Hs"
        for cl in perm:
            cells = [p for o in _ordered(objs, order_k) if o["new"] == cl for p in o["cells"]]
            if cells:
                ops.append(cl)
                sels.append(sel_of(cells))

    else:
        # "one kind of shape at a time, but shape by shape"
        for cl in perm:
            for o in _ordered([o for o in objs if o["new"] == cl], order_k):
                ops.append(cl)
                sels.append(sel_of(o["cells"]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e509e548"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e509e548"
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
                                f"for task e509e548"
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
                    f"Failed to build a complete episode for task e509e548 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e509e548-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
