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
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(lb, ub, rng):
    a, b = rng
    lo = a + int(round(max(0.0, min(1.0, lb)) * (b - a)))
    hi = a + int(round(max(0.0, min(1.0, ub)) * (b - a)))
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 6)]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 6)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ccols=None, **kwargs) -> dict:
    if ccols is None:
        cols = [c for c in range(10) if c not in (1, 2, 6, bgc)]
        ccols = random.sample(cols, random.randint(1, 6))

    def connect_v(r0, r1, c):
        return {(r, c) for r in range(r0, r1 + 1)}

    def connect_h(r, c0, c1):
        return {(r, c) for c in range(c0, c1 + 1)}

    def getL(h, w):
        return connect_v(0, h - 1, 0) | connect_h(0, 0, w - 1)

    def getU(h, w):
        return (connect_h(0, 0, w - 1) | connect_v(0, random.randint(1, h - 1), 0)
                | connect_v(0, random.randint(1, h - 1), w - 1))

    def getH(h, w):
        base = connect_v(0, h - 1, 0) | connect_h(h - 1, 0, random.randint(1, w - 1))
        s = random.randint(1, w - 2)
        return connect_h(0, 0, w - 1) | {(i, j + s) for i, j in base}

    def normalize(obj):
        mi = min(i for i, j in obj)
        mj = min(j for i, j in obj)
        return {(i - mi, j - mj) for i, j in obj}

    mfs = [lambda o: set(o),
           lambda o: {(j, i) for i, j in o},
           lambda o: {(-j, -i) for i, j in o},
           lambda o: {(i, -j) for i, j in o},
           lambda o: {(-i, j) for i, j in o}]

    pairs = ((2, 2, getL, 1), (2, 3, getU, 6), (3, 3, getH, 2))
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = _unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 10)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        minh, minw, getter, ocol = random.choice(pairs)
        oh = random.randint(minh, 6)
        ow = random.randint(minw, 6)
        obj = getter(oh, ow)
        for fn in random.sample(mfs, random.choice((1, 2))):
            obj = normalize(fn(obj))
        oh = max(i for i, j in obj) + 1
        ow = max(j for i, j in obj) + 1
        cands = [(i, j) for (i, j) in inds if i <= h - oh and j <= w - ow]
        if not cands:
            continue
        li, lj = random.choice(cands)
        plcd = {(i + li, j + lj) for i, j in obj}
        if plcd.issubset(inds):
            succ += 1
            nb = set()
            for i, j in plcd:
                nb |= {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}
            inds = (inds - plcd) - nb
            col = random.choice(ccols)
            for i, j in plcd:
                gi[i][j] = col
                go[i][j] = ocol
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: the canvas colour; shapes are sparse lines so it is the majority
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 4-connected same-colour objects (objects never touch orthogonally)
    seen = np.zeros((hi, wi), dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            objs.append(sorted(cells))

    fg = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    if fg:
        # base layer: every shape painted 2 (the "H" colour, the default class)
        ops.append(2); sels.append(sel_of(fg))

        Ls, Us = [], []
        for cells in objs:
            rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            oh, ow = r1 - r0 + 1, c1 - c0 + 1
            if len(cells) == oh + ow - 1:
                Ls.append(cells)                        # single corner path
            elif all(p[0] in (r0, r1) or p[1] in (c0, c1) for p in cells):
                Us.append(cells)                        # lies on its bbox border
        # L shapes -> 1, one flood fill per object
        for cells in Ls:
            ops.append(11); sels.append(sel_of([cells[0]]))
        # U shapes -> 6, one flood fill per object
        for cells in Us:
            ops.append(16); sels.append(sel_of([cells[0]]))

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
