"""
ARC Task: f8ff0b80 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + round((b - a) * diff_lb)
    hi = a + round((b - a) * diff_ub)
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; object colours are free
    # (the rule reads object sizes, not which colour they carry).
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (10, max_h))
    w = _unifint(diff_lb, diff_ub, (10, max_w))
    remcols = [c for c in range(10) if c != bgc]
    nobjs = _unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 25)))
    gi = [[bgc] * w for _ in range(h)]
    numcells = _unifint(diff_lb, diff_ub, (nobjs + 1, 36))
    base = [(i, j) for i in range(6) for j in range(6)]
    maxtr = 10
    inds = {(i, j) for i in range(h) for j in range(w)}
    go = []

    def nbrs4(cell):
        i, j = cell
        return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

    for k in range(nobjs):
        if len(inds) == 0 or numcells < 2:
            break
        numcells = _unifint(diff_lb, diff_ub, (nobjs - k, numcells - 1))
        if numcells == 0:
            break
        sp = random.choice(base)
        shp = {sp}
        reminds = set(base) - {sp}
        for _ in range(numcells - 1):
            front = set()
            for cell in shp:
                front |= nbrs4(cell)
            cand = sorted((reminds - shp) & front)
            if not cand:
                break
            shp.add(random.choice(cand))
        mr = min(i for i, _ in shp)
        mc = min(j for _, j in shp)
        shp = {(i - mr, j - mc) for i, j in shp}
        sh = max(i for i, _ in shp) + 1
        sw = max(j for _, j in shp) + 1
        if not any(i <= h - sh and j <= w - sw for i, j in inds):
            break
        validloc = False
        tr = 0
        plcd = None
        indl = sorted(inds)
        while not validloc and tr < maxtr:
            loc = random.choice(indl)
            plcd = {(i + loc[0], j + loc[1]) for i, j in shp}
            validloc = plcd.issubset(inds)
            tr += 1
        if validloc:
            # avoid a colour that touches this object diagonally, so every
            # object stays its own 8-connected component
            touching = set()
            for (i, j) in plcd:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and (ni, nj) not in plcd:
                            touching.add(gi[ni][nj])
            okcols = [c for c in remcols if c not in touching] or remcols
            col = random.choice(okcols)
            go.append(col)
            rem = set(plcd)
            for cell in plcd:
                rem |= nbrs4(cell)
            inds = inds - rem
            for (i, j) in plcd:
                gi[i][j] = col
    if not go:
        raise ValueError("no objects placed")
    out = [[c] for c in go]
    return {"input": gi, "output": out}


def derive_operations(I, O, examples=None):
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [(int(r), int(c)) for r, c in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background: canvas colour; objects cover a small fraction, so majority
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # objects: same-colour 8-connected components of non-background
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append((len(cells), min(cells), col))

    # Reading of the task: rank objects by how many cells they have,
    # LARGEST FIRST, and write that ranking straight down a column
    # (no ascending list followed by a mirror).
    comps.sort(key=lambda t: (-t[0], t[1]))
    desc = [t[2] for t in comps]
    n = max(1, len(desc))

    ops, sels = [], []

    # 1. resize canvas to the n x 1 column (full rectangle selection)
    ops.append(33)
    sels.append([0, 0, n - 1, 0])
    # transparent crop keeps nonzero input values in the first column
    cur = [int(I[i, 0]) if i < h else 0 for i in range(n)]

    # 2. write one cell per object, largest object at the top, going down
    for i, col in enumerate(desc):
        if cur[i] != col:
            ops.append(int(col))
            sels.append(sel_of([(i, 0)]))
            cur[i] = int(col)

    ops.append(34)
    sels.append([0, 0, n - 1, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8ff0b80"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8ff0b80"
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
                                f"for task f8ff0b80"
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
                    f"Failed to build a complete episode for task f8ff0b80 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8ff0b80-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
