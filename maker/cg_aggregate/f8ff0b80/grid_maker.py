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


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; object colours do not
    # affect the rule (output lists whatever colours the objects have).
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    def neighbors4(loc):
        r, c = loc
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    if bgc is None:
        bgc = random.choice(list(range(10)))
    remcols = [c for c in range(10) if c != bgc]
    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    nobjs = unifint(diff_lb, diff_ub, (1, min(30, max_h, (h * w) // 25)))
    gi = [[bgc] * w for _ in range(h)]
    numcells = unifint(diff_lb, diff_ub, (nobjs + 1, 36))
    base = {(i, j) for i in range(6) for j in range(6)}
    maxtr = 10
    inds = {(i, j) for i in range(h) for j in range(w)}
    go = []
    for k in range(nobjs):
        if len(inds) == 0 or numcells < 2:
            break
        numcells = unifint(diff_lb, diff_ub, (nobjs - k, numcells - 1))
        if numcells <= 0:
            break
        sp = random.choice(sorted(base))
        shp = {sp}
        reminds = base - {sp}
        for _ in range(numcells - 1):
            cand = set()
            for x in shp:
                cand |= neighbors4(x)
            cand = (reminds - shp) & cand
            if not cand:
                break
            shp.add(random.choice(sorted(cand)))
        mr = min(r for r, _ in shp)
        mc = min(c for _, c in shp)
        shp = {(r - mr, c - mc) for r, c in shp}
        sh = max(r for r, _ in shp) + 1
        sw = max(c for _, c in shp) + 1
        rems = [ij for ij in inds if ij[0] <= h - sh and ij[1] <= w - sw]
        if not rems:
            break
        validloc = False
        tr = 0
        plcd = None
        sinds = sorted(inds)
        while not validloc and tr < maxtr:
            loc = random.choice(sinds)
            plcd = {(r + loc[0], c + loc[1]) for r, c in shp}
            validloc = plcd.issubset(inds)
            tr += 1
        if validloc:
            col = random.choice(remcols)
            go.append(col)
            nb = set()
            for x in plcd:
                nb |= neighbors4(x)
            inds = (inds - plcd) - nb
            for r, c in plcd:
                gi[r][c] = col
    if not go:
        raise ValueError("no objects placed")
    out = [[c] for c in go]
    return {"input": gi, "output": out}


def derive_operations(I, O=None, examples=None):
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [(int(r), int(c)) for r, c in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # Background: the colour every demonstration input is painted on
    # (majority colour shared by all example inputs); fallback = majority of I.
    bgc = None
    if examples:
        cands = None
        for ex in examples:
            xi = np.asarray(ex[0], dtype=int)
            mc = Counter(xi.flatten().tolist()).most_common(1)[0][0]
            cands = {mc} if cands is None else (cands & {mc})
        if cands:
            c0 = next(iter(cands))
            if c0 in set(I.flatten().tolist()):
                bgc = c0
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Objects: same-colour, 8-connected components of non-background.
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

    # Object is the unit (colours may repeat across objects, so a per-colour
    # count reading would merge distinct objects). Smallest first.
    comps.sort(key=lambda t: (t[0], t[1]))
    asc = [t[2] for t in comps]
    n = max(1, len(asc))

    ops, sels = [], []

    # 1. Crop canvas to the n x 1 column (full rectangle selection).
    ops.append(33)
    sels.append([0, 0, n - 1, 0])
    cur = [int(I[i, 0]) if i < h else 0 for i in range(n)]

    # 2. One cell per object, ascending size from the top.
    for i, col in enumerate(asc):
        if cur[i] != col:
            ops.append(int(col))
            sels.append(sel_of([(i, 0)]))
            cur[i] = int(col)

    # 3. Mirror up<->down so the largest object is on top (whole column).
    if n > 1 and cur != cur[::-1]:
        ops.append(27)
        sels.append([0, 0, n - 1, 0])

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
