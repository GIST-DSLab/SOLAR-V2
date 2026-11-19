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

try:
    from dsl import *
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    # The background is the only colour role. Each object's colour is copied
    # into the output as is, so the rule never depends on which colour it is.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 25)))
    gi = canvas(bgc, (h, w))
    numcells = unifint(diff_lb, diff_ub, (nobjs + 1, 36))
    base = asindices(canvas(-1, (6, 6)))
    maxtr = 10
    inds = asindices(gi)
    go = []
    for k in range(nobjs):
        if len(inds) == 0 or numcells < 2:
            break
        numcells = unifint(diff_lb, diff_ub, (nobjs - k, numcells - 1))
        if numcells == 0:
            break
        sp = choice(totuple(base))
        shp = {sp}
        reminds = remove(sp, base)
        for kk in range(numcells - 1):
            shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
        shp = normalize(shp)
        validloc = False
        rems = sfilter(inds, lambda ij: ij[0] <= h - height(shp) and ij[1] <= w - width(shp))
        if len(rems) == 0:
            break
        loc = choice(totuple(rems))
        tr = 0
        while not validloc and tr < maxtr:
            loc = choice(totuple(inds))
            validloc = shift(shp, loc).issubset(inds)
            tr += 1
        if validloc:
            plcd = shift(shp, loc)
            col = choice(remcols)
            go.append(col)
            inds = (inds - plcd) - mapply(neighbors, plcd)
            gi = fill(gi, col, plcd)
    if len(go) == 0:
        raise ValueError("no objects placed")
    go = dmirror((tuple(go),))
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [(int(r), int(c)) for r, c in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # Background: the canvas colour. Objects cover only a small part of the
    # grid, so it is the most common colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Objects: same-colour components of non-background cells, joined
    # diagonally too (the generator grows shapes with 8-neighbours).
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

    # Golf reading: rank by size, LARGEST first, read straight down. No
    # ascending list that gets mirrored afterwards. The generator makes every
    # object a different size, so this order is fixed.
    comps.sort(key=lambda t: (-t[0], t[1]))
    desc = [t[2] for t in comps]
    n = max(1, len(desc))

    ops, sels = [], []

    # 1. Crop the canvas to an n x 1 column: one slot per object, n counted
    #    from I. The selection is the full crop rectangle.
    ops.append(33)
    sels.append([0, 0, n - 1, 0])
    cur = [int(I[i, 0]) for i in range(n)]  # the transparent crop keeps these values

    # 2. Treat each colour as one unit, as the golf reading does: one Color
    #    op paints every slot whose object has that colour. Colours go in
    #    order of the largest object that has them. Slots already holding the
    #    colour are left out (checked against the current grid).
    order = []
    for col in desc:
        if col not in order:
            order.append(col)
    for col in order:
        rows = [i for i, cc in enumerate(desc) if cc == col and cur[i] != col]
        if not rows:
            continue
        ops.append(int(col))
        sels.append(sel_of([(i, 0) for i in rows]))
        for i in rows:
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
