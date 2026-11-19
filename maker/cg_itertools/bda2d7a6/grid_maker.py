"""
ARC Task: bda2d7a6 (RE-ARC) — LLM-generated grid_maker
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

try:
    from dsl import *
    from utils import *
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    # Every instance samples its own palette and ring order. The rule is the same for
    # any palette: each nested ring's colour moves inward by one ring, and the outermost
    # ring takes the innermost colour. So no colour role has to be fixed for the episode.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **color_kwargs) -> dict:
    colopts = list(range(10))
    hmax = max(2, min(14, max_h // 2))
    wmax = max(2, min(14, max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    ncols = unifint(diff_lb, diff_ub, (2, 10))
    cols = random.sample(colopts, ncols)
    colord = [random.choice(cols) for j in range(min(h, w))]
    shp = (h * 2, w * 2)
    gi = canvas(0, shp)
    for idx in range(len(colord)):
        ci = colord[idx]
        ulc = (idx, idx)
        lrc = (h * 2 - 1 - idx, w * 2 - 1 - idx)
        bx = box(frozenset({ulc, lrc}))
        gi = fill(gi, ci, bx)
    I = gi
    objso = order(objects(I, T, F, F), compose(maximum, shape))
    if color(objso[0]) == color(objso[-1]):
        objso = (combine(objso[0], objso[-1]),) + objso[1:-1]
    res = mpapply(recolor, apply(color, objso), (objso[-1],) + objso[:-1])
    go = paint(gi, res)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Colours shift one ring inward: each nested ring takes the colour of the ring just
    outside it, and the outermost ring takes the innermost colour.

    How this compares with the golf reading: the golf entry recolours every cell by its
    colour, mapping each colour to the previous one in the ring sequence. That treats a
    colour class as the unit. It works when no colour appears twice. In this generator
    colours DO repeat (Example 1 has two separate rings of colour 2, which turn 5 and 9),
    so a colour class is not one unit here. The unit has to be the nested ring, i.e. the
    4-connected same-colour object of I. The two readings agree on the rule itself (a
    cyclic shift of colours through the nesting) and on its wrap-around. The golf
    entry's "same colour = same unit" idea shows up here only in the case the generator
    itself merges: when the innermost and outermost objects share a colour, they count
    as one unit. Otherwise the maker's carving is the right one, so it is kept.

    Every selection is one object of I (a whole ring, the core, or the merged
    core+outer pair), so it can be named from the input alone. Each Color mask is
    fixed by I, so the order of painting does not matter. Painting runs from the outside
    inward."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    # --- split I into 4-connected same-colour objects (the rings and the core) ---
    seen = np.zeros((H, W), dtype=bool)
    objs = []
    for r in range(H):
        for c in range(W):
            if seen[r, c]:
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
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            span = max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
            objs.append({"color": col, "cells": cells, "span": span})

    # --- sort innermost to outermost by extent ---
    objs.sort(key=lambda o: (o["span"], len(o["cells"])))

    # --- core and outermost ring share a colour -> treat them as one unit ---
    if len(objs) > 1 and objs[0]["color"] == objs[-1]["color"]:
        merged = {"color": objs[0]["color"],
                  "cells": objs[0]["cells"] + objs[-1]["cells"],
                  "span": objs[-1]["span"]}
        objs = [merged] + objs[1:-1]

    n = len(objs)
    ops, sels = [], []

    # --- paint each ring with the next-outer ring's colour (outermost gets the core's) ---
    for k in range(n - 1, -1, -1):
        newc = objs[(k + 1) % n]["color"]
        if newc == objs[k]["color"]:
            continue  # this ring already has that colour, so painting it would change nothing
        ops.append(int(newc))
        sels.append(sel_of(objs[k]["cells"]))

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task bda2d7a6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bda2d7a6"
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
                                f"for task bda2d7a6"
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
                    f"Failed to build a complete episode for task bda2d7a6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bda2d7a6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
