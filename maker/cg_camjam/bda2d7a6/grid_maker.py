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


def sample_colors(num_examples=None) -> dict:
    # The rule only uses the nesting of the rings: each ring takes the colour of
    # the ring around it, and the outermost ring takes the innermost colour.
    # It never depends on which colours are used, so the ring colours stay random
    # for each instance. The canvas is completely covered by rings, so there is
    # no background colour to fix either.
    return {}


def _segment(I):
    """Split the grid into 4-connected objects of one colour, background included."""
    H, W = I.shape
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
    # sort from the innermost object to the outermost
    objs.sort(key=lambda o: (o["span"], len(o["cells"])))
    # if the innermost and outermost objects share a colour, treat them as one object
    if len(objs) > 1 and objs[0]["color"] == objs[-1]["color"]:
        merged = {"color": objs[0]["color"],
                  "cells": objs[0]["cells"] + objs[-1]["cells"],
                  "span": objs[-1]["span"]}
        objs = [merged] + objs[1:-1]
    return objs


def _apply_rule(I):
    objs = _segment(I)
    n = len(objs)
    O = I.copy()
    for k in range(n):
        newc = objs[(k + 1) % n]["color"]
        for (r, c) in objs[k]["cells"]:
            O[r, c] = newc
    return O


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    hmax = max(2, min(14, max_h // 2))
    wmax = max(2, min(14, max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    ncols = unifint(diff_lb, diff_ub, (2, 10))
    cols = random.sample(range(10), ncols)
    colord = [random.choice(cols) for _ in range(min(h, w))]
    H, W = h * 2, w * 2
    gi = np.zeros((H, W), dtype=int)
    for idx, ci in enumerate(colord):
        r0, c0 = idx, idx
        r1, c1 = H - 1 - idx, W - 1 - idx
        gi[r0, c0:c1 + 1] = ci
        gi[r1, c0:c1 + 1] = ci
        gi[r0:r1 + 1, c0] = ci
        gi[r0:r1 + 1, c1] = ci
    go = _apply_rule(gi)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """Each ring takes the colour of the ring around it, and the outermost ring
    takes the innermost colour. Everything is measured from I.

    The golfed solution treats this as a colour permutation, so all cells of one
    colour are recoloured as a single unit. That works whenever a colour has one
    outcome. RE-ARC can reuse a colour in rings that need different new colours
    (e.g. 3 -> 0 on the outer ring and 3 -> 7 on an inner ring). So:
      * if every object of a colour gets the same new colour, recolour the whole
        colour class at once (all its cells in I, one Color op);
      * otherwise fall back to one Color op per object.
    Every selection comes from I and every cell is painted once, so the order of
    the ops cannot interfere. They are emitted from the outermost ring inward."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    objs = _segment(I)
    n = len(objs)

    # new colour of each object, taken from I only
    for k in range(n):
        objs[k]["new"] = objs[(k + 1) % n]["color"]

    # for each colour, the set of new colours its objects receive
    targets = {}
    for o in objs:
        targets.setdefault(o["color"], set()).add(o["new"])

    ops, sels = [], []
    done_colours = set()
    for k in range(n - 1, -1, -1):          # outermost -> innermost
        o = objs[k]
        old, newc = o["color"], o["new"]
        if len(targets[old]) == 1:
            # the colour maps the same way everywhere: recolour all its cells at once
            if old in done_colours:
                continue
            done_colours.add(old)
            if newc == old:
                continue
            cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(I == old))]
            ops.append(int(newc))
            sels.append(sel_of(cells))
        else:
            # the colour maps differently in different rings: recolour this object only
            if newc == old:
                continue
            ops.append(int(newc))
            sels.append(sel_of(o["cells"]))

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
