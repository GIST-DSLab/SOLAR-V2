"""
ARC Task: 0a938d79 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    cols = list(range(10))
    bgc, cola, colb = random.sample(cols, 3)
    variants = [{"orient": "landscape"}, {"orient": "portrait"}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, max(1, n_ex))]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "cola": cola, "colb": colb, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cola, colb, orient=None) -> dict:
    import random

    if orient is None:
        orient = random.choice(("landscape", "portrait"))

    # Work in a normalized "landscape" frame: H rows, W cols, W > H,
    # two dots on the top/bottom row, full-height lines repeating rightwards.
    # The portrait variant is the transpose of that frame.
    if orient == "landscape":
        Hlim, Wlim = min(29, max_h), min(30, max_w)
    else:
        Hlim, Wlim = min(29, max_w), min(30, max_h)
    Wlim = max(6, Wlim)
    Hmax = max(4, min(Hlim, Wlim - 1))

    def uf(lb, ub):
        if ub <= lb:
            return lb
        a = min(max(float(diff_lb), 0.0), 1.0)
        b = min(max(float(diff_ub), 0.0), 1.0)
        lo = lb + int(round((ub - lb) * a))
        hi = lb + int(round((ub - lb) * b))
        if hi < lo:
            lo, hi = hi, lo
        lo = max(lb, min(ub, lo))
        hi = max(lb, min(ub, hi))
        return random.randint(lo, hi)

    H = uf(4, Hmax)
    Wlo = max(H + 1, 6)
    Whi = max(Wlo, Wlim)
    W = uf(Wlo, Whi)

    # leftmost dot column (kept small so at least three lines fit)
    c1 = uf(1, max(1, (W - 1) // 3))
    dhi = max(2, (W - 1 - c1) // 2)
    d = uf(2, dhi)
    c2 = c1 + d
    if c2 > W - 2:
        c2 = W - 2
        d = c2 - c1

    r1 = random.choice((0, H - 1))
    r2 = random.choice((0, H - 1))
    if random.random() < 0.5:
        leftcol, rightcol = cola, colb
    else:
        leftcol, rightcol = colb, cola

    gi = [[bgc] * W for _ in range(H)]
    go = [[bgc] * W for _ in range(H)]
    gi[r1][c1] = leftcol
    gi[r2][c2] = rightcol

    k = 0
    while c1 + k * d <= W - 1:
        cc = c1 + k * d
        col = leftcol if k % 2 == 0 else rightcol
        for r in range(H):
            go[r][cc] = col
        k += 1

    if orient == "portrait":
        gi = [list(row) for row in zip(*gi)]
        go = [list(row) for row in zip(*go)]

    return {"input": [[int(v) for v in row] for row in gi],
            "output": [[int(v) for v in row] for row in go]}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in cells]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    dots = [(r, c, int(I[r, c])) for r in range(h) for c in range(w) if I[r, c] != bgc]

    if len(dots) != 2:
        # defensive fallback: paint the differing cells grouped by target colour
        for col in sorted({int(O[r, c]) for r in range(ho) for c in range(wo)
                           if O[r, c] != I[r, c]}):
            cells = [(r, c) for r in range(ho) for c in range(wo)
                     if O[r, c] != I[r, c] and int(O[r, c]) == col]
            if cells:
                ops.append(col)
                sels.append(sel_of(cells))
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    horizontal = h > w  # portrait instance -> full-width horizontal lines

    if not horizontal:
        dots.sort(key=lambda t: t[1])
        p1, k1 = dots[0][1], dots[0][2]
        p2, k2 = dots[1][1], dots[1][2]
        limit = w - 1
        flip_op = 26  # FlipH: left<->right, mirrors a vertical strip

        def line_cells(p):
            return [(r, p) for r in range(h)]

        def band(a, b):                 # full rectangle: whole columns a..b
            return [0, a, h - 1, b - a]

        def origin(b):                  # paste origin (top-left of the band)
            return [0, b, 0, 0]
    else:
        dots.sort(key=lambda t: t[0])
        p1, k1 = dots[0][0], dots[0][2]
        p2, k2 = dots[1][0], dots[1][2]
        limit = h - 1
        flip_op = 27  # FlipV: up<->down, mirrors a horizontal band

        def line_cells(p):
            return [(p, c) for c in range(w)]

        def band(a, b):                 # full rectangle: whole rows a..b
            return [a, 0, b - a, w - 1]

        def origin(b):
            return [b, 0, 0, 0]

    d = p2 - p1

    # 1. each dot grows into a full line across the grid
    ops.append(k1)
    sels.append(sel_of(line_cells(p1)))
    ops.append(k2)
    sels.append(sel_of(line_cells(p2)))

    # 2. every further line is the mirror image of the previous pair:
    #    copy the strip holding the last two lines, lay it down starting at the
    #    last line, and reflect that strip in place.
    K = (limit - p1) // d
    for k in range(2, K + 1):
        a = p1 + (k - 2) * d
        b = p1 + (k - 1) * d
        ops.append(29)                  # CopyO: the strip we ourselves drew
        sels.append(band(a, b))         # full rectangle (background included)
        ops.append(30)                  # Paste at the strip's new origin
        sels.append(origin(b))
        ops.append(flip_op)             # reflect the strip in place
        sels.append(band(b, b + d))     # full rectangle (background included)

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 0a938d79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0a938d79"
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
                                f"for task 0a938d79"
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
                    f"Failed to build a complete episode for task 0a938d79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0a938d79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
