"""
ARC Task: f5b8619d (RE-ARC) — LLM-generated grid_maker
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
    import random as _random
    cols = [c for c in range(10) if c != 8]          # 8 is the reserved fill colour
    bgc = _random.choice(cols)
    pal = [c for c in cols if c != bgc]
    _random.shuffle(pal)
    return {"bgc": bgc, "pal": pal}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, pal=None) -> dict:
    cols = remove(8, interval(0, 10, 1))
    if bgc is None:
        bgc = choice(cols)
    remcols = remove(bgc, cols)

    # the output is 2h x 2w, so the canvas may only use half the allowance
    hmax = max(2, min(15, max_h // 2))
    wmax = max(2, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))

    ncells = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 2 - 1)))
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    locs = sample(totuple(inds), ncells)
    blockcol = randint(0, w - 1)
    locs = sfilter(locs, lambda ij: ij[1] != blockcol)

    numcols = unifint(diff_lb, diff_ub, (1, 8))
    if pal:
        ccols = [c for c in pal if c != bgc][:numcols]
        if not ccols:
            ccols = sample(remcols, numcols)
    else:
        ccols = sample(remcols, numcols)

    obj = frozenset({(choice(ccols), ij) for ij in locs})
    gi = paint(gi, obj)
    go = fill(gi, 8, mapply(vfrontier, set(locs)) & (inds - set(locs)))
    go = hconcat(go, go)
    go = vconcat(go, go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [(int(r), int(c)) for r, c in cells]}

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 2 * hi, 2 * wi

    # --- the colour the rule paints with: read it from the worked examples
    #     (the one colour that shows up in an output but never in its input)
    fill = 8
    if examples:
        for ei, eo in examples:
            a = np.asarray(ei, dtype=int)
            b = np.asarray(eo, dtype=int)
            new = sorted(set(b.flatten().tolist()) - set(a.flatten().tolist()))
            if len(new) == 1:
                fill = int(new[0])
                break

    # --- background is the strict majority colour of the canvas
    bgc = int(np.bincount(I.flatten(), minlength=10).argmax())

    marks = {c: [r for r in range(hi) if I[r, c] != bgc] for c in range(wi)}
    blanks = {c: [r for r in range(hi) if I[r, c] == bgc] for c in range(wi)}
    # a column is worked only if it holds a marker AND still has canvas to fill
    targets = [c for c in range(wi) if marks[c] and blanks[c]]
    m = sum(len(v) for v in marks.values())

    # --- route choices, all read off the input so one instance always
    #     gives one route, and different instances give different ones
    style = (bgc + len(targets)) % 3        # 0 straight, 1 stripe-per-column, 2 stripe-all
    order_mode = (m + wi) % 3               # which column to deal with first
    tile_route = (hi + wi) % 3              # how the quarters get duplicated

    if order_mode == 0:
        cols = sorted(targets)                                   # sweep left -> right
    elif order_mode == 1:
        cols = sorted(targets, reverse=True)                     # sweep right -> left
    else:
        cols = sorted(targets, key=lambda c: (-len(marks[c]), c))  # busiest column first

    ops, sels = [], []

    def redraw(cs):
        # put the markers back on top of the stripe, one colour at a time;
        # colour 0 is an ordinary colour here and is laid down with Color0.
        order, groups = [], {}
        for c in cs:
            for r in marks[c]:
                v = int(I[r, c])
                if v not in groups:
                    groups[v] = []
                    order.append(v)
                groups[v].append((r, c))
        for v in order:
            ops.append(v)
            sels.append(sel_of(groups[v]))

    if style == 0:
        # draw straight onto the canvas: only the empty cells of each marked column
        for c in cols:
            ops.append(fill)
            sels.append(sel_of([(r, c) for r in blanks[c]]))
    elif style == 1:
        # lay the whole stripe down, then draw that column's markers back over it
        for c in cols:
            ops.append(fill)
            sels.append(sel_of([(r, c) for r in range(hi)]))
            redraw([c])
    else:
        # lay every stripe at once, then draw all the markers back over them
        if cols:
            ops.append(fill)
            sels.append(sel_of([(r, c) for c in cols for r in range(hi)]))
            redraw(cols)

    # --- the finished panel is repeated twice across and twice down
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])      # full output rectangle: grow the canvas

    panel = I.copy()
    for c in targets:
        for r in blanks[c]:
            panel[r, c] = fill

    if panel.any():
        if tile_route == 0:
            ops.append(29)
            sels.append([0, 0, hi - 1, wi - 1])          # whole panel rectangle
            for (dr, dc) in ((0, wi), (hi, 0), (hi, wi)):
                ops.append(30)
                sels.append([dr, dc, 0, 0])
        elif tile_route == 1:
            ops.append(29)
            sels.append([0, 0, hi - 1, wi - 1])          # panel rectangle
            ops.append(30)
            sels.append([0, wi, 0, 0])
            ops.append(29)
            sels.append([0, 0, hi - 1, wo - 1])          # the doubled top band
            ops.append(30)
            sels.append([hi, 0, 0, 0])
        else:
            ops.append(29)
            sels.append([0, 0, hi - 1, wi - 1])          # panel rectangle
            ops.append(30)
            sels.append([hi, 0, 0, 0])
            ops.append(29)
            sels.append([0, 0, ho - 1, wi - 1])          # the doubled left band
            ops.append(30)
            sels.append([0, wi, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task f5b8619d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f5b8619d"
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
                                f"for task f5b8619d"
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
                    f"Failed to build a complete episode for task f5b8619d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f5b8619d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
