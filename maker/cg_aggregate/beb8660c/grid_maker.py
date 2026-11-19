"""
ARC Task: beb8660c (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice([c for c in range(10) if c != 8])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append(dict(random.choice(plan)))
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, rot=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice([c for c in range(10) if c != 8])
    if rot is None:
        rot = random.choice(ROTS)
    cap = min(max_h, max_w, 30)
    w = unifint(diff_lb, diff_ub, (3, cap))
    h = unifint(diff_lb, diff_ub, (w, cap))
    remcols = [c for c in range(10) if c not in (8, bgc)]
    gi = canvas(bgc, (h, w))
    k = min(8, w - 1)
    k = unifint(diff_lb, diff_ub, (1, k))
    co = sample(remcols, k)
    wds = sorted(sample(interval(1, w, 1), k))
    for j, (c, l) in enumerate(zip(co, wds)):
        j = h - k - 1 + j
        gi = fill(gi, c, connect((j, 0), (j, l - 1)))
    gi = fill(gi, 8, connect((h - 1, 0), (h - 1, w - 1)))
    go = vmirror(gi)
    gi = list(list(r) for r in gi[:-1])
    shuffle(gi)
    gi = tuple(tuple(r) for r in gi)
    gi = gi + go[-1:]
    gif = tuple()
    for r in gi:
        nbc = r.count(bgc)
        ofs = randint(0, nbc)
        gif = gif + (r[-ofs:] + r[:-ofs],)
    gi = vmirror(gif)
    rotf = (identity, rot90, rot180, rot270)[rot]
    return {"input": rotf(gi), "output": rotf(go)}


def derive_operations(I, O, examples=None):
    """Reading (same as the golf `sorted(map(sorted, rows))`): every coloured bar
    is an object. It slides along its own axis into the aligned corner (the
    in-row sort), then the bars stack against the 8-wall, longest nearest
    (the sort of the rows). Each bar is translated with Move ops."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # the 8-wall: a full edge row/column of colour 8
    if np.all(I[0, :] == 8):
        edge = 'top'
    elif np.all(I[hi - 1, :] == 8):
        edge = 'bottom'
    elif np.all(I[:, 0] == 8):
        edge = 'left'
    else:
        edge = 'right'
    horizontal = edge in ('top', 'bottom')

    cnt = Counter(I.flatten().tolist())
    cnt.pop(8, None)
    bgc = cnt.most_common(1)[0][0]

    # bars: one per non-bg, non-8 colour; longest first
    bars = []
    for col in sorted(cnt):
        if col == bgc:
            continue
        cells = sorted(map(tuple, np.argwhere(I == col).tolist()))
        bars.append({"col": col, "cells": cells, "L": len(cells)})
    bars.sort(key=lambda b: (-b["L"], b["col"]))

    def target(i, L):
        if edge == 'top':
            return [(1 + i, c) for c in range(L)]
        if edge == 'bottom':
            return [(hi - 2 - i, c) for c in range(wi - L, wi)]
        if edge == 'left':
            return [(r, 1 + i) for r in range(hi - L, hi)]
        return [(r, wi - 2 - i) for r in range(L)]

    def perp(cells):  # the row (horizontal bars) or column (vertical bars)
        return cells[0][0] if horizontal else cells[0][1]

    for i, b in enumerate(bars):
        b["tgt"] = target(i, b["L"])

    wall = 0 if edge in ('top', 'left') else (hi - 1 if horizontal else wi - 1)
    n_lanes = hi if horizontal else wi

    ops, sels = [], []

    def shift(cells, dr, dc):
        return [(r + dr, c + dc) for r, c in cells]

    def relocate(b, new_cells):
        src = b["cells"]
        if b["col"] == 0:
            # colour 0 is invisible to ARCLE's object grab: erase and redraw
            ops.append(bgc); sels.append(sel_of(src))
            ops.append(0); sels.append(sel_of(new_cells))
        else:
            dr = new_cells[0][0] - src[0][0]
            dc = new_cells[0][1] - src[0][1]
            # along the bar's own axis first, then across toward the wall
            if horizontal:
                steps = [22 if dc > 0 else 23] * abs(dc) + [21 if dr > 0 else 20] * abs(dr)
            else:
                steps = [21 if dr > 0 else 20] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
            for s, op in enumerate(steps):
                ops.append(op)
                sels.append(sel_of(src) if s == 0 else sel_of([]))
            hole = sorted(set(src) - set(new_cells))
            if bgc != 0 and hole:
                ops.append(bgc); sels.append(sel_of(hole))
        b["cells"] = list(new_cells)

    pending = [b for b in bars if sorted(b["cells"]) != sorted(b["tgt"])]
    placed_lanes = {perp(b["tgt"]) for b in bars if b not in pending}
    while pending:
        occupied = {perp(b["cells"]) for b in pending}
        ready = [b for b in pending
                 if perp(b["tgt"]) == perp(b["cells"]) or perp(b["tgt"]) not in occupied]
        if ready:
            b = ready[0]  # longest available first
            relocate(b, b["tgt"])
            pending.remove(b)
            placed_lanes.add(perp(b["tgt"]))
            continue
        # cycle: park one bar in a free lane (along-axis unchanged)
        b = pending[0]
        taken = occupied | placed_lanes | {wall}
        targets = {perp(x["tgt"]) for x in pending}
        free = [l for l in range(n_lanes) if l not in taken]
        free.sort(key=lambda l: (l in targets, abs(l - perp(b["cells"]))))
        if free:
            l = free[0]
            d = l - perp(b["cells"])
            relocate(b, shift(b["cells"], d, 0) if horizontal else shift(b["cells"], 0, d))
        else:
            # no lane to park in: lift the bar off the board (erase), redraw later
            ops.append(bgc); sels.append(sel_of(b["cells"]))
            b["cells"] = []
            pending.remove(b)
            bars_redraw = b
            while True:
                occ = {perp(x["cells"]) for x in pending}
                if perp(bars_redraw["tgt"]) not in occ:
                    break
                rdy = [x for x in pending
                       if perp(x["tgt"]) not in occ or perp(x["tgt"]) == perp(x["cells"])]
                if not rdy:
                    break
                x = rdy[0]
                relocate(x, x["tgt"])
                pending.remove(x)
                placed_lanes.add(perp(x["tgt"]))
            ops.append(b["col"]); sels.append(sel_of(b["tgt"]))
            placed_lanes.add(perp(b["tgt"]))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task beb8660c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task beb8660c"
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
                                f"for task beb8660c"
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
                    f"Failed to build a complete episode for task beb8660c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"beb8660c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
