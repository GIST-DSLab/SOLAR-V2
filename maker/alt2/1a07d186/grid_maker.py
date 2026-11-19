"""
ARC Task: 1a07d186 (RE-ARC) — LLM-generated grid_maker
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
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# ---------------------------------------------------------------- 1. colors
VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    # Only the background must be shared across the whole episode: the rule
    # ("a loose cell whose colour equals a full line's colour slides to that
    # line, otherwise it vanishes") is about colour *matching*, not about any
    # particular palette entry.
    bgc = random.choice(list(range(10)))

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, transpose=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if transpose is None:
        transpose = choice((True, False))
    mh = max(8, min(30, int(max_h)))
    mw = max(8, min(30, int(max_w)))
    # a dmirror at the end swaps the axes, so swap the bounds up front
    if transpose:
        hbnd, wbnd = mw, mh
    else:
        hbnd, wbnd = mh, mw

    h = unifint(diff_lb, diff_ub, (8, hbnd))
    w = unifint(diff_lb, diff_ub, (8, wbnd))
    if bgc is None:
        bgc = choice(cols)
    remcols = remove(bgc, cols)
    nlines = unifint(diff_lb, diff_ub, (1, w // 5))
    linecols = sample(remcols, nlines)
    remcols = difference(remcols, linecols)
    nnoisecols = unifint(diff_lb, diff_ub, (0, len(remcols)))
    noisecols = sample(remcols, nnoisecols)
    locopts = interval(0, w, 1)
    locs = []
    for k in range(nlines):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        locopts = difference(locopts, interval(loc - 2, loc + 3, 1))
        locs.append(loc)
    locs = sorted(locs)
    nlines = len(locs)
    linecols = linecols[:nlines]
    gi = canvas(bgc, (h, w))
    for loc, col in zip(locs, linecols):
        gi = fill(gi, col, connect((0, loc), (h - 1, loc)))
    go = tuple(e for e in gi)
    nilocs = unifint(diff_lb, diff_ub, (1, h))
    ilocs = sample(interval(0, h, 1), nilocs)
    dotlocopts = difference(interval(0, w, 1), locs)
    for ii in ilocs:
        ndots = unifint(diff_lb, diff_ub, (1, min(nlines + nnoisecols, (w - nlines) // 2 - 1)))
        dotlocs = sample(dotlocopts, ndots)
        dotcols = sample(totuple(set(linecols) | set(noisecols)), ndots)
        for dotlocj, col in zip(dotlocs, dotcols):
            gi = fill(gi, col, {(ii, dotlocj)})
            if col in linecols:
                idx = linecols.index(col)
                linelocj = locs[idx]
                if dotlocj > linelocj:
                    go = fill(go, col, {(ii, linelocj + 1)})
                else:
                    go = fill(go, col, {(ii, linelocj - 1)})
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- 3. derive
def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: the colour the canvas is painted with before lines/dots
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # ---- the full-grid lines (frontiers) --------------------------------
    col_lines, row_lines = {}, {}
    for c in range(w):
        v = set(I[:, c].tolist())
        if len(v) == 1 and int(I[0, c]) != bgc:
            col_lines[c] = int(I[0, c])
    for r in range(h):
        v = set(I[r, :].tolist())
        if len(v) == 1 and int(I[r, 0]) != bgc:
            row_lines[r] = int(I[r, 0])

    if not col_lines and not row_lines:
        return [34], [[0, 0, h - 1, w - 1]]

    vertical = len(col_lines) >= len(row_lines)
    lines = col_lines if vertical else row_lines
    line_of_color = {col: pos for pos, col in lines.items()}

    line_cells = set()
    if vertical:
        for c in lines:
            line_cells.update((r, c) for r in range(h))
    else:
        for r in lines:
            line_cells.update((r, c) for c in range(w))

    # ---- the loose cells -------------------------------------------------
    dead = []          # colour matches no line -> vanishes
    travellers = []    # [colour, src, dst]
    for r in range(h):
        for c in range(w):
            col = int(I[r, c])
            if col == bgc or (r, c) in line_cells:
                continue
            if col not in line_of_color:
                dead.append((r, c))
                continue
            L = line_of_color[col]
            if vertical:
                dst = (r, L + 1) if c > L else (r, L - 1)
            else:
                dst = (L + 1, c) if r > L else (L - 1, c)
            if dst == (r, c):
                continue                      # already resting against its line
            travellers.append([col, (r, c), dst])

    g = I.copy()

    # every loose cell with no matching line disappears -> one Color(bgc)
    if dead:
        ops.append(bgc)
        sels.append(sel_of(dead))
        for (r, c) in dead:
            g[r, c] = bgc

    # ARCLE reads 0 as "nothing there", so colour-0 cells cannot ride a Move:
    # lift them off their square by hand now, put them down with Color0 later.
    zero_src = [m[1] for m in travellers if m[0] == 0]
    if zero_src:
        ops.append(bgc)
        sels.append(sel_of(zero_src))
        for (r, c) in zero_src:
            g[r, c] = bgc

    pending_paint = [m for m in travellers if m[0] == 0]
    pending_move = [m for m in travellers if m[0] != 0]

    def between(src, dst):
        (r0, c0), (r1, c1) = src, dst
        if r0 == r1:
            st = 1 if c1 > c0 else -1
            return [(r0, cc) for cc in range(c0 + st, c1, st)]
        st = 1 if r1 > r0 else -1
        return [(rr, c0) for rr in range(r0 + st, r1, st)]

    def slide(col, src, dst):
        """Move the single-cell object from src to dst, then repair only the
        square it truly vacated (ARCLE keeps the path it crossed alive)."""
        (r0, c0), (r1, c1) = src, dst
        if r0 != r1:
            op = 21 if r1 > r0 else 20
            steps = abs(r1 - r0)
        else:
            op = 22 if c1 > c0 else 23
            steps = abs(c1 - c0)
        ops.append(op)
        sels.append(sel_of([src]))            # first Move GRABS the cell
        for _ in range(steps - 1):
            ops.append(op)
            sels.append(sel_of([]))           # empty -> keep gliding same object
        g[r0, c0] = 0
        g[r1, c1] = col
        if bgc != 0:                          # the vacated square reads 0 now
            ops.append(bgc)
            sels.append(sel_of([src]))
            g[r0, c0] = bgc

    guard = 0
    while (pending_move or pending_paint) and guard < 10000:
        guard += 1
        progressed = False

        for m in list(pending_paint):
            if int(g[m[2]]) == bgc:
                ops.append(0)
                sels.append(sel_of([m[2]]))
                g[m[2]] = 0
                pending_paint.remove(m)
                progressed = True

        for m in list(pending_move):
            if int(g[m[2]]) == bgc:
                slide(m[0], m[1], m[2])
                pending_move.remove(m)
                progressed = True

        if progressed:
            continue

        # --- every remaining destination is held by another traveller ---
        # nudge one of them as far along its own route as is currently clear
        moved = False
        for m in pending_move:
            free = [cell for cell in between(m[1], m[2]) if int(g[cell]) == bgc]
            if free:
                slide(m[0], m[1], free[-1])
                m[1] = free[-1]
                moved = True
                break
        if moved:
            continue

        # otherwise step one traveller aside onto a clear square of its own
        # row/column, clearing the knot; it resumes its route afterwards
        reserved = {m[2] for m in pending_move} | {m[2] for m in pending_paint}
        for m in pending_move:
            src = m[1]
            axis = [(src[0], cc) for cc in range(w)] if vertical else [(rr, src[1]) for rr in range(h)]
            cand = [cell for cell in axis
                    if cell != src and cell not in reserved and int(g[cell]) == bgc]
            if cand:
                best = min(cand, key=lambda x: abs(x[0] - src[0]) + abs(x[1] - src[1]))
                slide(m[0], src, best)
                m[1] = best
                moved = True
                break
        if moved:
            continue

        # last resort (no clear square anywhere on the axis): lift then place
        for m in pending_move:
            ops.append(bgc)
            sels.append(sel_of([m[1]]))
            g[m[1]] = bgc
        for m in pending_move:
            ops.append(int(m[0]))
            sels.append(sel_of([m[2]]))
            g[m[2]] = m[0]
        pending_move = []
        for m in pending_paint:
            ops.append(0)
            sels.append(sel_of([m[2]]))
            g[m[2]] = 0
        pending_paint = []

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1a07d186"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1a07d186"
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
                                f"for task 1a07d186"
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
                    f"Failed to build a complete episode for task 1a07d186 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1a07d186-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
