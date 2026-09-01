"""
ARC Task: 890034e9 (RE-ARC) — LLM-generated grid_maker
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
try:
    from maker.sel_helpers import sel_of
except Exception:
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for r, c in cells]}


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    markercol = random.choice(cols)
    remcols = [c for c in cols if c != markercol]
    numbgc = random.randint(2, 8)
    bgcols = random.sample(remcols, numbgc)
    return {"markercol": markercol, "bgcols": bgcols}


def generate(diff_lb, diff_ub, max_h, max_w, markercol, bgcols) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    def box_cells(r0, c0, r1, c1):
        cells = set()
        for c in range(c0, c1 + 1):
            cells.add((r0, c)); cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c0)); cells.add((r, c1))
        return cells

    def zero_blocks(g, oh, ow):
        hh = len(g); ww = len(g[0])
        return [(r, c) for r in range(hh - oh + 1) for c in range(ww - ow + 1)
                if all(g[i][j] == 0 for i in range(r, r + oh) for j in range(c, c + ow))]

    def chosen_go(g, hh, ww, oh, ow, marker_block, mcol):
        # the task's rule applied to the finished input
        out = [row[:] for row in g]
        cands = [(r, c) for (r, c) in zero_blocks(g, oh, ow)
                 if r - 1 >= 0 and c - 1 >= 0 and r + oh < hh and c + ow < ww]
        used = set()
        for (r, c) in [marker_block] + [x for x in cands if x != marker_block]:
            ring = box_cells(r - 1, c - 1, r + oh, c + ow)
            if ring & used:
                continue
            used |= ring
            for (a, b) in ring:
                out[a][b] = mcol
        return out

    last = None
    raw = None
    for _attempt in range(400):
        h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
        oh = random.randint(2, max(2, h // 4))
        ow = random.randint(2, max(2, w // 4))
        if h < oh + 2 or w < ow + 2:
            continue
        gi = [[random.choice(bgcols) for _ in range(w)] for _ in range(h)]
        inds = {(i, j) for i in range(h) for j in range(w)}
        numbl = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
        for (i, j) in random.sample(sorted(inds), numbl):
            gi[i][j] = 0
        # make sure no oh x ow all-zero block exists by accident yet
        for (r, c) in zero_blocks(gi, oh, ow):
            i, j = random.choice([(a, b) for a in range(r, r + oh) for b in range(c, c + ow)])
            gi[i][j] = random.choice(bgcols)
        noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // ((oh + 2) * (ow + 2)))))
        go = [row[:] for row in gi]
        placed = []
        tr = 0
        maxtr = 5 * noccs
        while tr < maxtr and len(placed) < noccs:
            tr += 1
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                break
            loci, locj = random.choice(sorted(cands))
            ring = box_cells(loci - 1, locj - 1, loci + oh, locj + ow)
            if not ring.issubset(inds):
                continue
            first = (len(placed) == 0)
            placed.append((loci, locj))
            inds -= ring
            for a in range(loci, loci + oh):
                for b in range(locj, locj + ow):
                    gi[a][b] = 0
                    go[a][b] = 0
            for (a, b) in ring:
                go[a][b] = markercol
                if first:
                    gi[a][b] = markercol
            if not first:
                lns = [[(loci - 1, locj + k) for k in range(ow)],
                       [(loci + oh, locj + k) for k in range(ow)],
                       [(loci + k, locj - 1) for k in range(oh)],
                       [(loci + k, locj + ow) for k in range(oh)]]
                for ln in lns:
                    a, b = random.choice(ln)
                    gi[a][b] = random.choice(bgcols)
        raw = (gi, go)
        if len(placed) < 2:
            continue
        # keep only instances whose rule is unambiguously recoverable from the input:
        # frame every all-zero oh x ow block that has room for a frame, marker box
        # first, frames never overlapping each other
        marker_block = placed[0]
        cands = [(r, c) for (r, c) in zero_blocks(gi, oh, ow)
                 if r - 1 >= 0 and c - 1 >= 0 and r + oh < h and c + ow < w]
        used = set()
        chosen = []
        for (r, c) in [marker_block] + [x for x in cands if x != marker_block]:
            ring = box_cells(r - 1, c - 1, r + oh, c + ow)
            if ring & used:
                continue
            used |= ring
            chosen.append((r, c))
        if set(chosen) != set(placed):
            if len(chosen) >= 2:
                last = (gi, chosen_go(gi, h, w, oh, ow, marker_block, markercol))
            continue
        return {"input": tuple(tuple(row) for row in gi),
                "output": tuple(tuple(row) for row in go)}
    gi, go = last if last is not None else raw
    return {"input": tuple(tuple(row) for row in gi),
            "output": tuple(tuple(row) for row in go)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    grid = [[int(v) for v in row] for row in I.tolist()]
    ops, sels = [], []

    def box_cells(r0, c0, r1, c1):
        cells = set()
        for c in range(c0, c1 + 1):
            cells.add((r0, c)); cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c0)); cells.add((r, c1))
        return cells

    def frame_candidates(g):
        # the marker: a colour whose cells form exactly one hollow rectangle,
        # at least 4x4 (its hollow interior is at least 2x2 and entirely 0)
        out = []
        for col in range(1, 10):
            cells = {(r, c) for r in range(hi) for c in range(wi) if g[r][c] == col}
            if not cells:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            if cells != box_cells(r0, c0, r1, c1):
                continue
            if any(g[r][c] != 0 for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)):
                continue
            out.append((col, r0, c0, r1, c1))
        return out

    def blocks_for(g, oh, ow):
        # every all-zero oh x ow block that has room for a frame around it
        return [(r, c)
                for r in range(1, hi - oh)
                for c in range(1, wi - ow)
                if all(g[i][j] == 0
                       for i in range(r, r + oh)
                       for j in range(c, c + ow))]

    def select_frames(g, oh, ow, marker_block):
        used = set()
        chosen = []
        cands = blocks_for(g, oh, ow)
        for (r, c) in [marker_block] + [b for b in cands if b != marker_block]:
            ring = box_cells(r - 1, c - 1, r + oh, c + ow)
            if ring & used:          # frames never overlap each other
                continue
            used |= ring
            chosen.append((r, c))
        return chosen

    def find_marker(g):
        cands = frame_candidates(g)
        if not cands:
            return None
        scored = []
        for (col, r0, c0, r1, c1) in cands:
            n = len(select_frames(g, r1 - r0 - 1, c1 - c0 - 1, (r0 + 1, c0 + 1)))
            scored.append((0 if n >= 2 else 1, col, r0, c0, r1, c1))
        scored.sort()
        return scored[0][1:]

    found = find_marker(grid)
    if found is not None:
        markercol, mr0, mc0, mr1, mc1 = found
        oh = mr1 - mr0 - 1          # hollow interior height of the marker frame
        ow = mc1 - mc0 - 1          # hollow interior width
        if oh >= 2 and ow >= 2:
            marker_block = (mr0 + 1, mc0 + 1)
            # draw the marker's frame around every hidden box, one frame per op
            for (r, c) in select_frames(grid, oh, ow, marker_block):
                ring = box_cells(r - 1, c - 1, r + oh, c + ow)
                todo = sorted(cell for cell in ring
                              if grid[cell[0]][cell[1]] != markercol)
                if not todo:         # the marker box is already framed
                    continue
                for (a, b) in todo:
                    grid[a][b] = markercol
                ops.append(int(markercol))
                sels.append(sel_of(todo))

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
                        f"num_examples+1 ({num_examples + 1}) for task 890034e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 890034e9"
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
                                f"for task 890034e9"
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
                    f"Failed to build a complete episode for task 890034e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"890034e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
