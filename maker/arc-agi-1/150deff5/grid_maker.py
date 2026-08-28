"""
ARC Task: 150deff5 (RE-ARC) — LLM-generated grid_maker
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
import math
import random

import numpy as np

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The generator draws bgc/fgc from 0..9 without 2 and 8 (those are the
    # output-only marker colors).  Both are fixed for the whole episode.
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    def unifint(dlb, dub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        lo = int(math.ceil(a + dlb * (b - a)))
        hi = int(math.floor(a + dub * (b - a)))
        if hi < lo:
            hi = lo
        return min(max(a, random.randint(lo, hi)), b)

    bo = [(0, 0), (0, 1), (1, 0), (1, 1)]          # 2x2 square  -> 8
    ro1 = [(0, 0), (0, 1), (0, 2)]                 # horizontal 3-line -> 2
    ro2 = [(0, 0), (1, 0), (2, 0)]                 # vertical 3-line   -> 2
    shapes = {"bo": bo, "ro1": ro1, "ro2": ro2}

    hub = min(int(max_h), int(max_w), 30)
    lob = 8 if hub >= 8 else hub

    last = None
    for _attempt in range(60):
        # square canvas: the rule is applied in two orientations, so the grid
        # has to be rotatable in place
        h = unifint(diff_lb, diff_ub, (lob, hub))
        w = h
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        noccs = unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 10)))
        free = {(i, j) for i in range(h) for j in range(w)}
        boforb, reforb = set(), set()
        cnt = {"bo": 0, "ro1": 0, "ro2": 0}

        for _k in range(noccs):
            kind = "bo" if random.random() < 0.5 else random.choice(["ro1", "ro2"])
            obj = shapes[kind]
            oh = max(r for r, c in obj) + 1
            ow = max(c for r, c in obj) + 1
            forb = boforb if kind == "bo" else reforb
            cands = [
                (i, j)
                for (i, j) in sorted(free)
                if i <= h - oh and j <= w - ow
                and (i, j) not in forb
                and all((i + r, j + c) in free for r, c in obj)
            ]
            if not cands:
                break
            loc = random.choice(cands)
            li, lj = loc
            if kind == "bo":
                for dr, dc in ((-2, 0), (2, 0), (0, 2), (0, -2)):
                    boforb.add((li + dr, lj + dc))
            elif kind == "ro1":
                reforb.add((li, lj + 3))
                reforb.add((li, lj - 3))
            else:
                reforb.add((li + 1, lj))
                reforb.add((li - 1, lj))
            col = 8 if kind == "bo" else 2
            for r, c in obj:
                gi[li + r][lj + c] = fgc
                go[li + r][lj + c] = col
                free.discard((li + r, lj + c))
            cnt[kind] += 1

        last = {
            "input": tuple(tuple(row) for row in gi),
            "output": tuple(tuple(row) for row in go),
        }
        if cnt["bo"] and cnt["ro1"] and cnt["ro2"]:
            return last
    return last


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    cells8 = {(r, c) for r in range(h) for c in range(w) if O[r, c] == 8}
    cells2 = {(r, c) for r in range(h) for c in range(w) if O[r, c] == 2}

    # ---- split the 8-marked area into the 2x2 squares it is made of --------
    blocks = []
    stray8 = []
    left8 = set(cells8)
    for (r, c) in sorted(cells8):
        if (r, c) not in left8:
            continue
        blk = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
        if all(t in left8 for t in blk):
            blocks.append(blk)
            left8 -= set(blk)
        else:
            stray8.append((r, c))
            left8.discard((r, c))

    # ---- split the 2-marked area into the 3-cell lines it is made of ------
    def components(cells):
        seen = set()
        comps = []
        for start in sorted(cells):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in cells and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(sorted(comp))
        return comps

    def cover(comp):
        cs = set(comp)
        order = sorted(comp)
        used = set()
        found = []

        def rec():
            rem = None
            for cell in order:
                if cell not in used:
                    rem = cell
                    break
            if rem is None:
                return True
            r, c = rem
            for trip in ([(r, c), (r, c + 1), (r, c + 2)],
                         [(r, c), (r + 1, c), (r + 2, c)]):
                if all(t in cs and t not in used for t in trip):
                    for t in trip:
                        used.add(t)
                    found.append(trip)
                    if rec():
                        return True
                    found.pop()
                    for t in trip:
                        used.discard(t)
            return False

        return list(found) if rec() else None

    horiz_lines = []      # 3-lines that read horizontally in the input frame
    vert_lines = []       # 3-lines that read vertically  in the input frame
    leftover2 = []        # unexpected shapes: painted as-is, in the input frame
    for comp in components(cells2):
        tiling = cover(comp)
        if tiling is None:
            leftover2.append(comp)
            continue
        for trip in tiling:
            if trip[0][0] == trip[1][0]:
                horiz_lines.append(trip)
            else:
                vert_lines.append(trip)
    horiz_lines.sort()
    vert_lines.sort()

    ops, sels = [], []

    # 1. every 2x2 square becomes 8 (orientation independent)
    for blk in blocks:
        ops.append(8)
        sels.append(sel_of(blk))
    if stray8:
        ops.append(8)
        sels.append(sel_of(stray8))

    # 2. every line that lies across the grid becomes 2
    for trip in horiz_lines:
        ops.append(2)
        sels.append(sel_of(trip))
    for comp in leftover2:
        ops.append(2)
        sels.append(sel_of(comp))

    # 3. the same rule for the lines lying along the grid: turn the whole grid
    #    a quarter turn so those lines read across it, mark them, turn it back.
    if vert_lines:
        if h == w:
            full = [0, 0, h - 1, w - 1]     # the entire grid, on purpose
            ops.append(25)                  # Rotate CW: upright lines now lie across
            sels.append(full)
            for trip in vert_lines:
                rot = [(c, h - 1 - r) for (r, c) in trip]
                ops.append(2)
                sels.append(sel_of(rot))
            ops.append(24)                  # Rotate CCW: back to the reading frame
            sels.append(full)
        else:
            for trip in vert_lines:
                ops.append(2)
                sels.append(sel_of(trip))

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
                        f"num_examples+1 ({num_examples + 1}) for task 150deff5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 150deff5"
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
                                f"for task 150deff5"
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
                    f"Failed to build a complete episode for task 150deff5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"150deff5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
