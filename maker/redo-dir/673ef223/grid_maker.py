"""
ARC Task: 673ef223 (RE-ARC) — LLM-generated grid_maker
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

# Discrete structural variants: the generator applies mirror/transpose transforms to the
# whole scene, which decides the bar orientation (vertical/horizontal) and therefore the
# direction in which every dot gravitates toward its bar (left/right/up/down).
VARIANTS = [
    {"transforms": []},                       # bars vertical, dot-bar on the left  -> dots gravitate left
    {"transforms": ["vmirror"]},              # bars vertical, dot-bar on the right -> dots gravitate right
    {"transforms": ["dmirror"]},              # bars horizontal, dot-bar on top     -> dots gravitate up
    {"transforms": ["dmirror", "hmirror"]},   # bars horizontal, dot-bar at bottom  -> dots gravitate down
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]          # 4 is reserved as the marker color
    bgc = random.choice(cols)
    barc = random.choice([c for c in cols if c != bgc])
    # dot color must be non-zero: the dots are physically MOVED (ARCLE object mode only
    # grabs non-zero cells), so a 0-colored dot could not be gravitated.
    dotc = random.choice([c for c in cols if c not in (bgc, barc, 0)])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [{"transforms": list(v["transforms"])} for v in VARIANTS]
        examples += [{"transforms": list(random.choice(VARIANTS)["transforms"])}
                     for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [{"transforms": list(v["transforms"])}
                    for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [{"transforms": list(random.choice(examples)["transforms"])}]
    return {"bgc": bgc, "barc": barc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, dotc, transforms=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if transforms is None:
        transforms = list(random.choice(VARIANTS)["transforms"])
    transforms = list(transforms)

    # a transpose swaps the final dimensions -> sample within swapped bounds
    swap = (sum(1 for t in transforms if t == "dmirror") % 2) == 1
    hb = max(5, min(30, max_w if swap else max_h))
    wb = max(5, min(30, max_h if swap else max_w))

    h = unifint(diff_lb, diff_ub, (5, hb))
    w = unifint(diff_lb, diff_ub, (5, wb))
    barh = unifint(diff_lb, diff_ub, (2, (h - 1) // 2))
    ncells = unifint(diff_lb, diff_ub, (1, barh))

    top_i = [[bgc] * w for _ in range(barh)]
    for k in range(barh):
        top_i[k][0] = barc
    bot_i = [row[::-1] for row in top_i]          # vmirror -> bar on the right edge
    top_o = [row[:] for row in top_i]
    bot_o = [row[:] for row in bot_i]

    ilocs = random.sample(range(barh), ncells)
    for k in ilocs:
        jloc = random.randint(2, w - 2)
        top_i[k][jloc] = dotc
        top_o[k][jloc] = 4
        for j in range(1, jloc):
            top_o[k][j] = dotc
        for j in range(0, w - 1):
            bot_o[k][j] = dotc

    blocks_i = [top_i, bot_i]
    blocks_o = [top_o, bot_o]
    total = 2 * barh
    while total < h:
        idx = random.randint(0, len(blocks_i) - 1)
        blocks_i = blocks_i[:idx] + [[[bgc] * w]] + blocks_i[idx:]
        blocks_o = blocks_o[:idx] + [[[bgc] * w]] + blocks_o[idx:]
        total += 1

    gi = np.array([row for blk in blocks_i for row in blk], dtype=int)
    go = np.array([row for blk in blocks_o for row in blk], dtype=int)

    for t in transforms:
        if t == "dmirror":
            gi = gi.T.copy()
            go = go.T.copy()
        elif t == "vmirror":
            gi = gi[:, ::-1].copy()
            go = go[:, ::-1].copy()
        elif t == "hmirror":
            gi = gi[::-1, :].copy()
            go = go[::-1, :].copy()

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # ---- read the scene from I -------------------------------------------------
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def components(cells):
        cells = set(cells)
        seen, comps = set(), []
        for start in sorted(cells):
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (r + dr, c + dc)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
            comps.append(sorted(comp))
        return comps

    others = sorted({int(v) for v in I.flatten().tolist()} - {bgc})
    info = {}
    for col in others:
        cells = {(r, c) for r in range(h) for c in range(w) if I[r, c] == col}
        info[col] = (components(cells), cells)

    # the bar color: exactly two components, all of its cells sit on the grid border,
    # and every component is longer than one cell (the dots never satisfy this)
    barc = None
    for col in others:
        comps, cells = info[col]
        on_border = all(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
        if len(comps) == 2 and on_border and min(len(x) for x in comps) > 1:
            barc = col
            break
    if barc is None:
        barc = max(others, key=lambda col: len(info[col][1]))
    dotc = [col for col in others if col != barc][0]

    bars = info[barc][0]
    bar_cells = info[barc][1]
    dots = sorted(info[dotc][1])

    vertical = len({c for _, c in bars[0]}) == 1   # bar is a vertical segment

    def matches(bar):
        if vertical:
            rows = {r for r, _ in bar}
            return any(r in rows for r, _ in dots)
        cols = {c for _, c in bar}
        return any(c in cols for _, c in dots)

    dot_bar = bars[0] if matches(bars[0]) else bars[1]
    other_bar = bars[1] if dot_bar is bars[0] else bars[0]

    # ---- 1. gravitate: every dot slides until it touches its bar ---------------
    for (r, c) in dots:
        if vertical:
            cb = dot_bar[0][1]
            if c > cb:
                mv, step = 23, (0, -1)          # MoveL
                dest = (r, cb + 1)
            else:
                mv, step = 22, (0, 1)           # MoveR
                dest = (r, cb - 1)
            steps = abs(c - cb) - 1
        else:
            rb = dot_bar[0][0]
            if r > rb:
                mv, step = 20, (-1, 0)          # MoveU
                dest = (rb + 1, c)
            else:
                mv, step = 21, (1, 0)           # MoveD
                dest = (rb - 1, c)
            steps = abs(r - rb) - 1

        # first Move grabs the dot, the rest continue the same slide (empty selection)
        ops.append(mv)
        sels.append(sel_of([(r, c)]))
        for _ in range(steps - 1):
            ops.append(mv)
            sels.append(sel_of([]))

        # the track the dot ploughed: cells strictly between start and landing place
        trail = []
        rr, cc = r + step[0], c + step[1]
        while (rr, cc) != dest:
            trail.append((rr, cc))
            rr += step[0]
            cc += step[1]
        if trail:
            ops.append(int(dotc))
            sels.append(sel_of(trail))

        # the vacated starting cell is marked with 4
        ops.append(4)
        sels.append(sel_of([(r, c)]))

    # ---- 2. the mirrored block: full lines through the shifted dot positions ----
    ur_d = (min(r for r, _ in dot_bar), min(c for _, c in dot_bar))
    ur_o = (min(r for r, _ in other_bar), min(c for _, c in other_bar))
    if vertical:
        dr = ur_o[0] - ur_d[0]
        for (r, c) in dots:
            row = r + dr
            line = [(row, cc) for cc in range(w) if (row, cc) not in bar_cells]
            if line:
                ops.append(int(dotc))
                sels.append(sel_of(line))
    else:
        dc = ur_o[1] - ur_d[1]
        for (r, c) in dots:
            col = c + dc
            line = [(rr, col) for rr in range(h) if (rr, col) not in bar_cells]
            if line:
                ops.append(int(dotc))
                sels.append(sel_of(line))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox == whole grid rectangle (submit)
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
                        f"num_examples+1 ({num_examples + 1}) for task 673ef223"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 673ef223"
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
                                f"for task 673ef223"
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
                    f"Failed to build a complete episode for task 673ef223 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"673ef223-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
