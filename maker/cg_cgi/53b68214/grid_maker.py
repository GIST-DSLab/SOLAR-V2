"""
ARC Task: 53b68214 (RE-ARC) — LLM-generated grid_maker
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

H_OUT = 10


def _period(I):
    from collections import Counter
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = {(r, c): int(I[r, c]) for r in range(h) for c in range(w) if I[r, c] != bgc}
    # period of the repetition: the offset (dr, dc) under which the drawing
    # overlaps itself most, with every overlapping cell agreeing in colour
    best = None
    for dr in range(1, 6):
        for dc in range(-10, 10):
            n_idx = n_obj = 0
            for (r, c), col in fg.items():
                q = (r + dr, c + dc)
                if q in fg:
                    n_idx += 1
                    if fg[q] == col:
                        n_obj += 1
            if n_obj > 0 and n_obj == n_idx:
                key = (n_obj, dr * dc)
                if best is None or key > best[0]:
                    best = (key, dr, dc)
    if best is None:
        return bgc, fg, None, None
    return bgc, fg, best[1], best[2]


def _new_rows(fg, dr, dc, h, w):
    # each new row r continues the row one period above it, slid sideways by
    # the period's shift: input rows carried down whole periods
    rows = []
    for r in range(h, H_OUT):
        row = {}
        for (r0, c0), col in fg.items():
            if (r - r0) % dr:
                continue
            k = (r - r0) // dr
            if k < 1:
                continue
            c = c0 + k * dc
            if 0 <= c < w:
                row[(r, c)] = col
        rows.append(row)
    return rows


def sample_colors(num_examples=None) -> dict:
    bgc = random.randint(0, 9)
    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    return {"bgc": bgc, "instance_plan": ex + [dict(random.choice(ex))]}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, mirror=None, **kw) -> dict:
    def unifint(lb, ub):
        if ub <= lb:
            return lb
        lo = lb + int(round(diff_lb * (ub - lb)))
        hi = lb + int(round(diff_ub * (ub - lb)))
        return random.randint(min(lo, hi), max(lo, hi))
    if mirror is None:
        mirror = random.choice((True, False))
    remcols = [c for c in range(10) if c != bgc]
    while True:
        h = unifint(2, 6)
        w = unifint(8, max(8, min(30, max_w)))
        ncols = unifint(1, 9)
        ccols = random.sample(remcols, ncols)
        oh = unifint(1, h // 2)
        ow = unifint(1, w // 2 - 1)
        bounds = [(i, j) for i in range(oh) for j in range(ow)]
        ncells = unifint(1, oh * ow)
        cells = random.sample(bounds, ncells)
        mi = min(i for i, _ in cells)
        mj = min(j for _, j in cells)
        obj = [(random.choice(ccols), (i - mi, j - mj)) for i, j in cells]
        oh = max(i for _, (i, _) in obj) + 1
        ow = max(j for _, (_, j) in obj) + 1
        locj = random.randint(0, w // 2)
        go = [[bgc] * w for _ in range(H_OUT)]
        hoffs = random.randint(0, ow // 2 + 1)
        for k in range(H_OUT // oh + 1):
            for col, (i, j) in obj:
                r, c = i + k * oh, j + locj + k * hoffs
                if 0 <= r < H_OUT and 0 <= c < w:
                    go[r][c] = col
        if len({v for row in go[h:] for v in row}) <= 1:
            continue
        if mirror:
            go = [row[::-1] for row in go]
        # keep only instances whose input pins the repetition down
        _, fg, dr, dc = _period(go[:h])
        if dr is None:
            continue
        pred = [list(r) for r in go[:h]] + [[bgc] * w for _ in range(H_OUT - h)]
        for row in _new_rows(fg, dr, dc, h, w):
            for (r, c), col in row.items():
                pred[r][c] = col
        if pred == go:
            break
    gi = go[:h]
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc, fg, dr, dc = _period(I)

    ops, sels = [], []
    # 1. grow the canvas to the 10 rows the repetition fills (full rectangle)
    ops.append(33); sels.append([0, 0, H_OUT - 1, w - 1])
    # 2. lay the background over the newly added rows (full rectangle)
    if bgc != 0 and h < H_OUT:
        ops.append(int(bgc)); sels.append([h, 0, H_OUT - 1 - h, w - 1])
    # 3. extend row by row, top to bottom: each new row is the row one period
    #    above it slid sideways by the shift; drawn colour by colour
    for row in _new_rows(fg, dr, dc, h, w):
        bycol = {}
        for rc, col in row.items():
            bycol.setdefault(col, []).append(rc)
        for col in sorted(bycol, key=lambda cc: min(bycol[cc])):
            ops.append(int(col)); sels.append(sel_of(sorted(bycol[col])))
    ops.append(34); sels.append([0, 0, H_OUT - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 53b68214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 53b68214"
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
                                f"for task 53b68214"
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
                    f"Failed to build a complete episode for task 53b68214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"53b68214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
