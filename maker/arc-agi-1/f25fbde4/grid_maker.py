"""
ARC Task: f25fbde4 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


# ----------------------------------------------------------------------------
# 1. colors -- the generator samples bgc and fgc randomly; fix both per episode
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(lo, min(b, hi))
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, **kwargs) -> dict:
    mh = max(4, min(30, int(max_h)))
    mw = max(4, min(30, int(max_w)))

    h = w = bh = bw = None
    for _ in range(500):
        hh = _unifint(diff_lb, diff_ub, (2, mh))
        ww = _unifint(diff_lb, diff_ub, (2, mw))
        # shape bbox is capped at 15 (generator) and at half the canvas limit
        # so that the 2x upscaled output still fits inside max_h x max_w
        bbh = min(15, hh - 1, max(1, mh // 2))
        bbw = min(15, ww - 1, max(1, mw // 2))
        if bbh >= 1 and bbw >= 1 and bbh * bbw >= 2:
            h, w, bh, bw = hh, ww, bbh, bbw
            break
    if h is None:
        h, w = min(mh, 6), min(mw, 6)
        bh = min(15, h - 1, max(1, mh // 2))
        bw = min(15, w - 1, max(1, mw // 2))

    ncd = _unifint(diff_lb, diff_ub, (1, max(1, (bh * bw) // 2)))
    nc = min(max(1, ncd), bh * bw - 1)

    # 8-connected random polyomino grown inside the bh x bw bounds
    cells = {(random.randrange(bh), random.randrange(bw))}
    for _ in range(nc):
        cand = set()
        for (r, c) in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < bh and 0 <= cc < bw and (rr, cc) not in cells:
                        cand.add((rr, cc))
        if not cand:
            break
        cells.add(random.choice(sorted(cand)))

    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    cells = {(r - mr, c - mc) for (r, c) in cells}
    oh = max(r for r, _ in cells) + 1
    ow = max(c for _, c in cells) + 1

    li = random.randint(0, h - oh)
    lj = random.randint(0, w - ow)
    if li == 0 and lj == 0:
        # keep the shape off the exact top-left corner so that the crop step
        # of the trajectory is always a real, non-removable operation
        if h - oh >= 1:
            li = random.randint(1, h - oh)
        elif w - ow >= 1:
            lj = random.randint(1, w - ow)

    gi = [[bgc] * w for _ in range(h)]
    for (r, c) in cells:
        gi[r + li][c + lj] = fgc

    go = []
    for r in range(oh):
        row = []
        for c in range(ow):
            v = gi[r + li][c + lj]
            row.append(v)
            row.append(v)
        go.append(list(row))
        go.append(list(row))

    return {"input": gi, "output": go}


# ----------------------------------------------------------------------------
# 3. trajectory: crop to the shape's bounding box, expand the canvas, then
#    duplicate every column and every row (2x upscale).  Everything below is
#    measured from I only; O is never read.
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = the colour the canvas was painted with (strict majority here)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    rows = [r for r in range(hi) if any(I[r, c] != bgc for c in range(wi))]
    cols = [c for c in range(wi) if any(I[r, c] != bgc for r in range(hi))]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    ops, sels = [], []

    # (1) crop the canvas down to the shape's bounding box (full rectangle)
    ops.append(33); sels.append([r0, c0, h - 1, w - 1])
    # (2) expand the canvas to twice that size (full rectangle)
    ops.append(33); sels.append([0, 0, 2 * h - 1, 2 * w - 1])

    G = np.zeros((2 * h, 2 * w), dtype=int)
    G[:h, :w] = I[r0:r1 + 1, c0:c1 + 1]
    clip = None

    # (3) horizontal doubling: column j of the cropped shape -> columns 2j, 2j+1
    for j in range(w - 1, -1, -1):
        src = [int(v) for v in G[:h, j]]
        for t in (2 * j, 2 * j + 1):
            if all(int(G[i, t]) == src[i] for i in range(h)):
                continue
            clear = [(i, t) for i in range(h) if src[i] == 0 and int(G[i, t]) != 0]
            if clear:
                # Paste never writes colour 0 -- clear those cells explicitly
                ops.append(0); sels.append(sel_of(clear))
                for (a, b) in clear:
                    G[a, b] = 0
            if any(src[i] != 0 and int(G[i, t]) != src[i] for i in range(h)):
                key = ("col", tuple(src))
                if clip != key:
                    ops.append(29); sels.append([0, j, h - 1, 0])
                    clip = key
                ops.append(30); sels.append([0, t, 0, 0])
                for i in range(h):
                    if src[i] != 0:
                        G[i, t] = src[i]

    # (4) vertical doubling: row i -> rows 2i, 2i+1
    for i in range(h - 1, -1, -1):
        src = [int(v) for v in G[i, :2 * w]]
        for t in (2 * i, 2 * i + 1):
            if all(int(G[t, c]) == src[c] for c in range(2 * w)):
                continue
            clear = [(t, c) for c in range(2 * w) if src[c] == 0 and int(G[t, c]) != 0]
            if clear:
                ops.append(0); sels.append(sel_of(clear))
                for (a, b) in clear:
                    G[a, b] = 0
            if any(src[c] != 0 and int(G[t, c]) != src[c] for c in range(2 * w)):
                key = ("row", tuple(src))
                if clip != key:
                    ops.append(29); sels.append([i, 0, 0, 2 * w - 1])
                    clip = key
                ops.append(30); sels.append([t, 0, 0, 0])
                for c in range(2 * w):
                    if src[c] != 0:
                        G[t, c] = src[c]

    ops.append(34); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f25fbde4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f25fbde4"
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
                                f"for task f25fbde4"
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
                    f"Failed to build a complete episode for task f25fbde4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f25fbde4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
