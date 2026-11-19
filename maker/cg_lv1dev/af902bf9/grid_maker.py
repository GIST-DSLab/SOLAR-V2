"""
ARC Task: af902bf9 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, 8)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, ccols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    cols = [c for c in range(10) if c != 2]
    if bgc is None:
        bgc = random.choice(cols)
    if not ccols:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, unifint(diff_lb, diff_ub, (1, 8)))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = 0
    tr = 0
    maxtr = 5 * numsq
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = random.randint(3, 5)
        ow = random.randint(3, 5)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        sq = {(r, c) for r in range(loci, loci + oh) for c in range(locj, locj + ow)}
        if sq.issubset(inds):
            inds -= sq
            succ += 1
            col = random.choice(ccols)
            for r, c in [(loci, locj), (loci, locj + ow - 1),
                         (loci + oh - 1, locj), (loci + oh - 1, locj + ow - 1)]:
                gi[r][c] = col
                go[r][c] = col
            for r in range(loci + 1, loci + oh - 1):
                for c in range(locj + 1, locj + ow - 1):
                    go[r][c] = 2
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = [[int(v) for v in row] for row in I]
    h, w = len(I), len(I[0])
    # Background = the canvas colour the generator paints first (frames are <=20% of cells).
    bgc = Counter(v for row in I for v in row).most_common(1)[0][0]

    # Split the marks into frames: each frame is 4 same-coloured corners of a 3..5 x 3..5
    # box, and boxes never overlap. Each mark belongs to exactly one frame, so this is an
    # exact cover. The first uncovered mark in raster order is always a frame's top-left
    # corner. This partition throws out "frames" whose corners belong to two real frames.
    fg = sorted((r, c) for r in range(h) for c in range(w) if I[r][c] != bgc)
    used, covered, frames = set(), set(), []

    def rec():
        nxt = next((p for p in fg if p not in covered), None)
        if nxt is None:
            return True
        i, j = nxt
        col = I[i][j]
        for dh in (3, 4, 5):
            for dw in (3, 4, 5):
                if i + dh > h or j + dw > w:
                    continue
                crn = [(i, j), (i, j + dw - 1), (i + dh - 1, j), (i + dh - 1, j + dw - 1)]
                if any(I[r][c] != col or (r, c) in covered for r, c in crn):
                    continue
                box = {(r, c) for r in range(i, i + dh) for c in range(j, j + dw)}
                if box & used:
                    continue
                if any(I[r][c] != bgc for r, c in box - set(crn)):
                    continue
                used.update(box); covered.update(crn); frames.append((i, j, dh, dw))
                if rec():
                    return True
                used.difference_update(box); covered.difference_update(crn); frames.pop()
        return False

    rec()

    ops, sels = [], []
    # Row-wise reading (as in the code-golf entry): fill each frame one interior row at a
    # time. Each row is the segment strictly between the two corner columns. Finish one
    # frame before moving to the next.
    for (i, j, dh, dw) in frames:
        for r in range(i + 1, i + dh - 1):
            seg = [(r, c) for c in range(j + 1, j + dw - 1)]
            ops.append(2)
            sels.append(sel_of(seg))
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # submit: whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task af902bf9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task af902bf9"
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
                                f"for task af902bf9"
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
                    f"Failed to build a complete episode for task af902bf9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"af902bf9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
