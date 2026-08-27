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


def sample_colors(num_examples=None) -> dict:
    # generator: cols = interval(0,10,1) with 2 removed (2 is the reserved fill color)
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    # The rule keys only on the 4-corner frame pattern, never on which color a
    # frame uses, so only the background needs to be fixed for the episode.
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    def rule(gi, bg):
        # Exact simulation of this task's verifier: sweep every frame size in
        # 3..6 x 3..6 in ascending-area order (the DSL's product/order); for each
        # size, find every window whose 4 corners hold one single input color and
        # whose remaining cells are ALL background, then flood those windows'
        # inner rectangles with 2.  Fills from a smaller size can invalidate a
        # larger frame later, so the sweep must stay sequential.
        g = [list(r) for r in gi]
        h, w = len(g), len(g[0])
        fg = set(v for row in gi for v in row if v != bg)
        sizes = tuple(sorted(frozenset((i, j) for j in (3, 4, 5, 6) for i in (3, 4, 5, 6)),
                             key=lambda t: t[0] * t[1]))
        rects = []
        for dh, dw in sizes:
            if dh > h or dw > w:
                continue
            found = []
            for i in range(h - dh + 1):
                for j in range(w - dw + 1):
                    c = g[i][j]
                    if c == bg or c not in fg:
                        continue
                    if g[i][j + dw - 1] != c or g[i + dh - 1][j] != c or g[i + dh - 1][j + dw - 1] != c:
                        continue
                    ok = True
                    for r in range(i, i + dh):
                        for cc in range(j, j + dw):
                            if (r == i or r == i + dh - 1) and (cc == j or cc == j + dw - 1):
                                continue
                            if g[r][cc] != bg:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        found.append((i, j, dh, dw))
            for (i, j, dh_, dw_) in found:
                for r in range(i + 1, i + dh_ - 1):
                    for cc in range(j + 1, j + dw_ - 1):
                        g[r][cc] = 2
            rects.extend(found)
        return g, rects

    cols = [c for c in range(10) if c != 2]
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numcols)
    numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    gi = [[bgc] * w for _ in range(h)]
    inds = set((i, j) for i in range(h) for j in range(w))
    succ, tr, maxtr = 0, 0, 5 * numsq
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = random.randint(3, 5)
        ow = random.randint(3, 5)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if len(cands) == 0:
            continue
        loci, locj = random.choice(sorted(cands))
        sq = set((i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow))
        if sq <= inds:
            inds -= sq
            succ += 1
            col = random.choice(ccols)
            for (r, c) in ((loci, locj), (loci, locj + ow - 1),
                           (loci + oh - 1, locj), (loci + oh - 1, locj + ow - 1)):
                gi[r][c] = col
    # the output is the verifier's rule applied to the sampled input (this also
    # covers frames formed incidentally by corners of two different placements)
    go, _ = rule(gi, bgc)
    return {"input": gi, "output": go}


def derive_operations(I, O):
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    def rule(gi, bg):
        # same sweep as in generate(): detect corner-frames size by size
        # (ascending area) on the grid as it stands, filling interiors with 2.
        g = [list(r) for r in gi]
        h, w = len(g), len(g[0])
        fg = set(v for row in gi for v in row if v != bg)
        sizes = tuple(sorted(frozenset((i, j) for j in (3, 4, 5, 6) for i in (3, 4, 5, 6)),
                             key=lambda t: t[0] * t[1]))
        rects = []
        for dh, dw in sizes:
            if dh > h or dw > w:
                continue
            found = []
            for i in range(h - dh + 1):
                for j in range(w - dw + 1):
                    c = g[i][j]
                    if c == bg or c not in fg:
                        continue
                    if g[i][j + dw - 1] != c or g[i + dh - 1][j] != c or g[i + dh - 1][j + dw - 1] != c:
                        continue
                    ok = True
                    for r in range(i, i + dh):
                        for cc in range(j, j + dw):
                            if (r == i or r == i + dh - 1) and (cc == j or cc == j + dw - 1):
                                continue
                            if g[r][cc] != bg:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        found.append((i, j, dh, dw))
            for (i, j, dh_, dw_) in found:
                for r in range(i + 1, i + dh_ - 1):
                    for cc in range(j + 1, j + dw_ - 1):
                        g[r][cc] = 2
            rects.extend(found)
        return g, rects

    I = [[int(v) for v in row] for row in I]
    O = [[int(v) for v in row] for row in O]
    ho, wo = len(O), len(O[0])
    # background = the canvas color the generator paints before dropping corners
    bgc = Counter(v for row in I for v in row).most_common(1)[0][0]
    _, rects = rule(I, bgc)

    ops, sels = [], []
    # one Color2 per detected frame: fill that frame's whole interior region.
    # Detection order is the rule's own order (small frames first), and a frame's
    # interior is background at the moment it is filled, so every op paints new cells.
    for (i, j, dh, dw) in rects:
        cells = [(r, c) for r in range(i + 1, i + dh - 1) for c in range(j + 1, j + dw - 1)]
        ops.append(2)
        sels.append(sel_of(cells))
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])  # bbox = whole grid, submit
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
