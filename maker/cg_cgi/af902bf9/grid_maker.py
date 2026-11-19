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
    rem = [c for c in cols if c != bgc]
    ccols = random.sample(rem, random.randint(1, len(rem)))
    return {"bgc": bgc, "ccols": ccols}


def _find_frames(g, bgc):
    # The task rule (same as the verifier): frame sizes 3..6 x 3..6 in ascending area;
    # a frame = 4 same-coloured corners with only background elsewhere in its box.
    # Each size's interiors are set to 2 before larger sizes are checked.
    h, w = len(g), len(g[0])
    sizes = sorted({(a, b) for a in (3, 4, 5, 6) for b in (3, 4, 5, 6)}, key=lambda t: t[0] * t[1])
    work = [row[:] for row in g]
    rects = []
    for dh, dw in sizes:
        if dh > h or dw > w:
            continue
        found = []
        for i in range(h - dh + 1):
            for j in range(w - dw + 1):
                c = work[i][j]
                if c == bgc or c == 2:
                    continue
                if work[i][j+dw-1] != c or work[i+dh-1][j] != c or work[i+dh-1][j+dw-1] != c:
                    continue
                if all(work[r][cc] == bgc
                       for r in range(i, i + dh) for cc in range(j, j + dw)
                       if not ((r in (i, i+dh-1)) and (cc in (j, j+dw-1)))):
                    found.append((i, j, dh, dw))
        for (i, j, dh_, dw_) in found:
            for r in range(i + 1, i + dh_ - 1):
                for cc in range(j + 1, j + dw_ - 1):
                    work[r][cc] = 2
        rects.extend(found)
    return work, rects


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ccols=(1,)):
    def unifint(lb, ub):
        d = random.uniform(diff_lb, diff_ub)
        return min(ub, max(lb, int(round(lb + d * (ub - lb)))))
    h = unifint(10, max(10, max_h))
    w = unifint(10, max(10, max_w))
    numsq = unifint(1, (h * w) // 20)
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = tr = 0
    while tr < 5 * numsq and succ < numsq:
        tr += 1
        oh, ow = random.randint(3, 5), random.randint(3, 5)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        sq = {(i, j) for i in range(li, li + oh) for j in range(lj, lj + ow)}
        if sq <= inds:
            inds -= sq
            succ += 1
            col = random.choice(list(ccols))
            for (i, j) in [(li, lj), (li, lj + ow - 1), (li + oh - 1, lj), (li + oh - 1, lj + ow - 1)]:
                gi[i][j] = col
    # Corners of different frames can line up into an extra empty frame; the task's
    # rule (verifier) fills those too, so the output is the rule applied to gi.
    go, _ = _find_frames(gi, bgc)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Same rule as the maker and the golf entry: a cell turns 2 exactly when it lies
    # strictly between a matching pair of corner markers along its ROW and also along
    # its COLUMN (row span ∩ column span of one 4-corner frame). Which cells get filled
    # is the same; only the unit of work differs. The golf entry works in LINES: it
    # sweeps row by row and fills each horizontal span that the column check allows.
    # We do the same: frames are found from I alone (O is never read), then the 2s are
    # laid down top to bottom, one Color2 per horizontal interior span
    # (row r, between that frame's two corner columns).
    g = [[int(v) for v in row] for row in I]
    h, w = len(g), len(g[0])
    # background = the canvas colour the generator paints before dropping sparse corners
    bgc = Counter(v for row in g for v in row).most_common(1)[0][0]
    _, rects = _find_frames(g, bgc)
    spans = []
    for (i, j, dh, dw) in rects:
        for r in range(i + 1, i + dh - 1):
            spans.append((r, j + 1, j + dw - 2))
    spans.sort()  # row sweep, top to bottom, left to right within a row
    ops, sels = [], []
    for (r, c1, c2) in spans:
        ops.append(2)
        sels.append(sel_of([(r, c) for c in range(c1, c2 + 1)]))
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # whole grid, submit
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
