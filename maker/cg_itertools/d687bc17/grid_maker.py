"""
ARC Task: d687bc17 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, c1, c2, c3, c4 = random.sample(range(10), 5)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(max(a, round(a + (b - a) * d)), b)

    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    for j in range(w):
        gi[0][j] = c1
    for i in range(h):
        gi[i][0] = c2
    for i in range(h):
        gi[i][w - 1] = c3
    for j in range(w):
        gi[h - 1][j] = c4
    for (i, j) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        gi[i][j] = bgc
    go = [row[:] for row in gi]
    cands = [(i, j) for i in range(2, h - 2) for j in range(2, w - 2)]
    ndots = unifint(diff_lb, diff_ub, (1, min(len(cands), h + h + w + w)))
    dots = random.sample(cands, ndots)
    dots = [(random.choice((c1, c2, c3, c4)), ij) for ij in dots]

    def place(colr, key_idx, total, dest_fn):
        n = [ij for (cc, ij) in dots if cc == colr]
        cov = sorted(set(ij[key_idx] for ij in n))
        if len(cov) == total - 4 and total > 5:
            cov.remove(random.choice(cov))
        for k in cov:
            loc = random.choice([ij for ij in n if ij[key_idx] == k])
            gi[loc[0]][loc[1]] = colr
            dr, dc = dest_fn(k)
            go[dr][dc] = colr

    place(c1, 1, w, lambda j: (1, j))
    place(c2, 0, h, lambda i: (i, 1))
    place(c4, 1, w, lambda j: (h - 2, j))
    place(c3, 0, h, lambda i: (i, w - 2))

    noisecands = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc]
    noisecols = [c for c in cols if c not in (bgc, c1, c2, c3, c4)]
    ub = ((h * w) - 2 * h - 2 * (w - 2)) // 2 - ndots - 1
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, ub)))
    nnoise = min(nnoise, len(noisecands))
    for (i, j) in random.sample(noisecands, nnoise):
        gi[i][j] = random.choice(noisecols)
    return {"input": tuple(tuple(r) for r in gi), "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # Each side of the frame owns one colour; frame edge cells (minus corners)
    # are never touched by noise.
    top = int(I[0, 1])
    left = int(I[1, 0])
    right = int(I[1, w - 1])
    bottom = int(I[h - 1, 1])

    # Background: the demonstrations' outputs always show it in the four corners.
    bgc = None
    if examples:
        votes = Counter()
        for _, eo in examples:
            eo = np.asarray(eo, dtype=int)
            eh, ew = eo.shape
            for (r, c) in ((0, 0), (0, ew - 1), (eh - 1, 0), (eh - 1, ew - 1)):
                votes[int(eo[r, c])] += 1
        bgc = votes.most_common(1)[0][0]
    if bgc is None:
        bgc = Counter(I[1:h - 1, 1:w - 1].flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []

    # 1. Erase every cell that is neither background, frame, nor a side-coloured dot:
    #    noise colours (anything not owned by a side) and noisy corners.
    side_cols = {top, left, right, bottom}
    junk = [(r, c) for (r, c) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
            if int(I[r, c]) != bgc]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = int(I[r, c])
            if v != bgc and v not in side_cols:
                junk.append((r, c))
    if junk:
        ops.append(bgc)
        sels.append(sel_of(junk))

    # 2. Side by side: each line perpendicular to a side carries (at most) one dot of
    #    that side's colour; the dot slides along its line until it touches the frame.
    def slide(r, c, dr, dc, steps, color):
        if steps <= 0:
            return
        op = 20 if dr < 0 else 21 if dr > 0 else 22 if dc > 0 else 23
        if color == 0:
            # ARCLE's object buffer ignores 0 cells -> a 0 dot cannot be grabbed.
            ops.append(0)
            sels.append(sel_of([(r + dr * steps, c + dc * steps)]))
        else:
            ops.append(op)
            sels.append(sel_of([(r, c)]))         # grab the dot
            for _ in range(steps - 1):
                ops.append(op)
                sels.append(sel_of([]))           # keep sliding the same dot
        if bgc != 0:
            ops.append(bgc)                       # repair the grabbed footprint
            sels.append(sel_of([(r, c)]))

    # top side: in each column, its top-coloured dot goes up to row 1
    for c in range(1, w - 1):
        for r in range(1, h - 1):
            if int(I[r, c]) == top:
                slide(r, c, -1, 0, r - 1, top)
    # left side: in each row, its left-coloured dot goes left to column 1
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(I[r, c]) == left:
                slide(r, c, 0, -1, c - 1, left)
    # bottom side: in each column, its bottom-coloured dot goes down to row h-2
    for c in range(1, w - 1):
        for r in range(h - 2, 0, -1):
            if int(I[r, c]) == bottom:
                slide(r, c, 1, 0, (h - 2) - r, bottom)
    # right side: in each row, its right-coloured dot goes right to column w-2
    for r in range(1, h - 1):
        for c in range(w - 2, 0, -1):
            if int(I[r, c]) == right:
                slide(r, c, 0, 1, (w - 2) - c, right)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # bbox = exactly the whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task d687bc17"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d687bc17"
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
                                f"for task d687bc17"
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
                    f"Failed to build a complete episode for task d687bc17 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d687bc17-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
