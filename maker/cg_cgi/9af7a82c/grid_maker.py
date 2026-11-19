"""
ARC Task: 9af7a82c (RE-ARC) — LLM-generated grid_maker
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


# Comparison with the code-golf reading: the two readings agree, so the maker's route is kept.
# The golf entry flattens I into one bag of cells. For each colour it builds a
# column [-count] + [colour]*count + zeros, sorts those columns by count
# (largest first), transposes them, drops all-zero rows, and strips the
# helper rows. That is the same split as the maker:
#   unit     = one colour's cell population, shown as one top-aligned column bar
#   order    = count descending, left to right
#   fixed    = top alignment; height = largest count; width = number of colours.
# The generator keeps every count different, so ties never happen and both
# readings give the same order. The trajectory is: resize the canvas to the
# histogram frame, paint each colour's bar as one column (tallest first), then
# clear the leftover input pixels under each bar.


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    return random.randint(lo_i, hi_i)


def sample_colors(num_examples=None) -> dict:
    # The rule only depends on how many cells each colour has. Colours are
    # sampled for each instance, and the output background is fixed at 0.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    cols = list(range(1, 10))
    cap = min(30, max_h)                      # bar height must fit the output
    prods = dict()
    for a in range(1, max_h + 1):
        for b in range(1, max_w + 1):
            prods.setdefault(a * b, []).append((a, b))
    ncols = _unifint(diff_lb, diff_ub, (2, min(9, cap)))
    leastnc = sum(range(1, ncols + 1))
    maxnc = sum(range(cap, cap - ncols, -1))
    options = set()
    for k, v in prods.items():
        if leastnc <= k <= maxnc:
            for opt in v:
                options.add(opt)
    options = sorted(options, key=lambda ij: (ij[0] * ij[1], ij))
    idx = _unifint(diff_lb, diff_ub, (0, len(options) - 1))
    h, w = options[idx]
    ccols = random.sample(cols, ncols)
    counts = list(range(1, ncols + 1))
    eliginds = {ncols - 1}
    while sum(counts) < h * w:
        el = sorted(eliginds, reverse=True)
        i2 = el[_unifint(diff_lb, diff_ub, (0, len(el) - 1))]
        counts[i2] += 1
        if i2 > 0:
            eliginds.add(i2 - 1)
        if i2 < ncols - 1 and counts[i2] == counts[i2 + 1] - 1:
            eliginds = eliginds - {i2}
        if counts[i2] == cap:
            eliginds = eliginds - {i2}
    gi = [[-1] * w for _ in range(h)]
    go = [[0] * ncols for _ in range(max(counts))]
    inds = [(r, c) for r in range(h) for c in range(w)]
    counts = counts[::-1]
    for j, (col, cnt) in enumerate(zip(ccols, counts)):
        locs = random.sample(inds, cnt)
        ls = set(locs)
        for (r, c) in locs:
            gi[r][c] = col
        inds = [x for x in inds if x not in ls]
        for r in range(cnt):
            go[r][j] = col
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Rule (measured from I only): count the cells of each colour, sort the
    colours by count (largest first), and draw one top-aligned bar per colour,
    left to right, with height equal to its count. The canvas becomes
    (largest count) x (number of colours).
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    counter = Counter(I.flatten().tolist())
    bars = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ho = int(bars[0][1])
    wo = len(bars)

    ops, sels = [], []

    # 1. Resize the canvas to the histogram frame (the full rectangle).
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])

    # 2. Paint each colour's bar as one full column rectangle, tallest first.
    for j, (color, count) in enumerate(bars):
        ops.append(int(color))
        sels.append([0, j, int(count) - 1, 0])   # exact rectangle: rows 0..count-1 of column j

    # 3. Clear the leftover input pixels below each bar (a rectangle of rows).
    hlim = min(hi, ho)
    for j, (color, count) in enumerate(bars):
        if j >= wi:
            continue
        count = int(count)
        if count < hlim and np.any(I[count:hlim, j] != 0):
            ops.append(0)
            sels.append([count, j, hlim - count - 1, 0])  # exact rectangle in column j

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
                        f"num_examples+1 ({num_examples + 1}) for task 9af7a82c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9af7a82c"
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
                                f"for task 9af7a82c"
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
                    f"Failed to build a complete episode for task 9af7a82c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9af7a82c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
