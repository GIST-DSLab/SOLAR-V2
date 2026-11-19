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
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # Input background is not a separate colour (every input cell holds a colour from 1-9).
    # The output background is hardcoded to 0, and the rule depends only on colour counts,
    # not on which colours appear. So there is no colour role to fix for the episode.
    return {}


def _unif(lb, ub, rng):
    a, b = rng
    lo = a + round(lb * (b - a))
    hi = a + round(ub * (b - a))
    lo, hi = min(lo, hi), max(lo, hi)
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    cap = min(30, max_h)                      # no bar may be taller than the output canvas
    cols = list(range(1, 10))
    ncols = _unif(diff_lb, diff_ub, (2, min(9, max_w)))
    leastnc = sum(range(1, ncols + 1))
    maxnc = sum(range(cap, cap - ncols, -1))
    options = sorted({(a, b) for a in range(1, max_h + 1) for b in range(1, max_w + 1)
                      if leastnc <= a * b <= maxnc},
                     key=lambda ij: (ij[0] * ij[1], ij))
    idx = _unif(diff_lb, diff_ub, (0, len(options) - 1))
    h, w = options[idx]
    ccols = random.sample(cols, ncols)
    counts = list(range(1, ncols + 1))
    elig = {ncols - 1}
    while sum(counts) < h * w:
        el = sorted(elig, reverse=True)
        i = el[_unif(diff_lb, diff_ub, (0, len(el) - 1))]
        counts[i] += 1
        if i > 0:
            elig.add(i - 1)
        if i < ncols - 1 and counts[i] == counts[i + 1] - 1:
            elig = elig - {i}
        if counts[i] == cap:
            elig = elig - {i}
    gi = [[-1] * w for _ in range(h)]
    go = [[0] * ncols for _ in range(max(counts))]
    inds = [(r, c) for r in range(h) for c in range(w)]
    counts = counts[::-1]
    for j, (col, cnt) in enumerate(zip(ccols, counts)):
        locs = random.sample(inds, cnt)
        ls = set(locs)
        for r, c in locs:
            gi[r][c] = col
        inds = [x for x in inds if x not in ls]
        for r in range(cnt):
            go[r][j] = col
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Reading (shared with the golf entry): the answer is a BLANK frame of size
    (largest colour count) x (number of colours). On that blank frame, each colour
    is drawn as one top-aligned column whose height is its cell count, tallest first.
    The maker draws the bars first and then erases leftovers. This version lays the
    blank base first: it resizes to the frame, clears the input block the frame still
    carries (only when some of it would survive the bars), and then draws each bar
    as a whole column. Everything is measured from I alone. O is never read.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bars = sorted(Counter(I.flatten().tolist()).items(), key=lambda kv: (-kv[1], kv[0]))
    ho = int(bars[0][1])          # tallest bar -> frame height
    wo = len(bars)                # one column per colour -> frame width

    ops, sels = [], []

    # Track the working grid only to avoid emitting no-op actions.
    G = np.zeros((ho, wo), dtype=int)
    hr, wr = min(hi, ho), min(wi, wo)
    G[:hr, :wr] = I[:hr, :wr]

    # 1. Resize the canvas to the histogram frame (full rectangle).
    if (ho, wo) != (hi, wi):
        ops.append(33)
        sels.append([0, 0, ho - 1, wo - 1])

    # 2. Blank the frame. The carried-over input block is a full rectangle
    #    [0..hr-1] x [0..wr-1]. Clear it only if some input cell in it lies below
    #    its column's bar (otherwise the bars cover all of it and the clear is moot).
    leftover = any(int(bars[j][1]) < hr for j in range(wr))
    if leftover:
        ops.append(0)
        sels.append([0, 0, hr - 1, wr - 1])   # whole carried input block (full rectangle)
        G[:hr, :wr] = 0

    # 3. Draw each colour's bar as one whole column segment, tallest first.
    for j, (color, count) in enumerate(bars):
        count = int(count)
        if np.all(G[:count, j] == color):     # already holds this bar -> op would do nothing
            continue
        ops.append(int(color))
        sels.append([0, j, count - 1, 0])     # full bar rectangle (rows 0..count-1, col j)
        G[:count, j] = color

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
