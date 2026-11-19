"""
ARC Task: feca6190 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Background is hardcoded to 0 in the generator. The object colours are not
    # sampled per episode: the rule copies whatever colour is at each pixel, so
    # no colour role needs to be fixed.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **color_kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if a >= b:
            return a
        d = random.uniform(lb, ub)
        return min(b, max(a, a + int(round(d * (b - a)))))

    lim = min(max_h, max_w)
    bgc = 0
    remcols = [c for c in range(10) if c != bgc]
    for _ in range(100):
        w = unifint(diff_lb, diff_ub, (2, 6))
        ncols = unifint(diff_lb, diff_ub, (1, min(w, 5)))
        if w * ncols <= lim and w <= max_w:
            break
    else:
        w, ncols = 2, 1
    ccols = random.sample(remcols, ncols)
    locs = random.sample(list(range(w)), ncols)
    n = w * ncols
    gi = [[bgc] * w]
    go = [[bgc] * n for _ in range(n)]
    for col, j in zip(ccols, locs):
        gi[0][j] = col
        r, c = n - 1, j
        while r >= 0 and c < n:
            go[r][c] = col
            r -= 1
            c += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Side length of the square output, worked out from I alone:
    # input width * number of coloured pixels.
    js = [c for c in range(wi) if I[0, c] != 0]
    n = wi * len(js)

    ops, sels = [], []

    # 1. Grow the canvas from 1 x wi to n x n. bgc == 0, so the zero padding is
    #    already background. This bbox is the full rectangle on purpose.
    ops.append(33)
    sels.append([0, 0, n - 1, n - 1])

    # 2. The padded input row becomes the BOTTOM row of the output. Slide it from
    #    row 0 to row n-1: grab it once, then use empty selections to keep moving it.
    grid = np.zeros((n, n), dtype=int)
    grid[0, :wi] = I[0]
    if n - 1 > 0:
        src = [(0, c) for c in js]
        ops.append(21)
        sels.append(sel_of(src))
        for _ in range(n - 2):
            ops.append(21)
            sels.append(sel_of([]))
        grid[n - 1, :wi] = I[0]
        grid[0, :] = 0
    # Row 0, where the row started, is left at 0, which is the background, so it needs no repair.

    # 3. Build upward one row at a time: each row is the row just below it,
    #    shifted right by one. The rightmost cell drops off the edge.
    #    Copy cells 0..n-2 of the current row from the working grid (CopyO) and
    #    paste them one row up, one column right.
    #    Stop once the part of the row still to be shifted is empty.
    r = n - 1
    while r - 1 >= 0:
        seg = grid[r, 0:n - 1]
        if not np.any(seg != 0):
            break
        ops.append(29)                      # CopyO: the row just built (cells 0..n-2)
        sels.append([r, 0, 0, n - 2])
        ops.append(30)                      # Paste one row up, one column right
        sels.append([r - 1, 1, 0, 0])
        grid[r - 1, 1:n] = np.where(seg != 0, seg, grid[r - 1, 1:n])
        r -= 1

    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task feca6190"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task feca6190"
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
                                f"for task feca6190"
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
                    f"Failed to build a complete episode for task feca6190 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"feca6190-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
