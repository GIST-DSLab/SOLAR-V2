"""
ARC Task: 9172f3a0 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background colour is chosen up front. The rule (stretch every
    # cell 3x in both directions) works the same whatever the colours are,
    # so the scatter colours are drawn per instance.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    a = lo + (hi - lo) * max(0.0, min(1.0, diff_lb))
    b = lo + (hi - lo) * max(0.0, min(1.0, diff_ub))
    a, b = int(round(min(a, b))), int(round(max(a, b)))
    a, b = max(lo, a), min(hi, b)
    if a > b:
        a = b
    return random.randint(a, b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.randint(0, 9)
    hcap = max(1, min(10, max_h // 3))
    wcap = max(1, min(10, max_w // 3))
    h = _unifint(diff_lb, diff_ub, 1, hcap)
    w = _unifint(diff_lb, diff_ub, 1, wcap)
    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in range(10) if c != bgc]
    numc = _unifint(diff_lb, diff_ub, 0, min(9, h * w))
    colsch = random.sample(remcols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for col in colsch:
        num = _unifint(diff_lb, diff_ub, 1, max(1, len(inds) // numc))
        num = min(num, len(inds))
        if num <= 0:
            break
        chos = random.sample(inds, num)
        for (i, j) in chos:
            gi[i][j] = col
        chs = set(chos)
        inds = [x for x in inds if x not in chs]
    go = [[gi[i // 3][j // 3] for j in range(3 * w)] for i in range(3 * h)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading: upscale x3 splits into two separate stretches. First each COLUMN
    # becomes three columns, then each ROW of the widened grid becomes three
    # rows. Lines are the units here, not 3x3 cells.
    # Everything below depends only on I. O is never read.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 3 * wi
    ops, sels = [], []

    # 1. Grow the canvas to the output size. I stays at the top-left and the
    #    rest is 0. (Full rectangle, so a bbox is exact.)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    cur = np.zeros((ho, wo), dtype=int)
    cur[:hi, :wi] = I

    # 2. Column stretch, rightmost column first. Column c slides right to 3c.
    #    Every column to its right has already left, so its path is all 0.
    #    Then copy it into columns 3c+1 and 3c+2.
    for c in range(wi - 1, -1, -1):
        col_cells = [(r, c) for r in range(hi)]
        if not any(cur[r, c] != 0 for r, _ in col_cells):
            continue  # an all-zero column: nothing visible to move or copy
        steps = 2 * c
        if steps > 0:
            ops.append(22); sels.append(sel_of(col_cells))  # grab the column
            for _ in range(steps - 1):
                ops.append(22); sels.append(sel_of([]))      # keep sliding it
            vals = cur[:hi, c].copy()
            cur[:hi, c] = 0
            cur[:hi, 3 * c] = vals
        dst = [(r, 3 * c) for r in range(hi)]
        ops.append(29); sels.append(sel_of(dst))             # CopyO the column
        for k in (1, 2):
            ops.append(30); sels.append([0, 3 * c + k, 0, 0])
            cur[:hi, 3 * c + k] = np.where(cur[:hi, 3 * c] != 0,
                                           cur[:hi, 3 * c], cur[:hi, 3 * c + k])

    # 3. Row stretch on the widened grid, bottom row first. Row r slides down
    #    to 3r, then gets copied into rows 3r+1 and 3r+2.
    for r in range(hi - 1, -1, -1):
        row_cells = [(r, c) for c in range(wo)]
        if not any(cur[r, c] != 0 for c in range(wo)):
            continue
        steps = 2 * r
        if steps > 0:
            ops.append(21); sels.append(sel_of(row_cells))   # grab the row
            for _ in range(steps - 1):
                ops.append(21); sels.append(sel_of([]))
            vals = cur[r, :].copy()
            cur[r, :] = 0
            cur[3 * r, :] = vals
        dst = [(3 * r, c) for c in range(wo)]
        ops.append(29); sels.append(sel_of(dst))             # CopyO the row
        for k in (1, 2):
            ops.append(30); sels.append([3 * r + k, 0, 0, 0])
            cur[3 * r + k, :] = np.where(cur[3 * r, :] != 0,
                                         cur[3 * r, :], cur[3 * r + k, :])

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9172f3a0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9172f3a0"
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
                                f"for task 9172f3a0"
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
                    f"Failed to build a complete episode for task 9172f3a0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9172f3a0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
