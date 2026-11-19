"""
ARC Task: c59eb873 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background colour is sampled per episode. The foreground colours are
    # scattered at random and the rule (double every cell) does not depend on them.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    # The output is twice the input size, so the input may use at most half the canvas.
    hmax = max(1, min(15, max_h // 2))
    wmax = max(1, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (1, hmax))
    w = unifint(diff_lb, diff_ub, (1, wmax))
    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(r, c) for r in range(h) for c in range(w)]
    for col in colsch:
        if not inds:
            break
        num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // max(1, numc))))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (r, c) in chos:
            gi[r][c] = col
        chs = set(chos)
        inds = [x for x in inds if x not in chs]
    go = []
    for row in gi:
        wide = []
        for v in row:
            wide += [v, v]
        go.append(list(wide))
        go.append(list(wide))
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # How this reading works: upscaling by 2 can be split into two steps (zip(r,r), then 2*[row]).
    # First, every column is doubled across the row. Then every row of the widened grid is doubled
    # down the page. The scale factor 2 is fixed by the task, so nothing here is read from O.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    H, W = 2 * h, 2 * w
    ops, sels = [], []

    # 1. Grow the canvas to 2h x 2w. I stays at the top-left and the new area is 0.
    #    The bbox is exactly the full target rectangle.
    ops.append(33); sels.append([0, 0, H - 1, W - 1])
    cur = np.zeros((H, W), dtype=int)
    cur[:h, :w] = I

    # 2. Horizontal doubling. Columns are handled from right to left: input column c slides
    #    right by c cells to column 2c. Then it is copied into column 2c+1.
    #    Everything it slides over has already been vacated (or is new zero padding).
    for c in range(w - 1, -1, -1):
        if not cur[:h, c].any():
            continue  # an all-zero column: moving or copying it would change nothing
        if c > 0:
            ops.append(22); sels.append(sel_of([(r, c) for r in range(h)]))  # grab the column
            for _ in range(c - 1):
                ops.append(22); sels.append(sel_of([]))                       # keep sliding
            cur[:h, 2 * c] = cur[:h, c]
            cur[:h, c] = 0
        ops.append(29); sels.append(sel_of([(r, 2 * c) for r in range(h)]))   # copy the column
        ops.append(30); sels.append(sel_of([(0, 2 * c + 1)]))                 # paste next to it
        cur[:h, 2 * c + 1] = cur[:h, 2 * c]

    # 3. Vertical doubling. Rows of the widened grid are handled from bottom to top:
    #    row r slides down by r cells to row 2r. Then it is copied into row 2r+1.
    for r in range(h - 1, -1, -1):
        if not cur[r].any():
            continue
        if r > 0:
            ops.append(21); sels.append(sel_of([(r, c) for c in range(W)]))  # grab the row
            for _ in range(r - 1):
                ops.append(21); sels.append(sel_of([]))                       # keep sliding
            cur[2 * r] = cur[r]
            cur[r] = 0
        ops.append(29); sels.append(sel_of([(2 * r, c) for c in range(W)]))   # copy the row
        ops.append(30); sels.append(sel_of([(2 * r + 1, 0)]))                 # paste below it
        cur[2 * r + 1] = cur[2 * r]

    # Every output cell is now covered by moved or pasted content. Cells that were 0 in I
    # are still 0, which is correct, so no background repair is needed.
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c59eb873"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c59eb873"
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
                                f"for task c59eb873"
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
                    f"Failed to build a complete episode for task c59eb873 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c59eb873-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
