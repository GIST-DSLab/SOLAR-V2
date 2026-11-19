"""
ARC Task: 995c5fa3 (RE-ARC) — LLM-generated grid_maker
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


# The golf entry reads the task the same way as the existing maker, so the operations are unchanged.
# Each 4x4 block is classified from a few of its cells, and that block becomes one solid row of
# the num x num answer, in block order.


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if bgc is None:
        bgc = random.choice(range(10))
    o1 = {(i, j) for i in range(4) for j in range(4)}
    o2 = {(i, j) for (i, j) in o1 if i in (0, 3) or j in (0, 3)}
    o3 = o1 - {(1, 0), (2, 0), (1, 3), (2, 3)}
    o4 = o1 - {(2, 1), (2, 2), (3, 1), (3, 2)}
    mpr = [(o1, 2), (o2, 8), (o3, 3), (o4, 4)]
    max_num = max(1, min(6, (max_w + 1) // 5))
    num = unifint(diff_lb, diff_ub, (1, max_num))
    num = max(1, min(num, max_num))
    h = 4
    w = 5 * num - 1
    remcols = [c for c in range(10) if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    ccols = []
    for k in range(num):
        col = random.choice(remcols)
        obj, outcol = random.choice(mpr)
        for (i, j) in obj:
            gi[i][j + 5 * k] = col
        ccols.append(outcol)
    go = [[c] * num for c in ccols]
    return {"input": gi, "output": go}


def _classify_block(B):
    B = np.asarray(B, dtype=int)
    if len(set(B.flatten().tolist())) == 1:
        return 2                       # solid block
    if not np.array_equal(B, B[::-1, :]):
        return 4                       # not up/down symmetric: notch at the bottom
    if B[0, 0] != B[1, 0]:
        return 3                       # left/right sides open in the middle
    return 8                           # hollow box


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    num = (wi + 1) // 5                # number of 4x4 blocks along the row
    labels = [_classify_block(I[0:4, 5 * k:5 * k + 4]) for k in range(num)]

    ops, sels = [], []

    # Resize the canvas once to the answer's shape (num x num).
    # Full-rectangle bbox: a canvas resize acts on the whole rectangle.
    ops.append(33); sels.append([0, 0, num - 1, num - 1])

    # Keep track of the grid as it stands now: a transparent copy of I's top-left corner,
    # with zero padding where the canvas is larger than I.
    cur = np.zeros((num, num), dtype=int)
    for r in range(min(num, hi)):
        for c in range(min(num, wi)):
            if I[r, c] != 0:
                cur[r, c] = I[r, c]

    # Each block becomes one solid row, in block order.
    for k in range(num):
        lab = labels[k]
        if np.all(cur[k, :] == lab):
            continue                   # the row already holds this colour, so painting would change nothing
        ops.append(int(lab))
        sels.append(sel_of([(k, c) for c in range(num)]))
        cur[k, :] = lab

    ops.append(34); sels.append([0, 0, num - 1, num - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 995c5fa3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 995c5fa3"
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
                                f"for task 995c5fa3"
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
                    f"Failed to build a complete episode for task 995c5fa3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"995c5fa3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
