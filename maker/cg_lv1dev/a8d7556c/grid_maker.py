"""
ARC Task: a8d7556c (RE-ARC) — LLM-generated grid_maker
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
    # Colours 0 (holes) and 2 (paint) are fixed by the rule.
    # Only the foreground colour is sampled.
    cols = [c for c in range(10) if c not in (0, 2)]
    return {"fgc": random.choice(cols)}


def generate(diff_lb, diff_ub, max_h, max_w, fgc=None, **kwargs) -> dict:
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c not in (0, 2)])

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    gi = [[fgc] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]

    numblacks = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 3 * 2)))
    for (i, j) in random.sample(inds, numblacks):
        gi[i][j] = 0

    numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    for (i, j) in random.sample(inds, numsq):
        for di in (0, 1):
            for dj in (0, 1):
                if i + di < h and j + dj < w:
                    gi[i + di][j + dj] = 0

    go = [row[:] for row in gi]
    for a in range(h - 1):
        for b in range(w - 1):
            if gi[a][b] == 0 and gi[a + 1][b] == 0 and gi[a][b + 1] == 0 and gi[a + 1][b + 1] == 0:
                for di in (0, 1):
                    for dj in (0, 1):
                        go[a + di][b + dj] = 2
    return {"input": [tuple(r) for r in gi], "output": [tuple(r) for r in go]}


def derive_operations(I, O, examples=None):
    """
    Rule: a 0-cell turns 2 when it belongs to a 2x2 block of 0s.

    How this splits the work (following the golf solution, which makes 4 rotated
    passes and in each one marks a 0-cell when the 2x2 block extending in one fixed
    diagonal direction is all 0/2):
    the unit here is the corner role, not the connected blob. There is one
    Color2 pass for each of the 4 corner roles (top-left, top-right, bottom-right,
    bottom-left). Each pass selects every cell that holds that corner of an
    all-zero 2x2 window in I. Together the 4 passes cover exactly the union of
    the windows.

    Each selection comes from I and the current grid only. A cell is dropped
    from a pass only when it is already 2 right now, which means an earlier pass
    painted it. O is used only for the Submit bbox.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = np.asarray(O, dtype=int).shape
    HOLE, PAINT = 0, 2

    def zero_window(a, b):
        return (0 <= a < hi - 1 and 0 <= b < wi - 1
                and I[a, b] == HOLE and I[a + 1, b] == HOLE
                and I[a, b + 1] == HOLE and I[a + 1, b + 1] == HOLE)

    # For a cell (r, c), the window whose given corner sits at (r, c) has
    # top-left (r - dr, c - dc).
    corner_roles = [(0, 0),   # cell is the top-left of a zero window
                    (0, 1),   # top-right
                    (1, 1),   # bottom-right
                    (1, 0)]   # bottom-left

    cur = I.copy()
    ops, sels = [], []
    for dr, dc in corner_roles:
        cells = [(r, c) for r in range(hi) for c in range(wi)
                 if I[r, c] == HOLE and zero_window(r - dr, c - dc)
                 and cur[r, c] != PAINT]
        if not cells:
            continue
        ops.append(PAINT)
        sels.append(sel_of(cells))
        for r, c in cells:
            cur[r, c] = PAINT

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
                        f"num_examples+1 ({num_examples + 1}) for task a8d7556c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a8d7556c"
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
                                f"for task a8d7556c"
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
                    f"Failed to build a complete episode for task a8d7556c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a8d7556c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
