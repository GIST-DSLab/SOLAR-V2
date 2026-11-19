"""
ARC Task: eb5a1d5d (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors
# The rule is purely geometric (collapse the duplicated rows/columns back to
# the deduped quadrant, then reflect it out), so the concrete colors carry no
# meaning -- but the nesting order is kept identical across the whole episode
# so every instance shows the same colour scheme from the outside in.
# Colour 0 is excluded on purpose: ARCLE's Move/Copy/Paste treat 0 as "nothing",
# and 0 would be real object content here.
def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    random.shuffle(cols)
    return {"palette": cols}


# ---------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, palette=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if palette is None:
        palette = random.sample(range(1, 10), 9)

    max_h = min(int(max_h), 30)
    max_w = min(int(max_w), 30)

    # d = number of nested boxes; keep 2*d-1 strictly smaller than the canvas
    # limits so at least one row/column duplication is always possible.
    dcap = min(len(palette), min(max_h, max_w) // 2)
    dcap = max(3, dcap)
    d = unifint(diff_lb, diff_ub, (3, dcap)) if dcap > 3 else 3
    n = 2 * d - 1

    # concentric boxes (filling nested squares outside-in == drawing the boxes)
    go = [[0] * n for _ in range(n)]
    for j in range(d):
        c = palette[j]
        for r in range(j, n - j):
            for cc in range(j, n - j):
                go[r][cc] = c

    vroom = max(0, max_h - n)
    hroom = max(0, max_w - n)
    nv = unifint(diff_lb, diff_ub, (0, vroom))
    nh = unifint(diff_lb, diff_ub, (0, hroom))
    if nv + nh == 0:                      # never emit an identity instance
        if vroom > 0:
            nv = 1
        elif hroom > 0:
            nh = 1

    gi = [row[:] for row in go]
    while nv > 0 or nh > 0:
        opts = []
        if nv > 0 and len(gi) < max_h:
            opts.append(0)
        if nh > 0 and len(gi[0]) < max_w:
            opts.append(1)
        if not opts:
            break
        ch = random.choice(opts)
        if ch == 0:
            idx = random.randint(0, len(gi) - 1)
            gi = gi[:idx + 1] + [gi[idx][:]] + gi[idx + 1:]
            nv -= 1
        else:
            idx = random.randint(0, len(gi[0]) - 1)
            gi = [r[:idx + 1] + [r[idx]] + r[idx + 1:] for r in gi]
            nh -= 1

    return {'input': tuple(tuple(r) for r in gi),
            'output': tuple(tuple(r) for r in go)}


# ---------------------------------------------------------------- derivation
def derive_operations(I, O):
    """
    I is the nested-boxes figure with some rows/columns duplicated.
    Route:
      1. slide the grid up / left to squeeze out every duplicated row / column
         (Moves -- the content really slides over the copies),
      2. ResizeGrid down to the d x d quadrant the rule keeps
         (top-left quarter incl. the centre row/column),
      3. widen the canvas, copy the part left of the centre column, paste it
         right of the centre and FlipH it  -> left<->right reflection,
      4. heighten the canvas, copy the part above the centre row, paste it
         below the centre and FlipV it     -> up<->down reflection.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    n = int(O.shape[0])
    d = (n + 1) // 2

    # first occurrence of every distinct row / column == the kept ones
    seen, R = [], []
    for r in range(h):
        key = tuple(I[r].tolist())
        if key not in seen:
            seen.append(key)
            R.append(r)
    seen, C = [], []
    for c in range(w):
        key = tuple(I[:, c].tolist())
        if key not in seen:
            seen.append(key)
            C.append(c)
    R = R[:d]
    C = C[:d]

    ops, sels = [], []

    # 1a. close the vertical gaps: everything from the kept row downwards
    #     slides up over the copies of the previous row
    shift = 0
    for j in range(len(R)):
        cur = R[j] - shift
        k = cur - j
        if k <= 0:
            continue
        cells = [(r, c) for r in range(cur, h) for c in range(w)]
        ops.append(20); sels.append(sel_of(cells))          # grab once
        for _ in range(k - 1):
            ops.append(20); sels.append(sel_of([]))         # keep sliding
        shift += k

    # 1b. same for the duplicated columns
    shift = 0
    for j in range(len(C)):
        cur = C[j] - shift
        k = cur - j
        if k <= 0:
            continue
        cells = [(r, c) for r in range(h) for c in range(cur, w)]
        ops.append(23); sels.append(sel_of(cells))          # grab once
        for _ in range(k - 1):
            ops.append(23); sels.append(sel_of([]))         # keep sliding
        shift += k

    # 2. reframe to the quadrant the rule keeps (bbox == exactly that rectangle)
    ops.append(33); sels.append([0, 0, d - 1, d - 1])

    # 3. reflect left -> right about the centre column
    ops.append(33); sels.append([0, 0, d - 1, n - 1])        # widen canvas
    if d >= 2:
        ops.append(29); sels.append([0, 0, d - 1, d - 2])    # copy left of centre
        ops.append(30); sels.append([0, d, 0, 0])            # paste right of centre
        if d - 1 > 1:                                        # a 1-wide flip is a no-op
            ops.append(26); sels.append([0, d, d - 1, d - 2])

    # 4. reflect top -> bottom about the centre row
    ops.append(33); sels.append([0, 0, n - 1, n - 1])        # heighten canvas
    if d >= 2:
        ops.append(29); sels.append([0, 0, d - 2, n - 1])    # copy above centre
        ops.append(30); sels.append([d, 0, 0, 0])            # paste below centre
        if d - 1 > 1:                                        # a 1-high flip is a no-op
            ops.append(27); sels.append([d, 0, d - 2, n - 1])

    ops.append(34); sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
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
                                f"for task eb5a1d5d"
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
                    f"Failed to build a complete episode for task eb5a1d5d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"eb5a1d5d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
