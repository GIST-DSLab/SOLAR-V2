"""
ARC Task: 25d8a9c8 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The rule ("is this row a single colour?") is colour-agnostic, but the palette
    # is kept fixed across the whole episode so every instance looks like one task.
    ncols = random.randint(2, 10)
    ccols = random.sample(list(range(10)), ncols)
    return {"ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, ccols=None, **kwargs) -> dict:
    h = unifint(diff_lb, diff_ub, (2, max(2, max_h)))
    w = unifint(diff_lb, diff_ub, (2, max(2, max_w)))
    if ccols is None:
        ncols = unifint(diff_lb, diff_ub, (2, 10))
        ccols = sample(list(range(10)), ncols)
    ccols = list(ccols)
    gi = []
    go = []
    for k in range(h):
        singlecol = choice((True, False))
        col = choice(ccols)
        row = tuple([col] * w)
        if singlecol:
            gi.append(row)
            go.append(tuple([5] * w))
        else:
            remcols = [c for c in ccols if c != col]
            nothercinv = unifint(diff_lb, diff_ub, (1, w - 1))
            notherc = w - 1 - nothercinv
            notherc = min(max(1, notherc), w - 1)
            row = list(row)
            indss = list(range(w))
            for j in sample(indss, notherc):
                row[j] = choice(remcols)
            gi.append(tuple(row))
            go.append(tuple([0] * w))
    return {'input': tuple(gi), 'output': tuple(go)}


def derive_operations(I, O):
    """
    Rule: a row holding a single colour becomes a full row of 5s; any row with
    more than one colour becomes a full row of 0s.

    Expressed as repetition:
      1. lay one 5 down, then carry it across the row by repeated CopyO/Paste
         (doubling the run each time) until the whole row is 5s,
      2. carry that finished row down to every other single-colour row (CopyO + Paste),
      3. paint all the mixed rows black with a single Color0 (0 is "nothing" to Paste,
         so a 0 row cannot be carried).
    """
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    ops, sels = [], []

    uniform_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    uniform_set = set(uniform_rows)
    mixed_rows = [r for r in range(hi) if r not in uniform_set]

    # Single-colour rows that are not already all 5s (those already read 5 need no work).
    targets = [r for r in uniform_rows if int(I[r, 0]) != 5]

    if targets:
        r0 = targets[0]

        # 1. lay the colour down once ...
        ops.append(5)
        sels.append(sel_of([(r0, 0)]))

        # ... and carry it across the row, doubling the finished run each time.
        cur = 1
        while cur < wi:
            n = min(cur, wi - cur)
            ops.append(29)                                            # CopyO the finished run
            sels.append(sel_of([(r0, c) for c in range(n)]))
            ops.append(30)                                            # Paste it just after itself
            sels.append(sel_of([(r0, cur)]))
            cur += n

        # 2. carry the finished row down to the other single-colour rows.
        rest = targets[1:]
        if rest:
            ops.append(29)                                            # CopyO the whole finished row
            sels.append(sel_of([(r0, c) for c in range(wi)]))
            for r in rest:
                ops.append(30)                                        # Paste at that row's left edge
                sels.append(sel_of([(r, 0)]))

    # 3. the mixed rows go black in one Color0 (only the cells not already 0).
    zero_cells = [(r, c) for r in mixed_rows for c in range(wi) if int(I[r, c]) != 0]
    if zero_cells:
        ops.append(0)
        sels.append(sel_of(zero_cells))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 25d8a9c8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 25d8a9c8"
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
                                f"for task 25d8a9c8"
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
                    f"Failed to build a complete episode for task 25d8a9c8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"25d8a9c8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
