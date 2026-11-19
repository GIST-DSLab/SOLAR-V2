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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule depends only on whether a row holds a single colour, never on which
    # colour it is (the fill 5 and the blank 0 are hardcoded by the generator).
    # Still fix the episode palette so every instance shares one colour scheme.
    ncols = random.randint(2, 10)
    ccols = random.sample(list(range(10)), ncols)
    return {"ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, ccols=None) -> dict:
    def unifint(dlb, dub, bounds):
        a, b = bounds
        lo = a + int((b - a) * dlb)
        hi = a + int((b - a) * dub)
        if lo > hi:
            lo, hi = hi, lo
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        return random.randint(lo, hi)

    if ccols is None:
        ccols = random.sample(list(range(10)), random.randint(2, 10))
    ccols = list(ccols)

    h = unifint(diff_lb, diff_ub, (2, max(2, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(30, max_w))))

    # guarantee both row kinds are visible in every instance -> the rule is readable
    while True:
        flags = [random.choice((True, False)) for _ in range(h)]
        if any(flags) and not all(flags):
            break

    gi, go = [], []
    for k in range(h):
        col = random.choice(ccols)
        row = [col] * w
        if flags[k]:
            gi.append(tuple(row))
            go.append(tuple([5] * w))
        else:
            remcols = [c for c in ccols if c != col]
            nothercinv = unifint(diff_lb, diff_ub, (1, w - 1))
            notherc = w - 1 - nothercinv
            notherc = min(max(1, notherc), w - 1)
            for j in random.sample(list(range(w)), notherc):
                row[j] = random.choice(remcols)
            gi.append(tuple(row))
            go.append(tuple([0] * w))

    return {"input": tuple(gi), "output": tuple(go)}


def derive_operations(I, O, examples=None):
    """Rule: a row holding a single colour becomes a row of 5s; any other row
    becomes a row of 0s.  One 5-row is painted, then carried to the remaining
    5-rows with CopyO/Paste; the non-uniform rows take a single Color0."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    FILL = 5

    uniform = [r for r in range(h) if len(set(I[r].tolist())) == 1]
    uset = set(uniform)
    nonuniform = [r for r in range(h) if r not in uset]

    ops, sels = [], []

    # --- the single-colour rows -> rows of 5 -------------------------------
    # rows that are already all-5 in I already hold the answer: touching them
    # would be a no-op, so they get no operation at all.
    need5 = [r for r in uniform if I[r, 0] != FILL]
    if need5:
        src = need5[0]
        src_row = [(src, c) for c in range(w)]
        ops.append(FILL)
        sels.append(sel_of(src_row))                 # paint the first 5-row
        rest = need5[1:]
        if rest:
            ops.append(29)
            sels.append(sel_of(src_row))             # CopyO that finished row
            for r in rest:
                ops.append(30)
                sels.append(sel_of([(r, 0)]))        # Paste it at each other 5-row

    # --- the mixed rows -> rows of 0 ---------------------------------------
    # 0 cannot be pasted (ARCLE reads 0 as "nothing there"), so it is a Color0.
    # Cells of those rows that already read 0 in I are left out: the op would
    # do nothing there.
    zero_cells = [(r, c) for r in nonuniform for c in range(w) if I[r, c] != 0]
    if zero_cells:
        ops.append(0)
        sels.append(sel_of(zero_cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])                # whole grid, submit
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
