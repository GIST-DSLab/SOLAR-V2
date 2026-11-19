"""
ARC Task: 67e8384a (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    from random import choice, sample
    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    hmax = max(1, min(14, max_h // 2))
    wmax = max(1, min(14, max_w // 2))
    h = unifint(diff_lb, diff_ub, (1, hmax))
    w = unifint(diff_lb, diff_ub, (1, wmax))
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    remcols = sample(remcols, numcols)
    canv = [[bgc] * w for _ in range(h)]
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = choice(sorted(bx))
    obj = {start: choice(remcols)}
    bx.discard(start)
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            dns |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = choice(cand)
        obj[ch] = choice(remcols)
        bx.discard(ch)
    gi = [row[:] for row in canv]
    for (i, j), c in obj.items():
        gi[i][j] = c
    top = [row + row[::-1] for row in gi]
    go = top + top[::-1]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading: first stack I over its up<->down mirror (a column band),
    # then extend every row of that band with its left<->right mirror.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 2 * hi, 2 * wi
    ops, sels = [], []

    # 1. expand canvas to 2h x 2w (input stays at top-left)  -- full rectangle
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    has_content = bool((I != 0).any())
    # 2. copy the input, paste it directly below itself
    if has_content:
        ops.append(28); sels.append([0, 0, hi - 1, wi - 1])   # whole input rectangle
        ops.append(30); sels.append([hi, 0, 0, 0])            # paste origin
    # 3. flip that pasted copy up<->down (bbox = exactly the pasted rectangle)
    if has_content and not np.array_equal(I, I[::-1, :]):
        ops.append(27); sels.append([hi, 0, hi - 1, wi - 1])

    # left band as it now stands
    band = np.vstack([I, I[::-1, :]])
    # 4. copy the whole left column band from the working grid, paste to its right
    if has_content:
        ops.append(29); sels.append([0, 0, ho - 1, wi - 1])   # whole left band rectangle
        ops.append(30); sels.append([0, wi, 0, 0])            # paste origin
    # 5. mirror the right band left<->right (bbox = exactly the pasted rectangle)
    if has_content and not np.array_equal(band, band[:, ::-1]):
        ops.append(26); sels.append([0, wi, ho - 1, wi - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 67e8384a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 67e8384a"
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
                                f"for task 67e8384a"
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
                    f"Failed to build a complete episode for task 67e8384a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"67e8384a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
