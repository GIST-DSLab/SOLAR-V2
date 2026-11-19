"""
ARC Task: 49d1d64f (RE-ARC) — LLM-generated grid_maker
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
    # The rule (extend the border by one cell, copying the edge cells) does not depend on
    # colour. The generator samples a new palette for every instance and there is no
    # background, so there are no colour roles to fix.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    h = unifint(diff_lb, diff_ub, (2, max(2, min(28, max_h - 2))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(28, max_w - 2))))
    ncols = unifint(diff_lb, diff_ub, (1, 10))
    ccols = random.sample(range(10), ncols)
    gi = [[random.choice(ccols) for _ in range(w)] for _ in range(h)]
    go = [[0] * (w + 2) for _ in range(h + 2)]
    for r in range(h):
        for c in range(w):
            go[r + 1][c + 1] = gi[r][c]
    for c in range(w):
        go[0][c + 1] = gi[0][c]
        go[h + 1][c + 1] = gi[h - 1][c]
    for r in range(h):
        go[r + 1][0] = gi[r][0]
        go[r + 1][w + 1] = gi[r][w - 1]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi + 2, wi + 2
    ops, sels = [], []

    # Track the working grid only to drop ops that would not change it.
    g = np.zeros((ho, wo), dtype=int)
    g[:hi, :wi] = I

    def emit(op, sel, new):
        nonlocal g
        if not np.array_equal(new, g):
            ops.append(op); sels.append(sel); g = new

    def paste(r, c):
        # Paste the whole-input clipboard at (r, c). Paste is transparent: 0 cells do not overwrite.
        n = g.copy()
        sub = n[r:r + hi, c:c + wi]
        sub[I != 0] = I[I != 0]
        emit(30, [r, c, 0, 0], n)

    def color0(r, c, h, w):
        # Full rectangle painted to 0.
        n = g.copy()
        n[r:r + h, c:c + w] = 0
        emit(0, [r, c, h - 1, w - 1], n)

    # 1. Make the canvas one cell larger on every side (full rectangle).
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    if not I.any():
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    has_zero = bool((I == 0).any())
    if has_zero:
        # 0 is real content here and Paste cannot write it, so clear the old input
        # area (full rectangle) before stamping copies onto it.
        color0(0, 0, hi, wi)

    # 2. Copy the whole input once (full rectangle).
    ops.append(28); sels.append([0, 0, hi - 1, wi - 1])

    # 3. Paste it shifted up, down, left and right. Each border strip is
    #    written only by its own shifted copy.
    for (r, c) in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        paste(r, c)

    # 4. Paste the input itself in the centre, on top of the others.
    if has_zero:
        # Clear the centre rectangle first so the input's 0 cells come out as 0.
        color0(1, 1, hi, wi)
    paste(1, 1)

    if not has_zero:
        # The top-left corner still holds the input pixel that was there before the
        # resize. The output corners are 0.
        color0(0, 0, 1, 1)

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
                        f"num_examples+1 ({num_examples + 1}) for task 49d1d64f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 49d1d64f"
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
                                f"for task 49d1d64f"
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
                    f"Failed to build a complete episode for task 49d1d64f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"49d1d64f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
