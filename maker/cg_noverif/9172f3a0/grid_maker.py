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
    # Only the background is a fixed role; the rule (upscale x3) ignores colours.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(a, b):
        if b <= a:
            return a
        return a + int(round((b - a) * random.uniform(diff_lb, diff_ub)))

    if bgc is None:
        bgc = random.choice(range(10))
    # output is 3h x 3w, so keep it within max_h/max_w
    h = unifint(1, max(1, min(10, max_h // 3)))
    w = unifint(1, max(1, min(10, max_w // 3)))
    gi = np.full((h, w), bgc, dtype=int)
    remcols = [c for c in range(10) if c != bgc]
    numc = unifint(0, min(9, h * w))
    colsch = random.sample(remcols, numc)
    inds = [(r, c) for r in range(h) for c in range(w)]
    for col in colsch:
        num = unifint(1, max(1, len(inds) // numc))
        num = min(num, len(inds))
        if num <= 0:
            break
        chos = random.sample(inds, num)
        for r, c in chos:
            gi[r, c] = col
        cs = set(chos)
        inds = [x for x in inds if x not in cs]
    go = np.repeat(np.repeat(gi, 3, axis=0), 3, axis=1)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading: upscale x3 is separable -- every cell is repeated 3x along its
    # row (each input column becomes 3 columns), then every (widened) row is
    # repeated 3x (each row becomes 3 rows). Built column-by-column, then
    # row-by-row, from I alone.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 3 * wi
    ops, sels = [], []

    # Expand canvas to 3x size (full rectangle; I stays at top-left, rest 0)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    cur = np.zeros((ho, wo), dtype=int)
    cur[:hi, :wi] = I

    # Stage 1: stretch each input column c into columns 3c..3c+2 (right to left,
    # so stale input columns being overwritten are already processed).
    for c in range(wi - 1, -1, -1):
        src = I[:, c]
        copied = False
        for t in range(3 * c, 3 * c + 3):
            diff = [r for r in range(hi) if cur[r, t] != src[r]]
            if not diff:
                continue
            # stale leftover cells where the source column holds 0 (paste can't write 0)
            zero_cells = [(r, t) for r in diff if src[r] == 0]
            if zero_cells:
                ops.append(0); sels.append(sel_of(zero_cells))
                for r, _ in zero_cells:
                    cur[r, t] = 0
            if any(src[r] != 0 for r in diff):
                if not copied:
                    ops.append(28); sels.append([0, c, hi - 1, 0])  # whole input column c
                    copied = True
                ops.append(30); sels.append([0, t, 0, 0])  # paste column at t
                for r in range(hi):
                    if src[r] != 0:
                        cur[r, t] = src[r]

    # Stage 2: stretch each widened row r into rows 3r..3r+2 (bottom to top).
    for r in range(hi - 1, -1, -1):
        src = cur[r, :].copy()
        copied = False
        for t in range(3 * r, 3 * r + 3):
            diff = [c for c in range(wo) if cur[t, c] != src[c]]
            if not diff:
                continue
            zero_cells = [(t, c) for c in diff if src[c] == 0]
            if zero_cells:
                ops.append(0); sels.append(sel_of(zero_cells))
                for _, c in zero_cells:
                    cur[t, c] = 0
            if any(src[c] != 0 for c in diff):
                if not copied:
                    ops.append(29); sels.append([r, 0, 0, wo - 1])  # whole widened row r
                    copied = True
                ops.append(30); sels.append([t, 0, 0, 0])  # paste row at t
                for c in range(wo):
                    if src[c] != 0:
                        cur[t, c] = src[c]

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
