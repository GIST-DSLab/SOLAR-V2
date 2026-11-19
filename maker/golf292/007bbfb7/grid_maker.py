"""
ARC Task: 007bbfb7 (RE-ARC) — LLM-generated grid_maker
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
    # Background is hard-coded to 0. Only the foreground colour is random, so fix it per episode.
    fgc = random.choice(range(1, 10))
    return {"fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, fgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return int(round(random.uniform(lo, hi_)))

    if fgc is None:
        fgc = random.choice(range(1, 10))
    # The output is (h*h, w*w), so it must fit inside max_h x max_w.
    hmax = 2
    while (hmax + 1) ** 2 <= max_h and hmax + 1 <= 5:
        hmax += 1
    wmax = 2
    while (wmax + 1) ** 2 <= max_w and wmax + 1 <= 5:
        wmax += 1
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = random.choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    inds = [(i, j) for i in range(h) for j in range(w)]
    locs = random.sample(inds, numc)
    gi = [[0] * w for _ in range(h)]
    for (i, j) in locs:
        gi[i][j] = fgc
    go = [[0] * (w * w) for _ in range(h * h)]
    for (a, b) in locs:
        for (i, j) in locs:
            go[a * h + i][b * w + j] = fgc
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of  # noqa: F401  (all selections here are true rectangles / paste origins)
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi * hi, wi * wi  # The rule sets the output size: shape(I) squared.

    ops, sels = [], []
    # 1. Grow the canvas to (hi*hi, wi*wi). Resize copies I into the top-left block, which is block (0,0).
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    # 2. Block (0,0) must stay empty when I[0,0] is background. Clear that whole block rectangle.
    if I[0, 0] == 0:
        ops.append(0); sels.append([0, 0, hi - 1, wi - 1])

    # Group I's rows by pattern. A block-row of O depends only on its row of I.
    first_of = {}
    dup_groups = {}
    for r in range(hi):
        key = tuple(I[r].tolist())
        if not any(key):
            continue  # an all-zero row of I leaves its block-row empty
        if key in first_of:
            dup_groups.setdefault(first_of[key], []).append(r)
        else:
            first_of[key] = r

    # Decide which duplicate rows get a whole-block-row copy instead of one paste per block.
    copy_rows = {}
    for src, dups in dup_groups.items():
        n = int(np.count_nonzero(I[src]))
        if 1 + len(dups) < n * len(dups):
            copy_rows[src] = dups
    copied = {r for d in copy_rows.values() for r in d}

    # 3. Copy the input pattern once, then paste it into every block whose cell in I is foreground.
    paste_targets = []
    for r in range(hi):
        if r in copied:
            continue
        for c in range(wi):
            if I[r, c] != 0 and not (r == 0 and c == 0):
                paste_targets.append((r * hi, c * wi))
    if paste_targets:
        ops.append(28); sels.append([0, 0, hi - 1, wi - 1])  # CopyI: the whole input
        for (pr, pc) in paste_targets:
            ops.append(30); sels.append([pr, pc, 0, 0])

    # 4. A repeated input row gives an identical block-row: copy the built block-row and paste it.
    for src in sorted(copy_rows):
        ops.append(29); sels.append([src * hi, 0, hi - 1, wo - 1])  # CopyO: one whole block-row
        for r in copy_rows[src]:
            ops.append(30); sels.append([r * hi, 0, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 007bbfb7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 007bbfb7"
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
                                f"for task 007bbfb7"
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
                    f"Failed to build a complete episode for task 007bbfb7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"007bbfb7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
