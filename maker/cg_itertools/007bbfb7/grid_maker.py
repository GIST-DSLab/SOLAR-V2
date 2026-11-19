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
import math
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, int(round(a + d * (b - a)))), b)


def sample_colors(num_examples=None) -> dict:
    # bgc is hardcoded 0; the single foreground colour is sampled once per episode
    fgc = random.choice(list(range(1, 10)))
    return {"fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, fgc=None, **kwargs) -> dict:
    if fgc is None:
        fgc = random.choice(list(range(1, 10)))
    hmax = max(2, min(5, int(math.isqrt(max_h))))
    wmax = max(2, min(5, int(math.isqrt(max_w))))
    h = _unifint(diff_lb, diff_ub, (2, hmax))
    w = _unifint(diff_lb, diff_ub, (2, wmax))
    numcd = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = random.choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    inds = [(i, j) for i in range(h) for j in range(w)]
    locs = random.sample(inds, numc)
    gi = [[0] * w for _ in range(h)]
    for (i, j) in locs:
        gi[i][j] = fgc
    go = [[0] * (w * w) for _ in range(h * h)]
    for (bi, bj) in locs:
        for (i, j) in locs:
            go[bi * h + i][bj * w + j] = fgc
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O=None, examples=None):
    # Rule (fixed by I alone): the canvas becomes an hi x wi grid of blocks, each the
    # size of I. Block (r, c) holds a copy of I exactly when I[r, c] is coloured
    # (golf reading: block = min(I[r][c], I) -> I or empty). Same carving as the
    # maker: one Paste of I per coloured cell. Difference: the maker's whole-canvas
    # Color0 is replaced by clearing only the top-left block, the one place the
    # resize leaves a stale copy of I, and only when I[0,0] is empty. When I[0,0] is
    # coloured, that leftover copy is already the correct block (0,0), so it is kept.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi * hi, wi * wi
    ops, sels = [], []

    # 1. Expand canvas to (hi*hi, wi*wi); I stays in the top-left block (zero-padded = bgc 0)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    # 2. Block (0,0): the leftover copy of I is right if I[0,0] is coloured, else erase it.
    #    Selection = the full top-left block rectangle (the region the input occupies).
    if I[0, 0] == 0:
        ops.append(0); sels.append([0, 0, hi - 1, wi - 1])

    # 3. Blocks (r,c) != (0,0) whose cell in I is coloured each receive a copy of I
    targets = [(r, c) for r in range(hi) for c in range(wi)
               if I[r, c] != 0 and (r, c) != (0, 0)]
    if targets:
        ops.append(28); sels.append([0, 0, hi - 1, wi - 1])   # CopyI: whole input rect
        for r, c in targets:
            ops.append(30); sels.append([r * hi, c * wi, 0, 0])

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
