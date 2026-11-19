"""
ARC Task: feca6190 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Background is hardcoded to 0. The rule only cares where the coloured cells
    # sit in the row, not which colours they are, so no colour needs fixing.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **kwargs) -> dict:
    from random import randint, sample as rsample
    bgc = 0
    lim = min(max_h, max_w)
    w = randint(2, 6)
    max_ncols = min(w, 5)
    while w * 1 > lim and w > 2:
        w -= 1
    max_ncols = min(max_ncols, max(1, lim // w))
    ncols = randint(1, max_ncols)
    remcols = [c for c in range(10) if c != bgc]
    ccols = rsample(remcols, ncols)
    locs = rsample(list(range(w)), ncols)
    N = w * ncols
    gi = [[bgc] * w]
    go = [[bgc] * N for _ in range(N)]
    for col, j in zip(ccols, locs):
        gi[0][j] = col
        r, c = N - 1, j
        while r >= 0 and c < N:
            go[r][c] = col
            r -= 1
            c += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    w = wi
    nz = [j for j in range(w) if I[0, j] != 0]
    ncols = len(nz)
    N = w * ncols
    ops, sels = [], []

    # 1. Make the canvas N x N (N = width * number of coloured cells). The input
    #    row stays at row 0, and bgc 0 means the zero padding is already background.
    ops.append(33); sels.append([0, 0, N - 1, N - 1])

    # 2. Move the input row down to the bottom row, where O holds it unshifted.
    #    The first Move grabs it; each later Move uses an empty selection.
    cur = [(0, j) for j in nz]
    steps = N - 1
    if steps > 0:
        ops.append(21); sels.append(sel_of(cur))
        for _ in range(steps - 1):
            ops.append(21); sels.append(sel_of([]))
    # The background is 0, so the cells the row left behind need no repair.

    # 3. Copy the input row and paste it on each row above, moving one column
    #    further right per row up (the up-right diagonals).
    ops.append(28); sels.append([0, 0, 0, w - 1])  # the whole 1 x w input row
    jmin = min(nz)
    for r in range(N - 2, -1, -1):
        off = N - 1 - r
        # Skip a row if no coloured cell would land inside the grid.
        if not any(j + off <= N - 1 for j in nz):
            continue
        ops.append(30); sels.append([r, off, 0, 0])

    ops.append(34); sels.append([0, 0, N - 1, N - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task feca6190"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task feca6190"
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
                                f"for task feca6190"
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
                    f"Failed to build a complete episode for task feca6190 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"feca6190-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
