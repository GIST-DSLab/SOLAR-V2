"""
ARC Task: 1190e5a7 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # Rule depends only on frontier COUNTS, not on line colors -> fix bgc only.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    # sizes (replace hardcoded 30 with max_h/max_w)
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))

    grid = [[bgc] * w for _ in range(h)]

    nhf_bounds = (1, max(1, h // 3))
    nvf_bounds = (1, max(1, w // 3))
    nhf = unifint(diff_lb, diff_ub, nhf_bounds)
    nvf = unifint(diff_lb, diff_ub, nvf_bounds)

    remcols = [c for c in range(10) if c != bgc]

    # pick horizontal frontier rows (interior only, spaced by >=2)
    hf_selection = []
    opts = list(range(1, h - 1))
    for _ in range(nhf):
        if not opts:
            break
        hf = random.choice(opts)
        hf_selection.append(hf)
        opts = [x for x in opts if x not in (hf - 1, hf, hf + 1)]

    # pick vertical frontier cols
    vf_selection = []
    opts = list(range(1, w - 1))
    for _ in range(nvf):
        if not opts:
            break
        vf = random.choice(opts)
        vf_selection.append(vf)
        opts = [x for x in opts if x not in (vf - 1, vf, vf + 1)]

    # paint horizontal frontiers, then vertical frontiers on top
    for r in hf_selection:
        col = random.choice(remcols)
        for c in range(w):
            grid[r][c] = col
    for c in vf_selection:
        col = random.choice(remcols)
        for r in range(h):
            grid[r][c] = col

    nhf_a = len(hf_selection)
    nvf_a = len(vf_selection)
    out = [[bgc] * (nvf_a + 1) for _ in range(nhf_a + 1)]

    return {"input": np.array(grid, dtype=int), "output": np.array(out, dtype=int)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # bgc = most common corner color (corners are never on a frontier)
    corner_vals = [int(I[0, 0]), int(I[0, wi - 1]), int(I[hi - 1, 0]), int(I[hi - 1, wi - 1])]
    bgc = Counter(corner_vals).most_common(1)[0][0]

    # horizontal frontiers = rows entirely non-bgc; vertical frontiers = cols entirely non-bgc
    nhf = sum(1 for r in range(hi) if all(int(I[r, c]) != bgc for c in range(wi)))
    nvf = sum(1 for c in range(wi) if all(int(I[r, c]) != bgc for r in range(hi)))

    ho, wo = nhf + 1, nvf + 1

    ops, sels = [], []
    # 1. collapse frontier structure: fill whole grid with bgc (full rectangle)
    ops.append(int(bgc)); sels.append([0, 0, hi - 1, wi - 1])
    # 2. resize canvas to (nhf+1) x (nvf+1) measured from I's frontier counts
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    # 3. submit
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 1190e5a7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1190e5a7"
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
                                f"for task 1190e5a7"
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
                    f"Failed to build a complete episode for task 1190e5a7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1190e5a7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
