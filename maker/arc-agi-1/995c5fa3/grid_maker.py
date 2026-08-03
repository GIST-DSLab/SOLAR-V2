"""
ARC Task: 995c5fa3 (RE-ARC) — LLM-generated grid_maker
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

# shape index -> (cell set inside the 4x4 block, output label colour)
#   0: full block                      -> 2
#   1: box outline                     -> 8
#   2: full minus 2 left/right notches -> 3
#   3: full minus bottom 2x2 notch     -> 4
_FULL = {(r, c) for r in range(4) for c in range(4)}
_SHAPES = [
    set(_FULL),                                                   # o1
    {(r, c) for (r, c) in _FULL if r in (0, 3) or c in (0, 3)},   # o2 (box)
    _FULL - {(1, 0), (2, 0), (1, 3), (2, 3)},                     # o3
    _FULL - {(2, 1), (2, 2), (3, 1), (3, 2)},                     # o4
]
_LABELS = [2, 8, 3, 4]

# every variant contains all four shape types (and its first 4 entries do too,
# so a max_w-driven truncation still teaches the full shape->colour mapping)
VARIANTS = [
    {"shapes": [0, 1, 2, 3]},
    {"shapes": [3, 2, 1, 0]},
    {"shapes": [1, 3, 0, 2, 1]},
    {"shapes": [2, 0, 3, 1, 2, 0]},
]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, shapes=None) -> dict:
    max_num = max(1, min((max_w + 1) // 5, max_h, 6))
    if shapes is None:
        n = random.randint(1, max_num)
        shapes = [random.randrange(4) for _ in range(n)]
    else:
        shapes = list(shapes)[:max_num]
    num = len(shapes)

    h = 4
    w = 5 * num - 1
    remcols = [c for c in range(10) if c != bgc]

    gi = [[bgc] * w for _ in range(h)]
    for k, si in enumerate(shapes):
        col = random.choice(remcols)
        for (r, c) in _SHAPES[si]:
            gi[r][c + 5 * k] = col

    go = [[_LABELS[si]] * num for si in shapes]
    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # blocks are 4x4 patches at column stride 5
    num = (wi + 1) // 5

    # classify each block patch straight out of I -> its label colour
    labels = []
    for k in range(num):
        P = I[0:4, 5 * k:5 * k + 4]
        if len(set(P.flatten().tolist())) == 1:          # solid block
            lab = 2
        elif not np.array_equal(P, P[::-1, :]):          # not up/down symmetric
            lab = 4
        elif int(P[0, 0]) != int(P[1, 0]):               # top-left edge broken
            lab = 3
        else:                                            # hollow box
            lab = 8
        labels.append(lab)

    ops, sels = [], []

    # canvas needs one row per block; grow it when there are more blocks than rows
    if num > hi:
        ops.append(33)
        sels.append([0, 0, num - 1, wi - 1])

    # each block becomes one solid row of its label colour
    for k, lab in enumerate(labels):
        ops.append(lab)
        sels.append([k, 0, 0, num - 1])

    # keep only the num x num answer block
    ops.append(33)
    sels.append([0, 0, num - 1, num - 1])

    ops.append(34)
    sels.append([0, 0, num - 1, num - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 995c5fa3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 995c5fa3"
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
                                f"for task 995c5fa3"
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
                    f"Failed to build a complete episode for task 995c5fa3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"995c5fa3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
