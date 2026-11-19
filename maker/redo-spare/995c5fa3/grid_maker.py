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
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- 1. colors
# The four 4x4 block shapes (generator's `mpr`) are discrete structural
# variants, each mapping to a fixed label colour:
#   0 -> full square      -> 2
#   1 -> box (perimeter)  -> 8
#   2 -> full minus (1,0),(2,0),(1,3),(2,3) -> 3
#   3 -> full minus 2x2 at (2,1)            -> 4
# Every variant must be demonstrated, so the shape assignment is planned
# per instance up-front and merged into generate() by the caller.
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    n_ex = num_examples if num_examples else 3

    order = [0, 1, 2, 3]
    random.shuffle(order)
    chunks = [[] for _ in range(max(1, n_ex))]
    for i, s in enumerate(order):
        chunks[i % len(chunks)].append(s)

    examples = []
    for ch in chunks:
        shapes = list(ch)
        random.shuffle(shapes)
        extra = [random.randrange(4) for _ in range(6 - len(shapes))]
        examples.append({"shapes": shapes + extra,
                         "min_num": max(1, len(shapes))})
    random.shuffle(examples)

    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, shapes=None, min_num=1) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)

    o1 = asindices(canvas(-1, (4, 4)))
    o2 = box(asindices(canvas(-1, (4, 4))))
    o3 = asindices(canvas(-1, (4, 4))) - {(1, 0), (2, 0), (1, 3), (2, 3)}
    o4 = o1 - shift(asindices(canvas(-1, (2, 2))), (2, 1))
    mpr = [(o1, 2), (o2, 8), (o3, 3), (o4, 4)]

    # input is 4 x (5*num - 1), output is num x num
    num_cap = min(6, (max_w + 1) // 5, max_h, max_w)
    num_cap = max(1, num_cap)
    num = unifint(diff_lb, diff_ub, (1, num_cap))
    num = max(num, min(max(1, int(min_num)), num_cap))

    h = 4
    w = 4 * num + num - 1
    remcols = [c for c in cols if c != bgc]
    gi = canvas(bgc, (h, w))
    ccols = []
    for k in range(num):
        col = choice(remcols)
        if shapes:
            obj, outcol = mpr[shapes[k % len(shapes)] % 4]
        else:
            obj, outcol = choice(mpr)
        locj = 5 * k
        gi = fill(gi, col, shift(obj, (0, locj)))
        ccols.append(outcol)
    go = tuple(repeat(c, num) for c in ccols)
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- 3. ops
def _classify_block(B):
    """Label colour of one 4x4 block, read from the block alone."""
    B = np.asarray(B, dtype=int)
    if len(set(B.flatten().tolist())) == 1:
        return 2                       # solid square
    if not np.array_equal(B, B[::-1]):
        return 4                       # not up/down symmetric (2x2 bite at bottom)
    if B[0, 0] != B[1, 0]:
        return 3                       # side notches
    return 8                           # box


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    num = (wi + 1) // 5                # number of 4x4 blocks along the row
    labels = [_classify_block(I[0:4, 5 * k:5 * k + 4]) for k in range(num)]

    ops, sels = [], []

    # Size the canvas ONCE to the answer's shape (num x num).
    # Full-rectangle bbox: a canvas resize acts on the whole rectangle.
    ops.append(33); sels.append([0, 0, num - 1, num - 1])

    # Track the grid as it now stands (transparent copy of I's top-left corner,
    # zero padding below row 3 when num > 4).
    cur = np.zeros((num, num), dtype=int)
    for r in range(min(num, hi)):
        for c in range(min(num, wi)):
            cur[r, c] = I[r, c]

    # One block -> one solid row, in block order.
    for k in range(num):
        lab = labels[k]
        if np.all(cur[k, :] == lab):
            continue                   # row already holds this colour: no visible change
        ops.append(int(lab))
        sels.append(sel_of([(k, c) for c in range(num)]))
        cur[k, :] = lab

    ops.append(34); sels.append([0, 0, num - 1, num - 1])
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
