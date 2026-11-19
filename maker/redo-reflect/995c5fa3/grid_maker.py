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


# The four 4x4 stamps the generator uses, in the order (object, output colour)
# o1 = solid square            -> 2   (a single colour)
# o2 = box / hollow border     -> 8   (nothing special holds)
# o3 = square minus side dents -> 3   (top-left cell differs from the one below)
# o4 = square minus a 2x2 bite -> 4   (the only stamp that is NOT up-down symmetric)
def _stamps():
    o1 = [(r, c) for r in range(4) for c in range(4)]
    o2 = [(r, c) for (r, c) in o1 if r in (0, 3) or c in (0, 3)]
    o3 = [(r, c) for (r, c) in o1 if (r, c) not in {(1, 0), (2, 0), (1, 3), (2, 3)}]
    o4 = [(r, c) for (r, c) in o1 if (r, c) not in {(2, 1), (2, 2), (3, 1), (3, 2)}]
    return [(o1, 2), (o2, 8), (o3, 3), (o4, 4)]


def sample_colors(num_examples=None) -> dict:
    # Only the background is a fixed role: the rule reads SHAPE, not stamp colour,
    # so per-block colours may stay random.  What must be planned is the set of
    # stamp types, so every shape -> colour mapping the test can show is demoed.
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3

    order = [0, 1, 2, 3]
    random.shuffle(order)
    groups = [[] for _ in range(max(1, n_ex))]
    for i, s in enumerate(order):
        groups[i % len(groups)].append(s)

    plan = []
    for g in groups:
        shapes = list(g)
        random.shuffle(shapes)
        plan.append({"shapes": shapes})
    seen = plan[random.randrange(len(plan))]
    plan.append({"shapes": list(seen["shapes"])})       # test reuses a demoed combo
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, shapes=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    mpr = _stamps()
    cols = [c for c in range(10) if c != bgc]

    # input is 4 x (5*num - 1); output is num x num
    num_ub = max(1, min(6, (max_w + 1) // 5, max_h, max_w))
    num = unifint(diff_lb, diff_ub, (1, num_ub))

    plan = [s for s in (shapes or []) if isinstance(s, int) and 0 <= s < 4]
    if len(plan) > num:
        num = min(len(plan), num_ub)
    plan = plan[:num]
    while len(plan) < num:
        plan.append(random.randrange(4))
    random.shuffle(plan)

    h, w = 4, 5 * num - 1
    gi = [[bgc] * w for _ in range(h)]
    ccols = []
    for k, s in enumerate(plan):
        obj, outcol = mpr[s]
        col = random.choice(cols)
        for (r, c) in obj:
            gi[r][5 * k + c] = col
        ccols.append(outcol)

    go = [[c] * num for c in ccols]
    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    n = (wi + 1) // 5                      # number of 4x4 stamps, read from I's width

    # --- score every stamp, left to right, straight from the input -------------
    labels = []
    for k in range(n):
        B = I[0:4, 5 * k:5 * k + 4]
        if len(set(B.flatten().tolist())) == 1:
            lab = 2                        # one colour -> solid square
        elif not np.array_equal(B, np.flipud(B)):
            lab = 4                        # not up-down symmetric -> bitten square
        elif B[0, 0] != B[1, 0]:
            lab = 3                        # dented sides
        else:
            lab = 8                        # box
        labels.append(lab)

    ops, sels = [], []

    # 1. grow the canvas by n scratch rows under the strip of stamps.
    #    selection is exactly this full rectangle (whole new canvas).
    ops.append(33); sels.append([0, 0, hi + n - 1, wi - 1])

    # 2. write each stamp's score into the scratch row, in stamp order.
    for k, lab in enumerate(labels):
        ops.append(lab); sels.append([hi, k, 0, 0])     # exactly this one cell

    if n > 1:
        # 3. the stamps run left-to-right but the answer stacks them top-to-bottom:
        #    rotate the score run a quarter turn clockwise so it stands up.
        #    selection is exactly the n x n scratch square (square -> rotate is exact).
        ops.append(25); sels.append([hi, 0, n - 1, n - 1])
        # scores now occupy column n-1 of that square, in the same order.

        # 4. each score fills a whole output row: repeat the score column across.
        ops.append(29); sels.append([hi, n - 1, n - 1, 0])   # exactly that column
        for c in range(n - 1):
            ops.append(30); sels.append([hi, c, 0, 0])       # paste origin

    # 5. keep only the answer square.
    ops.append(33); sels.append([hi, 0, n - 1, n - 1])
    ops.append(34); sels.append([0, 0, n - 1, n - 1])
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
