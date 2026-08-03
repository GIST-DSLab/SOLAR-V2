"""
ARC Task: 1bfc4729 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    acol = random.choice(remcols)
    remcols = [c for c in remcols if c != acol]
    bcol = random.choice(remcols)
    return {"bgc": bgc, "acol": acol, "bcol": bcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol) -> dict:
    from random import randint, choice

    def unifint(lb, ub, rng):
        a, b = rng
        span = b - a
        lo = a + int(span * lb)
        hi = a + int(span * ub)
        if hi < lo:
            hi = lo
        return randint(lo, hi)

    max_h = max(4, int(max_h))
    max_w = max(4, int(max_w))

    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    if h % 2 == 1:
        h = choice((max(4, h - 1), min(max_h, h + 1)))
    if h % 2 == 1:
        h = max(4, h - 1)

    alocj = unifint(diff_lb, diff_ub, (w // 2, w - 1))
    if choice((True, False)):
        alocj = max(min(w // 2, alocj - w // 2), 1)
    aloci = randint(1, h // 2 - 1)
    blocj = unifint(diff_lb, diff_ub, (w // 2, w - 1))
    if choice((True, False)):
        blocj = max(min(w // 2, blocj - w // 2), 1)
    bloci = randint(h // 2, h - 2)

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    gi[aloci][alocj] = acol
    gi[bloci][blocj] = bcol

    go = [row[:] for row in gi]
    # horizontal frontier through each marker
    for c in range(w):
        go[aloci][c] = acol
        go[bloci][c] = bcol
    # top / bottom border rows
    for c in range(w):
        go[0][c] = acol
        go[h - 1][c] = bcol
    # side columns: top half acol, bottom half bcol
    for r in range(h // 2):
        go[r][0] = acol
        go[r][w - 1] = acol
    for r in range(h // 2, h):
        go[r][0] = bcol
        go[r][w - 1] = bcol

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background = the canvas colour the generator paints before placing 2 markers
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # measure the two markers FROM I: position and colour
    marks = [(r, c, int(I[r, c])) for r in range(h) for c in range(w) if I[r, c] != bgc]
    half = h // 2
    top = [m for m in marks if m[0] < half]
    bot = [m for m in marks if m[0] >= half]
    if not top or not bot:
        # degenerate fallback: recover roles from O's border rows
        acol = int(O[0, 0])
        bcol = int(O[h - 1, 0])
        ar = next((r for r in range(1, half) if all(O[r, c] == acol for c in range(w))), 1)
        br = next((r for r in range(half, h - 1) if all(O[r, c] == bcol for c in range(w))), h - 2)
    else:
        ar, _ac, acol = top[0]
        br, _bc, bcol = bot[0]

    ops, sels = [], []

    # 1. horizontal frontier through the top marker, in its own colour
    ops.append(acol)
    sels.append(sel_of([(ar, c) for c in range(w)]))
    # 2. horizontal frontier through the bottom marker, in its own colour
    ops.append(bcol)
    sels.append(sel_of([(br, c) for c in range(w)]))
    # 3. top border row belongs to the top marker
    ops.append(acol)
    sels.append(sel_of([(0, c) for c in range(w)]))
    # 4. bottom border row belongs to the bottom marker
    ops.append(bcol)
    sels.append(sel_of([(h - 1, c) for c in range(w)]))
    # 5-6. left/right side columns, upper half -> top marker's colour
    ops.append(acol)
    sels.append(sel_of([(r, 0) for r in range(half)]))
    ops.append(acol)
    sels.append(sel_of([(r, w - 1) for r in range(half)]))
    # 7-8. left/right side columns, lower half -> bottom marker's colour
    ops.append(bcol)
    sels.append(sel_of([(r, 0) for r in range(half, h)]))
    ops.append(bcol)
    sels.append(sel_of([(r, w - 1) for r in range(half, h)]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1bfc4729"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1bfc4729"
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
                                f"for task 1bfc4729"
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
                    f"Failed to build a complete episode for task 1bfc4729 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1bfc4729-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
