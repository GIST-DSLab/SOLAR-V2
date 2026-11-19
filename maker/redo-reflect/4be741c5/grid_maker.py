"""
ARC Task: 4be741c5 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    # The rule has two discrete structural cases (the generator's final coin-flip dmirror):
    #   transposed=False -> stripes run down columns, answer is a 1 x n row
    #   transposed=True  -> stripes run across rows,   answer is an n x 1 column
    # Both must be shown in the examples, so the episode is learnable.
    VARIANTS = [{"transposed": False}, {"transposed": True}]
    palette = random.sample(range(10), 4)          # fixed stripe palette for the whole episode
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test variant is one that was shown
    return {"ccols": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, ccols=None, transposed=None, **kwargs) -> dict:
    import random

    def unifint(lb, ub):
        if ub < lb:
            ub = lb
        return random.randint(lb + int((ub - lb) * diff_lb), lb + int((ub - lb) * diff_ub))

    if transposed is None:
        transposed = random.choice((True, False))
    if ccols is None:
        ccols = random.sample(range(10), 4)

    # grid is built as h rows x w cols, and transposed at the end when transposed=True
    rowcap = max_w if transposed else max_h
    colcap = max_h if transposed else max_w
    h = unifint(4, max(4, rowcap))
    w = unifint(6, max(6, colcap))
    numcolors = unifint(2, max(2, min(len(ccols), w // 3)))
    cc = list(ccols)[:numcolors]

    rows = [[c] * h for c in cc for _ in range(3)]
    while len(rows) < w:
        idx = random.randint(0, len(rows) - 1)
        rows = rows[:idx] + [list(rows[idx])] + rows[idx:]
    gi = [[rows[j][i] for j in range(w)] for i in range(h)]   # dmirror -> vertical stripes

    ndisturbances = unifint(0, 3 * h * numcolors)
    for _ in range(ndisturbances):
        options = []
        for a in range(h):
            for b in range(w - 3):
                if gi[a][b] == gi[a][b + 1] and gi[a][b + 2] == gi[a][b + 3]:
                    options.append((a, b, gi[a][b], gi[a][b + 2]))
        if len(options) == 0:
            break
        a, b, c1, c2 = random.choice(options)
        if random.choice((True, False)):
            gi[a][b + 1] = c2
        else:
            gi[a][b + 2] = c1

    go = [list(cc)]
    if transposed:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O):
    import numpy as np
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    def dedupe(seq):
        out = []
        for v in seq:
            if not out or out[-1] != int(v):
                out.append(int(v))
        return out

    row0 = [int(v) for v in I[0].tolist()]
    col0 = [int(v) for v in I[:, 0].tolist()]

    # Column 0 is never disturbed by the generator, so a uniform first row means the
    # stripes run ACROSS rows; then the stripe sequence is read down column 0 and the
    # answer is a column (the verifier's dmirror branch). Otherwise stripes are vertical,
    # the sequence is read along row 0, and the answer is a row.
    horizontal = (len(set(row0)) == 1)
    colors = dedupe(col0) if horizontal else dedupe(row0)
    n = len(colors)

    ops, sels = [], []

    # Shrink the canvas to a 1 x n strip at the top-left: one cell per stripe.
    # Whole rectangle is intended -> bbox selection.
    ops.append(33); sels.append([0, 0, 0, n - 1])

    # Write the stripe sequence into that strip, one stripe at a time, left to right.
    cur = [(row0[j] if j < wi else 0) for j in range(n)]
    for j in range(n):
        if cur[j] != colors[j]:
            ops.append(colors[j]); sels.append(sel_of([(0, j)]))

    if horizontal:
        # The stripes ran across rows, so the finished strip must be mirrored across the
        # main diagonal (dmirror) to lie along the other axis: rot90 CCW + flip up/down.
        ops.append(33); sels.append([0, 0, n - 1, n - 1])   # pad canvas to n x n (rotate needs a square) - whole rect
        ops.append(24); sels.append([0, 0, n - 1, n - 1])   # Rotate90 (CCW) on the whole square
        ops.append(27); sels.append([0, 0, n - 1, n - 1])   # FlipV on the whole square -> together = diagonal mirror
        ops.append(33); sels.append([0, 0, n - 1, 0])       # crop down to the n x 1 column
        ops.append(34); sels.append([0, 0, n - 1, 0])
    else:
        ops.append(34); sels.append([0, 0, 0, n - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 4be741c5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4be741c5"
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
                                f"for task 4be741c5"
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
                    f"Failed to build a complete episode for task 4be741c5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4be741c5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
