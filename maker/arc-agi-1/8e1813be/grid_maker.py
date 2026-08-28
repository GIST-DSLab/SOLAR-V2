"""
ARC Task: 8e1813be (RE-ARC) — LLM-generated grid_maker
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
import math
import random
import numpy as np

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback with the documented mask format
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# ---------------------------------------------------------------- helpers ---
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = int(math.ceil(a + (b - a) * diff_lb))
    hi = int(math.floor(a + (b - a) * diff_ub))
    lo = max(a, min(lo, b))
    hi = max(a, min(hi, b))
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


# the one discrete structural variant of this task: whether the whole instance
# is diagonally mirrored (bars run as columns instead of rows)
VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    rem = [c for c in cols if c not in (bgc, sqc)]
    random.shuffle(rem)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "colorder": rem, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, colorder,
             transposed=None, **kwargs) -> dict:
    if transposed is None:
        transposed = random.choice([True, False])

    # dimensions of the grid BEFORE the optional diagonal mirror
    H = max_w if transposed else max_h
    W = max_h if transposed else max_w

    hi_n = min(8, H // 3, W - 3, len(colorder))
    if hi_n < 3:
        raise ValueError("grid bounds too small for this task")
    nbars = _unifint(diff_lb, diff_ub, (3, hi_n))
    ccols = list(colorder[:nbars])

    w = _unifint(diff_lb, diff_ub, (nbars + 3, W))
    hmarg = _unifint(diff_lb, diff_ub, (2 * nbars, H - nbars))

    gi = [[c] * w for c in ccols]
    bgrow = [bgc] * w
    for _ in range(hmarg):
        idx = random.randint(0, nbars - 1)
        gi = gi[:idx] + [list(bgrow)] + gi[idx:]
    h2 = nbars + hmarg

    loci = random.randint(1, h2 - nbars - 2)
    locj = random.randint(1, w - nbars - 2)
    # the marker square (size == number of bars) ...
    for i in range(loci, loci + nbars):
        for j in range(locj, locj + nbars):
            gi[i][j] = sqc
    # ... isolated by a background ring
    for i in range(loci - 1, loci + nbars + 1):
        for j in range(locj - 1, locj + nbars + 1):
            if i in (loci - 1, loci + nbars) or j in (locj - 1, locj + nbars):
                gi[i][j] = bgc

    go = [[c] * nbars for c in ccols]

    if transposed:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    """
    Rule: the isolated square marks the answer canvas (its side == number of bars).
    The bar colours, taken in order, are written into that canvas as its columns
    (that is the colour list, repeated).  When the bars run horizontally the
    canvas is then reflected across its diagonal so the stripes lie along the
    bars -- the reflection is performed here (Rotate90 + FlipV == transpose),
    and without it the submitted grid is wrong.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    n = int(O.shape[0])
    ops, sels = [], []

    # orientation of the answer's stripes
    rows_const = all(len(set(O[i].tolist())) == 1 for i in range(O.shape[0]))
    if rows_const:
        # answer row i == colour i  ->  pre-reflection canvas has column i == colour i
        seq = [int(O[i, 0]) for i in range(n)]
        need_reflect = True
    else:
        seq = [int(O[0, j]) for j in range(O.shape[1])]
        need_reflect = False

    # locate the marker square: the only colour forming a solid n x n block
    sr, sc = 0, 0
    barset = set(seq)
    for c in np.unique(I):
        c = int(c)
        if c in barset:
            continue
        cells = np.argwhere(I == c)
        if len(cells) != n * n:
            continue
        r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
        r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
        if r1 - r0 + 1 == n and c1 - c0 + 1 == n:
            sr, sc = r0, c0
            break

    # write the bar colours, in bar order, as the columns of the marker square
    for j in range(n):
        col_cells = [(sr + i, sc + j) for i in range(n)]
        if all(int(I[r, c]) == seq[j] for r, c in col_cells):
            continue  # already this colour -> the op would do nothing
        ops.append(seq[j])
        sels.append(sel_of(col_cells))

    square_cells = [(sr + i, sc + j) for i in range(n) for j in range(n)]
    # crop the canvas down to the marker square (selection IS exactly that block)
    ops.append(33)
    sels.append(sel_of(square_cells))

    full = [(i, j) for i in range(n) for j in range(n)]
    if need_reflect:
        # diagonal reflection of the whole canvas = Rotate90 (CCW) then FlipV
        ops.append(24)
        sels.append(sel_of(full))
        ops.append(27)
        sels.append(sel_of(full))

    ops.append(34)
    sels.append(sel_of(full))
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
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
                                f"for task 8e1813be"
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
                    f"Failed to build a complete episode for task 8e1813be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e1813be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
