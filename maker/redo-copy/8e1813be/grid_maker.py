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
import random
import numpy as np
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- helpers
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        hi = lo
    if lo < a:
        lo = a
    if hi > b:
        hi = b
    return random.randint(lo, hi)


# the one discrete structural variant of this task: bars run in rows, or in columns
VARIANTS = [{"transpose": False}, {"transpose": True}]


# ---------------------------------------------------------------- 1. colors
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    # bar colours never 0: ARCLE's object ops ignore 0-valued cells, so a 0 bar
    # could not be Moved.  bgc / sqc may still be 0.
    barpool = [c for c in cols if c not in (bgc, sqc) and c != 0]

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "barpool": barpool, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, barpool, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])

    # build in "horizontal bars" orientation, then transpose at the end
    mh = max_w if transpose else max_h
    mw = max_h if transpose else max_w

    nb_hi = min(8, len(barpool), mh // 3, mw - 3)
    nb_hi = max(3, nb_hi)
    nbars = _unifint(diff_lb, diff_ub, (3, nb_hi))
    nbars = max(3, min(nbars, len(barpool)))

    ccols = random.sample(list(barpool), nbars)

    w = _unifint(diff_lb, diff_ub, (nbars + 3, max(nbars + 3, mw)))
    hmarg = _unifint(diff_lb, diff_ub, (2 * nbars, max(2 * nbars, mh - nbars)))

    rows = [[c] * w for c in ccols]
    for _ in range(hmarg):
        idx = random.randint(0, nbars - 1)
        rows.insert(idx, [bgc] * w)
    h2 = nbars + hmarg

    oh = ow = nbars
    loci = random.randint(1, max(1, h2 - oh - 2))
    locj = random.randint(1, max(1, w - ow - 2))

    # background ring around the square, then the square itself
    for r in range(loci - 1, loci + oh + 1):
        for c in range(locj - 1, locj + ow + 1):
            if 0 <= r < h2 and 0 <= c < w:
                rows[r][c] = bgc
    for r in range(loci, loci + oh):
        for c in range(locj, locj + ow):
            rows[r][c] = sqc

    go = [[c] * nbars for c in ccols]
    gi = rows
    if transpose:
        gi = [list(t) for t in zip(*gi)]
        go = [list(t) for t in zip(*go)]

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# ---------------------------------------------------------------- 3. derive
def derive_operations(I, O):
    """
    Rule (measured from I only):
      * the "bars" are the colours whose cells all lie in a single row (or in a
        single column) - they are full-length lines crossing the canvas.
      * n = number of bars.  The answer is the n bars, in their reading order,
        packed side by side into an n x n block.
    Trajectory: complete each bar across the n-wide window that will be kept
    (the square/its border interrupt some bars), slide the bars together with
    unit Moves so they form a solid n x n block, then crop that block.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    horiz, vert = [], []
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        if r1 == r0:
            horiz.append((r0, int(col)))
        elif c1 == c0:
            vert.append((c0, int(col)))

    n = len(horiz) + len(vert)
    horizontal = len(horiz) > len(vert)

    ops, sels = [], []

    if horizontal:
        bars = sorted(horiz)                      # (row, colour), top -> bottom

        # pick the n-wide column window the bars are most intact in
        best = None
        for C in range(0, w - n + 1):
            holes = 0
            for (r, col) in bars:
                holes += int(np.sum(I[r, C:C + n] != col))
            if best is None or holes < best[0]:
                best = (holes, C)
        C = best[1]

        # complete every bar across that window
        for (r, col) in bars:
            cells = [(r, c) for c in range(C, C + n) if I[r, c] != col]
            if cells:
                ops.append(int(col))
                sels.append(sel_of(cells))

        # slide the bars up until they are stacked directly under the first one
        base = bars[0][0]
        for i, (r, col) in enumerate(bars):
            steps = r - (base + i)
            if steps <= 0:
                continue
            cells = sorted(
                set((r, c) for c in range(w) if I[r, c] == col)
                | set((r, c) for c in range(C, C + n))
            )
            ops.append(20)                        # MoveU - grabs this bar
            sels.append(sel_of(cells))
            for _ in range(steps - 1):
                ops.append(20)                    # keep the same bar gliding
                sels.append(sel_of([]))

        # keep the solid block (full rectangle -> bbox selection is exact)
        ops.append(33)
        sels.append([base, C, n - 1, n - 1])

    else:
        bars = sorted(vert)                       # (col, colour), left -> right

        best = None
        for R in range(0, h - n + 1):
            holes = 0
            for (c, col) in bars:
                holes += int(np.sum(I[R:R + n, c] != col))
            if best is None or holes < best[0]:
                best = (holes, R)
        R = best[1]

        for (c, col) in bars:
            cells = [(r, c) for r in range(R, R + n) if I[r, c] != col]
            if cells:
                ops.append(int(col))
                sels.append(sel_of(cells))

        base = bars[0][0]
        for i, (c, col) in enumerate(bars):
            steps = c - (base + i)
            if steps <= 0:
                continue
            cells = sorted(
                set((r, c) for r in range(h) if I[r, c] == col)
                | set((r, c) for r in range(R, R + n))
            )
            ops.append(23)                        # MoveL - grabs this bar
            sels.append(sel_of(cells))
            for _ in range(steps - 1):
                ops.append(23)
                sels.append(sel_of([]))

        ops.append(33)
        sels.append([R, base, n - 1, n - 1])

    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])
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
