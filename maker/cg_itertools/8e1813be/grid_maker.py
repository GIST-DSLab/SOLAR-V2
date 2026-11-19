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
from collections import Counter
from maker.sel_helpers import sel_of

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, a + round(d * (b - a))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, transpose=None, **kwargs) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    lim = min(max_h, max_w, 30)
    remcols = [c for c in range(10) if c != bgc and c != sqc]
    nb_hi = max(3, min(8, lim // 3))
    nbars = _unifint(diff_lb, diff_ub, (3, nb_hi))
    ccols = random.sample(remcols, nbars)
    w = _unifint(diff_lb, diff_ub, (nbars + 3, lim))
    hmarg = _unifint(diff_lb, diff_ub, (2 * nbars, max(2 * nbars, lim - nbars)))
    go = [[col] * nbars for col in ccols]
    gi = [[col] * w for col in ccols]
    for _ in range(hmarg):
        idx = random.randint(0, len(gi) - 1)
        gi = gi[:idx] + [[bgc] * w] + gi[idx:]
    gi = np.array(gi, dtype=int)
    h2 = nbars + hmarg
    oh = ow = nbars
    loci = random.randint(1, h2 - oh - 2)
    locj = random.randint(1, w - ow - 2)
    gi[loci - 1:loci + oh + 1, locj - 1:locj + ow + 1] = bgc
    gi[loci:loci + oh, locj:locj + ow] = sqc
    go = np.array(go, dtype=int)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Golf reading (differs from the maker's): the unit is ONE edge line -- the far row
    (bars vertical) or far column (bars horizontal). The square's cleared border never
    reaches it, so it crosses every bar once, in order, with background in between.
    The answer is that line with the background squeezed out, repeated n times.
    Ops:
      1. CropGrid to that whole edge line.
      2. Slide each bar cell toward the start of the line, in bar order (grab once,
         then empty-selection Moves). Bars already in their slot are skipped. A
         colour-0 bar cannot be grabbed, because ARCLE objects keep only nonzero
         cells, so its slot is painted with Color0 instead, and only if the slot
         is not already 0. Vacated cells inside the first n get overwritten by later
         bars; the rest get cropped away, so no repair is needed.
      3. Resize the canvas to n x n.
      4. CopyO the squeezed line (this changes only the clipboard), then Paste it into
         every other row/column. Zero cells are already 0 there, so the transparent
         paste is exact.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # bars vertical <=> the last row crosses several distinct bar colours
    vertical = len(set(I[-1, :].tolist()) - {bgc}) >= 2
    line = I[-1, :] if vertical else I[:, -1]
    L = len(line)
    bars = [(p, int(line[p])) for p in range(L) if line[p] != bgc]
    n = len(bars)

    def cell(p):
        return (0, p) if vertical else (p, 0)

    ops, sels = [], []
    # 1. keep only the far edge line (a whole row / column of I)
    if vertical:
        ops.append(33); sels.append([hi - 1, 0, 0, wi - 1])
    else:
        ops.append(33); sels.append([0, wi - 1, hi - 1, 0])
    cur = [int(v) for v in line]

    # 2. squeeze bars together toward the start of the line, in their order
    step_op = 23 if vertical else 20          # MoveL along a row / MoveU along a column
    for k, (p, col) in enumerate(bars):
        if p == k:
            continue
        if col == 0:
            if cur[k] != 0:
                ops.append(0); sels.append(sel_of([cell(k)]))
                cur[k] = 0
            continue
        ops.append(step_op); sels.append(sel_of([cell(p)]))     # grab the bar cell
        for _ in range(p - k - 1):
            ops.append(step_op); sels.append(sel_of([]))         # keep sliding same object
        cur[p] = 0
        cur[k] = col

    # 3. size the canvas to the n x n answer block
    ops.append(33); sels.append([0, 0, n - 1, n - 1])

    # 4. copy the squeezed line and repeat it across the block
    if vertical:
        ops.append(29); sels.append([0, 0, 0, n - 1])
        for r in range(1, n):
            ops.append(30); sels.append([r, 0, 0, 0])
    else:
        ops.append(29); sels.append([0, 0, n - 1, 0])
        for c in range(1, n):
            ops.append(30); sels.append([0, c, 0, 0])

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
