"""
ARC Task: 99b1bc43 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors ----
# Discrete structural variant: the generator flips a coin and dmirrors the whole
# instance, so the two source grids are either side-by-side (vertical separator
# line) or stacked (horizontal separator line).  Both cases must be demonstrated.
VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]          # 3 is reserved for the answer
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)                         # separator line colour
    rem = [c for c in rem if c != linc]
    acol = random.choice(rem)                         # marks of the first grid
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)                         # marks of the second grid

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "linc": linc, "acol": acol, "bcol": bcol,
            "instance_plan": plan}


# ------------------------------------------------------------- generator ----
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, linc=None, acol=None, bcol=None, transpose=None) -> dict:
    if transpose is None:
        transpose = choice((True, False))

    # the input is h x (2w+1), transposed to (2w+1) x h when mirrored
    if transpose:
        h_ub = max(2, min(30, max_w))
        w_ub = max(2, min(14, (max_h - 1) // 2))
    else:
        h_ub = max(2, min(30, max_h))
        w_ub = max(2, min(14, (max_w - 1) // 2))

    h = unifint(diff_lb, diff_ub, (2, h_ub))
    w = unifint(diff_lb, diff_ub, (2, w_ub))

    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    res = (set(bset) - set(aset)) | (set(aset) - set(bset))
    go = fill(c, 3, res)
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# ------------------------------------------------------------- operations ---
def derive_operations(I, O, examples=None):
    """
    The input holds two same-sized grids separated by a solid line.  The answer
    keeps a mark (3) exactly where one grid is marked and the other is not.

    Account of the construction:
      1. the marks that only the SECOND grid has are turned into answer marks,
      2. that whole set of marks TRAVELS across the separator onto the first
         grid (one grab, then plain one-cell steps),
      3. the marks that only the FIRST grid has become answer marks too,
      4. the marks the two grids share are wiped back to the background,
      5. the canvas shrinks to the first grid, which now carries the answer.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # ---- locate the separator line and the shared background colour --------
    def common_bg(P, Q, linecol):
        pp = set(P.flatten().tolist())
        qq = set(Q.flatten().tolist())
        inter = pp & qq
        if len(inter) != 1:
            return None
        bg = inter.pop()
        return None if bg == linecol else bg

    mode, bgc, m = None, None, None
    if wi % 2 == 1:
        mc = wi // 2
        col = I[:, mc]
        if len(set(col.tolist())) == 1:
            bg = common_bg(I[:, :mc], I[:, mc + 1:], int(col[0]))
            if bg is not None:
                mode, bgc, m = 'v', bg, mc
    if mode is None and hi % 2 == 1:
        mr = hi // 2
        row = I[mr]
        if len(set(row.tolist())) == 1:
            bg = common_bg(I[:mr], I[mr + 1:], int(row[0]))
            if bg is not None:
                mode, bgc, m = 'h', bg, mr
    if mode is None:                                   # defensive fallback
        ho, wo = O.shape
        if wi == 2 * wo + 1:
            mode, m = 'v', wo
        else:
            mode, m = 'h', ho
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    if mode == 'v':                                    # first grid on the left
        h, w = hi, m
        A = I[:, :w]
        B = I[:, w + 1: 2 * w + 1]
        b_to_grid = lambda r, c: (r, c + w + 1)
        move_op, steps = 23, w + 1                     # MoveL
    else:                                              # first grid on top
        h, w = m, wi
        A = I[:h]
        B = I[h + 1: 2 * h + 1]
        b_to_grid = lambda r, c: (r + h + 1, c)
        move_op, steps = 20, h + 1                     # MoveU

    amark = {(r, c) for r in range(h) for c in range(w) if A[r, c] != bgc}
    bmark = {(r, c) for r in range(h) for c in range(w) if B[r, c] != bgc}
    a_only = sorted(amark - bmark)
    b_only = sorted(bmark - amark)
    shared = sorted(amark & bmark)

    # 1. the second grid's own marks become answer marks (still in their half)
    if b_only:
        src = [b_to_grid(r, c) for (r, c) in b_only]
        ops.append(3); sels.append(sel_of(src))

        # 2. they travel over the separator onto the first grid: grab once,
        #    then keep stepping with an empty selection so ARCLE restores the
        #    path (separator line included) behind them.
        ops.append(move_op); sels.append(sel_of(src))
        for _ in range(steps - 1):
            ops.append(move_op); sels.append(sel_of([]))
        # the vacated footprint stays in the second half, which is cropped away

    # 3. the first grid's own marks become answer marks
    if a_only:
        ops.append(3); sels.append(sel_of(list(a_only)))

    # 4. the marks both grids agree on are wiped back to the background
    if shared:
        ops.append(bgc); sels.append(sel_of(list(shared)))

    # 5. the canvas shrinks to the first grid (full rectangle, background
    #    included -> bbox selection is the right form here)
    ops.append(33); sels.append([0, 0, h - 1, w - 1])
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 99b1bc43"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 99b1bc43"
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
                                f"for task 99b1bc43"
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
                    f"Failed to build a complete episode for task 99b1bc43 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"99b1bc43-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
