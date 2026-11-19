"""
ARC Task: 94f9d214 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# The task: the input is TWO panels glued together (side by side, or — when the
# whole thing is transposed — stacked).  Each panel is the same size, drawn on a
# shared background colour `bgc`; panel A is speckled with `acol`, panel B with
# `bcol`.  The answer is one panel-sized grid: colour 2 exactly where BOTH panels
# are blank, background everywhere else.
#
# Account used by derive_operations (no raster scan):
#   1. Color2  on panel B's blank cells          -> B's blanks become marks
#   2. MoveL/MoveU x panel-width/height          -> panel B's marks SLIDE onto A
#                                                   (transparent composite: they
#                                                    land on top of A's cells)
#   3. Color<bgc> on panel A's speckles          -> erase A's marks, which wipes
#                                                   the marks that landed on them
#   4. ResizeGrid to panel A                     -> keep just the one panel
# ----------------------------------------------------------------------------

VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    rem2 = [c for c in rem if c != acol]
    bcol = random.choice(rem2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, acol=None, bcol=None, mirror=None) -> dict:
    if mirror is None:
        mirror = random.choice([True, False])
    if bgc is None or acol is None or bcol is None:
        cols_ = [c for c in range(10) if c != 2]
        bgc = random.choice(cols_)
        rem_ = [c for c in cols_ if c != bgc]
        acol = random.choice(rem_)
        bcol = random.choice([c for c in rem_ if c != acol])

    # panel grid is (h, w); glued grid is (h, 2w), transposed to (2w, h) if mirror
    if mirror:
        hmax = max(2, min(30, int(max_w)))
        wmax = max(2, min(14, int(max_h) // 2))
    else:
        hmax = max(2, min(30, int(max_h)))
        wmax = max(2, min(14, int(max_w) // 2))

    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))

    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
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
    gi = hconcat(A, B)
    res = (set(inds) - set(aset)) & (set(inds) - set(bset))
    go = fill(c, 2, res)
    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # ---- find how the two panels are glued together -------------------------
    # A panel pair is recognisable by its own structure: each half uses exactly
    # two colours and the halves share exactly one colour (the background).
    cands = []
    if hi % 2 == 0:
        hh = hi // 2
        cands.append(('v', (0, 0, hh, wi), (hh, 0, hh, wi)))
    if wi % 2 == 0:
        hw = wi // 2
        cands.append(('h', (0, 0, hi, hw), (0, hw, hi, hw)))

    scored = []
    for axis, abox, bbox in cands:
        ar, ac, ah, aw = abox
        br, bc, bh, bw = bbox
        A = I[ar:ar + ah, ac:ac + aw]
        B = I[br:br + bh, bc:bc + bw]
        pa = set(A.flatten().tolist())
        pb = set(B.flatten().tolist())
        common = pa & pb
        structural = (len(pa) == 2 and len(pb) == 2 and len(common) == 1)
        shape_ok = (A.shape == O.shape)
        scored.append(((structural, shape_ok, len(common) == 1), axis, abox, bbox, common))
    scored.sort(key=lambda t: (t[0][0], t[0][1], t[0][2]), reverse=True)
    _, axis, abox, bbox, common = scored[0]

    ar, ac, ah, aw = abox
    br, bc, bh, bw = bbox
    if len(common) == 1:
        bgc = int(next(iter(common)))
    else:
        from collections import Counter
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    ops, sels = [], []

    # ---- 1. mark panel B's blank cells with 2 -------------------------------
    b_blanks = [(r, c) for r in range(br, br + bh) for c in range(bc, bc + bw)
                if int(I[r, c]) == bgc]
    ops.append(2)
    sels.append(sel_of(b_blanks))

    # ---- 2. slide those marks over panel A ---------------------------------
    # first Move grabs the marks; the rest continue the same grab (empty sel),
    # so ARCLE restores every cell the marks glide over.
    if axis == 'h':
        move_op, steps = 23, aw          # panel B travels left onto panel A
    else:
        move_op, steps = 20, ah          # panel B travels up onto panel A
    ops.append(move_op)
    sels.append(sel_of(b_blanks))
    for _ in range(steps - 1):
        ops.append(move_op)
        sels.append(sel_of([]))

    # ---- 3. erase panel A's own marks --------------------------------------
    # (a cell that A marks can never be doubly-blank, so this also removes the
    #  marks that just landed on top of them)
    a_marks = [(r, c) for r in range(ar, ar + ah) for c in range(ac, ac + aw)
               if int(I[r, c]) != bgc]
    if a_marks:
        ops.append(bgc)
        sels.append(sel_of(a_marks))

    # ---- 4. keep only panel A ----------------------------------------------
    # full rectangle on purpose: the canvas shrinks to panel A's whole region.
    ops.append(33)
    sels.append([ar, ac, ah - 1, aw - 1])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 94f9d214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 94f9d214"
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
                                f"for task 94f9d214"
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
                    f"Failed to build a complete episode for task 94f9d214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"94f9d214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
