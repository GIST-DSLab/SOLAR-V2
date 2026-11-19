"""
ARC Task: 4093f84a (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:                                    # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# The bar can be horizontal (rows) or its transposed variant (columns) -> discrete
# structural variant, planned per instance so both orientations are shown.
VARIANTS = [{"flip": False}, {"flip": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    # barc must be non-zero: the dots take the bar colour and are then MOVED, and
    # ARCLE's object grab only carries non-zero cells.
    barc = random.choice([c for c in cols if c != 0])
    bgc = random.choice([c for c in cols if c != barc])
    dotc = random.choice([c for c in cols if c not in (bgc, barc)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "barc": barc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, dotc, flip=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if flip is None:
        flip = random.choice([True, False])
    max_h = max(7, min(int(max_h), 30))
    max_w = max(7, min(int(max_w), 30))
    # the final dmirror transposes the grid, so swap the bounds in that case
    hb = max_w if flip else max_h
    wb = max_h if flip else max_w
    h = unifint(diff_lb, diff_ub, (7, max(7, hb)))
    w = unifint(diff_lb, diff_ub, (7, max(7, wb)))

    loci1, loci2 = sorted(random.sample(list(range(2, h - 2)), 2))
    gi = [[bgc] * w for _ in range(h)]
    for ii in range(loci1, loci2 + 1):
        for j in range(w):
            gi[ii][j] = barc
    go = [row[:] for row in gi]

    num1 = min(unifint(diff_lb, diff_ub, (1, max(1, w // 2))), 5)
    num2 = min(unifint(diff_lb, diff_ub, (1, max(1, w // 2))), 5)
    for l1 in random.sample(list(range(w)), num1):
        k = unifint(diff_lb, diff_ub, (1, max(1, min(loci1 - 1, 3))))
        for r in random.sample(list(range(0, loci1)), k):
            gi[r][l1] = dotc
        for r in range(loci1 - k, loci1):
            go[r][l1] = barc
    for l2 in random.sample(list(range(w)), num2):
        k = unifint(diff_lb, diff_ub, (1, max(1, min(h - loci2 - 2, 3))))
        for r in random.sample(list(range(loci2 + 1, h)), k):
            gi[r][l2] = dotc
        for r in range(loci2 + 1, loci2 + k + 1):
            go[r][l2] = barc

    if flip:                                   # dmirror == transpose
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": tuple(tuple(r) for r in gi), "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    """Rule: the loose dots fall toward the solid bar, stacking against it (their
    order preserved), and fuse into it by taking the bar's colour.  Each dot is
    physically MOVED to its resting place with unit Move ops."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    full = [0, 0, h - 1, w - 1]                # whole grid: a true full rectangle

    # --- find the bar: the colour whose cells are exactly a contiguous block of
    #     complete rows (or complete columns) and appear nowhere else.
    barc, vertical, band = None, False, None
    for c in sorted(set(I.flatten().tolist())):
        cnt = int((I == c).sum())
        rr = sorted(set(np.where(I == c)[0].tolist()))
        if rr and rr[-1] - rr[0] + 1 == len(rr) and cnt == len(rr) * w:
            barc, vertical, band = c, False, (rr[0], rr[-1]); break
        cc = sorted(set(np.where(I == c)[1].tolist()))
        if cc and cc[-1] - cc[0] + 1 == len(cc) and cnt == len(cc) * h:
            barc, vertical, band = c, True, (cc[0], cc[-1]); break
    if barc is None:
        return [34], [full]

    # work in a canonical view where the bar is horizontal
    A = I.T if vertical else I
    H, W = A.shape
    r0, r1 = band
    outside = np.concatenate([A[:r0].flatten(), A[r1 + 1:].flatten()]).tolist()
    cnts = Counter(outside)
    bgc = cnts.most_common(1)[0][0]            # background dominates outside the bar
    rest = [c for c in cnts if c != bgc]
    if not rest:
        return [34], [full]
    dotc = rest[0]

    down_op = 22 if vertical else 21           # toward the bar from the near side
    up_op = 23 if vertical else 20             # toward the bar from the far side
    RC = (lambda r, c: (c, r)) if vertical else (lambda r, c: (r, c))
    # normally give the dots the bar colour first, then slide them; if the bar
    # colour is 0 the object could not be grabbed, so slide first and colour after.
    recolor_first = (barc != 0)

    # destinations: dot nearest the bar rests against it, the next one behind it, ...
    stacks = []
    for c in range(W):
        srcs = sorted([r for r in range(0, r0) if A[r, c] == dotc], reverse=True)
        if srcs:
            stacks.append((c, srcs, [r0 - 1 - i for i in range(len(srcs))], down_op))
    for c in range(W):
        srcs = sorted([r for r in range(r1 + 1, H) if A[r, c] == dotc])
        if srcs:
            stacks.append((c, srcs, [r1 + 1 + i for i in range(len(srcs))], up_op))

    for (c, srcs, dsts, mop) in stacks:
        if recolor_first:                       # this column's dots join the bar colour
            ops.append(barc); sels.append(sel_of([RC(r, c) for r in srcs]))
        for s, d in zip(srcs, dsts):            # nearest dot first, so the way stays clear
            steps = abs(d - s)
            if steps:
                ops.append(mop); sels.append(sel_of([RC(s, c)]))     # grab this dot
                for _ in range(steps - 1):
                    ops.append(mop); sels.append(sel_of([]))         # keep sliding it
        holes = [r for r in srcs if r not in set(dsts)]   # footprints it no longer covers
        if bgc != 0 and holes:
            ops.append(bgc); sels.append(sel_of([RC(r, c) for r in holes]))
        if not recolor_first:
            ops.append(barc); sels.append(sel_of([RC(r, c) for r in dsts]))

    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 4093f84a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4093f84a"
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
                                f"for task 4093f84a"
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
                    f"Failed to build a complete episode for task 4093f84a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4093f84a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
