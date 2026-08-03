"""
ARC Task: 23581191 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    colopts = [c for c in range(10) if c != 2]
    bgc = random.choice(colopts)
    rem = [c for c in colopts if c != bgc]
    acol = random.choice(rem)
    bcol = random.choice([c for c in rem if c != acol])
    return {"bgc": bgc, "acol": acol, "bcol": bcol}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, acol: int, bcol: int) -> dict:
    hb = max(3, min(30, max_h))
    wb = max(3, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (3, hb))
    w = unifint(diff_lb, diff_ub, (3, wb))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    card_bounds = (1, max(1, (h * w) // 4))
    na = unifint(diff_lb, diff_ub, card_bounds)
    nb = unifint(diff_lb, diff_ub, card_bounds)
    a = sample(inds, na)
    b = sample(difference(inds, a), nb)
    gi = fill(c, acol, a)
    gi = fill(gi, bcol, b)
    fa = apply(first, a)
    la = apply(last, a)
    fb = apply(first, b)
    lb = apply(last, b)
    alins = sfilter(inds, lambda ij: first(ij) in fa or last(ij) in la)
    blins = sfilter(inds, lambda ij: first(ij) in fb or last(ij) in lb)
    go = fill(c, acol, alins)
    go = fill(go, bcol, blins)
    go = fill(go, 2, intersection(set(alins), set(blins)))
    go = fill(go, acol, a)
    go = fill(go, bcol, b)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background = the canvas colour the points are scattered on (strict majority:
    # each point colour covers at most h*w//4 cells, bg at least half)
    cnt = Counter(I.flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])
    others = [int(c) for c, _ in cnt.most_common() if c != bgc]

    acol = others[0] if len(others) >= 1 else -1
    bcol = others[1] if len(others) >= 2 else -1

    def rows_of(col):
        if col < 0:
            return set()
        return {int(r) for r in np.where((I == col).any(axis=1))[0]}

    def cols_of(col):
        if col < 0:
            return set()
        return {int(c) for c in np.where((I == col).any(axis=0))[0]}

    # each source point emits a full horizontal line (its row) and a full
    # vertical line (its column); collect the two colour families' line sets
    rowsA, colsA = rows_of(acol), cols_of(acol)
    rowsB, colsB = rows_of(bcol), cols_of(bcol)

    def target(r, c):
        if I[r, c] != bgc:
            return int(I[r, c])              # original points keep their colour
        inA = (r in rowsA) or (c in colsA)
        inB = (r in rowsB) or (c in colsB)
        if inA and inB:
            return 2                          # the two families cross here
        if inA:
            return acol
        if inB:
            return bcol
        return bgc                            # untouched by either family

    painted = set()

    def draw_line(kind, idx):
        cells = [(idx, j) for j in range(wi)] if kind == 'r' else [(i, idx) for i in range(hi)]
        run = []
        for pos in range(len(cells) + 1):
            cell = cells[pos] if pos < len(cells) else None
            ok = False
            t = None
            if cell is not None:
                r, c = cell
                t = target(r, c)
                ok = (I[r, c] == bgc) and (t != bgc) and (cell not in painted)
            if ok and (not run or target(*run[-1]) == t):
                run.append(cell)
                continue
            if run:
                r0, c0 = run[0]
                r1, c1 = run[-1]
                ops.append(int(target(r0, c0)))
                sels.append([r0, c0, r1 - r0, c1 - c0])
                painted.update(run)
                run = []
            if ok:
                run.append(cell)

    # family A lines first (rows, then columns), then family B lines;
    # every cell a line adds is drawn in its own colour, or 2 where the
    # families overlap. Points on the line are never overwritten.
    order = []
    order += [('r', r) for r in sorted(rowsA)]
    order += [('c', c) for c in sorted(colsA)]
    order += [('r', r) for r in sorted(rowsB)]
    order += [('c', c) for c in sorted(colsB)]

    done_lines = set()
    for kind, idx in order:
        if (kind, idx) in done_lines:
            continue
        done_lines.add((kind, idx))
        draw_line(kind, idx)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 23581191"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 23581191"
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
                                f"for task 23581191"
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
                    f"Failed to build a complete episode for task 23581191 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"23581191-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
