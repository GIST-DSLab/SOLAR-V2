"""
ARC Task: ded97339 (RE-ARC) — LLM-generated grid_maker
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
    """Fix the two colour roles for the whole episode, and pre-plan which of the
    two structural variants (the guaranteed pair of marks lies in a row, or in a
    column) each instance gets, so both are demonstrated before the test."""
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])

    VARIANTS = [{"forced": "horizontal"}, {"forced": "vertical"}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, forced=None) -> dict:
    if forced not in ("horizontal", "vertical"):
        forced = random.choice(["horizontal", "vertical"])

    def _unifint(lb, ub, rng):
        lo, hi_ = rng
        if hi_ < lo:
            hi_ = lo
        return int(round(lo + (hi_ - lo) * (random.random() * (ub - lb) + lb)))

    h = max(5, _unifint(diff_lb, diff_ub, (5, min(30, max_h))))
    w = max(5, _unifint(diff_lb, diff_ub, (5, min(30, max_w))))

    gi = [[bgc] * w for _ in range(h)]
    ndots = max(2, _unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 9))))

    inds = set((i, j) for i in range(h) for j in range(w))
    dots = set()
    if forced == "horizontal":                       # guaranteed row pair
        idxi = random.randint(0, h - 1)
        locj1 = random.randint(0, w - 3)
        locj2 = random.randint(locj1 + 2, w - 1)
        dots.add((idxi, locj1))
        dots.add((idxi, locj2))
    else:                                            # guaranteed column pair
        idxj = random.randint(0, w - 1)
        loci1 = random.randint(0, h - 3)
        loci2 = random.randint(loci1 + 2, h - 1)
        dots.add((loci1, idxj))
        dots.add((loci2, idxj))

    def _nbrs(loc):
        i, j = loc
        return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

    for _ in range(ndots - 2):
        if not inds:
            break
        loc = random.choice(tuple(sorted(inds)))
        dots.add(loc)
        inds = (inds - {loc}) - _nbrs(loc)

    for (i, j) in dots:
        gi[i][j] = linc

    go = [row[:] for row in gi]
    for ii in range(h):
        cs = [jj for jj in range(w) if gi[ii][jj] == linc]
        if len(cs) >= 2:
            for jj in range(min(cs), max(cs) + 1):
                go[ii][jj] = linc
    for jj in range(w):
        rs = [ii for ii in range(h) if gi[ii][jj] == linc]
        if len(rs) >= 2:
            for ii in range(min(rs), max(rs) + 1):
                go[ii][jj] = linc

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O, examples=None):
    """Rule: every row and every column that carries two or more marks becomes a
    solid line of the mark colour between its outermost two marks.  One Color op
    per line.  Which lines get drawn first — rows or columns, top-down or
    bottom-up, boldest line first or a plain sweep — is read off the input, so
    the route is fixed for a given puzzle but differs from puzzle to puzzle.
    Nothing here looks at O."""
    import numpy as np
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            uniq = sorted({(int(r), int(c)) for r, c in cells})
            return {"cells": [[r, c] for r, c in uniq]}

    A = np.asarray(I, dtype=int)
    hi, wi = A.shape

    # --- the mark colour: the sparse colour the picture is drawn in ----------
    vals, counts = np.unique(A, return_counts=True)
    linc = int(vals[int(np.argmin(counts))])
    if examples:                       # the demonstrations settle the convention
        painted = set()
        for pair in examples:
            EI = np.asarray(pair[0], dtype=int)
            EO = np.asarray(pair[1], dtype=int)
            if EI.shape == EO.shape:
                painted |= {int(v) for v in EO[EI != EO].tolist()}
        present = {int(v) for v in vals}
        cand = [c for c in painted if c in present]
        if len(cand) == 1:
            linc = cand[0]

    # --- the lines to draw, measured from the input only --------------------
    row_segs, col_segs = [], []
    for r in range(hi):
        cs = [c for c in range(wi) if A[r, c] == linc]
        if len(cs) >= 2:
            row_segs.append(("row", r, min(cs), max(cs)))
    for c in range(wi):
        rs = [r for r in range(hi) if A[r, c] == linc]
        if len(rs) >= 2:
            col_segs.append(("col", c, min(rs), max(rs)))

    def length(s):
        return s[3] - s[2] + 1

    # --- route choices, all read off the input ------------------------------
    nmarks = int((A == linc).sum())
    longest_row = max((length(s) for s in row_segs), default=0)
    longest_col = max((length(s) for s in col_segs), default=0)
    if longest_row != longest_col:          # start with the axis holding the
        first_axis = "row" if longest_row > longest_col else "col"
    elif len(row_segs) != len(col_segs):    # longest line; else the busier axis
        first_axis = "row" if len(row_segs) > len(col_segs) else "col"
    else:
        first_axis = "row" if nmarks % 2 == 0 else "col"
    boldest_first = (nmarks % 2 == 1)       # draw long lines first, or sweep

    def route(segs, span):
        if not segs:
            return []
        if boldest_first:
            return sorted(segs, key=lambda s: (-length(s), s[1]))
        anchor = max(segs, key=lambda s: (length(s), -s[1]))[1]
        # sweep starting from the side the boldest line sits on
        return sorted(segs, key=lambda s: s[1], reverse=(2 * anchor > span - 1))

    plan = [route(row_segs, hi), route(col_segs, wi)]
    if first_axis == "col":
        plan.reverse()

    # --- emit: one Color(linc) per line, the whole line drawn in one stroke --
    grid = A.copy()
    ops, sels = [], []
    for segs in plan:
        for kind, idx, a, b in segs:
            if kind == "row":
                cells = [(idx, c) for c in range(a, b + 1)]
            else:
                cells = [(r, idx) for r in range(a, b + 1)]
            if all(grid[r, c] == linc for r, c in cells):
                continue                    # already solid: the op would do nothing
            for r, c in cells:
                grid[r, c] = linc
            # linc may itself be 0 — Color0 lays those cells down explicitly
            ops.append(int(linc))
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])     # whole grid: a true full rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task ded97339"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ded97339"
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
                                f"for task ded97339"
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
                    f"Failed to build a complete episode for task ded97339 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ded97339-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
