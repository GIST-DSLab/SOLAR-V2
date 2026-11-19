"""
ARC Task: 8d510a79 (RE-ARC) — LLM-generated grid_maker
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

# colours the generator may hand to the canvas / the bar (1 and 2 are reserved marks)
CANDIDATE_COLS = [0, 3, 4, 5, 6, 7, 8, 9]

# the one discrete structural variant: the bar lies along a row, or (after dmirror)
# along a column.  Both must be demonstrated or the test instance is unlearnable.
VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(CANDIDATE_COLS)
    barcol = random.choice([c for c in CANDIDATE_COLS if c != bgc])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "barcol": barcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))
    # after the diagonal mirror the grid is (w, h); make sure that still fits the caps
    if transpose and (max_w < 5 or max_h < 3):
        transpose = False
    hcap = int(max_w if transpose else max_h)
    wcap = int(max_h if transpose else max_w)
    hcap = max(5, min(30, hcap))
    wcap = max(3, min(30, wcap))

    h = unifint(diff_lb, diff_ub, (5, hcap))
    w = unifint(diff_lb, diff_ub, (3, wcap))
    barloci = random.randint(2, h - 3)

    gi = canvas(bgc, (h, w))
    bar = connect((barloci, 0), (barloci, w - 1))
    gi = fill(gi, barcol, bar)
    go = tuple(e for e in gi)

    jinds = list(range(w))
    numtop = unifint(diff_lb, diff_ub, (1, w - 1))
    numbot = unifint(diff_lb, diff_ub, (1, w - 1))
    tops = random.sample(jinds, numtop)
    bots = random.sample(jinds, numbot)

    for t in tops:
        loci = random.randint(0, barloci - 2)
        col = random.choice((1, 2))
        loc = (loci, t)
        gi = fill(gi, col, {loc})
        if col == 1:
            go = fill(go, col, connect(loc, (0, t)))
        else:
            go = fill(go, col, connect(loc, (barloci - 1, t)))

    for t in bots:
        loci = random.randint(barloci + 2, h - 1)
        col = random.choice((1, 2))
        loc = (loci, t)
        gi = fill(gi, col, {loc})
        if col == 1:
            go = fill(go, col, connect(loc, (h - 1, t)))
        else:
            go = fill(go, col, connect(loc, (barloci + 1, t)))

    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)

    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule (read off the demonstrations, never off O):
      one straight bar of a single colour cuts the grid in two.  Every 2-mark grows a
      line towards the bar and stops right beside it; every 1-mark grows a line the
      other way, out to the far edge.

    Every line is laid down with an explicit Color(1)/Color(2) over the exact cells of
    that line, so nothing is carried by Paste/Move/Copy and a canvas or bar of colour 0
    is handled like any other colour.

    The ROUTE (which half first, which way to sweep, one stroke per line or one stroke
    per colour) is read out of the input itself, so the same picture always produces the
    same route and different pictures produce different ones.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # ---- find the bar -------------------------------------------------------------
    # Only two colours besides the marks 1 and 2 are ever on the grid: the canvas and
    # the bar, and the canvas always covers more cells than the bar.  So try the rarer
    # one first and look for a line it fills completely.
    cnt = Counter(I.flatten().tolist())
    cands = sorted([c for c in cnt if c not in (1, 2)], key=lambda c: cnt[c])
    bar_row, bar_col = None, None
    for cand in cands:
        for r in range(h):
            if bool(np.all(I[r, :] == cand)):
                bar_row = r
                break
        if bar_row is None:
            for c in range(w):
                if bool(np.all(I[:, c] == cand)):
                    bar_col = c
                    break
        if bar_row is not None or bar_col is not None:
            break

    horizontal = bar_row is not None
    b = bar_row if horizontal else bar_col
    span = w if horizontal else h          # how many lines cross the bar

    # ---- the lines each mark has to grow -------------------------------------------
    rays = []
    for r in range(h):
        for c in range(w):
            col = int(I[r, c])
            if col not in (1, 2):
                continue
            if horizontal:
                if r < b:
                    rng = range(r, b) if col == 2 else range(0, r + 1)
                else:
                    rng = range(b + 1, r + 1) if col == 2 else range(r, h)
                cells = [(rr, c) for rr in rng]
                along, perp = c, r
            else:
                if c < b:
                    rng = range(c, b) if col == 2 else range(0, c + 1)
                else:
                    rng = range(b + 1, c + 1) if col == 2 else range(c, w)
                cells = [(r, cc) for cc in rng]
                along, perp = r, c
            # the mark itself already holds the colour; painting it would do nothing
            need = [(rr, cc) for rr, cc in cells if int(I[rr, cc]) != col]
            if not need:
                continue                    # mark already sits on its edge: no stroke
            rays.append({"col": col, "cells": need, "along": along,
                         "side": 0 if perp < b else 1})

    if rays:
        n = len(rays)
        n_near = sum(1 for x in rays if x["side"] == 0)
        n_far = n - n_near

        # busier half first; if the halves are equally busy, begin where the first mark is
        if n_near != n_far:
            first_side = 0 if n_near > n_far else 1
        else:
            first_side = rays[0]["side"]
        side_rank = {first_side: 0, 1 - first_side: 1}

        asc = (b % 2 == 0)                       # which way to sweep along the bar
        crowded = (n >= span)                    # more marks than lines -> broad strokes
        balanced = abs(n_near - n_far) <= 1      # even picture -> work line by line
        first_col = rays[0]["col"]               # start with the colour seen first
        col_other = 3 - first_col

        def key_along(x):
            return x["along"] if asc else -x["along"]

        if crowded:
            # a crowded picture gets painted a whole colour at a time, half by half
            for s in (first_side, 1 - first_side):
                for col in (first_col, col_other):
                    grp = [x for x in rays if x["side"] == s and x["col"] == col]
                    if not grp:
                        continue
                    grp.sort(key=key_along)
                    cells = []
                    for x in grp:
                        cells.extend(x["cells"])
                    ops.append(col)
                    sels.append(sel_of(cells))
        else:
            if balanced:
                # sweep across the bar, dealing with both halves of each line together
                rays.sort(key=lambda x: (key_along(x), side_rank[x["side"]]))
            else:
                # finish one half of the grid, then the other
                rays.sort(key=lambda x: (side_rank[x["side"]], key_along(x)))
            for x in rays:
                ops.append(x["col"])
                sels.append(sel_of(x["cells"]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 8d510a79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8d510a79"
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
                                f"for task 8d510a79"
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
                    f"Failed to build a complete episode for task 8d510a79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8d510a79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
