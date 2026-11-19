"""
ARC Task: 97999447 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # colour 5 is the hardcoded "spacer" colour of the rule -> never a role colour
    cols = [c for c in range(10) if c != 5]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    numc = random.randint(1, 8)
    ccols = random.sample(rem, numc)          # dot palette, fixed for the whole episode
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols) -> dict:
    def _unifint(lb, ub, bounds):
        a, b = bounds
        s = int(math.ceil(a + lb * (b - a)))
        e = int(math.floor(a + ub * (b - a)))
        s = max(a, min(b, s))
        e = max(a, min(b, e))
        if e < s:
            s, e = e, s
        return random.randint(s, e)

    ccols = list(ccols)
    h = _unifint(diff_lb, diff_ub, (3, max_h))
    w = _unifint(diff_lb, diff_ub, (3, max_w))
    num = _unifint(diff_lb, diff_ub, (1, h))
    locs = random.sample(list(range(h)), num)   # distinct rows: one dot per row

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    go = [[bgc for _ in range(w)] for _ in range(h)]
    for idx in locs:
        col = random.choice(ccols)
        j = random.randint(0, w - 1)
        gi[idx][j] = col
        for x in range(j, w, 2):
            go[idx][x] = col
        for x in range(j + 1, w, 2):
            go[idx][x] = 5
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Every non-background dot shoots a ray to the right: its own colour on every
    second cell, colour 5 on the cells in between.

    The rule is fixed; the ROUTE through it is read off the input grid, the way two
    people would tackle the same puzzle differently: which ray to draw first, which
    way to sweep the rows, and whether to lay a 5-lattice down first and draw the
    colour over it or draw the two colours straight onto the grid.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    dots = [(r, c, int(I[r, c]))
            for r in range(hi) for c in range(wi) if int(I[r, c]) != bgc]

    if not dots:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    def ray(c):
        """cells of this dot's ray: (own-colour cells, 5-cells), columns only"""
        return list(range(c + 2, wi, 2)), list(range(c + 1, wi, 2))

    # ---- route choice, derived from the input picture alone -------------------
    n_dots = len(dots)
    n_cols = len({col for _, _, col in dots})
    c_min = min(c for _, c, _ in dots)
    r_top = min(r for r, _, _ in dots)

    order_mode = (n_dots + c_min) % 4          # which ray goes first
    paint_mode = (int(bgc) + n_cols + r_top) % 4   # how a ray gets drawn

    # rays never share a row in this task; if one ever did, fall back to the
    # grouped route, which lays every colour first and the 5-lattice last.
    if len({r for r, _, _ in dots}) != n_dots:
        paint_mode = 3

    if order_mode == 0:                        # sweep downwards
        dots.sort(key=lambda d: (d[0], d[1]))
    elif order_mode == 1:                      # sweep upwards
        dots.sort(key=lambda d: (-d[0], d[1]))
    elif order_mode == 2:                      # longest rays first
        dots.sort(key=lambda d: (d[1], d[0]))
    else:                                      # shortest rays first
        dots.sort(key=lambda d: (-d[1], d[0]))

    # ---- draw ----------------------------------------------------------------
    if paint_mode == 3:
        # all dots of one colour at a time, then the whole 5-lattice in one go
        by_col = {}
        for d in dots:
            by_col.setdefault(d[2], []).append(d)
        for col in dict.fromkeys(d[2] for d in dots):
            cells = []
            for (r, c, _) in by_col[col]:
                cells += [(r, x) for x in ray(c)[0]]
            if cells:
                ops.append(int(col))
                sels.append(sel_of(cells))
        five_cells = []
        for (r, c, _) in dots:
            five_cells += [(r, x) for x in ray(c)[1]]
        if five_cells:
            ops.append(5)
            sels.append(sel_of(five_cells))

    else:
        for (r, c, col) in dots:
            same_cols, five_cols = ray(c)
            same = [(r, x) for x in same_cols]
            five = [(r, x) for x in five_cols]

            if paint_mode == 0:                # colour first, spacers after
                if same:
                    ops.append(int(col))
                    sels.append(sel_of(same))
                if five:
                    ops.append(5)
                    sels.append(sel_of(five))

            elif paint_mode == 1:              # spacers first, colour after
                if five:
                    ops.append(5)
                    sels.append(sel_of(five))
                if same:
                    ops.append(int(col))
                    sels.append(sel_of(same))

            else:                              # lay a 5 base over the whole tail,
                span = [(r, x) for x in range(c + 1, wi)]   # then draw the colour on top
                if span:
                    ops.append(5)
                    sels.append(sel_of(span))
                if same:
                    ops.append(int(col))
                    sels.append(sel_of(same))

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
                        f"num_examples+1 ({num_examples + 1}) for task 97999447"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 97999447"
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
                                f"for task 97999447"
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
                    f"Failed to build a complete episode for task 97999447 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"97999447-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
