"""
ARC Task: 9dfd6313 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors / plan
# The four discrete structural cases of this task are the four possible mirror
# axes (main diagonal / anti diagonal / middle column / middle row).  They must
# be planned up front so every case a test instance can show has been shown.
VARIANTS = [{"lni": 1}, {"lni": 2}, {"lni": 3}, {"lni": 4}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]      # test case was shown

    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


# ---------------------------------------------------------------------- generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, lni=None) -> dict:
    cols = interval(0, 10, 1)

    dmax = min(max_h, max_w)
    dh_ub = max(1, min(14, (dmax - 1) // 2))
    dh = unifint(diff_lb, diff_ub, (1, dh_ub))
    d = 2 * dh + 1

    remcols = remove(bgc, cols)
    remcols = remove(linc, remcols)

    gi = canvas(bgc, (d, d))
    inds = asindices(gi)

    if lni is None:
        lni = randint(1, 4)

    if lni == 1:
        ln = connect((dh, 0), (dh, d - 1))
        mirrf = hmirror
        cands = sfilter(inds, lambda ij: ij[0] > dh)
    elif lni == 2:
        ln = connect((0, dh), (d - 1, dh))
        mirrf = vmirror
        cands = sfilter(inds, lambda ij: ij[1] > dh)
    elif lni == 3:
        ln = connect((0, 0), (d - 1, d - 1))
        mirrf = dmirror
        cands = sfilter(inds, lambda ij: ij[0] > ij[1])
    else:
        ln = connect((d - 1, 0), (0, d - 1))
        mirrf = cmirror
        cands = sfilter(inds, lambda ij: (ij[0] + ij[1]) > d)

    gi = fill(gi, linc, ln)

    mp = (d * (d - 1)) // 2
    numcols = unifint(diff_lb, diff_ub, (1, min(7, mp)))
    colsch = sample(remcols, numcols)
    numpix = unifint(diff_lb, diff_ub, (1, len(cands)))
    pixs = sample(totuple(cands), numpix)
    for pix in pixs:
        gi = fill(gi, choice(colsch), {pix})

    go = mirrf(gi)
    if choice((True, False)):
        gi, go = go, gi

    return {'input': gi, 'output': go}


# ------------------------------------------------------------- derive_operations
def derive_operations(I, O, examples=None):
    """Mirror the whole grid across the single monochrome line drawn inside it.

    Everything below is read from I alone (the line, the asymmetric rows/columns,
    the line's colour).  Handed a different answer grid, this returns the same ops.

    The route the rule takes is fixed — mirror across that axis — but the way it is
    carried out is chosen from what the input shows:
      * axis-parallel line (middle row / middle column):  a mirror across a column
        is a mirror of every row, so it may be done as one sweep of the whole
        picture, band by band of marked rows, or one marked row at a time, and the
        sweep may run either way.  Rows/columns that already read the same both
        ways are skipped — flipping them would change nothing.
      * diagonal line: a transpose (resp. anti-transpose) is a quarter turn plus a
        mirror, and there are four orders/turn-directions that reach it.  Which one
        is picked comes from the line's colour and the grid's size.
    Flips/rotations here always cover a self-contained rectangle, so cells holding
    colour 0 land on cells that were themselves cleared by the grab — no 0 is lost
    and no Color(0) repair is owed.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    d = min(hi, wi)
    dh = (d - 1) // 2
    counts = Counter(I.flatten().tolist())

    candidates = [
        ("dmirror", [(i, i) for i in range(d)]),          # main diagonal
        ("cmirror", [(i, d - 1 - i) for i in range(d)]),  # anti diagonal
        ("vmirror", [(i, dh) for i in range(d)]),         # middle column
        ("hmirror", [(dh, j) for j in range(d)]),         # middle row
    ]

    kind, linc = None, 0
    # the line is monochrome AND its colour occurs exactly as often as the line is
    # long (so it is a drawn line, not the background showing through)
    for name, cells in candidates:
        vals = {int(I[r, c]) for (r, c) in cells}
        if len(vals) == 1 and counts[next(iter(vals))] == len(cells):
            kind, linc = name, next(iter(vals))
            break
    if kind is None:                                      # fallback: just monochrome
        for name, cells in candidates:
            vals = {int(I[r, c]) for (r, c) in cells}
            if len(vals) == 1:
                kind, linc = name, next(iter(vals))
                break
    if kind is None:
        kind, linc = "hmirror", int(I[dh, 0])

    ops, sels = [], []
    full = [0, 0, hi - 1, wi - 1]   # intended cells ARE exactly this full rectangle

    if kind in ("dmirror", "cmirror"):
        # transpose / anti-transpose: a quarter turn and a mirror, in either order
        if kind == "dmirror":
            routes = [[25, 26], [26, 24], [24, 27], [27, 25]]
        else:
            routes = [[25, 27], [27, 24], [24, 26], [26, 25]]
        for op in routes[(linc + d) % 4]:
            ops.append(op)
            sels.append(list(full))                       # whole square grid
    else:
        if kind == "vmirror":       # line is the middle column -> reverse each row
            flip_op = 26            # FlipH (fliplr)
            marked = [r for r in range(hi)
                      if list(I[r, :]) != list(I[r, ::-1])]

            def band_sel(a, b):     # full-width strip of rows a..b
                return [a, 0, b - a, wi - 1]
        else:                       # line is the middle row -> reverse each column
            flip_op = 27            # FlipV (flipud)
            marked = [c for c in range(wi)
                      if list(I[:, c]) != list(I[::-1, c])]

            def band_sel(a, b):     # full-height strip of columns a..b
                return [0, a, hi - 1, b - a]

        if marked:
            runs = []
            for x in marked:
                if runs and x == runs[-1][1] + 1:
                    runs[-1][1] = x
                else:
                    runs.append([x, x])

            style = (linc + len(marked)) % 3
            if style == 0:                                # one sweep of the picture
                sel_list = [list(full)]
            elif style == 1:                              # block of marked lines
                sel_list = [band_sel(a, b) for a, b in runs]
            else:                                         # one marked line at a time
                sel_list = [band_sel(x, x) for x in marked]

            if style != 0 and (linc + marked[0]) % 2 == 1:
                sel_list = sel_list[::-1]                 # sweep the other way

            for s in sel_list:
                ops.append(flip_op)
                sels.append(s)

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
                        f"num_examples+1 ({num_examples + 1}) for task 9dfd6313"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9dfd6313"
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
                                f"for task 9dfd6313"
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
                    f"Failed to build a complete episode for task 9dfd6313 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9dfd6313-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
