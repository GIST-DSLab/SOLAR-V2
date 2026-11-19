"""
ARC Task: 855e0971 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    """The rule's colour is the 'dot' colour (the mark that stretches through its
    bar), so it is fixed for the whole episode.  Bar colours are irrelevant to the
    rule (only their stripe structure matters) and stay free.
    The bar orientation (horizontal stripes / vertical stripes after dmirror) is a
    discrete structural variant -> planned per instance."""
    dotc = random.choice(list(range(1, 10)))   # non-zero: it is a real object here
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= len(variants):
        plan = [dict(v) for v in variants]
        plan += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(plan)
    else:
        plan = [dict(v) for v in random.sample(variants, n_ex)]
    plan = plan + [dict(random.choice(plan))]   # test case must have been shown
    return {"dotc": dotc, "instance_plan": plan}


# -------------------------------------------------------------- generator ---
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             dotc: int, transpose=None, **kwargs) -> dict:
    if transpose is None:
        transpose = choice((True, False))

    cols = interval(0, 10, 1)

    # the grid is built h x w and transposed afterwards -> swap the limits
    hlim = min(30, max_w if transpose else max_h)
    wlim = min(30, max_h if transpose else max_w)
    wlim = max(3, wlim)

    nbarsd = unifint(diff_lb, diff_ub, (1, 4))
    nbars = choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    nbars = max(2, min(nbars, max(1, hlim // 2)))

    h = unifint(diff_lb, diff_ub, (nbars, max(nbars, hlim)))
    w = unifint(diff_lb, diff_ub, (3, wlim))

    barsizes = [2] * nbars
    while sum(barsizes) < h:
        j = randint(0, nbars - 1)
        barsizes[j] += 1

    gi = tuple()
    go = tuple()
    locs = interval(0, w, 1)
    remcols = remove(dotc, cols)
    lastcol = -1
    nloclbs = [choice((0, 1)) for k in range(len(barsizes))]
    if sum(nloclbs) < 2:
        loc1, loc2 = sample(interval(0, len(nloclbs), 1), 2)
        nloclbs[loc1] = 1
        nloclbs[loc2] = 1
    for bs, nloclb in zip(barsizes, nloclbs):
        col = choice(remove(lastcol, remcols))
        gim = canvas(col, (bs, w))
        gom = canvas(col, (bs, w))
        nl = unifint(diff_lb, diff_ub, (nloclb, max(nloclb, w // 2)))
        chlocs = sample(locs, nl)
        for jj in chlocs:
            idx = (randint(0, bs - 1), jj)
            gim = fill(gim, dotc, {idx})
            gom = fill(gom, dotc, vfrontier(idx))
        lastcol = col
        gi = gi + gim
        go = go + gom

    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)

    return {'input': gi, 'output': go}


# ------------------------------------------------------------- operations ---
def derive_operations(I, O, examples=None):
    """Rule: the grid is a stack of solid bars.  Every mark inside a bar stretches
    across the bar's thickness, so in the output all lines of one bar are IDENTICAL
    copies of the bar's merged line.

    Account used here, bar by bar:
      1. slide every mark of the bar onto the bar's first line  (Move)  -> that line
         now carries all the bar's marks,
      2. copy that merged line (CopyO) and stamp it onto every other line of the
         bar (Paste).
    The cell a mark leaves behind is exactly one of the stamped columns, so the
    stamp itself restores it - no repainting anywhere.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- 1. which colour is the mark: the one the rule paints (grows in examples) --
    dotc = None
    if examples:
        gained = None
        for pair in examples:
            if isinstance(pair, dict):
                ei, eo = pair.get("input"), pair.get("output")
            else:
                ei, eo = pair[0], pair[1]
            a = np.asarray(ei, dtype=int)
            b = np.asarray(eo, dtype=int)
            g = {c for c in range(10) if int((b == c).sum()) > int((a == c).sum())}
            gained = g if gained is None else (gained & g)
        if gained and len(gained) == 1:
            dotc = int(sorted(gained)[0])
    if dotc is None:                      # fallback: most 8-isolated colour
        best, bestn = None, -1
        for c in sorted(set(I.flatten().tolist())):
            n = 0
            rs, cs = np.where(I == c)
            for r, cc in zip(rs.tolist(), cs.tolist()):
                iso = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, c2 = r + dr, cc + dc
                        if 0 <= rr < h and 0 <= c2 < w and int(I[rr, c2]) == c:
                            iso = False
                if iso:
                    n += 1
            if n > bestn:
                bestn, best = n, c
        dotc = int(best)

    # --- 2. orientation of the bars -------------------------------------------
    def uniform(lines):
        for line in lines:
            if len({v for v in line if v != dotc}) > 1:
                return False
        return True

    horiz = uniform([I[r, :].tolist() for r in range(h)])

    n = h if horiz else w           # number of lines across the bars
    m = w if horiz else h           # length of one line

    def cell(i, j):
        return (i, j) if horiz else (j, i)

    # --- 3. cut the grid into bars (runs of equal line colour) -----------------
    linecol = []
    for i in range(n):
        vs = [int(I[cell(i, j)[0], cell(i, j)[1]]) for j in range(m)]
        vs = [v for v in vs if v != dotc]
        linecol.append(Counter(vs).most_common(1)[0][0] if vs else None)

    bars = []
    start = 0
    for i in range(1, n + 1):
        if i == n or (linecol[i] is not None and linecol[i] != linecol[start]):
            bars.append((start, i - 1))
            start = i

    # --- 4. bar by bar: gather the marks, then stamp the merged line -----------
    move_op = 20 if horiz else 23          # up / left, towards the bar's first line
    for (i0, i1) in bars:
        dots = []
        for i in range(i0, i1 + 1):
            for j in range(m):
                r, c = cell(i, j)
                if int(I[r, c]) == dotc:
                    dots.append((i, j))
        if not dots:
            continue                       # bar without marks stays as it is

        if dotc == 0:                      # defensive: a 0 mark is not a movable object
            for (i, j) in dots:
                ops.append(0)
                sels.append(sel_of([cell(ii, j) for ii in range(i0, i1 + 1)]))
            continue

        # 4a. slide each mark onto the bar's first line
        for (i, j) in dots:
            steps = i - i0
            if steps <= 0:
                continue
            ops.append(move_op)
            sels.append(sel_of([cell(i, j)]))      # grab this mark
            for _ in range(steps - 1):
                ops.append(move_op)
                sels.append(sel_of([]))            # keep gliding the same mark

        # 4b. the first line now holds every mark of the bar: stamp it across the bar
        ops.append(29)                             # CopyO the merged line
        sels.append(sel_of([cell(i0, j) for j in range(m)]))
        for i in range(i0 + 1, i1 + 1):
            ops.append(30)                         # Paste at the line's origin
            sels.append(sel_of([cell(i, 0)]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])             # whole grid, rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 855e0971"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 855e0971"
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
                                f"for task 855e0971"
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
                    f"Failed to build a complete episode for task 855e0971 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"855e0971-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
