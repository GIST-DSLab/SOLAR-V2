"""
ARC Task: 239be575 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------------------
# The only structural degree of freedom this task has is the classification
# label itself: the marker colour either bridges the two 2x2 blocks or it does
# not.  Both cases must be demonstrated, so they are planned up front.
# ---------------------------------------------------------------------------
VARIANTS = [{"shouldhaveconn": True}, {"shouldhaveconn": False}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    markcol, sqcol = random.sample(cols, 2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"markcol": markcol, "sqcol": sqcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, markcol, sqcol, shouldhaveconn=None) -> dict:
    sq = {(0, 0), (1, 1), (0, 1), (1, 0)}
    if shouldhaveconn is None:
        shouldhaveconn = choice((True, False))

    hub = max(6, max_h)
    wub = max(6, max_w)

    while True:
        h = unifint(diff_lb, diff_ub, (6, hub))
        w = unifint(diff_lb, diff_ub, (6, wub))
        c = canvas(0, (h, w))
        fullcands = totuple(asindices(canvas(0, (h - 1, w - 1))))
        a = choice(fullcands)
        b = choice(remove(a, fullcands))
        mindist = unifint(diff_lb, diff_ub, (3, min(h, w) - 3))
        ctr = 0
        while not manhattan({a}, {b}) > mindist:
            a = choice(fullcands)
            b = choice(remove(a, fullcands))
            ctr += 1
            if ctr > 200:
                break
        if not manhattan({a}, {b}) > mindist:
            continue

        aset = shift(sq, a)
        bset = shift(sq, b)
        gi = fill(c, sqcol, aset | bset)

        cands = totuple(ofcolor(gi, 0))
        num = unifint(diff_lb, diff_ub, (int(0.25 * len(cands)), int(0.75 * len(cands))))
        mc = sample(cands, num)
        gi = fill(gi, markcol, mc)

        cond = fork(both, rbind(adjacent, aset), rbind(adjacent, bset))
        bobjs = colorfilter(objects(gi, T, F, F), markcol)
        ss = sfilter(bobjs, cond)

        if shouldhaveconn and len(ss) == 0:
            while len(ss) == 0:
                opts2 = totuple(ofcolor(gi, 0))
                if len(opts2) == 0:
                    break
                gi = fill(gi, markcol, {choice(opts2)})
                bobjs = colorfilter(objects(gi, T, F, F), markcol)
                ss = sfilter(bobjs, cond)
        elif (not shouldhaveconn) and len(ss) > 0:
            while len(ss) > 0:
                opts2 = totuple(ofcolor(gi, markcol))
                if len(opts2) == 0:
                    break
                gi = fill(gi, 0, {choice(opts2)})
                bobjs = colorfilter(objects(gi, T, F, F), markcol)
                ss = sfilter(bobjs, cond)

        if (len(ss) > 0) != shouldhaveconn:
            continue
        if len(palette(gi)) != 3:
            continue
        break

    oc = markcol if shouldhaveconn else 0
    go = canvas(oc, (1, 1))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule: two 2x2 blocks of one colour sit on a field of marker cells.  If one
    marker object touches BOTH blocks, the answer is a 1x1 cell of the marker
    colour; otherwise a 1x1 cell of background (0).

    O is never read.  Every choice below comes from I alone.

    Two honest routes exist for a 1x1 answer, and real solvers split between
    them:
      A) shrink the grid straight onto a cell that already holds the verdict
         colour (the bridging marker object, or an empty cell);
      B) shrink onto one of the two blocks -- the landmarks the rule is about --
         and then paint the verdict onto it.
    Route B is also the route that puts a colour-0 verdict down explicitly with
    Color0 instead of letting a cropped 0 stand in for it.
    Which route, which block/end of the bridge, and which way the empty-cell
    sweep runs are all fixed by counts and dimensions read off I.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def components(color):
        cells = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] == color}
        seen = set()
        out = []
        for cell in sorted(cells):
            if cell in seen:
                continue
            stack = [cell]
            seen.add(cell)
            comp = set()
            while stack:
                r, c = stack.pop()
                comp.add((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (r + dr, c + dc)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
            out.append(comp)
        return out

    def is_2x2(comp):
        if len(comp) != 4:
            return False
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        return max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1

    def halo(cells):
        out = set()
        for (r, c) in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                out.add((r + dr, c + dc))
        return out - set(cells)

    # background is 0 (the generator paints the canvas 0, then blocks, then markers)
    colors = sorted({int(v) for v in I.flatten().tolist() if v != 0})

    # block colour: exactly two components, each a full 2x2 square (8 cells)
    sqcol, squares = None, None
    best = None
    for col in colors:
        cs = components(col)
        if len(cs) == 2 and all(is_2x2(x) for x in cs):
            cnt = int((I == col).sum())
            if best is None or cnt < best:
                best = cnt
                sqcol, squares = col, cs
    if sqcol is None:
        sqcol = colors[0]
        squares = components(sqcol)[:2]

    markcol = next((c for c in colors if c != sqcol), sqcol)

    squares = sorted(squares, key=lambda s: min(sorted(s)))
    sq_first, sq_second = squares[0], squares[1]
    halo_first, halo_second = halo(sq_first), halo(sq_second)

    # a marker object touching BOTH blocks -> verdict is markcol, else 0
    bridge = None
    for comp in components(markcol):
        if (comp & halo_first) and (comp & halo_second):
            bridge = comp
            break
    ans = markcol if bridge is not None else 0

    # ---- solver-style choices, all read off the input -----------------------
    n_mark = int((I == markcol).sum())
    k_route = (hi * wi + n_mark) % 2          # crop-onto-answer vs crop-then-paint
    k_side = (hi + n_mark) % 2                # which block / bridge end is dealt with first
    k_cell = (wi + n_mark) % 4                # which cell of that block
    k_sweep = (hi + wi) % 2                   # row-wise or column-wise sweep for empties

    zeros = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 0]

    anchor = None
    if k_route == 0:
        # Route A: land directly on a cell that already holds the verdict colour.
        if bridge is not None:
            ends = sorted(bridge & (halo_second if k_side else halo_first))
            if ends:
                anchor = ends[0]
        elif zeros:
            # prefer an empty cell hugging the block looked at first: that gap is
            # exactly what the rule was checking
            near = sorted((halo_second if k_side else halo_first) & set(zeros))
            if not near:
                near = sorted((halo_first if k_side else halo_second) & set(zeros))
            if near:
                anchor = near[0]
            else:
                key = (lambda p: p) if k_sweep == 0 else (lambda p: (p[1], p[0]))
                ordered = sorted(zeros, key=key)
                anchor = ordered[0] if k_side == 0 else ordered[-1]

    if anchor is None:
        # Route B: land on one of the two blocks, then paint the verdict on top.
        # A block cell can never already hold the verdict (sqcol != markcol, != 0),
        # so the paint always changes the grid.
        block = sorted(sq_second if k_side else sq_first)
        anchor = block[k_cell % len(block)]
        ar, ac = anchor
        ops = [33, ans, 34]
        sels = [
            [ar, ac, 0, 0],   # 1x1 rectangle: exactly the anchor cell
            [0, 0, 0, 0],     # paint the verdict onto the whole 1x1 canvas
            [0, 0, 0, 0],
        ]
        return ops, sels

    ar, ac = anchor
    ops = [33, 34]
    sels = [
        [ar, ac, 0, 0],       # 1x1 rectangle: exactly the anchor cell
        [0, 0, 0, 0],
    ]
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
                        f"num_examples+1 ({num_examples + 1}) for task 239be575"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 239be575"
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
                                f"for task 239be575"
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
                    f"Failed to build a complete episode for task 239be575 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"239be575-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
