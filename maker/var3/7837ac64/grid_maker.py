"""
ARC Task: 7837ac64 (RE-ARC) — LLM-generated grid_maker
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
    """Episode-level colour roles: the colour filling the block interiors (bgc)
    and the colour of the lattice lines (linc).  The marker colours sitting on
    the line intersections are read off the grid itself, so they may vary freely
    from instance to instance without hurting learnability."""
    import random as _random
    cols = list(range(10))
    bgc, linc = _random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or linc is None:
        bgc, linc = sample(cols, 2)

    # keep oh/ow small enough that a lattice with cells of size >= 2 still fits
    oh_ub = max(2, min(6, (max_h - 1) // 3))
    ow_ub = max(2, min(6, (max_w - 1) // 3))
    oh = unifint(diff_lb, diff_ub, (2, oh_ub))
    ow = unifint(diff_lb, diff_ub, (2, ow_ub))

    remcols = remove(bgc, remove(linc, cols))
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcols)
    go = canvas(bgc, (oh, ow))
    inds = asindices(go)
    fullinds = asindices(go)
    nocc = unifint(diff_lb, diff_ub, (1, oh * ow))
    for k in range(nocc):
        mpr = {
            cc: sfilter(
                inds | mapply(neighbors, ofcolor(go, cc)),
                lambda ij: (neighbors(ij) & fullinds).issubset(inds | ofcolor(go, cc))
            ) for cc in ccols
        }
        mpr = [(kk, vv) for kk, vv in mpr.items() if len(vv) > 0]
        if len(mpr) == 0:
            break
        col, locs = choice(mpr)
        loc = choice(totuple(locs))
        go = fill(go, col, {loc})
        inds = remove(loc, inds)
    obj = fullinds - ofcolor(go, bgc)
    go = subgrid(obj, go)
    oh, ow = shape(go)

    # Close the reading of the lattice before drawing it.  A block is shown on
    # the lattice by colouring its four corner intersections, so a blank block
    # whose four corners all happen to carry one colour is indistinguishable
    # from a coloured one -- fill those in (to a fixpoint) so that input and
    # output stay in step and the puzzle has exactly one answer.
    gl = [list(row) for row in go]
    while True:
        pts = {}
        for i in range(oh):
            for j in range(ow):
                c = gl[i][j]
                if c != bgc:
                    for p in ((i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)):
                        pts[p] = c
        changed = False
        for i in range(oh):
            for j in range(ow):
                if gl[i][j] == bgc:
                    cs = {pts.get((i, j)), pts.get((i, j + 1)),
                          pts.get((i + 1, j)), pts.get((i + 1, j + 1))}
                    if len(cs) == 1 and None not in cs:
                        gl[i][j] = cs.pop()
                        changed = True
        if not changed:
            break
    go = tuple(tuple(row) for row in gl)

    sqsizh = unifint(diff_lb, diff_ub, (2, max(2, (max_h - oh - 1) // oh)))
    sqsizw = unifint(diff_lb, diff_ub, (2, max(2, (max_w - ow - 1) // ow)))
    fullh = oh + 1 + oh * sqsizh
    fullw = ow + 1 + ow * sqsizw
    gi = canvas(linc, (fullh, fullw))
    sq = backdrop(frozenset({(0, 0), (sqsizh - 1, sqsizw - 1)}))
    obj = asobject(go)
    for col, ij in obj:
        plcd = shift(sq, add((1, 1), multiply(ij, (sqsizh + 1, sqsizw + 1))))
        gi = fill(gi, bgc, plcd)
        if col != bgc:
            gi = fill(gi, col, corners(outbox(plcd)))
    gih = unifint(diff_lb, diff_ub, (fullh, max(fullh, max_h)))
    giw = unifint(diff_lb, diff_ub, (fullw, max(fullw, max_w)))
    loci = randint(0, gih - fullh)
    locj = randint(0, giw - fullw)
    gigi = canvas(bgc, (gih, giw))
    plcd = shift(asobject(gi), (loci, locj))
    gigi = paint(gigi, plcd)
    ulci, ulcj = ulcorner(plcd)
    lrci, lrcj = lrcorner(plcd)
    for a in range(ulci, gih + 1, sqsizh + 1):
        gigi = fill(gigi, linc, hfrontier((a, 0)))
    for a in range(ulci, -1, -sqsizh - 1):
        gigi = fill(gigi, linc, hfrontier((a, 0)))
    for b in range(ulcj, giw + 1, sqsizw + 1):
        gigi = fill(gigi, linc, vfrontier((0, b)))
    for b in range(ulcj, -1, -sqsizw - 1):
        gigi = fill(gigi, linc, vfrontier((0, b)))
    gi = paint(gigi, plcd)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Read the lattice, shrink the sheet to one cell per lattice block, and
    paint in the blocks whose four corner intersections carry one marker colour.

    Everything below is measured from I only.  The route a solver takes is not
    unique, and the choices it faces -- crop out a blank block interior and draw
    straight onto it or crop anywhere and lay a background first, which corner to
    start from, whether to fill a shape at once, a colour at a time, or block by
    block -- are decided here from what the lattice itself looks like, so the same
    input always yields the same route and a different answer grid would not
    change a single operation.
    """
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:                                    # pragma: no cover
        def sel_of(cells):
            uniq = sorted({(int(r), int(c)) for r, c in cells})
            return {"cells": [[r, c] for r, c in uniq]}

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # ── structural roles ─────────────────────────────────────────────────────
    # The lattice period is >= 3 on both axes, so bgc is the strict majority of
    # every non-line row/column and non-line rows/columns outnumber line ones:
    # voting the per-row and per-column modes elects bgc, runner-up is linc.
    row_modes = [Counter(I[r].tolist()).most_common(1)[0][0] for r in range(hi)]
    col_modes = [Counter(I[:, c].tolist()).most_common(1)[0][0] for c in range(wi)]
    score = Counter(row_modes) + Counter(col_modes)
    bgc = int(score.most_common(1)[0][0])
    runners = [c for c, _ in score.most_common() if c != bgc]
    if runners:
        linc = int(runners[0])
    else:
        tally = Counter(I.flatten().tolist())
        rest = [c for c, _ in tally.most_common() if c != bgc]
        linc = int(rest[0]) if rest else bgc

    # lattice lines = rows / columns that hold no interior colour at all
    line_rows = [r for r in range(hi) if bgc not in I[r].tolist()]
    line_cols = [c for c in range(wi) if bgc not in I[:, c].tolist()]

    # markers sit on line intersections; the lattice of interest is their span
    markers = [(r, c) for r in line_rows for c in line_cols
               if int(I[r, c]) != bgc and int(I[r, c]) != linc]
    if markers:
        rmin = min(r for r, _ in markers); rmax = max(r for r, _ in markers)
        cmin = min(c for _, c in markers); cmax = max(c for _, c in markers)
        hlines = [r for r in line_rows if rmin <= r <= rmax]
        vlines = [c for c in line_cols if cmin <= c <= cmax]
    else:
        hlines = list(line_rows)
        vlines = list(line_cols)
    if len(hlines) < 2:
        hlines = list(line_rows)
    if len(vlines) < 2:
        vlines = list(line_cols)

    ho = max(1, len(hlines) - 1)
    wo = max(1, len(vlines) - 1)

    # ── read the blocks: a block takes a colour iff its four corner
    #    intersections all carry that one (non-line) colour ──────────────────
    out = [[bgc] * wo for _ in range(ho)]
    for i in range(ho):
        for j in range(wo):
            r0, r1 = hlines[i], hlines[min(i + 1, len(hlines) - 1)]
            c0, c1 = vlines[j], vlines[min(j + 1, len(vlines) - 1)]
            vals = {int(I[r0, c0]), int(I[r0, c1]), int(I[r1, c0]), int(I[r1, c1])}
            if len(vals) == 1:
                v = vals.pop()
                if v != linc and v != bgc:
                    out[i][j] = v

    # ── the solver's choices, all read off the lattice in I ─────────────────
    sqh = hlines[1] - hlines[0] - 1          # block interior height
    sqw = vlines[1] - vlines[0] - 1          # block interior width
    painted = sum(1 for i in range(ho) for j in range(wo) if out[i][j] != bgc)
    key = hlines[0] + vlines[0] + len(markers) + painted + ho + wo
    sweep = key % 4            # which corner to start from / which way to sweep
    grouping = (key // 4) % 3  # shape at a time / colour at a time / block by block

    def cell_key(rc):
        r, c = rc
        if sweep == 0:
            return (r, c)        # from the top-left, row by row
        if sweep == 1:
            return (c, r)        # from the top-left, column by column
        if sweep == 2:
            return (-r, -c)      # from the bottom-right, row by row
        return (-c, -r)          # from the bottom-right, column by column

    ops, sels = [], []

    # ── 1) get a bgc canvas of the answer size ──────────────────────────────
    # Either crop out one blank block interior (already nothing but bgc, so the
    # answer can be drawn straight onto it), or crop the lattice's top-left
    # corner and lay the background down over the lines that came with it.
    blank = None
    if ho <= sqh and wo <= sqw:
        br, bc = hlines[0] + 1, vlines[0] + 1
        if (I[br:br + ho, bc:bc + wo] == bgc).all():
            blank = (br, bc)
    if blank is not None and key % 2 == 0:
        # bbox == exactly the blank interior patch we mean to keep
        ops.append(33); sels.append([blank[0], blank[1], ho - 1, wo - 1])
    else:
        r0, c0 = hlines[0], vlines[0]
        # bbox == exactly the corner patch of the lattice we shrink onto
        ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])
        kept = I[r0:r0 + ho, c0:c0 + wo]
        if (kept != bgc).any():
            # lay the background over the lattice fragment the crop carried in
            ops.append(int(bgc)); sels.append([0, 0, ho - 1, wo - 1])

    # ── 2) draw the coloured blocks on top ──────────────────────────────────
    cells_by_colour = {}
    for i in range(ho):
        for j in range(wo):
            v = out[i][j]
            if v != bgc:
                cells_by_colour.setdefault(v, set()).add((i, j))

    shapes = []                      # (colour, cells of one connected block group)
    for col, cells in cells_by_colour.items():
        seen = set()
        for start in sorted(cells, key=cell_key):
            if start in seen:
                continue
            comp, stack = [], [start]
            seen.add(start)
            while stack:
                p = stack.pop()
                comp.append(p)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    q = (p[0] + dr, p[1] + dc)
                    if q in cells and q not in seen:
                        seen.add(q); stack.append(q)
            shapes.append((col, sorted(comp, key=cell_key)))
    shapes.sort(key=lambda t: (cell_key(t[1][0]), t[0]))

    if grouping == 1:
        # one colour at a time, in the order that colour is first met
        order, groups = [], {}
        for col, comp in shapes:
            if col not in groups:
                groups[col] = []
                order.append(col)
            groups[col].extend(comp)
        for col in order:
            cells = sorted(groups[col], key=cell_key)
            ops.append(int(col)); sels.append(sel_of(cells))
    elif grouping == 2:
        # block by block, finishing one shape before starting the next
        for col, comp in shapes:
            for (r, c) in comp:
                ops.append(int(col)); sels.append([r, c, 0, 0])
    else:
        # one whole shape per stroke
        for col, comp in shapes:
            ops.append(int(col)); sels.append(sel_of(comp))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 7837ac64"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7837ac64"
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
                                f"for task 7837ac64"
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
                    f"Failed to build a complete episode for task 7837ac64 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7837ac64-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
