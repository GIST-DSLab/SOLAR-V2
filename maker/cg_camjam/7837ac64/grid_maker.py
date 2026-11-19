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
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or linc is None:
        bgc, linc = sample(cols, 2)
    oh = unifint(diff_lb, diff_ub, (2, 6))
    ow = unifint(diff_lb, diff_ub, (2, 6))
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
    sqsizh = unifint(diff_lb, diff_ub, (2, max(2, (max_h - oh - 1) // oh)))
    sqsizw = unifint(diff_lb, diff_ub, (2, max(2, (max_w - ow - 1) // ow)))
    fullh = oh + 1 + oh * sqsizh
    fullw = ow + 1 + ow * sqsizw
    if fullh > max_h or fullw > max_w:
        raise ValueError("grid too large")
    gi = canvas(linc, (fullh, fullw))
    stph, stpw = sqsizh + 1, sqsizw + 1
    sq = backdrop(frozenset({(0, 0), (sqsizh - 1, sqsizw - 1)}))
    obj = asobject(go)
    for col, ij in obj:
        plcd = shift(sq, add((1, 1), multiply(ij, (sqsizh + 1, sqsizw + 1))))
        gi = fill(gi, bgc, plcd)
        if col != bgc:
            gi = fill(gi, col, corners(outbox(plcd)))
    gih = unifint(diff_lb, diff_ub, (fullh, max_h))
    giw = unifint(diff_lb, diff_ub, (fullw, max_w))
    loci = randint(0, gih - fullh)
    locj = randint(0, giw - fullw)
    gigi = canvas(bgc, (gih, giw))
    plcd = shift(asobject(gi), (loci, locj))
    gigi = paint(gigi, plcd)
    ulci, ulcj = ulcorner(plcd)
    for a in range(ulci, gih + 1, sqsizh + 1):
        gigi = fill(gigi, linc, hfrontier((a, 0)))
    for a in range(ulci, -1, -sqsizh - 1):
        gigi = fill(gigi, linc, hfrontier((a, 0)))
    for b in range(ulcj, giw + 1, sqsizw + 1):
        gigi = fill(gigi, linc, vfrontier((0, b)))
    for b in range(ulcj, -1, -sqsizw - 1):
        gigi = fill(gigi, linc, vfrontier((0, b)))
    gi = paint(gigi, plcd)
    gop = palette(go)
    while True:
        go2 = identity(go)
        for c in set(ccols) & gop:
            o1 = frozenset({(c, ORIGIN), (bgc, RIGHT), (c, (0, 2))})
            o2 = dmirror(o1)
            go2 = fill(go2, c, combine(
                shift(occurrences(go, o1), RIGHT),
                shift(occurrences(go, o2), DOWN)
            ))
        if go2 == go:
            break
        go = go2
    # keep only instances where the output agrees with the four-corner rule
    # (diagonal same-colour cells can otherwise frame a background cell,
    #  which makes the instance contradict the verifier's reading)
    for i in range(oh):
        for j in range(ow):
            vals = {gi[loci + a * stph][locj + b * stpw]
                    for a in (i, i + 1) for b in (j, j + 1)}
            exp = bgc
            if len(vals) == 1:
                v = next(iter(vals))
                if v != linc and v != bgc:
                    exp = v
            if go[i][j] != exp:
                raise ValueError("ambiguous corner framing")
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    # Reading: the maker and the golf entry agree on the rule. Strip the cell
    # interiors so only the lattice of line intersections is left. Each output
    # cell is one 2x2 window of that lattice. It takes the window's colour when
    # all four corners agree on a non-line colour; otherwise it is background.
    # Where they differ is the unit: the golf entry treats every corner colour
    # the same way, so here all the agreeing windows of one colour are painted
    # as ONE unit (one Color op per colour) instead of cell by cell.
    # O is never read.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def mode(vals):
        return Counter(vals).most_common(1)[0][0]

    # bgc = colour of the rectangular cell interiors (the modal colour of most rows);
    # linc = colour of the lattice lines (the modal colour of the remaining rows).
    row_modes = Counter(mode(I[r].tolist()) for r in range(hi))
    bgc = row_modes.most_common(1)[0][0]
    others = [c for c in row_modes if c != bgc]
    if others:
        linc = max(others, key=lambda c: row_modes[c])
    else:
        rest = [c for c, _ in Counter(I.flatten().tolist()).most_common() if c != bgc]
        linc = rest[0]

    # lattice lines = rows / cols with no interior colour
    line_rows = [r for r in range(hi) if bgc not in I[r].tolist()]
    line_cols = [c for c in range(wi) if bgc not in I[:, c].tolist()]

    # coloured intersections; their bbox spans the encoded picture
    markers = [(r, c) for r in line_rows for c in line_cols
               if I[r, c] not in (bgc, linc)]
    rmin = min(r for r, _ in markers); rmax = max(r for r, _ in markers)
    cmin = min(c for _, c in markers); cmax = max(c for _, c in markers)
    hl = [r for r in line_rows if rmin <= r <= rmax]
    vl = [c for c in line_cols if cmin <= c <= cmax]
    ho, wo = len(hl) - 1, len(vl) - 1

    ops, sels = [], []
    # 1. canvas -> one cell per 2x2 intersection window, anchored at the
    #    lattice's top-left intersection (the window index origin).
    ops.append(33); sels.append([rmin, cmin, ho - 1, wo - 1])  # full rectangle
    # 2. base: every window starts as background.
    ops.append(int(bgc)); sels.append([0, 0, ho - 1, wo - 1])  # full rectangle

    # 3. for each corner colour, all windows whose four corners agree on it.
    by_col = {}
    order = []
    for i in range(ho):
        for j in range(wo):
            vals = {int(I[hl[i], vl[j]]), int(I[hl[i], vl[j + 1]]),
                    int(I[hl[i + 1], vl[j]]), int(I[hl[i + 1], vl[j + 1]])}
            if len(vals) == 1:
                v = vals.pop()
                if v != linc and v != bgc:
                    if v not in by_col:
                        by_col[v] = []
                        order.append(v)
                    by_col[v].append((i, j))
    for v in order:
        ops.append(v); sels.append(sel_of(by_col[v]))

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
