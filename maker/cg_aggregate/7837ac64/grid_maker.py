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
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, linc)]
    numcols = random.randint(1, 8)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "linc": linc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or linc is None:
        bgc, linc = sample(cols, 2)
    if ccols is None:
        remcols = remove(bgc, remove(linc, cols))
        ccols = sample(remcols, unifint(diff_lb, diff_ub, (1, 8)))
    ccols = list(ccols)
    # lattice needs oh + 1 + 2*oh <= max_h
    oh_cap = max(2, min(6, (max_h - 1) // 3))
    ow_cap = max(2, min(6, (max_w - 1) // 3))
    oh = unifint(diff_lb, diff_ub, (2, oh_cap))
    ow = unifint(diff_lb, diff_ub, (2, ow_cap))
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
        raise ValueError("lattice does not fit")
    gi = canvas(linc, (fullh, fullw))
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
    # reject ambiguous lattices (diagonal contacts make a block's four corners
    # agree although the block itself stays blank) -- same reading as verifier
    rows = list(range(loci, loci + fullh, sqsizh + 1))
    colsl = list(range(locj, locj + fullw, sqsizw + 1))
    for i in range(oh):
        for j in range(ow):
            vals = {gi[rows[i]][colsl[j]], gi[rows[i]][colsl[j + 1]],
                    gi[rows[i + 1]][colsl[j]], gi[rows[i + 1]][colsl[j + 1]]}
            v = vals.pop() if len(vals) == 1 else bgc
            if v == linc:
                v = bgc
            if v != go[i][j]:
                raise ValueError("ambiguous corner reading")
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): the lattice is read line by line.
    # Along each horizontal lattice line, a segment between two consecutive
    # intersections "carries" a colour when both end markers agree (and are
    # not the plain line colour); a block between two horizontal lines gets
    # that colour when the segment above and the segment below carry the same
    # one.  That is exactly "all four corners agree", so maker and golf
    # coincide on WHAT each output cell is; the golf reading changes the unit
    # of work: it walks band by band and a horizontal run of agreeing
    # segments is one stroke.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # bgc = colour of the block interiors: the largest solid single-colour
    # rectangle; linc = the most frequent remaining colour (the lattice)
    seen = np.zeros((hi, wi), bool)
    best, bgc = -1, int(I[0, 0])
    for r0 in range(hi):
        for c0 in range(wi):
            if seen[r0, c0]:
                continue
            col = I[r0, c0]
            stack = [(r0, c0)]; seen[r0, c0] = True; comp = []
            while stack:
                r, c = stack.pop(); comp.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True; stack.append((nr, nc))
            rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
            area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if area == len(comp) and area > best:
                best, bgc = area, int(col)
    cnt = Counter(I.flatten().tolist())
    rest = [c for c, _ in cnt.most_common() if c != bgc]
    linc = rest[0] if rest else bgc

    line_rows = [r for r in range(hi) if bgc not in I[r].tolist()]
    line_cols = [c for c in range(wi) if bgc not in I[:, c].tolist()]
    markers = [(r, c) for r in range(hi) for c in range(wi)
               if I[r, c] != bgc and I[r, c] != linc]
    if markers:
        rmin = min(r for r, _ in markers); rmax = max(r for r, _ in markers)
        cmin = min(c for _, c in markers); cmax = max(c for _, c in markers)
        hlines = [r for r in line_rows if rmin <= r <= rmax]
        vlines = [c for c in line_cols if cmin <= c <= cmax]
    else:
        hlines, vlines = list(line_rows), list(line_cols)
    ho = max(1, len(hlines) - 1)
    wo = max(1, len(vlines) - 1)

    # horizontal pass: segment colour on each horizontal line
    def seg(r, j):
        a, b = int(I[r, vlines[j]]), int(I[r, vlines[j + 1]])
        return a if (a == b and a != linc) else None

    ops, sels = [], []
    # canvas -> one cell per lattice block (full rectangle)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    # base: the blank block interiors over the whole canvas (full rectangle)
    if any(int(I[r, c]) != bgc for r in range(ho) for c in range(wo)):
        ops.append(int(bgc)); sels.append([0, 0, ho - 1, wo - 1])

    # vertical pass, band by band: block colour = top segment == bottom segment
    for i in range(ho):
        band = []
        for j in range(wo):
            t = seg(hlines[i], j)
            b = seg(hlines[i + 1], j)
            band.append(t if (t is not None and t == b) else None)
        j = 0
        while j < wo:
            col = band[j]
            if col is None or col == bgc:
                j += 1
                continue
            k = j
            while k + 1 < wo and band[k + 1] == col:
                k += 1
            # one stroke: consecutive blocks of this band sharing the colour
            ops.append(int(col)); sels.append(sel_of([(i, x) for x in range(j, k + 1)]))
            j = k + 1

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
