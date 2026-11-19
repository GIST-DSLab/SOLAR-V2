"""
ARC Task: c8cbb738 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """Episode-level colours: background + ordered foreground palette.

    palette[0] is the frame colour (the four corner cells) — fixed for the whole
    episode so the rule stays readable.  Colour 0 is kept out of the foreground
    palette: ARCLE's object ops grab only NONZERO cells, so a 0-coloured piece
    could never be dragged across the grid.
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    palette = [c for c in cols if c != bgc and c != 0]
    random.shuffle(palette)
    return {"bgc": bgc, "palette": palette}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int = None, palette=None) -> dict:
    if bgc is None:
        bgc = random.choice(list(range(10)))
    if palette is None:
        palette = [c for c in range(10) if c != bgc and c != 0]
        random.shuffle(palette)

    gh_ub = max(3, min(10, max_h // 2))
    gw_ub = max(3, min(10, max_w // 2))
    gh = unifint(diff_lb, diff_ub, (3, gh_ub))
    gw = unifint(diff_lb, diff_ub, (3, gw_ub))
    h = unifint(diff_lb, diff_ub, (min(gh * 2, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(gw * 2, max_w), max_w))

    ncols = unifint(diff_lb, diff_ub, (1, min(9, len(palette))))
    ccols = list(palette[:ncols])

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (gh, gw))
    goinds = asindices(go)
    ring = box(goinds)
    crns = corners(ring)
    remring = ring - crns
    nrr = len(remring)
    sc = ccols[0]
    go = fill(go, sc, crns)
    loci = randint(0, h - gh)
    locj = randint(0, w - gw)
    gi = fill(gi, sc, shift(crns, (loci, locj)))
    ccols = ccols[1:]

    bL = connect((0, 0), (gh - 1, 0))
    bR = connect((0, gw - 1), (gh - 1, gw - 1))
    bT = connect((0, 0), (0, gw - 1))
    bB = connect((gh - 1, 0), (gh - 1, gw - 1))
    validpairs = [(bL, bT), (bL, bB), (bR, bT), (bR, bB)]

    for c in ccols:
        if len(remring) < 3:
            break
        nc = unifint(diff_lb, diff_ub, (3, max(3, min(len(remring), nrr // len(ccols)))))
        obj = set(sample(totuple(remring), nc))
        flag = False
        for b1, b2 in validpairs:
            if len(obj & b1) > 0 and len(obj & b2) > 0:
                flag = True
                break
        if flag:
            oh, ow = shape(obj)
            locs = ofcolor(gi, bgc)
            cands = sfilter(locs, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) > 0:
                objn = normalize(obj)
                cands2 = sfilter(cands, lambda ij: shift(objn, ij).issubset(locs))
                if len(cands2) > 0:
                    loc = choice(totuple(cands2))
                    gi = fill(gi, c, shift(objn, loc))
                    go = fill(go, c, obj)
                    remring -= obj

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """The four corner dots mark a frame.  Every other colour-piece is a chunk of
    that frame's border: it slides onto the one spot of the border it fits (the
    placement that puts all of its cells on the border ring and sits closest to
    the frame's centre).  Finally the canvas is cropped down to the frame.

    Everything below — frame size, frame position, and each piece's landing spot
    — is measured from I; O is never inspected.
    """
    I = np.asarray(I, dtype=int)
    # O is deliberately never read: frame size, frame position and every landing
    # spot below come from I and from the rule the generator implements.
    hi, wi = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def bbox(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    icols = [int(v) for v in np.unique(I) if int(v) != bgc]
    iobj = {c: sorted((int(r), int(cc)) for r, cc in zip(*np.where(I == c)))
            for c in icols}
    if not iobj:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # ---- frame size: the widest / tallest piece spans the whole frame
    ho = max(bbox(v)[2] for v in iobj.values())
    wo = max(bbox(v)[3] for v in iobj.values())

    # ---- frame position in I: the piece that is exactly the 4 corners of a ho x wo box
    R0 = C0 = None
    for col in icols:
        cells = iobj[col]
        if len(cells) != 4:
            continue
        r0, c0, hh, ww = bbox(cells)
        if (hh, ww) != (ho, wo):
            continue
        if set(cells) == {(r0, c0), (r0, c0 + ww - 1),
                          (r0 + hh - 1, c0), (r0 + hh - 1, c0 + ww - 1)}:
            R0, C0 = r0, c0
            break
    if R0 is None:                      # no corner marks: any ho x wo window works
        R0, C0 = 0, 0
    R0 = max(0, min(R0, hi - ho))
    C0 = max(0, min(C0, wi - wo))

    # ---- landing spot of each piece on the frame border, measured from I
    ring = frozenset((r, c) for r in range(ho) for c in range(wo)
                     if r in (0, ho - 1) or c in (0, wo - 1))
    spots = frozenset((i, j) for i in range(ho) for j in range(wo))
    ctr_r, ctr_c = ho // 2, wo // 2
    weight = 2 * max(ho, wo)

    dest = {}
    for col in icols:
        r0, c0, nh, nw = bbox(iobj[col])
        norm = [(r - r0, c - c0) for r, c in iobj[col]]

        def fit(ij, norm=norm, nh=nh, nw=nw):
            i, j = ij
            on_ring = sum(1 for (r, c) in norm if (r + i, c + j) in ring)
            off_centre = abs(i + nh // 2 - ctr_r) + abs(j + nw // 2 - ctr_c)
            return weight * on_ring - off_centre

        bi, bj = max(spots, key=fit)
        cand = [(R0 + r + bi, C0 + c + bj) for r, c in norm]
        if all(0 <= r < hi and 0 <= c < wi for r, c in cand):
            dest[col] = sorted(cand)

    def in_frame(p):
        return R0 <= p[0] < R0 + ho and C0 <= p[1] < C0 + wo

    grid = I.copy()
    cur = {c: list(v) for c, v in iobj.items()}
    lifted = set()          # pieces taken off the grid: they get drawn, not slid

    def lift(col):
        """Wipe a piece off the grid (it is drawn on the frame later instead)."""
        keep = set(dest[col])
        gone = sorted(p for p in cur[col] if p not in keep and int(grid[p[0], p[1]]) == col)
        if gone:
            ops.append(int(bgc))
            sels.append(sel_of(gone))
            for (r, c) in gone:
                grid[r, c] = bgc
        cur[col] = [p for p in cur[col] if p in keep]
        lifted.add(col)

    def drop(col):
        tgt = sorted(dest[col])
        ops.append(int(col))
        sels.append(sel_of(tgt))
        for (r, c) in tgt:
            grid[r, c] = col
        cur[col] = tgt

    def carry(col):
        """Slide one piece onto the frame."""
        if col in lifted:
            drop(col)
            return
        src = sorted(cur[col])
        tgt = sorted(dest[col])
        dr = tgt[0][0] - src[0][0]
        dc = tgt[0][1] - src[0][1]
        if (dr, dc) == (0, 0):
            cur[col] = tgt
            return
        if col == 0:
            # ARCLE's object buffer holds only nonzero cells, so a 0-coloured
            # piece cannot be dragged: wipe it and draw it on the frame instead.
            lift(col)
            drop(col)
            return
        seq = [21 if dr > 0 else 20] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
        for k, op in enumerate(seq):
            ops.append(op)
            # the first step grabs the piece; the rest keep the same grab, so
            # ARCLE restores every cell the piece glides over
            sels.append(sel_of(src) if k == 0 else sel_of([]))
        for (r, c) in src:
            grid[r, c] = 0
        for (r, c) in tgt:
            grid[r, c] = col
        cur[col] = tgt
        # only the vacated footprint reads 0 now; repair the part of it that the
        # final crop keeps -- the rest is thrown away by the crop anyway
        hole = [p for p in sorted(set(src) - set(tgt)) if in_frame(p)]
        if bgc != 0 and hole:
            ops.append(int(bgc))
            sels.append(sel_of(hole))
            for (r, c) in hole:
                grid[r, c] = bgc

    def blocked(col):
        occupied = set(cur[col])
        for (r, c) in dest[col]:
            if (r, c) not in occupied and int(grid[r, c]) != bgc:
                return True
        return False

    # a 0-coloured piece can never be dragged, so take it off the grid right away:
    # left lying there it would block the pieces that have to land on top of it
    for col in list(dest):
        if col == 0 and sorted(cur[col]) != sorted(dest[col]):
            lift(col)

    # fill the frame in reading order of the landing spots
    pending = sorted((c for c in icols if c in dest),
                     key=lambda c: (dest[c][0], c))
    guard = 0
    while pending and guard < 400:
        guard += 1
        moved = False
        for col in list(pending):
            if blocked(col):
                continue
            carry(col)
            pending.remove(col)
            moved = True
        if moved or not pending:
            continue
        # cyclic stand-off: shove one blocking piece onto free ground outside the
        # frame first, so the piece waiting for that spot can land
        stuck = pending[0]
        occupied = set(cur[stuck])
        blockers = []
        for (r, c) in dest[stuck]:
            v = int(grid[r, c])
            if (r, c) not in occupied and v != bgc and v in pending and v != stuck:
                if v not in blockers:
                    blockers.append(v)
        parked = False
        for b in blockers:
            if b in lifted or not cur[b]:
                continue
            bc = sorted(cur[b])
            br, bcc, bh, bw = bbox(bc)
            norm = [(r - br, c - bcc) for r, c in bc]
            for rr in range(hi - bh + 1):
                for cc in range(wi - bw + 1):
                    if (rr, cc) == (br, bcc):
                        continue
                    spot = [(rr + a, cc + d) for a, d in norm]
                    if any(in_frame(p) for p in spot):
                        continue
                    if any(int(grid[p[0], p[1]]) != bgc for p in spot):
                        continue
                    keep = dest[b]
                    dest[b] = sorted(spot)
                    carry(b)
                    dest[b] = keep
                    parked = True
                    break
                if parked:
                    break
            if parked:
                break
        if parked:
            continue
        # nowhere to shove it: take the blocking pieces off the grid instead --
        # they are drawn on the frame when their spot comes free
        if blockers:
            for b in blockers:
                if b not in lifted:
                    lift(b)
        else:
            carry(stuck)
            pending.remove(stuck)

    # ---- crop to the frame (full rectangle: the whole window, background included)
    ops.append(33)
    sels.append([R0, C0, ho - 1, wo - 1])
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    raise ValueError(
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
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
                                f"for task c8cbb738"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task c8cbb738 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c8cbb738-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
