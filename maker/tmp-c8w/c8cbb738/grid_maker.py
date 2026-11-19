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


def _frame_rule(I, bgc):
    """Shared reading of the rule (used by derive_operations and as a generator check).
    Returns (fcol, R0, C0, gh, gw, cells, dest) or None."""
    I = np.asarray(I, dtype=int)
    icols = sorted(int(v) for v in np.unique(I) if int(v) != bgc)
    if not icols:
        return None
    cells = {c: sorted((int(r), int(q)) for r, q in zip(*np.where(I == c))) for c in icols}

    def bbox(cs):
        rs = [r for r, _ in cs]
        qs = [q for _, q in cs]
        return min(rs), min(qs), max(rs) - min(rs) + 1, max(qs) - min(qs) + 1

    gh = max(bbox(cells[c])[2] for c in icols)
    gw = max(bbox(cells[c])[3] for c in icols)
    fcol, R0, C0 = None, None, None
    for c in icols:
        r0, c0, hh, ww = bbox(cells[c])
        if len(cells[c]) == 4 and (hh, ww) == (gh, gw) and set(cells[c]) == {
                (r0, c0), (r0, c0 + ww - 1), (r0 + hh - 1, c0), (r0 + hh - 1, c0 + ww - 1)}:
            fcol, R0, C0 = c, r0, c0
            break
    if fcol is None:
        return None
    ring = {(r, q) for r in range(gh) for q in range(gw)
            if r in (0, gh - 1) or q in (0, gw - 1)}
    K = 2 * max(gh, gw)
    cr, cc = gh // 2, gw // 2
    dest = {}
    for c in icols:
        if c == fcol:
            continue
        r0, c0, oh, ow = bbox(cells[c])
        norm = [(r - r0, q - c0) for r, q in cells[c]]
        best, bshift = None, None
        for i in range(gh):
            for j in range(gw):
                sh = {(a + i, b + j) for a, b in norm}
                sc = K * len(sh & ring) - (abs(i + oh // 2 - cr) + abs(j + ow // 2 - cc))
                if best is None or sc > best:
                    best, bshift = sc, (i, j)
        dest[c] = sorted((R0 + bshift[0] + a, C0 + bshift[1] + b) for a, b in norm)
    return fcol, R0, C0, gh, gw, cells, dest


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; the frame colour is recognised
    # structurally (four lone corner cells), fragment colours carry no meaning.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **_unused) -> dict:
    if bgc is None:
        bgc = random.choice(list(range(10)))
    cols = interval(0, 10, 1)
    for _attempt in range(50):
        gh = unifint(diff_lb, diff_ub, (3, max(3, min(10, max_h // 2))))
        gw = unifint(diff_lb, diff_ub, (3, max(3, min(10, max_w // 2))))
        h = unifint(diff_lb, diff_ub, (min(gh * 2, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(gw * 2, max_w), max_w))
        remcols = remove(bgc, cols)
        ncols = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, ncols)
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
            obj = set(sample(totuple(remring), unifint(diff_lb, diff_ub, (3, max(3, min(len(remring), nrr // len(ccols)))))))
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
        # keep only instances where the rule (read from I alone) reproduces O
        GI = np.array(gi, dtype=int)
        if Counter(GI.flatten().tolist()).most_common(1)[0][0] != bgc:
            continue
        res = _frame_rule(GI, bgc)
        if res is None:
            continue
        fcol, R0, C0, rh, rw, cells, dest = res
        pred = np.full((rh, rw), bgc, dtype=int)
        for (a, b) in [(0, 0), (0, rw - 1), (rh - 1, 0), (rh - 1, rw - 1)]:
            pred[a, b] = fcol
        for c, d in dest.items():
            for r, q in d:
                pred[r - R0, q - C0] = c
        if pred.shape == np.array(go).shape and np.array_equal(pred, np.array(go)):
            return {'input': gi, 'output': go}
    raise ValueError("could not build an unambiguous instance")


def derive_operations(I, O=None, examples=None):
    """Rule, read from I (and the demonstrations' shared background): the four lone
    corner cells of one colour mark a gh x gw frame (gh/gw = tallest/widest colour).
    Every other colour is a piece of that frame's border; each piece slides, unchanged,
    to the border position it covers best (ties -> nearest the frame centre).

    Compared with the golf reading (centre each colour's cell cloud on the frame centre,
    which only works when every piece is already symmetric), the unit here is the same --
    one colour = one rigid piece, the frame fixed by the corner colour -- so the ops are
    the maker's: one Move chain per piece, then crop to the frame.  Difference from the
    earlier trajectory: the footprints the grabbed pieces leave (cells that now read 0
    inside the frame) are repaired once, after every piece has landed, as one Color(bgc)
    over 'the frame cells that read 0 and are not a 0-coloured piece', a selection read
    off the visible grid.  O is never consulted."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    if examples:
        bg_votes = Counter()
        for ei, _eo in examples:
            ei = np.asarray(ei, dtype=int)
            bg_votes[Counter(ei.flatten().tolist()).most_common(1)[0][0]] += 1
        bgc = bg_votes.most_common(1)[0][0]
        if not np.any(I == bgc):
            bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    res = _frame_rule(I, bgc)
    if res is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    fcol, R0, C0, gh, gw, cells, dest = res

    def in_win(p):
        return R0 <= p[0] < R0 + gh and C0 <= p[1] < C0 + gw

    def bbox(cs):
        rs = [r for r, _ in cs]
        qs = [q for _, q in cs]
        return min(rs), min(qs), max(rs) - min(rs) + 1, max(qs) - min(qs) + 1

    grid = I.copy()                    # visible working grid (simulated)
    occ = {}                           # cell -> colour of the piece standing there
    for c, cs in cells.items():
        for p in cs:
            occ[p] = c
    cur = {c: list(v) for c, v in cells.items()}
    zero_piece = set()                 # cells currently holding the 0-coloured piece
    if 0 in cur:
        zero_piece = set(cur[0])
    dest_cells = set(p for v in dest.values() for p in v)

    def translate(col, tgt):
        nonlocal zero_piece
        src = list(cur[col])
        dr = tgt[0][0] - src[0][0]
        dc = tgt[0][1] - src[0][1]
        if dr == 0 and dc == 0:
            return
        for p in src:
            if occ.get(p) == col:
                del occ[p]
        if col == 0:
            # object ops cannot carry a 0-coloured piece (0 = empty to ARCLE):
            # set it down at its new spot with Color0; its old cells are left as
            # stale 0s and swept up by the final background repair.
            if any(grid[p] != 0 for p in tgt):
                ops.append(0); sels.append(sel_of(tgt))
            for p in tgt:
                grid[p] = 0
            zero_piece = set(tgt)
        else:
            seq = [21 if dr > 0 else 20] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
            for k, op in enumerate(seq):
                ops.append(op)
                sels.append(sel_of(src) if k == 0 else sel_of([]))
            for p in src:
                grid[p] = 0
            for p in tgt:
                grid[p] = col
            zero_piece -= set(tgt)
        for p in tgt:
            occ[p] = col
        cur[col] = sorted(tgt)

    def blockers(col):
        mine = set(cur[col])
        return {occ[p] for p in dest[col] if p not in mine and p in occ}

    pending = [c for c in sorted(dest) if dest[c] != sorted(cur[c])]
    guard = 0
    while pending and guard < 500:
        guard += 1
        moved = False
        for col in list(pending):
            if not (blockers(col) & set(pending)):
                translate(col, dest[col])
                pending.remove(col)
                moved = True
        if moved:
            continue
        # cyclic blocking: park one blocker on free background outside the frame
        col = pending[0]
        b = sorted(blockers(col) & set(pending))[0]
        br, bc_, bh, bw = bbox(cur[b])
        norm = [(r - br, q - bc_) for r, q in cur[b]]
        cand = None
        for rr in range(hi - bh + 1):
            for qq in range(wi - bw + 1):
                pts = [(rr + a, qq + e) for a, e in norm]
                if all(p not in occ and grid[p] == bgc and p not in dest_cells
                       and not in_win(p) for p in pts):
                    d = abs(rr - br) + abs(qq - bc_)
                    if cand is None or d < cand[0]:
                        cand = (d, pts)
        if cand is None:
            break
        translate(b, sorted(cand[1]))

    # ---- repair: frame cells left reading 0 by the grabs (not the 0-coloured piece)
    if bgc != 0:
        holes = sorted((r, q) for r in range(R0, R0 + gh) for q in range(C0, C0 + gw)
                       if grid[r, q] == 0 and (r, q) not in zero_piece)
        if holes:
            ops.append(int(bgc)); sels.append(sel_of(holes))
            for p in holes:
                grid[p] = bgc

    # ---- crop the canvas to the frame
    ops.append(33); sels.append([R0, C0, gh - 1, gw - 1])
    ops.append(34); sels.append([0, 0, gh - 1, gw - 1])
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
