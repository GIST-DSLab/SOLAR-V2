"""
ARC Task: 3631a71a (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    patchcol = random.choice(cols)
    bgc = random.choice([c for c in cols if c != patchcol])
    return {"bgc": bgc, "patchcol": patchcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, patchcol) -> dict:
    cols = interval(0, 10, 1)
    hi_h = min(15, (max_h + 2) // 2, (max_w + 2) // 2)
    if hi_h < 6:
        hi_h = 6
    h = unifint(diff_lb, diff_ub, (6, hi_h))
    w = h
    remcols = difference(cols, (bgc, patchcol))
    c = canvas(bgc, (h, w))
    inds = sfilter(asindices(c), lambda ij: ij[0] >= ij[1])
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    ncells = unifint(diff_lb, diff_ub, (1, len(inds)))
    cells = set(sample(totuple(inds), ncells))
    obj = {(choice(ccols), ij) for ij in cells}
    c = paint(dmirror(paint(c, obj)), obj)
    c = hconcat(c, vmirror(c))
    c = vconcat(c, hmirror(c))
    cutoff = 2
    go = dmirror(dmirror(c[:-cutoff])[:-cutoff])
    gi = tuple(e for e in go)
    forbidden = asindices(canvas(-1, (cutoff, cutoff)))
    dmirrareaL = shift(asindices(canvas(-1, (h * 2 - 2 * cutoff, cutoff))), (cutoff, 0))
    dmirrareaT = shift(asindices(canvas(-1, (cutoff, 2 * w - 2 * cutoff))), (0, cutoff))
    inds1 = sfilter(asindices(gi), lambda ij: cutoff <= ij[0] < h and cutoff <= ij[1] < w and ij[0] >= ij[1])
    inds2 = dmirror(inds1)
    inds3 = shift(hmirror(inds1), (h - cutoff, 0))
    inds4 = shift(hmirror(inds2), (h - cutoff, 0))
    inds5 = shift(vmirror(inds1), (0, w - cutoff))
    inds6 = shift(vmirror(inds2), (0, w - cutoff))
    inds7 = shift(hmirror(vmirror(inds1)), (h - cutoff, w - cutoff))
    inds8 = shift(hmirror(vmirror(inds2)), (h - cutoff, w - cutoff))
    f1 = identity
    f2 = dmirror
    f3 = lambda x: hmirror(shift(x, invert((h - cutoff, 0))))
    f4 = lambda x: dmirror(hmirror(shift(x, invert((h - cutoff, 0)))))
    f5 = lambda x: vmirror(shift(x, invert((0, w - cutoff))))
    f6 = lambda x: dmirror(vmirror(shift(x, invert((0, w - cutoff)))))
    f7 = lambda x: vmirror(hmirror(shift(x, invert((h - cutoff, w - cutoff)))))
    f8 = lambda x: dmirror(vmirror(hmirror(shift(x, invert((h - cutoff, w - cutoff))))))
    indsarr = [inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8]
    farr = [f1, f2, f3, f4, f5, f6, f7, f8]
    ndist = unifint(diff_lb, diff_ub, (1, int((2 * h * 2 * w) ** 0.5)))
    succ = 0
    tr = 0
    maxtr = 10 * ndist
    fullh, fullw = shape(gi)
    while succ < ndist and tr < maxtr:
        tr += 1
        oh = randint(2, h // 2 + 1)
        ow = randint(2, w // 2 + 1)
        loci = randint(0, fullh - oh)
        locj = randint(0, fullw - ow)
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        isleft = set()
        gi2 = fill(gi, patchcol, bd)
        if patchcol in palette(toobject(forbidden, gi2)):
            continue
        oo1 = toindices(sfilter(toobject(dmirrareaL, gi2), lambda cij: cij[0] != patchcol))
        oo2 = toindices(sfilter(toobject(dmirrareaT, gi2), lambda cij: cij[0] != patchcol))
        oo2 = frozenset({(ij[1], ij[0]) for ij in oo2})
        if oo1 | oo2 != dmirrareaL:
            continue
        for ii, ff in zip(indsarr, farr):
            oo = toobject(ii, gi2)
            rem = toindices(sfilter(oo, lambda cij: cij[0] != patchcol))
            if len(rem) > 0:
                isleft = isleft | ff(rem)
        if isleft != inds1:
            continue
        succ += 1
        gi = gi2
    if gi == go:
        # nothing ended up occluded: there would be no transformation to show
        raise ValueError("no patch was placed")
    if _read_patch_colour(np.array(gi, dtype=int)) != patchcol:
        # the occluding colour has to be readable from the input alone
        raise ValueError("occluding colour not identifiable from the input")
    return {'input': gi, 'output': go}


# ── derive_operations ────────────────────────────────────────────────────────
#
# WHAT THE INPUT IS.  A transpose-symmetric square block is mirrored to the
# right and downwards, and the last two rows and columns are then cut off.  On
# the n x n grid that is left the picture is invariant under
#
#       (r, c) -> (c, r)          the diagonal through the top-left corner
#       (r, c) -> (n+1-r, c)      the row axis      (defined for r >= 2)
#       (r, c) -> (r, n+1-c)      the column axis   (defined for c >= 2)
#
# and under the five further maps those three generate.  Solid rectangles of a
# single colour are dropped on top of it; that colour occurs nowhere else in
# the picture, and no rectangle ever covers a whole symmetry orbit.
#
# WHAT THE OPERATIONS DO.  While anything is still hidden: take the largest
# rectangle of the blot that ONE of the eight symmetries carries onto an
# unoccluded region of the input, CopyI that region, Paste it over the
# rectangle and Flip/Rotate the copy in place so it lands the way the symmetry
# maps it.  Everything the route needs — which colour occludes, which cells it
# covers, which symmetry restores each piece and hence which region to copy and
# which turns to apply — is measured from the input.  A single cell left over
# is painted with the colour its own orbit still shows in the input.

# the seven non-identity symmetries, as (transpose?, flip rows?, flip cols?)
_SYMS = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

# the turns that carry a pasted copy of the source region onto the target
# rectangle.  26 = FlipH (left-right), 27 = FlipV (up-down),
# 25 = rotate clockwise, 24 = rotate counter-clockwise.
_POST = {
    (0, 0, 1): (26,),
    (0, 1, 0): (27,),
    (0, 1, 1): (26, 27),
    (1, 0, 0): (25, 26),
    (1, 0, 1): (25,),
    (1, 1, 0): (24,),
    (1, 1, 1): (24, 26),
}


def _sym_cell(sym, r, c, H, W):
    """Where symmetry `sym` carries cell (r, c), or None if that leaves the grid."""
    t, a, b = sym
    p = (H + 1 - r) if a else r
    q = (W + 1 - c) if b else c
    if t:
        p, q = q, p
    if 0 <= p < H and 0 <= q < W:
        return p, q
    return None


def _sym_rect(sym, r1, c1, r2, c2, H, W):
    """The region `sym` carries onto the rectangle (r1,c1)-(r2,c2)."""
    t, a, b = sym
    rows = (r1, r2) if not a else (H + 1 - r2, H + 1 - r1)
    cols = (c1, c2) if not b else (W + 1 - c2, W + 1 - c1)
    if t:
        rows, cols = cols, rows
    (sr1, sr2), (sc1, sc2) = rows, cols
    if sr1 < 0 or sc1 < 0 or sr2 >= H or sc2 >= W:
        return None
    return sr1, sc1, sr2, sc2


def _read_patch_colour(I):
    """The colour of the occluding rectangles, read off the input alone.

    Setting that colour aside has to leave every pair of cells the symmetries
    relate showing one and the same colour, and every cell of that colour has
    to keep a symmetric partner that is not occluded.  Only the occluding
    colour manages both: it appears nowhere in the picture underneath, so for
    any other colour the occlusions stay in place and contradict the symmetry.
    """
    H, W = I.shape
    best = None
    for v in sorted(set(I.flatten().tolist())):
        hidden = I == v
        if not hidden.any():
            continue
        ok = True
        for r in range(H):
            for c in range(W):
                if hidden[r, c]:
                    continue
                for sym in _SYMS:
                    p = _sym_cell(sym, r, c, H, W)
                    if p is not None and not hidden[p] and I[p] != I[r, c]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            for rc in np.argwhere(hidden):
                r, c = int(rc[0]), int(rc[1])
                seen = [_sym_cell(sym, r, c, H, W) for sym in _SYMS]
                if not any(p is not None and not hidden[p] for p in seen):
                    ok = False
                    break
        if not ok:
            continue
        cnt = int(hidden.sum())
        if best is None or cnt > best[0]:
            best = (cnt, int(v))
    return None if best is None else best[1]


def _uncovered(I, patchcol):
    """The picture under the occlusions: every hidden cell shows the colour one
    of its symmetric partners still has in the input."""
    H, W = I.shape
    R = I.copy()
    for rc in np.argwhere(I == patchcol):
        r, c = int(rc[0]), int(rc[1])
        for sym in _SYMS:
            p = _sym_cell(sym, r, c, H, W)
            if p is not None and I[p] != patchcol:
                R[r, c] = I[p]
                break
    return R


def _blots(mask):
    """The 4-connected blots of the occluding colour, in reading order."""
    todo = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    out = []
    while todo:
        seed = min(todo)
        todo.discard(seed)
        stack, comp = [seed], [seed]
        while stack:
            r, c = stack.pop()
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in todo:
                    todo.discard(nb)
                    stack.append(nb)
                    comp.append(nb)
        out.append((min(p[0] for p in comp), min(p[1] for p in comp),
                    max(p[0] for p in comp), max(p[1] for p in comp)))
    out.sort()
    return out


def _turn(region, op):
    if op == 26:
        return np.fliplr(region)
    if op == 27:
        return np.flipud(region)
    if op == 25:
        return np.rot90(region, 3)
    return np.rot90(region, 1)


def _widest_rect(feas, hidden, box, square):
    """The rectangle inside `box` that covers the most still-hidden cells while
    every one of its cells has an unoccluded partner under this symmetry."""
    R0, C0, R1, C1 = box
    best = None
    ncol = C1 - C0 + 1
    for top in range(R0, R1 + 1):
        ok = [True] * ncol
        colhid = [0] * ncol
        for bottom in range(top, R1 + 1):
            for k in range(ncol):
                c = C0 + k
                if not feas[bottom, c]:
                    ok[k] = False
                if hidden[bottom, c]:
                    colhid[k] += 1
            n = bottom - top + 1
            k = 0
            while k < ncol:
                if not ok[k]:
                    k += 1
                    continue
                j = k
                while j < ncol and ok[j]:
                    j += 1
                if square:
                    for s in range(k, j - n + 1):
                        cnt = sum(colhid[s:s + n])
                        if cnt:
                            cand = (-cnt, n * n, top, C0 + s)
                            if best is None or cand < best[0]:
                                best = (cand, (top, C0 + s, bottom, C0 + s + n - 1))
                else:
                    cnt = sum(colhid[k:j])
                    if cnt:
                        cand = (-cnt, n * (j - k), top, C0 + k)
                        if best is None or cand < best[0]:
                            best = (cand, (top, C0 + k, bottom, C0 + j - 1))
                k = j
    return best


def _replay(I, ops, sels):
    """The grid these operations leave, following ARCLE: Copy takes a region of
    the input, Paste writes its non-zero cells at a corner, a turn transforms
    the selected rectangle where it stands."""
    g = np.asarray(I, dtype=int).copy()
    clip = None
    for op, sel in zip(ops, sels):
        if isinstance(sel, dict):
            cells = sel["cells"]
            if op < 10:
                for r, c in cells:
                    g[r, c] = op
            continue
        r, c, h, w = sel
        if op < 10:
            g[r:r + h + 1, c:c + w + 1] = op
        elif op == 28:
            clip = np.asarray(I, dtype=int)[r:r + h + 1, c:c + w + 1].copy()
        elif op == 30 and clip is not None:
            ch, cw = clip.shape
            ch, cw = min(ch, g.shape[0] - r), min(cw, g.shape[1] - c)
            part = clip[:ch, :cw]
            tgt = g[r:r + ch, c:c + cw]
            g[r:r + ch, c:c + cw] = np.where(part != 0, part, tgt)
        elif op in (24, 25, 26, 27):
            g[r:r + h + 1, c:c + w + 1] = _turn(g[r:r + h + 1, c:c + w + 1], op)
    return g


def derive_operations(I, O=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []
    patchcol = _read_patch_colour(I)

    if patchcol is not None:
        R = _uncovered(I, patchcol)
        cur = I.copy()
        strokes = []                       # the operations of one rectangle each
        feas = {}
        for sym in _SYMS:
            m = np.zeros((H, W), dtype=bool)
            for r in range(H):
                for c in range(W):
                    p = _sym_cell(sym, r, c, H, W)
                    m[r, c] = p is not None and I[p] != patchcol
            feas[sym] = m

        def mirror_rect(sym, r1, c1, r2, c2):
            """Rebuild the rectangle out of the region this symmetry carries
            onto it: copy that region, paste it over, turn it into place."""
            h, w = r2 - r1 + 1, c2 - c1 + 1
            sr1, sc1, sr2, sc2 = _sym_rect(sym, r1, c1, r2, c2, H, W)
            src = I[sr1:sr2 + 1, sc1:sc2 + 1]
            reg = cur[r1:r2 + 1, c1:c2 + 1].copy()
            grp = []
            # a copy carries no black, so black cells of it are painted first
            blanks = [(r1 + int(i), c1 + int(j))
                      for i, j in np.argwhere((src == 0) & (reg != 0))]
            if blanks:
                grp.append((0, sel_of(blanks)))
                for r, c in blanks:
                    reg[r - r1, c - c1] = 0
            pasted = np.where(src != 0, src, reg)
            if not np.array_equal(pasted, reg):
                grp.append((28, [sr1, sc1, sr2 - sr1, sc2 - sc1]))
                grp.append((30, [r1, c1, 0, 0]))
                reg = pasted
            turned = reg
            for op in _POST[sym]:
                turned = _turn(turned, op)
            if not np.array_equal(turned, reg):     # a turn that changes nothing
                for op in _POST[sym]:               # is left out, and so is a pair
                    nxt = _turn(reg, op)            # of turns that undo each other
                    if not np.array_equal(nxt, reg):
                        grp.append((op, [r1, c1, h - 1, w - 1]))
                        reg = nxt
            cur[r1:r2 + 1, c1:c2 + 1] = reg
            strokes.append(grp)

        def paint_cell(r, c):
            strokes.append([(int(R[r, c]), sel_of([(r, c)]))])
            cur[r, c] = R[r, c]

        def restore(box):
            """Uncover everything still hidden inside this box, largest
            mirrorable rectangle first."""
            while True:
                hidden = cur != R
                sub = hidden[box[0]:box[2] + 1, box[1]:box[3] + 1]
                if not sub.any():
                    return
                rs, cs = np.nonzero(sub)
                tight = (box[0] + int(rs.min()), box[1] + int(cs.min()),
                         box[0] + int(rs.max()), box[1] + int(cs.max()))
                pick = None
                for sym in _SYMS:
                    got = _widest_rect(feas[sym], hidden, tight, bool(sym[0]))
                    if got is not None and (pick is None or got[0] < pick[0]):
                        pick = (got[0], sym, got[1])
                if pick is None:                    # no symmetry reaches it
                    return
                r1, c1, r2, c2 = pick[2]
                if r1 == r2 and c1 == c2:
                    paint_cell(r1, c1)              # a lone cell is just painted
                else:
                    mirror_rect(pick[1], r1, c1, r2, c2)

        for blot in _blots(I == patchcol):
            restore(blot)
        restore((0, 0, H - 1, W - 1))

        def replay(items):
            return _replay(I, [o for o, _ in items], [s for _, s in items])

        # a rectangle a later, larger one covers again is not a stroke of its own
        k = 0
        while k < len(strokes):
            rest = [it for j, g in enumerate(strokes) if j != k for it in g]
            if np.array_equal(replay(rest), R):
                del strokes[k]
                k = 0
            else:
                k += 1
        items = [it for g in strokes for it in g]

        # a copy of what the clipboard already holds is left out
        kept, clip = [], None
        for op, sel in items:
            if op == 28:
                r, c, h, w = sel
                src = I[r:r + h + 1, c:c + w + 1]
                if clip is not None and clip.shape == src.shape and np.array_equal(clip, src):
                    continue
                clip = src.copy()
            kept.append((op, sel))

        # and so is any single operation the result does not depend on
        while True:
            for k in range(len(kept)):
                if np.array_equal(replay(kept[:k] + kept[k + 1:]), R):
                    del kept[k]
                    break
            else:
                break

        ops = [op for op, _ in kept]
        sels = [sel for _, sel in kept]

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3631a71a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3631a71a"
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
                                f"for task 3631a71a"
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
                    f"Failed to build a complete episode for task 3631a71a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3631a71a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
