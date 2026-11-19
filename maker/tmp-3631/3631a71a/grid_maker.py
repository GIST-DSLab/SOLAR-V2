"""
ARC Task: 3631a71a (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

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
from maker.sel_helpers import sel_of

import importlib
for _m in ["utils", "dsl", "generators"]:
    if _m in sys.modules:
        del sys.modules[_m]
from utils import *  # noqa
from dsl import *    # noqa

# ── LLM-generated ────────────────────────────────────────────────────────────
import random
import numpy as np


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    patchcol = random.choice(cols)
    bgc = random.choice([c for c in cols if c != patchcol])
    return {"bgc": bgc, "patchcol": patchcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, patchcol) -> dict:
    cols = interval(0, 10, 1)
    lim = min(max_h, max_w)
    hi_h = max(6, min(15, (lim + 2) // 2))
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
        # nothing was occluded: there would be no transformation to show
        raise ValueError("no patch placed")
    gia = np.array(gi, dtype=int)
    orbs = _symmetry_orbits(gia.shape[0])
    if _find_patchcol(gia, orbs) != patchcol:
        # the occluding colour must be recognisable from the input alone
        raise ValueError("patch colour not identifiable from input")
    return {'input': gi, 'output': go}


# ── derive_operations ────────────────────────────────────────────────────────
#
# What the input hides: the picture is a transpose-symmetric block mirrored to
# the right and down, with the last two rows and columns dropped.  On the n x n
# canvas that leaves three mirror symmetries,
#       (r, c) -> (c, r)            main diagonal
#       (r, c) -> (n+1-r, c)        row axis      (only for r >= 2)
#       (r, c) -> (r, n+1-c)        column axis   (only for c >= 2)
# and rectangular patches of one colour sit on top of it.  Repairing a patch
# means copying the region it mirrors onto and mirroring that copy back over it,
# which is what the operations below do: CopyI the (unoccluded) mirror region,
# Paste it over the patch, then Flip/Rotate it in place about the axis that
# relates the two.  Every parameter — which colour occludes, which rectangles,
# which axis, how far away the source lies — is measured from the input.


def _symmetry_orbits(n):
    """Cells of an n x n grid grouped into their symmetry orbits (see above)."""
    parent = list(range(n * n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in range(n):
        for c in range(n):
            i = r * n + c
            union(i, c * n + r)
            if r >= 2:
                union(i, (n + 1 - r) * n + c)
            if c >= 2:
                union(i, r * n + (n + 1 - c))
    groups = {}
    for r in range(n):
        for c in range(n):
            groups.setdefault(find(r * n + c), []).append((r, c))
    return list(groups.values())


def _find_patchcol(I, orbits):
    """The colour of the occluding patches, measured from I alone.

    A patch is a solid rectangle of at least 2x2, it never reaches the top-left
    2x2 corner (that corner has no mirror partner), and once its cells are set
    aside every symmetry orbit must still show exactly one colour.
    """
    H, W = I.shape
    corner = set(I[0:2, 0:2].flatten().tolist())
    best = None
    for v in sorted(set(I.flatten().tolist())):
        if v in corner:
            continue
        mask = (I == v)
        # every cell of a patch lies inside a solid 2x2 block of the patch
        blocky = True
        for r, c in np.argwhere(mask):
            hit = False
            for r0 in (r - 1, r):
                for c0 in (c - 1, c):
                    if 0 <= r0 < H - 1 and 0 <= c0 < W - 1 and mask[r0:r0 + 2, c0:c0 + 2].all():
                        hit = True
            if not hit:
                blocky = False
                break
        if not blocky:
            continue
        good = True
        for g in orbits:
            rest = {int(I[r, c]) for r, c in g if int(I[r, c]) != v}
            if len(rest) != 1:
                good = False
                break
        if not good:
            continue
        cnt = int(mask.sum())
        if best is None or cnt > best[0]:
            best = (cnt, int(v))
    return None if best is None else best[1]


def _restore(I, patchcol, orbits):
    """The picture the input hides: each orbit shows the one colour it kept."""
    R = I.copy()
    for g in orbits:
        rest = {int(I[r, c]) for r, c in g if int(I[r, c]) != patchcol}
        if len(rest) != 1:
            continue
        v = rest.pop()
        for r, c in g:
            R[r, c] = v
    return R


def _components(cells):
    """4-connected blots of the occluding colour, ordered by top-left cell."""
    todo = set(cells)
    comps = []
    while todo:
        seed = min(todo)
        todo.discard(seed)
        stack, comp = [seed], []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in todo:
                    todo.discard((nr, nc))
                    stack.append((nr, nc))
        comps.append(sorted(comp))
    comps.sort(key=lambda comp: comp[0])
    return comps


def _mirror_sources(r1, c1, r2, c2, n):
    """Regions of the input that mirror onto the rectangle (r1,c1)-(r2,c2).

    Each entry is (source rectangle, ops that map the copy onto the target):
    27 = FlipV (row axis), 26 = FlipH (column axis), 25 = Rotate CW.
    Transpose = Rotate CW + FlipH, anti-transpose = Rotate CW + FlipV; both need
    a square selection, so they are only offered for a square rectangle.
    """
    out = []
    if r1 >= 2:                                  # mirror across the row axis
        out.append(((n + 1 - r2, c1, n + 1 - r1, c2), (27,)))
    if c1 >= 2:                                  # mirror across the column axis
        out.append(((r1, n + 1 - c2, r2, n + 1 - c1), (26,)))
    if r1 >= 2 and c1 >= 2:                      # both axes at once
        out.append(((n + 1 - r2, n + 1 - c2, n + 1 - r1, n + 1 - c1), (26, 27)))
    if (r2 - r1) == (c2 - c1):
        out.append(((c1, r1, c2, r2), (25, 26)))                 # main diagonal
        if r1 >= 2 and c1 >= 2:                                  # minor diagonal
            out.append(((n + 1 - c2, n + 1 - r2, n + 1 - c1, n + 1 - r1), (25, 27)))
    return out


def _apply_flip(region, op):
    if op == 26:
        return np.fliplr(region)
    if op == 27:
        return np.flipud(region)
    return np.rot90(region, 3)      # op 25 = rotate clockwise


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    orbits = _symmetry_orbits(hi) if hi == wi else []
    patchcol = _find_patchcol(I, orbits) if orbits else None

    if patchcol is not None:
        R = _restore(I, patchcol, orbits)
        cur = I.copy()

        def paint_cells(cells):
            """Last resort for cells no whole rectangle could be mirrored onto:
            colour them with what their own orbit still shows in the input."""
            by_col = {}
            for r, c in cells:
                by_col.setdefault(int(R[r, c]), []).append((r, c))
            for col in sorted(by_col, key=lambda k: (-len(by_col[k]), k)):
                ops.append(int(col))
                sels.append(sel_of(by_col[col]))
                for r, c in by_col[col]:
                    cur[r, c] = col

        def mirror_rect(r1, c1, r2, c2):
            """Rebuild the rectangle from the region it mirrors onto it, when
            that region is unoccluded.  Returns True when it worked."""
            th, tw = r2 - r1 + 1, c2 - c1 + 1
            target = R[r1:r2 + 1, c1:c2 + 1]
            for (sr1, sc1, sr2, sc2), flips in _mirror_sources(r1, c1, r2, c2, hi):
                if not (0 <= sr1 <= sr2 < hi and 0 <= sc1 <= sc2 < wi):
                    continue
                if not (I[sr1:sr2 + 1, sc1:sc2 + 1] == patchcol).any():
                    copy_op = 28                      # mirror region of the input
                    clip = I[sr1:sr2 + 1, sc1:sc2 + 1]
                elif (not (cur[sr1:sr2 + 1, sc1:sc2 + 1] != R[sr1:sr2 + 1, sc1:sc2 + 1]).any()
                      and (sr2 < r1 or r2 < sr1 or sc2 < c1 or c2 < sc1)):
                    copy_op = 29                      # mirror region already rebuilt
                    clip = cur[sr1:sr2 + 1, sc1:sc2 + 1]
                else:
                    continue                          # mirror region still hidden
                clip = np.asarray(clip)
                if clip.shape != (th, tw):
                    continue
                for clear in (False, True):
                    reg = cur[r1:r2 + 1, c1:c2 + 1].copy()
                    steps = []
                    if clear:
                        if not (reg != 0).any():
                            continue                  # nothing to clear
                        steps.append(("clear", None))
                        reg = np.zeros_like(reg)
                    pasted = np.where(clip != 0, clip, reg)
                    if not np.array_equal(pasted, reg):
                        steps.append(("paste", None))
                        reg = pasted
                    for f in flips:
                        nxt = _apply_flip(reg, f)
                        if not np.array_equal(nxt, reg):
                            steps.append(("flip", f))
                            reg = nxt
                    if not steps or not np.array_equal(reg, target):
                        continue
                    # both selections are whole rectangles, background included
                    rect_sel = [r1, c1, th - 1, tw - 1]
                    src_sel = [sr1, sc1, sr2 - sr1, sc2 - sc1]
                    if any(k == "paste" for k, _ in steps):
                        ops.append(copy_op)
                        sels.append(src_sel)
                    for kind, f in steps:
                        if kind == "clear":
                            ops.append(0)
                            sels.append(rect_sel)
                        elif kind == "paste":
                            ops.append(30)
                            sels.append([r1, c1, 0, 0])
                        else:
                            ops.append(f)
                            sels.append(rect_sel)
                    cur[r1:r2 + 1, c1:c2 + 1] = reg
                    return True
            return False

        def repair(r1, c1, r2, c2, allow_paint):
            """Restore everything still hidden inside this rectangle."""
            bad = cur[r1:r2 + 1, c1:c2 + 1] != R[r1:r2 + 1, c1:c2 + 1]
            if not bad.any():
                return
            rs, cs = np.nonzero(bad)
            r1, r2 = r1 + int(rs.min()), r1 + int(rs.max())
            c1, c2 = c1 + int(cs.min()), c1 + int(cs.max())
            if mirror_rect(r1, c1, r2, c2):
                return
            th, tw = r2 - r1 + 1, c2 - c1 + 1
            if th == 1 and tw == 1:
                if allow_paint:
                    paint_cells([(r1, c1)])
                return
            if th >= tw:                       # split the rectangle and mirror the halves
                mid = (r1 + r2) // 2
                repair(r1, c1, mid, c2, allow_paint)
                repair(mid + 1, c1, r2, c2, allow_paint)
            else:
                mid = (c1 + c2) // 2
                repair(r1, c1, r2, mid, allow_paint)
                repair(r1, mid + 1, r2, c2, allow_paint)

        blots = []
        for comp in _components([(int(r), int(c)) for r, c in np.argwhere(I == patchcol)]):
            rs = [int(r) for r, _ in comp]
            cs = [int(c) for _, c in comp]
            blots.append((min(rs), min(cs), max(rs), max(cs)))

        # repeat while blots keep getting rebuilt: a blot whose mirror region was
        # itself hidden can be rebuilt once that region has been restored
        while True:
            before = cur.copy()
            for rect in blots:
                repair(*rect, allow_paint=False)
            if np.array_equal(before, cur):
                break
        for rect in blots:
            repair(*rect, allow_paint=True)

        left = [(int(r), int(c)) for r, c in np.argwhere(cur != R)]
        if left:
            paint_cells(left)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
    return ops, sels


# ── GridMaker ────────────────────────────────────────────────────────────────

class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[
        List[NDArray], List[NDArray],
        List[NDArray], List[NDArray],
        Dict[str, Any],
    ]]:
        num_samples = kwargs.get("num_samples", 1)
        num_examples = kwargs.get("num_examples", 3)
        max_h, max_w = kwargs.get("max_grid_dim", [30, 30])
        dataset = []

        for _sn in range(num_samples):
            pr_in: List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in: List[NDArray] = []
            ex_out: List[NDArray] = []
            ops: List[int] = []
            sels: List[Any] = []

            colors = sample_colors(num_examples)
            plan = colors.pop("instance_plan", None)

            j = 0
            while j < num_examples + 1:
                ok = False
                kw = dict(colors)
                if plan is not None:
                    kw.update(plan[j])
                for _ in range(20):
                    try:
                        r = generate(
                            random.uniform(0.2, 0.5),
                            random.uniform(0.5, 0.8),
                            max_h, max_w,
                            **kw,
                        )
                        I = np.array(r["input"], dtype=np.uint8)
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
                    j += 1
                    continue
                if j == num_examples:
                    pr_in.append(I)
                    pr_out.append(O)
                    ops, sels = derive_operations(I, O)
                else:
                    ex_in.append(I)
                    ex_out.append(O)
                j += 1

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id": f"3631a71a-rearc-llm_{_sn + 1}",
                "concept": "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
