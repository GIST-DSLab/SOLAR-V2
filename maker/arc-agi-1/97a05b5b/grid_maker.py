"""
ARC Task: 97a05b5b (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback if helper unavailable
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # generator samples exactly two structural colors: bgc (canvas) and sqc
    # (the big board / output background).  The per-object "key" colors are
    # role-free (the rule only says "a hole gets its matching key's colour"),
    # so they stay random inside generate().
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC generator with max_h/max_w and fixed bgc/sqc)
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, sqc: int) -> dict:
    cols = interval(0, 10, 1)
    hhi = min(30, max_h)
    hlo = min(15, hhi)
    whi = min(30, max_w)
    wlo = min(15, whi)
    h = unifint(diff_lb, diff_ub, (hlo, hhi))
    w = unifint(diff_lb, diff_ub, (wlo, whi))
    sgh = randint(h // 3, h // 3 * 2)
    sgw = randint(w // 3, w // 3 * 2)
    sgh = max(4, min(sgh, h))
    sgw = max(4, min(sgw, w))
    remcols = remove(bgc, remove(sqc, cols))
    gi = canvas(bgc, (h, w))
    oh = randint(2, sgh // 2)
    ow = randint(2, sgw // 2)
    nobjs = unifint(diff_lb, diff_ub, (1, 8))
    objs = set()
    cands = asindices(canvas(-1, (oh, ow)))
    forbidden = set()
    tr = 0
    maxtr = 4 * nobjs
    while len(objs) != nobjs and tr < maxtr:
        tr += 1
        obj = {choice(totuple(cands))}
        ncells = randint(1, oh * ow - 1)
        for k in range(ncells - 1):
            rem = totuple((cands - obj) & mapply(neighbors, obj))
            if len(rem) == 0:
                break
            obj.add(choice(rem))
        obj |= choice((dmirror, cmirror, vmirror, hmirror))(obj)
        if len(obj) == height(obj) * width(obj):
            continue
        obj = frozenset(obj)
        objn = normalize(obj)
        if objn not in forbidden:
            objs.add(objn)
        for augmf1 in (identity, dmirror, cmirror, hmirror, vmirror):
            for augmf2 in (identity, dmirror, cmirror, hmirror, vmirror):
                forbidden.add(augmf1(augmf2(objn)))
    tr = 0
    maxtr = 5 * nobjs
    succ = 0
    loci = randint(0, h - sgh)
    locj = randint(0, w - sgw)
    bd = backdrop(frozenset({(loci, locj), (loci + sgh - 1, locj + sgw - 1)}))
    gi = fill(gi, sqc, bd)
    go = canvas(sqc, (sgh, sgw))
    goinds = asindices(go)
    giinds = asindices(gi) - shift(goinds, (loci, locj))
    giinds = giinds - mapply(neighbors, shift(goinds, (loci, locj)))
    while succ < nobjs and tr < maxtr and len(objs) > 0:
        tr += 1
        obj = choice(totuple(objs))
        col = choice(remcols)
        subgi = fill(canvas(col, shape(obj)), sqc, obj)
        if len(palette(subgi)) == 1:
            continue
        f1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        f2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        f = compose(f1, f2)
        subgo = f(subgi)
        giobj = asobject(subgi)
        goobj = asobject(subgo)
        ohi, owi = shape(giobj)
        oho, owo = shape(goobj)
        gocands = sfilter(goinds, lambda ij: ij[0] <= sgh - oho and ij[1] <= sgw - owo)
        if len(gocands) == 0:
            continue
        goloc = choice(totuple(gocands))
        goplcd = shift(goobj, goloc)
        goplcdi = toindices(goplcd)
        if goplcdi.issubset(goinds):
            gicands = sfilter(giinds, lambda ij: ij[0] <= h - ohi and ij[1] <= owi)
            if len(gicands) == 0:
                continue
            giloc = choice(totuple(gicands))
            giplcd = shift(giobj, giloc)
            giplcdi = toindices(giplcd)
            if giplcdi.issubset(giinds):
                succ += 1
                remcols = remove(col, remcols)
                objs = remove(obj, objs)
                goinds = goinds - goplcdi
                giinds = (giinds - giplcdi) - mapply(neighbors, giplcdi)
                gi = paint(gi, giplcd)
                gi = fill(gi, bgc, sfilter(shift(goplcd, (loci, locj)), lambda cij: cij[0] == sqc))
                go = paint(go, goplcd)
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def _components(grid, bg):
    """4-connected components of non-background cells (colours may mix)."""
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = x + dx, y + dy
                    if 0 <= a < h and 0 <= b < w and not seen[a, b] and grid[a, b] != bg:
                        seen[a, b] = True
                        stack.append((a, b))
            comps.append(cells)
    return comps


def _variants(m):
    """The 8 dihedral variants of a boolean shape mask, de-duplicated."""
    out, seen = [], set()
    for v in (m, np.rot90(m, 2), np.rot90(m, 3), np.rot90(m, 1),
              np.flipud(m), np.fliplr(m), m.T, np.rot90(m.T, 2)):
        v = np.ascontiguousarray(v)
        key = (v.shape, v.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def derive_operations(I, O):
    """
    Rule: the big single-colour board (colour sqc) carries background-coloured
    HOLES.  Each hole cluster is the shape of one small two-colour "key" patch
    lying outside the board (in some rotation/mirroring).  The answer is the
    board alone, with every hole replaced by that key patch itself: the key's
    rectangle stamped in the key's colour, the key's shape carved back to sqc.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- separate the key patches (solid 2-colour rectangles) from the board --
    comps = _components(I, bgc)
    patch_boxes, rest = [], []
    for cells in comps:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        colors = {int(I[r, c]) for r, c in cells}
        if len(cells) == area and len(colors) == 2:
            patch_boxes.append((r0, c0, r1, c1))
        else:
            rest.extend(cells)
    if not rest:
        rest = [cell for cells in comps for cell in cells]
        patch_boxes = []

    sqc = Counter(int(I[r, c]) for r, c in rest).most_common(1)[0][0]
    R0 = min(r for r, _ in rest)
    R1 = max(r for r, _ in rest)
    C0 = min(c for _, c in rest)
    C1 = max(c for _, c in rest)

    # board bbox must have the output's size; if a whole border row/col of the
    # board is holed, slide a ho x wo window that still holds the whole board.
    if (R1 - R0 + 1, C1 - C0 + 1) != (ho, wo):
        pm = np.zeros((hi, wi), bool)
        for (a, b, c, d) in patch_boxes:
            pm[a:c + 1, b:d + 1] = True
        found = None
        for r in range(max(0, R1 - ho + 1), max(0, min(R0, hi - ho)) + 1):
            for c in range(max(0, C1 - wo + 1), max(0, min(C0, wi - wo)) + 1):
                if pm[r:r + ho, c:c + wo].any():
                    continue
                if not np.isin(I[r:r + ho, c:c + wo], [sqc, bgc]).all():
                    continue
                found = (r, c)
                break
            if found:
                break
        if found:
            R0, C0 = found

    crop = I[R0:R0 + ho, C0:C0 + wo]
    holes = (crop == bgc)

    # --- describe every key patch: its colour and its shape mask -------------
    descs = []
    for (a, b, c, d) in patch_boxes:
        sub = I[a:c + 1, b:d + 1]
        colors = [int(x) for x in np.unique(sub)]
        if sqc not in colors or len(colors) != 2:
            continue
        col = colors[0] if colors[1] == sqc else colors[1]
        descs.append((col, sub == sqc))

    # --- where can each key's shape sit?  (exact hole/no-hole match) ---------
    cand_lists = []
    for col, m in descs:
        cl = []
        for v in _variants(m):
            vh, vw = v.shape
            if vh > ho or vw > wo:
                continue
            n = int(v.sum())
            for r in range(ho - vh + 1):
                for c in range(wo - vw + 1):
                    if np.array_equal(holes[r:r + vh, c:c + vw], v):
                        cl.append((r, c, vh, vw, n, v))
        cand_lists.append(cl)

    # --- exact cover: key rectangles are disjoint and cover every hole -------
    n_p = len(cand_lists)
    total_holes = int(holes.sum())
    order = sorted(range(n_p), key=lambda i: len(cand_lists[i]))
    sol = [None] * n_p
    occ = np.zeros((ho, wo), bool)
    state = {"cov": 0, "nodes": 0}

    def rec(k):
        state["nodes"] += 1
        if state["nodes"] > 200000:
            return False
        if k == n_p:
            return state["cov"] == total_holes
        i = order[k]
        for cand in cand_lists[i]:
            r, c, vh, vw, n, v = cand
            if occ[r:r + vh, c:c + vw].any():
                continue
            occ[r:r + vh, c:c + vw] = True
            sol[i] = cand
            state["cov"] += n
            if rec(k + 1):
                return True
            state["cov"] -= n
            sol[i] = None
            occ[r:r + vh, c:c + vw] = False
        return False

    if not rec(0):
        # greedy fallback: first non-overlapping placement per key
        sol = [None] * n_p
        occ[:] = False
        for i in order:
            for cand in cand_lists[i]:
                r, c, vh, vw, n, v = cand
                if occ[r:r + vh, c:c + vw].any():
                    continue
                occ[r:r + vh, c:c + vw] = True
                sol[i] = cand
                break

    placements = []
    for i, cand in enumerate(sol):
        if cand is None:
            continue
        r, c, vh, vw, n, v = cand
        placements.append((r, c, vh, vw, descs[i][0], v))
    placements.sort(key=lambda p: (p[0], p[1]))

    # --- paint each key into its hole, on the still-complete grid ------------
    sim = crop.copy()
    for (r, c, vh, vw, col, v) in placements:
        # stamp the key's whole rectangle (every cell here changes: the holes
        # were bgc, the surround was sqc) -- bbox IS exactly the intended cells
        ops.append(int(col))
        sels.append([int(R0 + r), int(C0 + c), int(vh - 1), int(vw - 1)])
        sim[r:r + vh, c:c + vw] = col
        # carve the key's own shape back to the board colour
        cells = [(R0 + r + i, C0 + c + j)
                 for i in range(vh) for j in range(vw) if v[i, j]]
        ops.append(int(sqc))
        sels.append(sel_of(cells))
        for i in range(vh):
            for j in range(vw):
                if v[i, j]:
                    sim[r + i, c + j] = sqc

    # safety net (only if the structural match above did not fully resolve)
    diff = np.argwhere(sim != O)
    if len(diff) > 0:
        for color in sorted({int(O[r, c]) for r, c in diff}):
            cells = [(int(R0 + r), int(C0 + c)) for r, c in diff if int(O[r, c]) == color]
            ops.append(color)
            sels.append(sel_of(cells))

    # --- keep only the board -------------------------------------------------
    # full rectangle: the board region itself
    ops.append(33)
    sels.append([int(R0), int(C0), int(ho - 1), int(wo - 1)])
    ops.append(34)
    sels.append([0, 0, int(ho - 1), int(wo - 1)])
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
                        f"num_examples+1 ({num_examples + 1}) for task 97a05b5b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 97a05b5b"
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
                                f"for task 97a05b5b"
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
                    f"Failed to build a complete episode for task 97a05b5b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"97a05b5b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
