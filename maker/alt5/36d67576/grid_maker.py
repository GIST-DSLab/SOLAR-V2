"""
ARC Task: 36d67576 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    # 0 is "nothing" to Copy/Paste, so it may only be used as the background:
    # a 0-coloured object cell would never travel with the stamp.
    pool = [c for c in cols if c != bgc and (bgc == 0 or c != 0)]
    mainc, markerc = random.sample(pool, 2)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, mainc=None, markerc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or mainc is None or markerc is None:
        bgc = choice(totuple(cols))
        pool = [c for c in cols if c != bgc and (bgc == 0 or c != 0)]
        mainc, markerc = sample(pool, 2)
    remcols = [c for c in cols
               if c not in (bgc, mainc, markerc) and (bgc == 0 or c != 0)]
    hlo, wlo = min(10, max_h), min(10, max_w)

    def _norm(o):
        mi = min(p[0] for _, p in o)
        mj = min(p[1] for _, p in o)
        return frozenset((v, (p[0] - mi, p[1] - mj)) for v, p in o)

    syms = [lambda o: frozenset((v, (-i, j)) for v, (i, j) in o),
            lambda o: frozenset((v, (i, -j)) for v, (i, j) in o),
            lambda o: frozenset((v, (-i, -j)) for v, (i, j) in o),
            lambda o: frozenset((v, (j, i)) for v, (i, j) in o),
            lambda o: frozenset((v, (-j, -i)) for v, (i, j) in o),
            lambda o: frozenset((v, (j, -i)) for v, (i, j) in o),
            lambda o: frozenset((v, (-j, i)) for v, (i, j) in o)]

    def _conn8(cells):
        cells = set(cells)
        start = next(iter(cells))
        stack, seen = [start], {start}
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    p = (x + dx, y + dy)
                    if p in cells and p not in seen:
                        seen.add(p)
                        stack.append(p)
        return len(seen) == len(cells)

    def _workspace(cells, oh, ow, h, w):
        """Rectangles the copy can be stamped/turned inside of: its own box, or
        — when the copy lies turned a quarter way round — a square big enough
        for the quarter turn."""
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        dh, dw = max(rs) - r0 + 1, max(cs) - c0 + 1
        if dh == dw or (dh, dw) == (oh, ow):
            return [(r0, c0, dh, dw)]
        s = max(oh, ow)
        out = []
        for sr in range(r0 - (s - dh), r0 + 1):
            for sc in range(c0 - (s - dw), c0 + 1):
                if sr < 0 or sc < 0 or sr + s > h or sc + s > w:
                    continue
                out.append((sr, sc, s, s))
        return out

    def _fits(cells, occupied, oh, ow, h, w):
        own = set(cells)
        for (sr, sc, sh, sw) in _workspace(cells, oh, ow, h, w):
            reg = {(r, c) for r in range(sr, sr + sh) for c in range(sc, sc + sw)}
            if (reg & occupied) <= own:
                return True
        return False

    while True:
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        gi = canvas(bgc, (h, w))
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if choice((True, False)):
            oh, ow = ow, oh
        if oh > h or ow > w:
            continue
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (4, len(bounds)))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            cands = totuple((bounds - obj) & mapply(neighbors, obj))
            if len(cands) == 0:
                break
            obj.add(choice(cands))
        if len(obj) < 4:
            continue
        ncells = len(obj)
        obj = normalize(obj)
        oh, ow = shape(obj)
        ntocompc = unifint(diff_lb, diff_ub, (1, ncells - 3))
        markercell = choice(totuple(obj))
        remobj = remove(markercell, obj)
        markercellobj = {(markerc, markercell)}
        tocompc = set(sample(totuple(remobj), ntocompc))
        mainpart = (obj - {markercell}) - tocompc
        mainpartobj = recolor(mainc, mainpart)
        tocompcobj = {(choice(remcols), ij) for ij in tocompc}
        obj = tocompcobj | mainpartobj | markercellobj
        smobj = mainpartobj | markercellobj
        # every copy must show ONE connected fragment, so that a copy can be
        # told apart from its neighbours by looking at connected pieces
        if not _conn8(toindices(smobj)):
            continue
        # the fragment must pin the orientation down: any symmetry of the
        # fragment has to be a symmetry of the whole stamp as well, otherwise
        # a copy could be finished in two different ways
        smobjn = _norm(smobj)
        objn = _norm(obj)
        isfakesymm = False
        for symmf in syms:
            if _norm(symmf(smobjn)) == smobjn and _norm(symmf(objn)) != objn:
                isfakesymm = True
                break
        if isfakesymm:
            continue
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        plcd = shift(obj, (loci, locj))
        plcdi = toindices(plcd)
        tbox = {(i, j) for i in range(loci, loci + oh)
                for j in range(locj, locj + ow)}
        inds = (asindices(gi) - plcdi) - mapply(neighbors, plcdi)
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        succ = 0
        tr = 0
        maxtr = noccs * 5
        placements = []
        while tr < maxtr and succ < noccs:
            tr += 1
            mf1 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf2 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf = compose(mf1, mf2)
            outobj = normalize(mf(obj))
            inobj = sfilter(outobj, lambda cij: cij[0] in [mainc, markerc])
            ohh, oww = shape(outobj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - ohh and ij[1] <= w - oww)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            outobjp = shift(outobj, loc)
            inobjp = shift(inobj, loc)
            outobjpi = toindices(outobjp)
            # a copy also keeps out of the complete stamp's rectangle, so that
            # the stamp can be lifted as one rectangle
            if outobjpi.issubset(inds) and len(outobjpi & tbox) == 0:
                succ += 1
                inds = (inds - outobjpi) - mapply(neighbors, outobjpi)
                placements.append((outobjp, inobjp))
        if not placements:
            continue
        # keep only the copies that have room to be stamped and turned
        cur = list(placements)
        while cur:
            occupied = set(plcdi)
            for p in cur:
                occupied |= set(toindices(p[0]))
            nxt = [p for p in cur
                   if _fits(toindices(p[0]), occupied, oh, ow, h, w)]
            if len(nxt) == len(cur):
                break
            cur = nxt
        if not cur:
            continue
        go = paint(gi, plcd)
        gi = go
        for (outobjp, inobjp) in cur:
            gi = paint(gi, inobjp)
            go = paint(go, outobjp)
        return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    submit = [0, 0, h - 1, w - 1]

    # background: the colour the canvas is filled with before anything is drawn
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            try:
                cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
            except Exception:
                pass
    bgc = int(cnt.most_common(1)[0][0])

    def bbox(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return int(min(rs)), int(min(cs)), \
            int(max(rs) - min(rs) + 1), int(max(cs) - min(cs) + 1)

    # ── the pieces lying on the canvas ───────────────────────────────────────
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack, cells = [(r, c)], []
                seen[r, c] = True
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and \
                                    not seen[nx, ny] and I[nx, ny] != bgc:
                                seen[nx, ny] = True
                                stack.append((nx, ny))
                comps.append(cells)
    if len(comps) < 2:
        return [34], [submit]

    # the complete stamp is the only piece carrying its completion colours,
    # so it is the biggest one
    comps.sort(key=lambda cc: -len(cc))
    tmpl, others = comps[0], comps[1:]
    tr0, tc0, oh, ow = bbox(tmpl)
    TPL = np.full((oh, ow), bgc, dtype=int)
    for (r, c) in tmpl:
        TPL[r - tr0, c - tc0] = I[r, c]
    # the half-finished copies show only these colours
    pal = sorted({int(I[r, c]) for comp in others for (r, c) in comp})
    FRAG = np.where(np.isin(TPL, pal), TPL, bgc)   # the stamp's visible part

    # every way the stamp can lie, with the ARCLE ops that put it that way:
    # flips inside the stamped rectangle first, then — when the copy lies a
    # quarter turn round — one rotation of the square that rectangle sits in
    tfs = [
        (lambda a: a, [], None),
        (lambda a: np.fliplr(a), [26], None),
        (lambda a: np.flipud(a), [27], None),
        (lambda a: np.rot90(a, 1), [], 24),
        (lambda a: np.rot90(a, 3), [], 25),
        (lambda a: np.flipud(np.fliplr(a)), [26, 27], None),
        (lambda a: np.ascontiguousarray(a.T), [26], 24),
        (lambda a: np.rot90(np.ascontiguousarray(a.T), 2), [27], 24),
    ]

    # ── read each copy: how it lies, and where its whole stamp belongs ───────
    copies = []
    for comp in others:
        cr, cc, chh, cww = bbox(comp)
        CARR = np.full((chh, cww), bgc, dtype=int)
        for (r, c) in comp:
            CARR[r - cr, c - cc] = I[r, c]
        for (f, flips, rot) in tfs:                  # shortest route first
            A = f(FRAG)
            nz = np.argwhere(A != bgc)
            if nz.size == 0:
                continue
            ar, ac = int(nz[:, 0].min()), int(nz[:, 1].min())
            ah, aw = int(nz[:, 0].max()) - ar + 1, int(nz[:, 1].max()) - ac + 1
            if (ah, aw) != (chh, cww):
                continue
            if not np.array_equal(A[ar:ar + ah, ac:ac + aw], CARR):
                continue
            copies.append({"cells": [(int(r), int(c)) for (r, c) in comp],
                           "flips": flips, "rot": rot,
                           "r": cr - ar, "c": cc - ac, "A": f(TPL)})
            break
    if not copies:
        return [34], [submit]
    copies.sort(key=lambda k: (k["r"], k["c"]))

    # what the canvas has to end up holding — read off I and the copies, never O
    target = I.copy()
    for k in copies:
        A, r0, c0 = k["A"], k["r"], k["c"]
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] != bgc:
                    target[r0 + i, c0 + j] = A[i, j]
    occupied = {(int(r), int(c)) for comp in comps for (r, c) in comp}
    occupied |= {(r, c) for r in range(h) for c in range(w)
                 if target[r, c] != bgc}

    # ── ARCLE's own semantics, to keep every op honest ───────────────────────
    def do_paste(g, clip, pr, pc):
        for i in range(clip.shape[0]):
            for j in range(clip.shape[1]):
                if clip[i, j] != 0 and pr + i < h and pc + j < w:
                    g[pr + i, pc + j] = clip[i, j]

    def do_obj(g, r0, c0, hh, ww, op):
        reg = g[r0:r0 + hh, c0:c0 + ww]
        new = {26: np.fliplr, 27: np.flipud,
               24: lambda a: np.rot90(a, 1),
               25: lambda a: np.rot90(a, 3)}[op](reg)
        g[r0:r0 + hh, c0:c0 + ww] = np.where(new != 0, new, 0)

    def replay(rops, rsels):
        g = I.copy()
        clip = np.zeros((1, 1), dtype=int)
        for op, sel in zip(rops, rsels):
            if op == 34:
                break
            if op == 28:
                r, c, hh, ww = sel
                clip = I[r:r + hh + 1, c:c + ww + 1].copy()
            elif op == 30:
                do_paste(g, clip, sel[0], sel[1])
            elif op in (24, 25, 26, 27):
                do_obj(g, sel[0], sel[1], sel[2] + 1, sel[3] + 1, op)
            elif op < 10:
                if isinstance(sel, dict):
                    for (r, c) in sel["cells"]:
                        g[r, c] = op
                else:
                    g[sel[0]:sel[0] + sel[2] + 1, sel[1]:sel[1] + sel[3] + 1] = op
        return g

    # ── stamp every copy ─────────────────────────────────────────────────────
    clip = TPL                       # what CopyI puts on the clipboard
    grid = I.copy()
    ops, sels = [], []
    wipes = []                       # where a fragment gets wiped first
    used = set()                     # squares already worked in
    # the complete stamp is lifted once and stamped onto every copy
    ops.append(28)
    sels.append([tr0, tc0, oh - 1, ow - 1])

    for k in copies:
        A, r0, c0 = k["A"], k["r"], k["c"]
        dh, dw = int(A.shape[0]), int(A.shape[1])
        own = {(r0 + i, c0 + j) for i in range(dh) for j in range(dw)
               if A[i, j] != bgc} | set(k["cells"])
        # room to stamp and turn: the copy's own box, or — for a quarter turn —
        # a square around it, slid to wherever it stands clear
        if dh == dw or (dh, dw) == (oh, ow):
            spots = [(r0, c0, dh, dw)]
        else:
            s = max(oh, ow)
            spots = [(sr, sc, s, s)
                     for sr in range(r0 - (s - dh), r0 + 1)
                     for sc in range(c0 - (s - dw), c0 + 1)
                     if 0 <= sr and 0 <= sc and sr + s <= h and sc + s <= w]
        free = []
        for (sr, sc, sh, sw) in spots:
            reg = {(r, c) for r in range(sr, sr + sh) for c in range(sc, sc + sw)}
            if (reg & occupied) <= own:
                free.append(((sr, sc, sh, sw), reg))
        spot = None
        for (rect, reg) in free:                 # keep out of earlier squares
            if not (reg & used):
                spot = rect
                break
        if spot is None and free:
            spot = free[0][0]
        if spot is None:
            # hemmed in by a neighbour: this copy is finished mark by mark
            bycol = {}
            for i in range(dh):
                for j in range(dw):
                    v = int(A[i, j])
                    if v != bgc and grid[r0 + i, c0 + j] != v:
                        bycol.setdefault(v, []).append((r0 + i, c0 + j))
            for v in sorted(bycol):
                ops.append(v)
                sels.append(sel_of(bycol[v]))
                for (r, c) in bycol[v]:
                    grid[r, c] = v
            continue
        wr, wc, wh, ww = spot
        used |= {(r, c) for r in range(wr, wr + wh) for c in range(wc, wc + ww)}
        if k["rot"] is None:
            pr, pc = r0, c0
        else:
            dr, dc = r0 - wr, c0 - wc
            if k["rot"] == 24:
                pr, pc = wr + dc, wc + wh - ow - dr
            else:
                pr, pc = wr + wh - oh - dc, wc + dr
        # wipe the fragment first when the stamp does not cover it and the turn
        # would otherwise drag its cells off to the wrong places
        probe = grid.copy()
        do_paste(probe, clip, pr, pc)
        for fop in k["flips"]:
            do_obj(probe, pr, pc, oh, ow, fop)
        if k["rot"] is not None:
            do_obj(probe, wr, wc, wh, ww, k["rot"])
        if not np.array_equal(probe[wr:wr + wh, wc:wc + ww],
                              target[wr:wr + wh, wc:wc + ww]):
            wipes.append(len(ops))
            ops.append(int(bgc))
            sels.append(sel_of(k["cells"]))
            for (r, c) in k["cells"]:
                grid[r, c] = bgc
        ops.append(30)
        sels.append([int(pr), int(pc), 0, 0])
        do_paste(grid, clip, pr, pc)
        for fop in k["flips"]:
            ops.append(int(fop))
            sels.append([int(pr), int(pc), oh - 1, ow - 1])
            do_obj(grid, pr, pc, oh, ow, fop)
        if k["rot"] is not None:
            ops.append(int(k["rot"]))
            sels.append([int(wr), int(wc), wh - 1, ww - 1])
            do_obj(grid, wr, wc, wh, ww, k["rot"])

    # a wipe whose cells a neighbouring stamp happens to sweep away later is
    # not part of the account — drop it
    if wipes:
        keep = [True] * len(ops)
        for idx in wipes:
            keep[idx] = False
            trial_ops = [o for i, o in enumerate(ops) if keep[i]]
            trial_sels = [s for i, s in enumerate(sels) if keep[i]]
            if not np.array_equal(replay(trial_ops, trial_sels), target):
                keep[idx] = True
        ops = [o for i, o in enumerate(ops) if keep[i]]
        sels = [s for i, s in enumerate(sels) if keep[i]]

    if not np.array_equal(replay(ops, sels), target):
        # safety net: never leave a copy half finished
        ops, sels = [], []
        grid = I.copy()
        for k in copies:
            A, r0, c0 = k["A"], k["r"], k["c"]
            bycol = {}
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    v = int(A[i, j])
                    if v != bgc and grid[r0 + i, c0 + j] != v:
                        bycol.setdefault(v, []).append((r0 + i, c0 + j))
            for v in sorted(bycol):
                ops.append(v)
                sels.append(sel_of(bycol[v]))
                for (r, c) in bycol[v]:
                    grid[r, c] = v

    ops.append(34)
    sels.append(submit)
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
                        f"num_examples+1 ({num_examples + 1}) for task 36d67576"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 36d67576"
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
                                f"for task 36d67576"
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
                    f"Failed to build a complete episode for task 36d67576 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"36d67576-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
