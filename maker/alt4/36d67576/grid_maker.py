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
# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
import numpy as np
from collections import Counter

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # 0 is deliberately left out of every role: the clipboard treats 0 as
    # "nothing", so a stamp that used 0 could not be copied and pasted whole.
    cols = list(range(1, 10))
    bgc, mainc, markerc = random.sample(cols, 3)
    remcols = [c for c in cols if c not in (bgc, mainc, markerc)]
    ncols = random.randint(1, len(remcols))
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, mainc, markerc, ccols) -> dict:
    while True:
        h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
        gi = canvas(bgc, (h, w))
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if choice((True, False)):
            oh, ow = ow, oh
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (4, len(bounds)))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        if oh > h or ow > w:
            continue
        ntocompc = unifint(diff_lb, diff_ub, (1, max(1, ncells - 3)))
        markercell = choice(totuple(obj))
        remobj = remove(markercell, obj)
        markercellobj = {(markerc, markercell)}
        ntocompc = min(ntocompc, len(remobj))
        tocompc = set(sample(totuple(remobj), ntocompc))
        mainpart = (obj - {markercell}) - tocompc
        if len(mainpart) == 0:
            continue
        mainpartobj = recolor(mainc, mainpart)
        tocompcobj = {(choice(ccols), ij) for ij in tocompc}
        obj = tocompcobj | mainpartobj | markercellobj
        smobj = mainpartobj | markercellobj
        smobjn = normalize(smobj)
        # A stamp whose main+marker part is left unchanged by a symmetry that
        # does move the completed stamp would make the answer a coin flip, so
        # every symmetry the placement loop can draw is checked, not only four.
        isfakesymm = False
        mfs = (identity, dmirror, cmirror, hmirror, vmirror)
        for mf1 in mfs:
            for mf2 in mfs:
                symmf = compose(mf1, mf2)
                if normalize(symmf(smobjn)) == smobjn \
                        and normalize(symmf(obj)) != normalize(obj):
                    isfakesymm = True
                    break
            if isfakesymm:
                break
        if isfakesymm:
            continue
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        plcd = shift(obj, (loci, locj))
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (asindices(gi) - plcdi) - mapply(neighbors, plcdi)
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        succ = 0
        tr = 0
        maxtr = noccs * 5
        go = tuple(e for e in gi)
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
            if outobjpi.issubset(inds):
                succ += 1
                inds = (inds - outobjpi) - mapply(neighbors, outobjpi)
                gi = paint(gi, inobjp)
                go = paint(go, outobjp)
        break
    return {'input': gi, 'output': go}


# ── derive_operations ────────────────────────────────────────────────────────
# The canvas holds one complete stamp plus a number of half-finished copies of
# it, each copy turned by one of the eight symmetries of the square and showing
# only its main+marker cells.  Every copy is finished by stamping the complete
# one onto it: CopyI the complete stamp once, then, per copy, Paste it and turn
# the pasted rectangle (FlipH / FlipV / a clockwise quarter turn) into that
# copy's orientation.  Nothing here reads O — the answer is read off the one
# finished stamp, which is what the puzzle asks a solver to do.

_FLIP_TF = ("id", "lr", "ud", "rot180")
_TURN_TF = ("cw", "tp", "at", "ccw")          # these transpose the rectangle
_TF_FLIPS = {"id": (), "lr": (26,), "ud": (27,), "rot180": (26, 27),
             "cw": (), "tp": (26,), "at": (27,), "ccw": (26, 27)}


def _tf(A, name):
    """The eight symmetries of a rectangle, as array operations."""
    if name == "id":
        return A.copy()
    if name == "lr":
        return np.fliplr(A).copy()
    if name == "ud":
        return np.flipud(A).copy()
    if name == "rot180":
        return np.rot90(A, 2).copy()
    if name == "cw":
        return np.rot90(A, 3).copy()
    if name == "ccw":
        return np.rot90(A, 1).copy()
    if name == "tp":                            # = FlipH after a clockwise turn
        return np.fliplr(np.rot90(A, 3)).copy()
    if name == "at":                            # = FlipV after a clockwise turn
        return np.flipud(np.rot90(A, 3)).copy()
    raise ValueError(name)


def _components(G, bgc):
    h, w = G.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if G[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] \
                                and G[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(comp)
    return comps


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    submit = [0, 0, h - 1, w - 1]            # full-grid bbox: submit the canvas

    # background: the colour the generator fills the canvas with before it
    # places any stamp — by far the most common value on a sparse canvas.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    comps = _components(I, bgc)
    if len(comps) < 2:
        return [34], [submit]

    # the complete stamp is the only one carrying its completion colours too,
    # so it is the largest piece on the canvas
    template = max(comps, key=len)
    tset = set(template)
    sr = min(r for r, _ in template)
    sc = min(c for _, c in template)
    oh = max(r for r, _ in template) - sr + 1
    ow = max(c for _, c in template) - sc + 1
    src = np.full((oh, ow), bgc, dtype=int)
    for (r, c) in template:
        src[r - sr, c - sc] = I[r, c]
    # a neighbouring copy can poke a cell into the complete stamp's rectangle;
    # then the rectangle cannot be copied as a whole and the copies are
    # finished cell by cell instead
    src_clean = bool(np.array_equal(src, I[sr:sr + oh, sc:sc + ow]))

    # the half-finished copies show only these colours
    pal = set()
    for comp in comps:
        if comp is template:
            continue
        for (r, c) in comp:
            pal.add(int(I[r, c]))
    if not pal:
        return [34], [submit]

    # ── locate every half-finished copy: a placement of some symmetry of the
    # stamp whose palette cells are all present and whose completion cells are
    # still background.  Coincidences happen — three cells of two neighbouring
    # copies can read as a fourth copy — so the placements are then required to
    # account for every marked cell on the canvas exactly once, each standing
    # clear of the others, which is how the copies were laid down.
    cands = []
    for name in _FLIP_TF + _TURN_TF:
        A = _tf(src, name)
        gh, gw = A.shape
        body = [(u, v, int(A[u, v])) for u in range(gh) for v in range(gw)
                if A[u, v] != bgc]
        keep = [(u, v, val) for (u, v, val) in body if val in pal]
        drop = [(u, v) for (u, v, val) in body if val not in pal]
        if not keep or not drop:
            continue
        for i in range(h - gh + 1):
            for j in range(w - gw + 1):
                if any(I[i + u, j + v] != val for (u, v, val) in keep):
                    continue
                if any(I[i + u, j + v] != bgc for (u, v) in drop):
                    continue
                shown = frozenset((i + u, j + v) for (u, v, val) in keep)
                full = frozenset((i + u, j + v) for (u, v, _val) in body)
                cells = {(i + u, j + v): int(A[u, v]) for (u, v, _val) in body}
                cands.append({"name": name, "r": i, "c": j, "A": A,
                              "shown": shown, "full": full, "cells": cells})
    if not cands:
        return [34], [submit]

    def halo(cellset):
        out = set()
        for (r, c) in cellset:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    out.add((r + dr, c + dc))
        return out

    # a copy never touches the complete stamp or another copy
    tsub = halo(tset)
    cands = [k for k in cands if not (k["full"] & tsub)]
    merged = {}
    for k in cands:
        key = (k["shown"], frozenset(k["cells"].items()))
        merged.setdefault(key, []).append(k)
    cands = []
    for key, group in merged.items():
        base = dict(group[0])
        base["options"] = [(g["name"], g["r"], g["c"], g["A"]) for g in group]
        cands.append(base)
    for k in cands:
        k["halo"] = halo(k["full"])

    todo = frozenset((r, c) for r in range(h) for c in range(w)
                     if I[r, c] != bgc and (r, c) not in tset)
    by_cell = {}
    for k in cands:
        for cell in k["shown"]:
            by_cell.setdefault(cell, []).append(k)

    def cover(left, chosen):
        if not left:
            return chosen
        cell = min(left, key=lambda x: len(by_cell.get(x, ())))
        for k in by_cell.get(cell, ()):
            if not k["shown"] <= left:
                continue
            if any(k["halo"] & p["full"] or p["halo"] & k["full"]
                   for p in chosen):
                continue
            got = cover(left - k["shown"], chosen + [k])
            if got is not None:
                return got
        return None

    picked = cover(todo, [])
    if picked is None:
        return [34], [submit]

    stamps = []
    for k in picked:
        stamps.append({"r": k["r"], "c": k["c"], "A": k["A"],
                       "cells": k["cells"], "options": k["options"]})
    stamps.sort(key=lambda s: (s["r"], s["c"]))

    occupied = set(tset)
    for st in stamps:
        occupied |= set(st["cells"])

    target = I.copy()
    for st in stamps:
        for (r, c), val in st["cells"].items():
            target[r, c] = val

    # ── emit ─────────────────────────────────────────────────────────────────
    grid = I.copy()
    clip = None
    ops, sels = [], []

    def do_paste(pr, pc):
        ch, cw = clip.shape
        for u in range(ch):
            for v in range(cw):
                if clip[u, v] != 0 and pr + u < h and pc + v < w:
                    grid[pr + u, pc + v] = clip[u, v]

    def do_obj(r0, c0, hh, ww, name):
        new = _tf(grid[r0:r0 + hh, c0:c0 + ww], name)
        grid[r0:r0 + hh, c0:c0 + ww] = np.where(new > 0, new, 0)

    for st in stamps:
        dr, dc = st["r"], st["c"]
        gh, gw = st["A"].shape
        forbidden = occupied - set(st["cells"])
        routes = []
        for (name, i, j, A) in st["options"]:
            if name in _FLIP_TF:
                rect = [(r, c) for r in range(dr, dr + gh)
                        for c in range(dc, dc + gw)]
                if any(cell in forbidden for cell in rect):
                    continue
                cost = 1 + len(_TF_FLIPS[name])
                routes.append((cost, "flip", name, None))
            else:
                # a quarter turn needs a square selection: the copy's rectangle
                # plus the strip alongside it, slid to wherever it is free
                s = max(oh, ow)
                offs = ([(a, 0) for a in range(s - oh + 1)] if oh <= ow
                        else [(0, b) for b in range(s - ow + 1)])
                for (a, b) in offs:
                    R0, C0 = dr - b, dc - s + a + oh
                    if R0 < 0 or C0 < 0 or R0 + s > h or C0 + s > w:
                        continue
                    sq = [(r, c) for r in range(R0, R0 + s)
                          for c in range(C0, C0 + s)]
                    if any(cell in forbidden for cell in sq):
                        continue
                    prect = {(R0 + a + u, C0 + b + v)
                             for u in range(oh) for v in range(ow)}
                    stale = sorted(cell for cell in sq
                                   if cell not in prect and I[cell] != bgc)
                    cost = 2 + len(_TF_FLIPS[name]) + (1 if stale else 0)
                    routes.append((cost, "turn", name, (R0, C0, s, a, b, stale)))
        if routes and src_clean:
            routes.sort(key=lambda t: t[0])
            cost, kind, name, extra = routes[0]
            if clip is None:
                # take the complete stamp once; it is stamped over every copy
                ops.append(28)
                sels.append([sr, sc, oh - 1, ow - 1])   # its whole rectangle
                clip = np.where(src > 0, src, 0)
            if kind == "flip":
                ops.append(30)
                sels.append([dr, dc, 0, 0])      # Paste uses the corner only
                do_paste(dr, dc)
                for flip_op in _TF_FLIPS[name]:
                    ops.append(flip_op)
                    # whole rectangle, background included: it is turned too
                    sels.append([dr, dc, gh - 1, gw - 1])
                    do_obj(dr, dc, gh, gw, "lr" if flip_op == 26 else "ud")
            else:
                R0, C0, s, a, b, stale = extra
                if stale:
                    # the copy's own cells that fall outside the rectangle the
                    # stamp is pasted on, cleared so the turn carries nothing
                    ops.append(int(bgc))
                    sels.append(sel_of(stale))
                    for (r, c) in stale:
                        grid[r, c] = bgc
                ops.append(30)
                sels.append([R0 + a, C0 + b, 0, 0])
                do_paste(R0 + a, C0 + b)
                ops.append(25)                   # clockwise quarter turn
                sels.append([R0, C0, s - 1, s - 1])   # the whole square
                do_obj(R0, C0, s, s, "cw")
                for flip_op in _TF_FLIPS[name]:
                    ops.append(flip_op)
                    sels.append([dr, dc, gh - 1, gw - 1])
                    do_obj(dr, dc, gh, gw, "lr" if flip_op == 26 else "ud")
        else:
            # boxed in by a neighbour: finish this copy cell by cell instead
            bycol = {}
            for (r, c), val in sorted(st["cells"].items()):
                if grid[r, c] != val:
                    bycol.setdefault(val, []).append((r, c))
            for val in sorted(bycol, key=lambda k: (-len(bycol[k]), k)):
                ops.append(int(val))
                sels.append(sel_of(bycol[val]))
                for (r, c) in bycol[val]:
                    grid[r, c] = val

    if not np.array_equal(grid, target):
        # safety net: never leave a copy half finished
        ops, sels = [], []
        grid = I.copy()
        for st in stamps:
            bycol = {}
            for (r, c), val in sorted(st["cells"].items()):
                if grid[r, c] != val:
                    bycol.setdefault(val, []).append((r, c))
            for val in sorted(bycol, key=lambda k: (-len(bycol[k]), k)):
                ops.append(int(val))
                sels.append(sel_of(bycol[val]))
                for (r, c) in bycol[val]:
                    grid[r, c] = val

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
