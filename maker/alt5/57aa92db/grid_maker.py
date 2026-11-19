"""
ARC Task: 57aa92db (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(cols)
    fixc = random.choice([c for c in cols if c != bgc])
    # the key is copied around as an object, so keep its body colour non-zero
    # (ARCLE's clipboard treats 0 as "nothing here" and would not carry it)
    mainc = random.choice([c for c in cols if c not in (0, bgc, fixc)])
    return {"bgc": bgc, "fixc": fixc, "mainc": mainc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, fixc: int, mainc: int) -> dict:
    cols = interval(0, 10, 1)
    lo_h = max(6, min(10, max_h))
    lo_w = max(6, min(10, max_w))
    h = unifint(diff_lb, diff_ub, (lo_h, max_h))
    w = unifint(diff_lb, diff_ub, (lo_w, max_w))
    oh = randint(2, 5)
    ow = randint(2, 5)
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {choice(totuple(bounds))}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    fixp = choice(totuple(obj))
    remcols = difference(cols, (bgc, fixc, mainc))
    gi = canvas(bgc, (h, w))
    obj = {(fixc, fixp)} | recolor(mainc, remove(fixp, obj))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, (h * w) // (4 * len(obj))))
    tr = 0
    succ = 0
    maxtr = 5 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = randint(1, 4)
        objups = upscale(obj, fac)
        hh, ww = shape(objups)
        cands = sfilter(inds, lambda ij: ij[0] <= h - hh and ij[1] <= w - ww)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        objupsplcd = shift(objups, loc)
        objupsplcdi = toindices(objupsplcd)
        if objupsplcdi.issubset(inds):
            succ += 1
            newc = choice(remcols)
            fixp2 = sfilter(objupsplcd, lambda cij: cij[0] == fixc)
            inds = inds - mapply(neighbors, objupsplcdi)
            gi = paint(gi, fixp2)
            go = paint(go, fixp2)
            remobjfull = toindices(objupsplcd - fixp2)
            ntorem = unifint(diff_lb, diff_ub, (0, max(0, len(remobjfull) - 1)))
            ntokeep = len(remobjfull) - ntorem
            tokeep = {choice(totuple(remobjfull & outbox(fixp2)))}
            fixp2i = toindices(fixp2)
            for k in range(ntokeep - 1):
                fullopts = remobjfull & mapply(neighbors, tokeep | fixp2i)
                remopts = fullopts - tokeep
                tokeep.add(choice(totuple(remopts)))
            gi = fill(gi, newc, tokeep)
            go = fill(go, newc, remobjfull)
    return {'input': gi, 'output': go}


# ── derive_operations helpers ────────────────────────────────────────────────

def _components(grid, bgc):
    """8-connected components of non-background cells."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bgc:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] \
                                and grid[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(cells)
    return comps


def _four_components(cells):
    """Split a set of cells into 4-connected groups (order preserved)."""
    remaining = set(cells)
    groups = []
    for start in sorted(remaining):
        if start not in remaining:
            continue
        stack = [start]
        remaining.discard(start)
        grp = []
        while stack:
            r, c = stack.pop()
            grp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.discard((nr, nc))
                    stack.append((nr, nc))
        groups.append(sorted(grp))
    return groups


def _flood(grid, seed, color):
    """4-connected flood fill on a numpy grid (mirrors ARCLE FloodFill)."""
    h, w = grid.shape
    src = grid[seed]
    if src == color:
        return False
    stack = [seed]
    seen = {seed}
    while stack:
        r, c = stack.pop()
        grid[r, c] = color
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen \
                    and grid[nr, nc] == src:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return True


def _flood_region(grid, seed):
    """Cells of the 4-connected same-colour region an ARCLE FloodFill reaches."""
    h, w = grid.shape
    src = grid[seed]
    seen = {seed}
    stack = [seed]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen \
                    and grid[nr, nc] == src:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _paste(grid, clip, r0, c0):
    """ARCLE Paste: the clipboard's non-zero cells land at (r0, c0)."""
    h, w = grid.shape
    for a in range(clip.shape[0]):
        for b in range(clip.shape[1]):
            v = int(clip[a, b])
            if v and r0 + a < h and c0 + b < w:
                grid[r0 + a, c0 + b] = v


def derive_operations(I, O, examples=None):
    """Every object is the key shape hung on a marker, drawn at some scale but
    left unfinished. Finish each one: at scale 1 by laying a copy of the key
    over the marker and repainting that copy in the object's own colour, at
    scale f by drawing one f x f tile and stamping it at every key cell."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    nothing = ([34], [[0, 0, hi - 1, wi - 1]])

    bgc = int(np.bincount(I.reshape(-1), minlength=10).argmax())
    comps = _components(I, bgc)
    if not comps:
        return nothing

    # the marker colour is the one every stamp of the key carries
    tally = {}
    for cells in comps:
        for col in {int(I[r, c]) for r, c in cells}:
            tally[col] = tally.get(col, 0) + 1
    fixc = max(sorted(tally), key=lambda col: tally[col])

    info = []
    for cells in comps:
        fix = [(r, c) for r, c in cells if I[r, c] == fixc]
        if not fix:
            continue
        fr0 = min(r for r, _ in fix)
        fc0 = min(c for _, c in fix)
        bodycols = [int(I[r, c]) for r, c in cells if I[r, c] != fixc]
        info.append({
            "cells": cells, "fr0": fr0, "fc0": fc0,
            "fac": max(r for r, _ in fix) - fr0 + 1, "nfix": len(fix),
            "col": max(set(bodycols), key=bodycols.count) if bodycols else None,
        })
    if not info:
        return nothing

    # the key: the object drawn at the smallest scale, and complete (so, largest)
    minfix = min(e["nfix"] for e in info)
    cands = [e for e in info if e["nfix"] == minfix]
    big = max(len(e["cells"]) for e in cands)
    cands = [e for e in cands if len(e["cells"]) == big]
    key = cands[0]
    for e in cands:
        if e["col"] is not None and tally.get(e["col"], 0) == 1:
            key = e
            break

    # the key's shape, in key-cell units, measured from its marker
    kf = key["fac"]
    base = sorted({((r - key["fr0"]) // kf, (c - key["fc0"]) // kf)
                   for r, c in key["cells"]})
    kbody = [p for p in base if p != (0, 0)]
    if not kbody:
        return nothing
    kr0 = min(p[0] for p in base)
    kc0 = min(p[1] for p in base)
    kh = max(p[0] for p in base) - kr0 + 1
    kw = max(p[1] for p in base) - kc0 + 1
    keycol = key["col"]
    kbr, kbc = key["fr0"] + kr0 * kf, key["fc0"] + kc0 * kf   # key's bbox in I
    kbh, kbw = kh * kf, kw * kf

    # every other stamp, completed: the key's shape at that stamp's scale,
    # hung on its marker block
    targets = []
    for e in info:
        if e is key or e["col"] is None:
            continue
        f = e["fac"]
        blocks = [(e["fr0"] + f * dr, e["fc0"] + f * dc) for dr, dc in kbody]
        cells = [(r + a, c + b) for r, c in blocks
                 for a in range(f) for b in range(f)]
        if any(not (0 <= r < hi and 0 <= c < wi) for r, c in cells):
            continue
        targets.append({"e": e, "f": f, "col": e["col"], "blocks": blocks,
                        "cells": cells, "set": set(cells)})
    if not targets:
        return nothing

    occupied = set()
    for t in targets:
        occupied |= t["set"]
    filled = I.copy()                    # every stamp completed (rule-derived)
    for t in targets:
        for p in t["cells"]:
            filled[p] = t["col"]

    grid = I.copy()
    clip = None

    def missing(t):
        f = t["f"]
        return [(br, bc) for (br, bc) in t["blocks"]
                if any(grid[br + a, bc + b] != t["col"]
                       for a in range(f) for b in range(f))]

    def stamp_plan(t):
        """A 1:1 stamp can be finished by laying a copy of the key over it and
        repainting that copy. Returns (dr0, dc0, seeds) or None."""
        if t["f"] != 1 or keycol in (None, 0):
            return None
        dr0, dc0 = t["e"]["fr0"] + kr0, t["e"]["fc0"] + kc0
        if dr0 < 0 or dc0 < 0 or dr0 + kh > hi or dc0 + kw > wi:
            return None
        own = t["set"] | {(t["e"]["fr0"], t["e"]["fc0"])}
        for a in range(kh):
            for b in range(kw):
                p = (dr0 + a, dc0 + b)
                if p not in own and (int(grid[p]) != bgc or p in occupied):
                    return None          # the key's box would cover something
        pg = filled.copy()
        for p in t["cells"]:
            pg[p] = grid[p]
        seeds = []
        for grp in _four_components(t["cells"]):
            seed = None
            for p in grp:                # a cell already in this stamp's colour
                if int(grid[p]) == t["col"]:
                    seed = p
                    break
            if seed is None:             # else one whose region is not the shape
                for p in grp:
                    if int(pg[p]) == bgc and \
                            any(q not in t["set"] for q in _flood_region(pg, p)):
                        seed = p
                        break
            if seed is None:
                return None
            seeds.append(seed)
        return dr0, dc0, seeds

    rest = []
    for t in targets:
        if not missing(t):
            continue                     # this stamp is already complete
        plan = stamp_plan(t)
        if plan is None:
            rest.append(t)
            continue
        dr0, dc0, seeds = plan
        if clip is None or clip.shape != (kbh, kbw) or \
                not np.array_equal(clip, I[kbr:kbr + kbh, kbc:kbc + kbw]):
            ops.append(28)                           # take a copy of the key
            sels.append([kbr, kbc, kbh - 1, kbw - 1])   # bbox: the whole key box
            clip = I[kbr:kbr + kbh, kbc:kbc + kbw].copy()
        ops.append(30)                               # lay it over this marker
        sels.append([dr0, dc0, 0, 0])
        _paste(grid, clip, dr0, dc0)
        for seed in seeds:                           # repaint it in its colour
            if int(grid[seed]) == t["col"]:
                continue
            ops.append(10 + t["col"])
            sels.append([seed[0], seed[1], 0, 0])    # single seed cell
            _flood(grid, seed, t["col"])

    rest.sort(key=lambda t: (t["f"], t["col"], t["e"]["fr0"], t["e"]["fc0"]))
    for t in rest:
        todo = missing(t)
        if not todo:
            continue
        f, col = t["f"], t["col"]
        tile = np.full((f, f), col, dtype=int)
        if f == 1:                       # one cell per key cell: nothing tiles
            cells = [p for p in todo if int(grid[p]) != col]
            ops.append(col)
            sels.append(sel_of(cells))
            for p in cells:
                grid[p] = col
            continue
        if clip is not None and clip.shape == (f, f) and np.array_equal(clip, tile):
            for (rr, cc) in todo:        # that tile is already on the clipboard
                ops.append(30)
                sels.append([rr, cc, 0, 0])
                grid[rr:rr + f, cc:cc + f] = col
            continue
        br, bc = todo[0]
        after = grid.copy()              # what drawing + stamping would give
        after[br:br + f, bc:bc + f] = col
        stale = after.copy()             # what a stale clipboard would give
        for (rr, cc) in todo[1:]:
            after[rr:rr + f, cc:cc + f] = col
            if clip is not None:
                _paste(stale, clip, rr, cc)
        reuse = col != 0 and len(todo) > 1 and not np.array_equal(after, stale)
        ops.append(col)                  # draw one tile of this stamp
        sels.append([br, bc, f - 1, f - 1])   # bbox: exactly that f x f tile
        grid[br:br + f, bc:bc + f] = col
        if reuse:
            ops.append(29)               # ... take a copy of that tile ...
            sels.append([br, bc, f - 1, f - 1])
            clip = tile
        for (rr, cc) in todo[1:]:        # ... and stamp it at every other one
            if reuse:
                ops.append(30)
                sels.append([rr, cc, 0, 0])
            else:
                ops.append(col)
                sels.append([rr, cc, f - 1, f - 1])
            grid[rr:rr + f, cc:cc + f] = col

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 57aa92db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 57aa92db"
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
                                f"for task 57aa92db"
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
                    f"Failed to build a complete episode for task 57aa92db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"57aa92db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
