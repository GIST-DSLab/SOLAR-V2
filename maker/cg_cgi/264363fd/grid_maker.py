"""
ARC Task: 264363fd (RE-ARC) — LLM-generated grid_maker
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

def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc, linc = random.sample(cols, 3)
    rem = [c for c in cols if c not in (bgc, sqc, linc)]
    cpcol, nbhcol = random.sample(rem, 2)
    return {"bgc": bgc, "sqc": sqc, "linc": linc, "cpcol": cpcol, "nbhcol": nbhcol}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, linc=2,
             cpcol=3, nbhcol=4, **kwargs) -> dict:
    def unifint(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + round((hi - lo) * d)))

    def nbrs(cell):
        r, c = cell
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)}

    def outbox(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs) - 1, max(rs) + 1, min(cs) - 1, max(cs) + 1
        return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)}

    cp = (2, 2)
    neighs = nbrs(cp)
    spikes_all = {
        (-1, 0): {(1, 2), (0, 2)},
        (0, -1): {(2, 1), (2, 0)},
        (1, 0): {(3, 2), (4, 2)},
        (0, 1): {(2, 3), (2, 4)},
    }
    lo_h = min(15, max_h); lo_w = min(15, max_w)
    h = unifint(lo_h, max(lo_h, min(30, max_h)))
    w = unifint(lo_w, max(lo_w, min(30, max_w)))
    nspikes = random.randint(1, 4)
    spikes = random.sample(list(spikes_all.keys()), nspikes)
    lns = set()
    for s in spikes:
        lns |= spikes_all[s]
    obj = {cp: cpcol}
    for c in lns:
        obj[c] = linc
    for c in neighs - lns:
        obj[c] = nbhcol
    loci = random.randint(0, h - 5)
    locj = random.randint(0, w - 5)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    objcells = set()
    for (r, c), col in obj.items():
        gi[r + loci][c + locj] = col
        objcells.add((r + loci, c + locj))
    nb = set()
    for c in objcells:
        nb |= nbrs(c)
    inds = {(r, c) for r in range(h) for c in range(w) if gi[r][c] == bgc} - nb
    numsq = unifint(1, max(1, (h * w) // 100))
    succ = tr = 0
    maxtr = 10 * numsq
    while succ < numsq and tr < maxtr:
        tr += 1
        gh = random.randint(5, h // 2 + 1)
        gw = random.randint(5, w // 2 + 1)
        cands = [ij for ij in inds if ij[0] <= h - gh and ij[1] <= w - gw]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        g1 = [[sqc] * gw for _ in range(gh)]
        g2 = [[sqc] * gw for _ in range(gh)]
        ginds = {(i, j) for i in range(gh) for j in range(gw)}
        full = set(ginds)
        bck = {(i + loc[0], j + loc[1]) for i, j in full}
        if not bck.issubset(inds):
            continue
        noccs = unifint(1, max(1, (gh * gw) // 25))
        succ2 = tr2 = 0
        maxtr2 = 5 * noccs
        while succ2 < noccs and tr2 < maxtr2:
            tr2 += 1
            cands2 = [ij for ij in ginds if ij[0] <= gh - 5 and ij[1] <= gw - 5]
            if not cands2:
                break
            l2 = random.choice(sorted(cands2))
            lns2 = set()
            for (dr, dc) in spikes:
                r, c = cp[0] + l2[0] + dr, cp[1] + l2[1] + dc
                while 0 <= r < gh and 0 <= c < gw:
                    lns2.add((r, c)); r += dr; c += dc
            plcd = {(r + l2[0], c + l2[1]): col for (r, c), col in obj.items()}
            pi = set(plcd)
            lincs = {(i, j) for i in range(gh) for j in range(gw) if g2[i][j] == linc}
            if pi.issubset(ginds) and lns2.issubset(ginds | lincs) and len(lns2 - pi) > 0:
                succ2 += 1
                pn = set()
                for c in pi:
                    pn |= nbrs(c)
                ginds = ((ginds - pi) - pn) - lns2
                g1[cp[0] + l2[0]][cp[1] + l2[1]] = cpcol
                for (r, c), col in plcd.items():
                    g2[r][c] = col
                for (r, c) in lns2:
                    g2[r][c] = linc
        if succ2 > 0:
            succ += 1
            inds = (inds - bck) - outbox(bck)
            for i in range(gh):
                for j in range(gw):
                    gi[i + loc[0]][j + loc[1]] = g1[i][j]
                    go[i + loc[0]][j + loc[1]] = g2[i][j]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def components(grid, bg):
        seen = np.zeros((h, w), dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != bg and not seen[r, c]:
                    stack = [(r, c)]; seen[r, c] = True; cells = []
                    while stack:
                        rr, cc = stack.pop(); cells.append((rr, cc))
                        for dr, dc in DIRS:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] \
                                    and grid[nr, nc] != bg:
                                seen[nr, nc] = True; stack.append((nr, nc))
                    comps.append(cells)
        return comps

    def bg_ok(bg):
        comps = components(I, bg)
        if len(comps) < 2:
            return None
        comps.sort(key=len)
        if len(comps[0]) > 13:
            return None
        for comp in comps[1:]:
            rs = [r for r, _ in comp]; cs = [c for _, c in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if (r1 - r0 + 1) < 5 or (c1 - c0 + 1) < 5:
                return None
            if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
                return None
        return comps

    # background: the canvas the key glyph and the rectangles sit on
    counts = Counter(I.flatten().tolist())
    bgc = counts.most_common(1)[0][0]
    comps = None
    for cand, _n in counts.most_common():
        got = bg_ok(cand)
        if got is not None:
            bgc, comps = cand, got
            break
    if comps is None:
        comps = components(I, bgc); comps.sort(key=len)

    ops, sels = [], []
    G = I.copy()
    ALL = [(r, c) for r in range(h) for c in range(w)]

    def emit(color, cells):
        cells = [(int(r), int(c)) for r, c in cells if 0 <= r < h and 0 <= c < w]
        if not cells or all(G[r, c] == color for r, c in cells):
            return
        for r, c in cells:
            G[r, c] = color
        ops.append(int(color)); sels.append(sel_of(cells))

    if not comps:
        ops.append(34); sels.append(sel_of(ALL))
        return ops, sels

    key = comps[0]
    keyset = set(key)
    keycol = {(r, c): int(I[r, c]) for (r, c) in key}

    # rectangles and the odd marker pixels inside them
    rects = []
    mcols = Counter()
    for sq in comps[1:]:
        rs = [r for r, _ in sq]; cs = [c for _, c in sq]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        sqcol = Counter(int(I[r, c]) for r, c in sq).most_common(1)[0][0]
        markers = sorted((r, c) for r, c in sq if int(I[r, c]) != sqcol)
        for m in markers:
            mcols[int(I[m])] += 1
        rects.append((r0, c0, r1, c1, markers))
    rects.sort()
    cpcol = mcols.most_common(1)[0][0] if mcols else None

    # hub of the key glyph
    def center_ok(cr, cc):
        for (r, c) in keyset:
            dr, dc = r - cr, c - cc
            m = max(abs(dr), abs(dc))
            if m > 2:
                return False
            if m == 2:
                if dr != 0 and dc != 0:
                    return False
                if (cr + dr // 2, cc + dc // 2) not in keyset:
                    return False
        return True

    def score(r, c):
        return sum(1 for (rr, cc) in key if max(abs(rr - r), abs(cc - c)) == 1)

    best, bs = None, -1
    for strict in (True, False):
        for (r, c) in key:
            if strict and cpcol is not None and keycol[(r, c)] != cpcol:
                continue
            if center_ok(r, c) and score(r, c) > bs:
                bs, best = score(r, c), (r, c)
        if best is not None:
            break
    if best is None:
        rs = [r for r, _ in key]; cs = [c for _, c in key]
        best = ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
    cr, cc = best

    # Reading: the key is a hub with 8 neighbour directions. Each direction is
    # either a single ring pixel, or an arm (the key continues 2 cells out),
    # which on a marker becomes one ray reaching the rectangle's edge.
    for (r0, c0, r1, c1, markers) in rects:
        def inside(r, c):
            return r0 <= r <= r1 and c0 <= c <= c1
        for (mr, mc) in markers:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) == (0, 0) or (cr + dr, cc + dc) not in keyset:
                        continue
                    col = keycol[(cr + dr, cc + dc)]
                    is_arm = (dr == 0 or dc == 0) and (cr + 2 * dr, cc + 2 * dc) in keyset
                    if not is_arm:
                        if inside(mr + dr, mc + dc):
                            emit(col, [(mr + dr, mc + dc)])
                        continue
                    ray = []
                    r, c = mr + dr, mc + dc
                    while inside(r, c):
                        ray.append((r, c)); r += dr; c += dc
                    emit(col, ray)

    # the key glyph has been applied everywhere: erase it to background
    emit(bgc, key)

    ops.append(34); sels.append(sel_of(ALL))
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
                        f"num_examples+1 ({num_examples + 1}) for task 264363fd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 264363fd"
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
                                f"for task 264363fd"
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
                    f"Failed to build a complete episode for task 264363fd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"264363fd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
