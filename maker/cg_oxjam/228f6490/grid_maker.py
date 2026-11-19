"""
ARC Task: 228f6490 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, **kwargs) -> dict:
    def unifint(lb, ub):
        if ub < lb:
            ub = lb
        lo = lb + int(round((ub - lb) * diff_lb))
        hi = lb + int(round((ub - lb) * diff_ub))
        lo = max(lb, min(lo, ub))
        hi = max(lo, min(hi, ub))
        return random.randint(lo, hi)

    def dneigh(cells):
        out = set()
        for (i, j) in cells:
            out |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
        return out

    def normalize(cells):
        mi = min(i for i, _ in cells)
        mj = min(j for _, j in cells)
        return frozenset((i - mi, j - mj) for i, j in cells)

    def shift(cells, loc):
        return frozenset((i + loc[0], j + loc[1]) for i, j in cells)

    h = unifint(min(10, max_h), max_h)
    w = unifint(min(10, max_w), max_w)
    nsq = unifint(1, max(1, (h * w) // 50))
    remcols = [c for c in range(10) if c not in (bgc, sqc)]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    forbidden = []
    succ, tr, maxtr = 0, 0, 5 * nsq
    while tr < maxtr and succ < nsq:
        tr += 1
        oh = random.randint(3, 6)
        ow = random.randint(3, 6)
        bd = frozenset((i, j) for i in range(oh) for j in range(ow))
        bounds = {(i, j) for i in range(1, oh - 1) for j in range(1, ow - 1)}
        obj = {random.choice(sorted(bounds))}
        ncells = random.randint(1, (oh - 2) * (ow - 2))
        for _ in range(ncells - 1):
            cands = (bounds - obj) & dneigh(obj)
            if not cands:
                break
            obj.add(random.choice(sorted(cands)))
        sqcands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not sqcands:
            continue
        loc = random.choice(sorted(sqcands))
        bdplcd = shift(bd, loc)
        if bdplcd.issubset(inds):
            tmpinds = inds - bdplcd
            inobjn = normalize(obj)
            ih = max(i for i, _ in inobjn) + 1
            iw = max(j for _, j in inobjn) + 1
            inobjcands = [ij for ij in inds if ij[0] <= h - ih and ij[1] <= w - iw]
            if not inobjcands:
                continue
            loc2 = random.choice(sorted(inobjcands))
            inobjplcd = shift(inobjn, loc2)
            bdnorm = frozenset(bd - obj)
            if inobjplcd.issubset(tmpinds) and bdnorm not in forbidden and inobjn not in forbidden:
                forbidden.append(bdnorm)
                forbidden.append(inobjn)
                succ += 1
                inds = (inds - (bdplcd | inobjplcd)) - dneigh(inobjplcd)
                col = random.choice(remcols)
                oplcd = shift(obj, loc)
                for (i, j) in bdplcd - oplcd:
                    gi[i][j] = sqc
                for (i, j) in bdplcd:
                    go[i][j] = sqc
                for (i, j) in oplcd:
                    go[i][j] = col
                for (i, j) in inobjplcd:
                    gi[i][j] = col
    if succ == 0:
        raise ValueError("no box placed")
    nremobjs = unifint(0, len(inds) // 25)
    succ, tr, maxtr = 0, 0, 10 * nremobjs
    while tr < maxtr and succ < nremobjs:
        tr += 1
        oh = random.randint(1, 4)
        ow = random.randint(1, 4)
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        obj = {random.choice(sorted(bounds))}
        ncells = random.randint(1, oh * ow)
        for _ in range(ncells - 1):
            cands = (bounds - obj) & dneigh(obj)
            if not cands:
                break
            obj.add(random.choice(sorted(cands)))
        obj = normalize(obj)
        if obj in forbidden:
            continue
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            inds = (inds - plcd) - dneigh(plcd)
            col = random.choice(remcols)
            for (i, j) in plcd:
                gi[i][j] = col
                go[i][j] = col
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    import random
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in sorted(set(cells))]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    exs = [np.asarray(ei, dtype=int) for ei, _ in (examples or [])]

    # background: the canvas colour the generator paints first (clear majority)
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    def comps4(G):
        gh, gw = G.shape
        seen = np.zeros((gh, gw), bool)
        out = []
        for r0 in range(gh):
            for c0 in range(gw):
                if seen[r0, c0]:
                    continue
                col = int(G[r0, c0])
                st = [(r0, c0)]
                seen[r0, c0] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gh and 0 <= nx < gw and not seen[ny, nx] and int(G[ny, nx]) == col:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                out.append((col, sorted(cells)))
        return out

    def norm(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    def pockets(G):
        """Background regions sitting inside a solid frame: every cell of the
        region's bbox grown by one, other than the region itself, is one colour."""
        gh, gw = G.shape
        res = []
        for col, cells in comps4(G):
            if col != bgc:
                continue
            r0 = min(r for r, _ in cells) - 1
            r1 = max(r for r, _ in cells) + 1
            c0 = min(c for _, c in cells) - 1
            c1 = max(c for _, c in cells) + 1
            if r0 < 0 or c0 < 0 or r1 >= gh or c1 >= gw:
                continue
            s = set(cells)
            frame = {int(G[r, c]) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                     if (r, c) not in s}
            if len(frame) == 1:
                fc = frame.pop()
                if fc != bgc:
                    res.append((cells, fc))
        return res

    STEP = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}

    def visible(grid, src, col, seq):
        # simulate ARCLE: grab once (snapshot with src zeroed), re-paste each step
        snap = grid.copy()
        for p in src:
            snap[p] = 0
        prev = grid
        oy = ox = 0
        for op in seq:
            oy += STEP[op][0]; ox += STEP[op][1]
            g = snap.copy()
            for r, c in src:
                if 0 <= r + oy < h and 0 <= c + ox < w:
                    g[r + oy, c + ox] = col
            if np.array_equal(g, prev):
                return False
            prev = g
        return True

    def route(grid, src, col, dr, dc):
        V = [20 if dr < 0 else 21] * abs(dr)
        H = [23 if dc < 0 else 22] * abs(dc)
        cands = [V + H, H + V]
        stair = []
        vi = hi_ = 0
        while vi < len(V) or hi_ < len(H):
            if vi < len(V):
                stair.append(V[vi]); vi += 1
            if hi_ < len(H):
                stair.append(H[hi_]); hi_ += 1
        cands.append(stair)
        rng = random.Random(len(src) * 1009 + dr * 31 + dc)
        for _ in range(40):
            mix = V + H
            rng.shuffle(mix)
            cands.append(mix)
        for seq in cands:
            if visible(grid, src, col, seq):
                return seq
        return None

    ops, sels = [], []
    pk = pockets(I)
    if pk:
        # box colour: the frame colour seen in every demonstration (fixed per
        # episode); fall back to the frame colour of most pockets in I
        votes = Counter(fc for _, fc in pk)
        boxc = None
        if exs:
            common = set(votes)
            for ei in exs:
                common &= {fc for _, fc in pockets(ei)}
            if len(common) == 1:
                boxc = common.pop()
        if boxc is None:
            boxc = votes.most_common(1)[0][0]
        holes = sorted(cells for cells, fc in pk if fc == boxc)
        pieces = [(col, cells) for col, cells in comps4(I) if col not in (bgc, boxc)]
        used = set()
        grid = I.copy()
        for hole in holes:
            hn = norm(hole)
            cand = [k for k, (col, cells) in enumerate(pieces)
                    if k not in used and len(cells) == len(hole) and norm(cells) == hn]
            if len(cand) != 1:
                continue
            k = cand[0]
            used.add(k)
            col, src = pieces[k]
            dr = min(r for r, _ in hole) - min(r for r, _ in src)
            dc = min(c for _, c in hole) - min(c for _, c in src)
            if col == 0:
                for p in hole:
                    grid[p] = 0
                for p in src:
                    grid[p] = bgc
                # a colour-0 piece is invisible to ARCLE's object layer (a grab
                # keeps only nonzero cells), so it cannot be carried: paint it
                # into the hole of its own shape and clear where it lay
                ops.append(0); sels.append(sel_of(hole))
                ops.append(bgc); sels.append(sel_of(src))
                continue
            # carry the piece into the hole of its shape: grab once, then glide.
            # Pick a route (rows-first, columns-first, or a staircase) on which
            # every unit step visibly changes the grid -- gliding over a run of
            # the piece's own colour can make a step invisible.
            seq = route(grid, src, col, dr, dc)
            if seq is None:
                ops.append(col); sels.append(sel_of(hole))
                ops.append(bgc); sels.append(sel_of(src))
                for p in hole:
                    grid[p] = col
                for p in src:
                    grid[p] = bgc
                continue
            for i, op in enumerate(seq):
                ops.append(op)
                sels.append(sel_of(src) if i == 0 else sel_of([]))
            for p in src:
                grid[p] = 0
            for r, c in src:
                grid[r + dr, c + dc] = col
            vac = [p for p in src if p not in {(r + dr, c + dc) for r, c in src}]
            if bgc != 0 and vac:
                # the grab left 0 where the piece started; restore background
                ops.append(bgc); sels.append(sel_of(vac))
                for p in vac:
                    grid[p] = bgc

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 228f6490"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 228f6490"
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
                                f"for task 228f6490"
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
                    f"Failed to build a complete episode for task 228f6490 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"228f6490-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
