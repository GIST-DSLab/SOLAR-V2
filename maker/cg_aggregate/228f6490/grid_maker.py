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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None) -> dict:
    from random import randint, choice

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    def dn(c):
        r, cc = c
        return {(r - 1, cc), (r + 1, cc), (r, cc - 1), (r, cc + 1)}

    def mapdn(s):
        out = set()
        for c in s:
            out |= dn(c)
        return out

    def normalize(s):
        mr = min(r for r, _ in s)
        mc = min(c for _, c in s)
        return frozenset((r - mr, c - mc) for r, c in s)

    def shift(s, d):
        return frozenset((r + d[0], c + d[1]) for r, c in s)

    def rect(h, w):
        return frozenset((i, j) for i in range(h) for j in range(w))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if sqc is None:
        sqc = random.choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, sqc)]

    while True:
        h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 50)))
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]

        def fill(g, col, cells):
            for r, c in cells:
                if 0 <= r < h and 0 <= c < w:
                    g[r][c] = col

        inds = set(rect(h, w))
        forbidden = []
        succ = 0
        tr = 0
        maxtr = 5 * nsq
        while tr < maxtr and succ < nsq:
            tr += 1
            oh = randint(3, 6)
            ow = randint(3, 6)
            bd = rect(oh, ow)
            bounds = shift(rect(oh - 2, ow - 2), (1, 1))
            obj = {choice(sorted(bounds))}
            ncells = randint(1, (oh - 2) * (ow - 2))
            for _ in range(ncells - 1):
                obj.add(choice(sorted((bounds - obj) & mapdn(obj))))
            sqcands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not sqcands:
                continue
            loc = choice(sorted(sqcands))
            bdplcd = shift(bd, loc)
            if bdplcd <= inds:
                tmpinds = inds - bdplcd
                inobjn = normalize(obj)
                ih = max(r for r, _ in inobjn) + 1
                iw = max(c for _, c in inobjn) + 1
                cands = [ij for ij in inds if ij[0] <= h - ih and ij[1] <= w - iw]
                if not cands:
                    continue
                loc2 = choice(sorted(cands))
                inobjplcd = shift(inobjn, loc2)
                bdnorm = frozenset(bd - obj)
                if inobjplcd <= tmpinds and bdnorm not in forbidden and inobjn not in forbidden:
                    forbidden.append(bdnorm)
                    forbidden.append(inobjn)
                    succ += 1
                    inds = (inds - (bdplcd | inobjplcd)) - mapdn(inobjplcd)
                    col = choice(remcols)
                    oplcd = shift(obj, loc)
                    fill(gi, sqc, bdplcd - oplcd)
                    fill(go, sqc, bdplcd)
                    fill(go, col, oplcd)
                    fill(gi, col, inobjplcd)
        if succ == 0:
            continue

        nrem = unifint(diff_lb, diff_ub, (0, len(inds) // 25))
        succ = 0
        tr = 0
        maxtr = 10 * nrem
        while tr < maxtr and succ < nrem:
            tr += 1
            oh = randint(1, 4)
            ow = randint(1, 4)
            bounds = rect(oh, ow)
            obj = {choice(sorted(bounds))}
            ncells = randint(1, oh * ow)
            for _ in range(ncells - 1):
                obj.add(choice(sorted((bounds - obj) & mapdn(obj))))
            obj = normalize(obj)
            if obj in forbidden:
                continue
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            loc = choice(sorted(cands))
            plcd = shift(obj, loc)
            if plcd <= inds:
                succ += 1
                inds = (inds - plcd) - mapdn(plcd)
                col = choice(remcols)
                fill(gi, col, plcd)
                fill(go, col, plcd)
        return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared by the maker and the golf entry): every background hole enclosed
    # by a box is the exact silhouette of one loose piece; that piece travels into
    # the hole and leaves its old spot as background.  The piece keeps shape and
    # colour, so the transfer is a translation -> one grab + unit Moves.
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    D4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def comps4(G):
        H, W = G.shape
        seen = np.zeros((H, W), bool)
        out = []
        for r0 in range(H):
            for c0 in range(W):
                if seen[r0, c0]:
                    continue
                col = int(G[r0, c0])
                st = [(r0, c0)]
                seen[r0, c0] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in D4:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and int(G[ny, nx]) == col:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                out.append((col, cells))
        return out

    def outer(cells, H, W):
        s = set(cells)
        nb = set()
        for y, x in cells:
            for dy, dx in D4:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in s:
                    nb.add((ny, nx))
        return nb

    def borders(cells, H, W):
        return any(r in (0, H - 1) or c in (0, W - 1) for r, c in cells)

    # --- box colour: read from the demonstrations (the frame around the cells that get filled)
    sqc = None
    if examples:
        cnt = Counter()
        for EI, EO in examples:
            EI = np.asarray(EI, int)
            EO = np.asarray(EO, int)
            if EI.shape != EO.shape:
                continue
            eb = int(Counter(EI.flatten().tolist()).most_common(1)[0][0])
            filled = {(r, c) for r in range(EI.shape[0]) for c in range(EI.shape[1])
                      if EI[r, c] == eb and EO[r, c] != eb}
            for (r, c) in filled:
                for dy, dx in D4:
                    ny, nx = r + dy, c + dx
                    if 0 <= ny < EI.shape[0] and 0 <= nx < EI.shape[1] and (ny, nx) not in filled \
                            and EI[ny, nx] != eb:
                        cnt[int(EI[ny, nx])] += 1
        if cnt:
            sqc = cnt.most_common(1)[0][0]

    comps = comps4(I)
    holes = [cells for col, cells in comps if col == bgc and not borders(cells, h, w)]
    if sqc is None:
        rc = [Counter(int(I[p]) for p in outer(c, h, w)).most_common(1)[0][0] for c in holes]
        sqc = Counter(rc).most_common(1)[0][0] if rc else None
    # a box hole: enclosed background whose whole rim is the box colour
    boxholes = [c for c in holes if all(int(I[p]) == sqc for p in outer(c, h, w))]

    def norm(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    loose = [(col, cells) for col, cells in comps if col not in (bgc, sqc)]
    used = set()
    tasks = []
    for hole in sorted(boxholes, key=lambda cs: min(cs)):
        hn = norm(hole)
        for k, (col, cells) in enumerate(loose):
            if k in used or len(cells) != len(hole) or norm(cells) != hn:
                continue
            used.add(k)
            hr = min(r for r, _ in hole)
            hc = min(c for _, c in hole)
            sr = min(r for r, _ in cells)
            sc = min(c for _, c in cells)
            tasks.append((sorted(hole), sorted(cells), int(col), hr - sr, hc - sc))
            break

    grid = I.copy()
    for hole, src, col, dr, dc in tasks:
        seq = []
        if dr:
            seq += [(20 if dr < 0 else 21, (-1 if dr < 0 else 1, 0))] * abs(dr)
        if dc:
            seq += [(23 if dc < 0 else 22, (0, -1 if dc < 0 else 1))] * abs(dc)
        usable = col != 0 and len(seq) > 0
        states = []
        if usable:
            snap = grid.copy()
            for p in src:
                snap[p] = 0
            prev = grid
            off = (0, 0)
            for _, (sy, sx) in seq:
                off = (off[0] + sy, off[1] + sx)
                g = snap.copy()
                for y, x in src:
                    ny, nx = y + off[0], x + off[1]
                    if 0 <= ny < h and 0 <= nx < w:
                        g[ny, nx] = col
                if np.array_equal(g, prev):
                    usable = False
                    break
                states.append(g)
                prev = g
        if usable:
            # one grab of the piece, then empty selections slide the same object into its box
            for i, (op, _) in enumerate(seq):
                ops.append(op)
                sels.append(sel_of(src) if i == 0 else sel_of([]))
            grid = states[-1]
            dst = {(y + dr, x + dc) for y, x in src}
            vac = [p for p in src if p not in dst]
            if bgc != 0 and vac:
                ops.append(bgc)
                sels.append(sel_of(vac))   # only the piece's original footprint
                for p in vac:
                    grid[p] = bgc
        else:
            # colour-0 piece is invisible to ARCLE's object layer: paint into box, clear source
            ops.append(col)
            sels.append(sel_of(hole))
            for p in hole:
                grid[p] = col
            ops.append(bgc)
            sels.append(sel_of(src))
            for p in src:
                grid[p] = bgc

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # full-grid rectangle: submit
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
