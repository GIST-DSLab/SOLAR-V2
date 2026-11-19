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
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **kw) -> dict:
    from random import choice, randint

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    def shift(cells, d):
        return {(r + d[0], c + d[1]) for r, c in cells}

    def rect(h, w):
        return {(r, c) for r in range(h) for c in range(w)}

    def dn(cells):
        out = set()
        for r, c in cells:
            out |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        return out

    def normalize(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    def shape(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    cols = list(range(10))
    if bgc is None:
        bgc = choice(cols)
    if sqc is None or sqc == bgc:
        sqc = choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, sqc)]
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 50)))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    inds = rect(h, w)
    forbidden = []
    succ = tr = 0
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
            obj.add(choice(sorted((bounds - obj) & dn(obj))))
        sqcands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not sqcands:
            continue
        loc = choice(sorted(sqcands))
        bdplcd = shift(bd, loc)
        if bdplcd.issubset(inds):
            tmpinds = inds - bdplcd
            inobjn = normalize(obj)
            ooh, oow = shape(obj)
            cands = [ij for ij in inds if ij[0] <= h - ooh and ij[1] <= w - oow]
            if not cands:
                continue
            loc2 = choice(sorted(cands))
            inobjplcd = shift(inobjn, loc2)
            bdnorm = frozenset(bd - obj)
            if inobjplcd.issubset(tmpinds) and bdnorm not in forbidden and inobjn not in forbidden:
                forbidden.append(bdnorm)
                forbidden.append(inobjn)
                succ += 1
                inds = (inds - (bdplcd | inobjplcd)) - dn(inobjplcd)
                col = choice(remcols)
                oplcd = shift(obj, loc)
                for r, c in bdplcd - oplcd:
                    gi[r, c] = sqc
                for r, c in bdplcd:
                    go[r, c] = sqc
                for r, c in oplcd:
                    go[r, c] = col
                for r, c in inobjplcd:
                    gi[r, c] = col
    nrem = unifint(diff_lb, diff_ub, (0, len(inds) // 25))
    succ = tr = 0
    maxtr = 10 * nrem
    while tr < maxtr and succ < nrem:
        tr += 1
        oh = randint(1, 4)
        ow = randint(1, 4)
        bounds = rect(oh, ow)
        obj = {choice(sorted(bounds))}
        ncells = randint(1, oh * ow)
        for _ in range(ncells - 1):
            obj.add(choice(sorted((bounds - obj) & dn(obj))))
        obj = normalize(obj)
        if obj in forbidden:
            continue
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loc = choice(sorted(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            inds = (inds - plcd) - dn(plcd)
            col = choice(remcols)
            for r, c in plcd:
                gi[r, c] = col
                go[r, c] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
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

    # 4-connected single-colour components (the generator keeps pieces 4-apart,
    # but same-coloured pieces may touch diagonally, so 4-connectivity is the unit)
    def comps4(G):
        H, W = G.shape
        seen = np.zeros((H, W), dtype=bool)
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
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and int(G[ny, nx]) == col:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                out.append((col, sorted(cells)))
        return out

    def bg_of(G):
        return int(Counter(G.flatten().tolist()).most_common(1)[0][0])

    def enclosing(G, bgc, cells):
        # the single non-background colour that walls the region in on all 4 sides
        H, W = G.shape
        s = set(cells)
        nb = set()
        for y, x in cells:
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if not (0 <= ny < H and 0 <= nx < W):
                    return None
                if (ny, nx) not in s:
                    nb.add(int(G[ny, nx]))
        if len(nb) == 1:
            c = nb.pop()
            return c if c != bgc else None
        return None

    # box colour is fixed per episode: vote over demonstration inputs and I
    votes = Counter()
    grids = [np.asarray(e[0], dtype=int) for e in (examples or [])] + [I]
    for G in grids:
        gb = bg_of(G)
        for col, cells in comps4(G):
            if col == gb:
                e = enclosing(G, gb, cells)
                if e is not None:
                    votes[e] += 1
    bgc = bg_of(I)
    if not votes:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    sqc = votes.most_common(1)[0][0]

    comps = comps4(I)

    def norm(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    # holes: background pockets walled in entirely by the box colour
    holes = [cells for col, cells in comps if col == bgc and enclosing(I, bgc, cells) == sqc]
    free = {norm(cells): cells for cells in holes}
    pieces = [(col, cells) for col, cells in comps if col not in (bgc, sqc)]

    grid = I.copy()
    # reading: walk the loose pieces; each piece whose shape is a hole slides into it
    for col, src in pieces:
        key = norm(src)
        if key not in free:
            continue
        hole = free.pop(key)
        sr = min(r for r, _ in src); sc = min(c for _, c in src)
        hr = min(r for r, _ in hole); hc = min(c for _, c in hole)
        dr, dc = hr - sr, hc - sc
        seq = []
        if dr:
            seq += [(20 if dr < 0 else 21, (-1 if dr < 0 else 1, 0))] * abs(dr)
        if dc:
            seq += [(23 if dc < 0 else 22, (0, -1 if dc < 0 else 1))] * abs(dc)
        usable = col != 0 and bool(seq)
        states = []
        if usable:
            # simulate ARCLE's grab-once snapshot slide; every step must be visible
            snap = grid.copy()
            for y, x in src:
                snap[y, x] = 0
            prev = grid
            off = (0, 0)
            for op, (sy, sx) in seq:
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
            for i, (op, _) in enumerate(seq):
                ops.append(op)
                sels.append(sel_of(src) if i == 0 else sel_of([]))  # grab once, then continue
            grid = states[-1]
            dst = {(y + dr, x + dc) for y, x in src}
            vac = [p for p in src if p not in dst]
            if bgc != 0 and vac:
                ops.append(int(bgc)); sels.append(sel_of(vac))
                for y, x in vac:
                    grid[y, x] = bgc
        elif seq:
            # colour-0 piece: ARCLE's object layer cannot grab 0 cells,
            # so paint the piece into its box and clear its old spot
            ops.append(int(col)); sels.append(sel_of(hole))
            for y, x in hole:
                grid[y, x] = col
            ops.append(int(bgc)); sels.append(sel_of(src))
            for y, x in src:
                grid[y, x] = bgc

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
