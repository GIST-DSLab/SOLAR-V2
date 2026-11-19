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


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    def dneigh(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def normalize(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    def shape_of(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if sqc is None:
        sqc = random.choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c != bgc and c != sqc]

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 50)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}
    forbidden = []
    succ, tr, maxtr = 0, 0, 5 * nsq
    while tr < maxtr and succ < nsq:
        tr += 1
        oh = random.randint(3, 6)
        ow = random.randint(3, 6)
        bd = {(r, c) for r in range(oh) for c in range(ow)}
        bounds = {(r, c) for r in range(1, oh - 1) for c in range(1, ow - 1)}
        obj = {random.choice(sorted(bounds))}
        ncells = random.randint(1, (oh - 2) * (ow - 2))
        for _ in range(ncells - 1):
            cand = set()
            for p in obj:
                cand |= dneigh(p)
            cand = (bounds - obj) & cand
            if not cand:
                break
            obj.add(random.choice(sorted(cand)))
        sqcands = [p for p in inds if p[0] <= h - oh and p[1] <= w - ow]
        if not sqcands:
            continue
        loc = random.choice(sorted(sqcands))
        bdplcd = {(r + loc[0], c + loc[1]) for r, c in bd}
        if not bdplcd.issubset(inds):
            continue
        tmpinds = inds - bdplcd
        inobjn = normalize(obj)
        ih, iw = shape_of(obj)
        cands2 = [p for p in inds if p[0] <= h - ih and p[1] <= w - iw]
        if not cands2:
            continue
        loc2 = random.choice(sorted(cands2))
        inobjplcd = {(r + loc2[0], c + loc2[1]) for r, c in inobjn}
        bdnorm = frozenset(bd - obj)
        if inobjplcd.issubset(tmpinds) and bdnorm not in forbidden and inobjn not in forbidden:
            forbidden.append(bdnorm)
            forbidden.append(inobjn)
            succ += 1
            nb = set()
            for p in inobjplcd:
                nb |= dneigh(p)
            inds = (inds - (bdplcd | inobjplcd)) - nb
            col = random.choice(remcols)
            oplcd = {(r + loc[0], c + loc[1]) for r, c in obj}
            for (r, c) in bdplcd - oplcd:
                gi[r][c] = sqc
            for (r, c) in bdplcd:
                go[r][c] = sqc
            for (r, c) in oplcd:
                go[r][c] = col
            for (r, c) in inobjplcd:
                gi[r][c] = col
    nremobjs = unifint(diff_lb, diff_ub, (0, len(inds) // 25))
    succ, tr, maxtr = 0, 0, 10 * nremobjs
    while tr < maxtr and succ < nremobjs:
        tr += 1
        oh = random.randint(1, 4)
        ow = random.randint(1, 4)
        bounds = {(r, c) for r in range(oh) for c in range(ow)}
        obj = {random.choice(sorted(bounds))}
        ncells = random.randint(1, oh * ow)
        for _ in range(ncells - 1):
            cand = set()
            for p in obj:
                cand |= dneigh(p)
            cand = (bounds - obj) & cand
            if not cand:
                break
            obj.add(random.choice(sorted(cand)))
        obj = normalize(obj)
        if obj in forbidden:
            continue
        cands = [p for p in inds if p[0] <= h - oh and p[1] <= w - ow]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        plcd = {(r + loc[0], c + loc[1]) for r, c in obj}
        if plcd.issubset(inds):
            succ += 1
            nb = set()
            for p in plcd:
                nb |= dneigh(p)
            inds = (inds - plcd) - nb
            col = random.choice(remcols)
            for (r, c) in plcd:
                gi[r][c] = col
                go[r][c] = col
    return {"input": gi, "output": go}


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

    # Background: the generator's canvas colour (clear majority; confirmed by demos).
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # Box colour: read from the demonstrations -- the colour that rims the
    # background cells which become filled in each demo output.
    sqc = None
    if examples:
        cnt = Counter()
        for (ei, eo) in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            eb = int(Counter(ei.flatten().tolist()).most_common(1)[0][0])
            eh, ew = ei.shape
            for r in range(eh):
                for c in range(ew):
                    if ei[r, c] == eb and eo[r, c] != eb:
                        for dy, dx in D4:
                            y, x = r + dy, c + dx
                            if 0 <= y < eh and 0 <= x < ew and ei[y, x] != eb and eo[y, x] == ei[y, x]:
                                cnt[int(ei[y, x])] += 1
        if cnt:
            sqc = cnt.most_common(1)[0][0]

    def comps4(pred):
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r0 in range(h):
            for c0 in range(w):
                if seen[r0, c0] or not pred(r0, c0):
                    continue
                col = int(I[r0, c0])
                st = [(r0, c0)]
                seen[r0, c0] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in D4:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(I[ny, nx]) == col:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                out.append((col, cells))
        return out

    if sqc is None:
        # fallback: the colour that rims enclosed background pockets in I
        cnt = Counter()
        for col, cells in comps4(lambda r, c: I[r, c] == bgc):
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
                continue
            s = set(cells)
            rim = Counter()
            for (y, x) in cells:
                for dy, dx in D4:
                    p = (y + dy, x + dx)
                    if p not in s:
                        rim[int(I[p])] += 1
            if rim:
                cnt[rim.most_common(1)[0][0]] += 1
        if cnt:
            sqc = cnt.most_common(1)[0][0]
    if sqc is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # Loose pieces: single-colour 4-connected objects that are neither background nor box.
    pieces = comps4(lambda r, c: I[r, c] != bgc and I[r, c] != sqc)

    # Piece-driven reading: each loose piece looks for the offset at which it slots
    # exactly into a background pocket whose every outside 4-neighbour is box colour.
    def fits(cells, dr, dc):
        sh = set((r + dr, c + dc) for r, c in cells)
        for (y, x) in sh:
            if not (0 <= y < h and 0 <= x < w) or I[y, x] != bgc:
                return False
        for (y, x) in sh:
            for dy, dx in D4:
                p = (y + dy, x + dx)
                if p in sh:
                    continue
                if not (0 <= p[0] < h and 0 <= p[1] < w) or I[p] != sqc:
                    return False
        return True

    plan = []
    for col, cells in pieces:
        mr = min(r for r, _ in cells); mc = min(c for _, c in cells)
        Mr = max(r for r, _ in cells); Mc = max(c for _, c in cells)
        found = None
        for tr in range(0, h - (Mr - mr)):
            for tc in range(0, w - (Mc - mc)):
                dr, dc = tr - mr, tc - mc
                if (dr or dc) and fits(cells, dr, dc):
                    found = (dr, dc)
                    break
            if found:
                break
        if found:
            plan.append((sorted(cells), col, found[0], found[1]))

    grid = I.copy()
    for (src, col, dr, dc) in plan:
        seq = []
        if dr:
            seq += [(20 if dr < 0 else 21, (-1 if dr < 0 else 1, 0))] * abs(dr)
        if dc:
            seq += [(23 if dc < 0 else 22, (0, -1 if dc < 0 else 1))] * abs(dc)
        dst = [(y + dr, x + dc) for (y, x) in src]

        usable = col != 0
        states = []
        if usable:
            # one grab, one background snapshot; check each unit step is visible
            snap = grid.copy()
            for (y, x) in src:
                snap[y, x] = 0
            prev = grid
            off = (0, 0)
            for (_, (sy, sx)) in seq:
                off = (off[0] + sy, off[1] + sx)
                g = snap.copy()
                for (y, x) in src:
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
                sels.append(sel_of(src) if i == 0 else sel_of([]))
            grid = states[-1]
            dset = set(dst)
            vacated = [p for p in src if p not in dset]
            if bgc != 0 and vacated:
                ops.append(int(bgc)); sels.append(sel_of(vacated))
                for (y, x) in vacated:
                    grid[y, x] = bgc
        else:
            # colour-0 piece is invisible to ARCLE's object layer: paint it into
            # the pocket it fits, then clear its old spot.
            ops.append(int(col)); sels.append(sel_of(dst))
            for (y, x) in dst:
                grid[y, x] = col
            ops.append(int(bgc)); sels.append(sel_of(src))
            for (y, x) in src:
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
