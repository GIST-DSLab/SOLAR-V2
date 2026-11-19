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


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None) -> dict:
    from random import randint, choice
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = list(range(10))
    if bgc is None:
        bgc = choice(cols)
    if sqc is None:
        sqc = choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, sqc)]

    def dneigh(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def norm(s):
        mr = min(r for r, _ in s); mc = min(c for _, c in s)
        return frozenset((r - mr, c - mc) for r, c in s)

    for _attempt in range(100):
        h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        h = min(h, max_h); w = min(w, max_w)
        nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 50)))
        succ = 0; tr = 0; maxtr = 5 * nsq
        gi = np.full((h, w), bgc, dtype=int)
        go = np.full((h, w), bgc, dtype=int)
        inds = {(r, c) for r in range(h) for c in range(w)}
        forbidden = []
        while tr < maxtr and succ < nsq:
            tr += 1
            oh = randint(3, 6); ow = randint(3, 6)
            bd = {(r, c) for r in range(oh) for c in range(ow)}
            bounds = {(r, c) for r in range(1, oh - 1) for c in range(1, ow - 1)}
            obj = {choice(sorted(bounds))}
            ncells = randint(1, (oh - 2) * (ow - 2))
            for _ in range(ncells - 1):
                cand = set()
                for p in obj:
                    cand |= dneigh(p)
                cand = (cand & bounds) - obj
                obj.add(choice(sorted(cand)))
            sqcands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not sqcands:
                continue
            loc = choice(sorted(sqcands))
            bdplcd = {(r + loc[0], c + loc[1]) for r, c in bd}
            if bdplcd.issubset(inds):
                tmpinds = inds - bdplcd
                inobjn = norm(obj)
                ih = max(r for r, _ in inobjn) + 1; iw = max(c for _, c in inobjn) + 1
                inobjcands = [ij for ij in inds if ij[0] <= h - ih and ij[1] <= w - iw]
                if not inobjcands:
                    continue
                loc2 = choice(sorted(inobjcands))
                inobjplcd = {(r + loc2[0], c + loc2[1]) for r, c in inobjn}
                bdnorm = frozenset(bd - obj)
                if inobjplcd.issubset(tmpinds) and bdnorm not in forbidden and inobjn not in forbidden:
                    forbidden.append(bdnorm); forbidden.append(inobjn)
                    succ += 1
                    nb = set()
                    for p in inobjplcd:
                        nb |= dneigh(p)
                    inds = (inds - (bdplcd | inobjplcd)) - nb
                    col = choice(remcols)
                    oplcd = {(r + loc[0], c + loc[1]) for r, c in obj}
                    for p in bdplcd - oplcd:
                        gi[p] = sqc
                    for p in bdplcd:
                        go[p] = sqc
                    for p in oplcd:
                        go[p] = col
                    for p in inobjplcd:
                        gi[p] = col
        if succ == 0:
            continue
        nremobjs = unifint(diff_lb, diff_ub, (0, len(inds) // 25))
        succ = 0; tr = 0; maxtr = 10 * nremobjs
        while tr < maxtr and succ < nremobjs:
            tr += 1
            oh = randint(1, 4); ow = randint(1, 4)
            bounds = {(r, c) for r in range(oh) for c in range(ow)}
            obj = {choice(sorted(bounds))}
            ncells = randint(1, oh * ow)
            for _ in range(ncells - 1):
                cand = set()
                for p in obj:
                    cand |= dneigh(p)
                cand = (cand & bounds) - obj
                obj.add(choice(sorted(cand)))
            obj = norm(obj)
            if obj in forbidden:
                continue
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            loc = choice(sorted(cands))
            plcd = {(r + loc[0], c + loc[1]) for r, c in obj}
            if plcd.issubset(inds):
                succ += 1
                nb = set()
                for p in plcd:
                    nb |= dneigh(p)
                inds = (inds - plcd) - nb
                col = choice(remcols)
                for p in plcd:
                    gi[p] = col; go[p] = col
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


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

    # background: the canvas colour (shared by every demonstration input)
    if examples:
        cnt = Counter()
        for (ei, _eo) in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
        cnt.update(I.flatten().tolist())
        bgc = int(cnt.most_common(1)[0][0])
        if bgc not in set(I.flatten().tolist()):
            bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    else:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # 4-connected monochrome components
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            col = int(I[r0, c0])
            stack = [(r0, c0)]; seen[r0, c0] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(I[ny, nx]) == col:
                        seen[ny, nx] = True; stack.append((ny, nx))
            comps.append((col, sorted(cells)))

    def norm(cells):
        mr = min(r for r, _ in cells); mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    def ring(cells):
        s = set(cells); nb = set()
        for (y, x) in cells:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in s:
                        continue
                    if not (0 <= ny < h and 0 <= nx < w):
                        return None
                    nb.add((ny, nx))
        cs = set(int(I[p]) for p in nb)
        return cs.pop() if len(cs) == 1 else None

    # black holes: enclosed background pockets wholly framed by one colour
    holes = []
    for col, cells in comps:
        if col != bgc:
            continue
        rc = ring(cells)
        if rc is not None and rc != bgc:
            holes.append((rc, cells))
    if not holes:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    # box colour: the frame colour of the pockets the demonstrations filled
    sqc = None
    if examples:
        vote = Counter()
        for (ei, eo) in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            diff = (ei != eo) & (ei == bgc)
            ys, xs = np.nonzero(diff)
            for y, x in zip(ys, xs):
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < ei.shape[0] and 0 <= nx < ei.shape[1] and ei[ny, nx] != bgc and diff[ny, nx] == False and eo[ny, nx] == ei[ny, nx]:
                        vote[int(ei[ny, nx])] += 1
        if vote:
            sqc = vote.most_common(1)[0][0]
    if sqc is None or sqc not in set(rc for rc, _ in holes):
        sqc = Counter(rc for rc, _ in holes).most_common(1)[0][0]
    holes = [cells for rc, cells in holes if rc == sqc]
    hole_by_shape = {}
    for cells in holes:
        hole_by_shape.setdefault(norm(cells), cells)

    # each non-box object: does it fit a hole like a puzzle piece?
    pieces = [(col, cells) for col, cells in comps if col not in (bgc, sqc)]
    grid = I.copy()
    used = set()
    for col, src in pieces:
        key = norm(src)
        if key not in hole_by_shape or key in used:
            continue
        hole = hole_by_shape[key]
        used.add(key)
        dr = min(r for r, _ in hole) - min(r for r, _ in src)
        dc = min(c for _, c in hole) - min(c for _, c in src)
        if dr == 0 and dc == 0:
            continue
        vseq = [((20 if dr < 0 else 21), (-1 if dr < 0 else 1, 0))] * abs(dr)
        hseq = [((23 if dc < 0 else 22), (0, -1 if dc < 0 else 1))] * abs(dc)

        def simulate(seq):
            snap = grid.copy()
            for p in src:
                snap[p] = 0
            prev = grid; off = (0, 0)
            for (_op, (sy, sx)) in seq:
                off = (off[0] + sy, off[1] + sx)
                g = snap.copy()
                for (y, x) in src:
                    ny, nx = y + off[0], x + off[1]
                    if 0 <= ny < h and 0 <= nx < w:
                        g[ny, nx] = col
                if np.array_equal(g, prev):
                    return None
                prev = g
            return prev

        chosen = None
        if col != 0:
            for seq in (vseq + hseq, hseq + vseq):
                res = simulate(seq)
                if res is not None:
                    chosen = (seq, res)
                    break
        if chosen is not None:
            seq, res = chosen
            for i, (op, _) in enumerate(seq):
                ops.append(op)
                sels.append(sel_of(src) if i == 0 else sel_of([]))  # grab once, then glide
            grid = res
            dst = set((y + dr, x + dc) for (y, x) in src)
            vacated = [p for p in src if p not in dst]
            if bgc != 0 and vacated:
                ops.append(bgc); sels.append(sel_of(vacated))
                for p in vacated:
                    grid[p] = bgc
        else:
            # colour-0 piece is invisible to ARCLE's object layer: remove + blit
            ops.append(bgc); sels.append(sel_of(src))
            for p in src:
                grid[p] = bgc
            ops.append(col); sels.append(sel_of(hole))
            for p in hole:
                grid[p] = col

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # full-grid submit
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
