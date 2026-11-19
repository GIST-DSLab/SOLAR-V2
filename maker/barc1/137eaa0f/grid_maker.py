"""
ARC Task: 137eaa0f (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.randrange(10)
    dotc = random.choice([c for c in range(10) if c != bgc])
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, dotc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.randrange(10)
    if dotc is None or dotc == bgc:
        dotc = random.choice([c for c in range(10) if c != bgc])

    def unifint(a, b):
        d = random.uniform(diff_lb, diff_ub)
        return min(max(a, round(a + (b - a) * d)), b)

    def simulate(I, ops, sels):
        g = np.array(I, dtype=int)
        obj, bg = None, None
        for op, s in zip(ops, sels):
            prev = g.copy()
            if isinstance(s, dict):
                cells = [tuple(x) for x in s["cells"]]
            else:
                r, c, hh, ww = s
                cells = [(i, j) for i in range(r, r + hh + 1) for j in range(c, c + ww + 1)]
            if op == 34:
                return g
            if 0 <= op <= 9:
                obj = None
                for (i, j) in cells:
                    g[i, j] = op
            elif 20 <= op <= 23:
                dr, dc = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[op]
                if cells:
                    obj = {(i, j): int(g[i, j]) for (i, j) in cells if g[i, j] != 0}
                    bg = g.copy()
                    for (i, j) in cells:
                        bg[i, j] = 0
                if obj is None:
                    return None
                obj = {(i + dr, j + dc): v for (i, j), v in obj.items()
                       if 0 <= i + dr < g.shape[0] and 0 <= j + dc < g.shape[1]}
                g = bg.copy()
                for (i, j), v in obj.items():
                    g[i, j] = v
            elif op == 33:
                obj = None
                rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
                r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
                g = g[r0:r1 + 1, c0:c1 + 1].copy()
            else:
                return None
            if g.shape == prev.shape and np.array_equal(g, prev):
                return None
        return None

    for _attempt in range(500):
        h = unifint(2, 4)
        w = unifint(2, 4)
        while min(h, w) * 2 > min(max_h, max_w) and max(h, w) > 2:
            if h >= w:
                h -= 1
            else:
                w -= 1
        lo = min(h, w) * 2
        if lo > max_h or lo > max_w:
            continue
        remcols = [c for c in range(10) if c not in (bgc, dotc)]
        go = np.full((h, w), dotc, dtype=int)
        inds = [(i, j) for i in range(h) for j in range(w)]
        loc = random.choice(inds)
        reminds = [x for x in inds if x != loc]
        nc = unifint(1, min(h * w - 1, 8))
        choscols = random.sample(remcols, nc)
        cd = {c: set() for c in choscols}
        for c in choscols:
            ij = random.choice(reminds)
            cd[c].add(ij)
            reminds.remove(ij)
        for ri in reminds:
            cd[random.choice(choscols)].add(ri)
        for c, idxes in cd.items():
            for (i, j) in idxes:
                go[i, j] = c
        gih = unifint(lo, max_h)
        giw = unifint(lo, max_w)
        objs = []
        for c in choscols:
            cells = [(dotc, loc)] + [(c, ij) for ij in cd[c]]
            mr = min(p[1][0] for p in cells); mc = min(p[1][1] for p in cells)
            objs.append([(v, (i - mr, j - mc)) for v, (i, j) in cells])
        maxtr = min(h, w) * 2
        gi = None
        for _round in range(60):
            succ = True
            grid = np.full((gih, giw), bgc, dtype=int)
            free = {(i, j) for i in range(gih) for j in range(giw)}
            for obj in objs:
                oh = max(p[1][0] for p in obj) + 1
                ow = max(p[1][1] for p in obj) + 1
                placed = None
                for _ in range(maxtr):
                    if gih - oh < 0 or giw - ow < 0:
                        break
                    li = random.randint(0, gih - oh)
                    lj = random.randint(0, giw - ow)
                    pl = [(v, (i + li, j + lj)) for v, (i, j) in obj]
                    if all(p[1] in free for p in pl):
                        placed = pl
                        break
                if placed is None:
                    succ = False
                    break
                for v, (i, j) in placed:
                    grid[i, j] = v
                for _, (i, j) in placed:
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            free.discard((i + di, j + dj))
            if succ:
                gi = grid
                break
            maxtr = int(maxtr * 1.5)
            gih = random.randint(gih, max_h)
            giw = random.randint(giw, max_w)
        if gi is None:
            continue
        try:
            ops, sels = derive_operations(gi, go)
        except Exception:
            continue
        res = simulate(gi, ops, sels)
        if res is None or res.shape != go.shape or not np.array_equal(res, go):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("137eaa0f: could not generate a valid instance")


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    # background: the colour the canvas is painted with
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    present = sorted(c for c in set(I.flatten().tolist()) if c != bgc)

    # dot colour: one per fragment, appears exactly once in every demo output
    pref = []
    if examples:
        for c in present:
            if all(int(np.sum(np.asarray(o) == c)) == 1 for _, o in examples):
                pref.append(c)
    order = pref + [c for c in present if c not in pref]

    def assemble(d):
        dots = [tuple(p) for p in np.argwhere(I == d).tolist()]
        others = [c for c in present if c != d]
        if not dots or len(others) != len(dots):
            return None
        cellsof = {c: [tuple(p) for p in np.argwhere(I == c).tolist()] for c in others}
        cand = {}
        for c in others:
            opts = []
            for k, (a, b) in enumerate(dots):
                offs = [(r - a, cc - b) for r, cc in cellsof[c]]
                if max(abs(x) for o in offs for x in o) <= 3:
                    opts.append((min(abs(x) + abs(y) for x, y in offs), k, offs))
            if not opts:
                return None
            cand[c] = sorted(opts)
        cols_order = sorted(others, key=lambda c: len(cand[c]))
        assign = {}

        def dfs(i, used, rel):
            if i == len(cols_order):
                rmin = min(p[0] for p in rel); rmax = max(p[0] for p in rel)
                cmin = min(p[1] for p in rel); cmax = max(p[1] for p in rel)
                return (rmax - rmin + 1) * (cmax - cmin + 1) == len(rel)
            c = cols_order[i]
            for _, k, offs in cand[c]:
                if k in used or any(o in rel for o in offs):
                    continue
                nrel = rel | set(offs)
                if max(p[0] for p in nrel) - min(p[0] for p in nrel) > 3 or \
                   max(p[1] for p in nrel) - min(p[1] for p in nrel) > 3:
                    continue
                assign[c] = k
                if dfs(i + 1, used | {k}, nrel):
                    return True
            return False

        if not dfs(0, frozenset(), frozenset({(0, 0)})):
            return None
        frags = [{"color": c, "cells": cellsof[c], "dot": dots[assign[c]]} for c in others]
        rel = {(0, 0)}
        total = 1
        for f in frags:
            dr0, dc0 = f["dot"]
            for (r, c) in f["cells"]:
                rel.add((r - dr0, c - dc0))
                total += 1
        if len(rel) != total:
            return None
        rmin = min(p[0] for p in rel); rmax = max(p[0] for p in rel)
        cmin = min(p[1] for p in rel); cmax = max(p[1] for p in rel)
        th, tw = rmax - rmin + 1, cmax - cmin + 1
        if th * tw != total:
            return None
        return {"dotc": d, "frags": frags, "loc": (-rmin, -cmin), "th": th, "tw": tw}

    plan = None
    for d in order:
        plan = assemble(d)
        if plan is not None:
            break
    if plan is None:
        raise ValueError("137eaa0f: could not read the fragments out of the input")

    dotc, frags, loc, th, tw = plan["dotc"], plan["frags"], plan["loc"], plan["th"], plan["tw"]

    # meeting point: every dot is brought to the middle of the canvas; fall back
    # to another spot only if that one cannot host the assembly (off-grid, or
    # two fragments would trample each other).
    cands = [(H // 2, W // 2)] + [f["dot"] for f in frags] + \
            [(r, c) for r in range(H) for c in range(W)]
    chosen = None
    for m in cands:
        tr, tc = m[0] - loc[0], m[1] - loc[1]
        if tr < 0 or tc < 0 or tr + th > H or tc + tw > W:
            continue
        for f in frags:
            f["shift"] = (m[0] - f["dot"][0], m[1] - f["dot"][1])
            sr, sc = f["shift"]
            f["sel"] = set(f["cells"]) | {f["dot"]}
            f["land"] = {(r + sr, c + sc) for (r, c) in f["sel"]}
        if dotc != 0 and not any(f["color"] != 0 or f["shift"] == (0, 0) for f in frags):
            continue
        movers = [k for k, f in enumerate(frags) if f["color"] != 0 and f["shift"] != (0, 0)]
        # G must leave before F lands on any of G's cells
        before = {k: {g for g in movers if g != k and frags[k]["land"] & frags[g]["sel"]}
                  for k in movers}
        seq, done = [], set()
        rem = sorted(movers, key=lambda k: frags[k]["dot"])
        ok = True
        while rem:
            nxt = next((k for k in rem if before[k] <= done), None)
            if nxt is None:
                ok = False
                break
            seq.append(nxt); done.add(nxt); rem.remove(nxt)
        if ok:
            chosen = (m, tr, tc, seq)
            break
    if chosen is None:
        raise ValueError("137eaa0f: no meeting point for the fragments")
    m, tr, tc, seq = chosen

    ops, sels = [], []
    g = I.copy()
    for k in seq:
        f = frags[k]
        dr, dc = f["shift"]
        steps = [20 if dr < 0 else 21] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
        # grab the fragment (its colour cells plus its dot) once, then glide
        ops.append(steps[0]); sels.append(sel_of(sorted(f["sel"])))
        for st in steps[1:]:
            ops.append(st); sels.append(sel_of([]))
        vals = {p: int(I[p]) for p in f["sel"]}
        for p in f["sel"]:
            g[p] = 0
        for (r, c), v in vals.items():
            if v != 0:
                g[r + dr, c + dc] = v

    # cells the 0-coloured pieces (and a 0 dot) land on: ARCLE cannot carry 0,
    # so those landing cells are emptied directly
    zero = set()
    for f in frags:
        if f["color"] == 0:
            sr, sc = f["shift"]
            zero |= {(r + sr, c + sc) for (r, c) in f["cells"]}
    if dotc == 0:
        zero.add(m)
    zero = sorted(p for p in zero if g[p] != 0)
    if zero:
        ops.append(0); sels.append(sel_of(zero))

    # the assembled frame is the answer (full rectangle crop)
    ops.append(33); sels.append([tr, tc, th - 1, tw - 1])
    ops.append(34); sels.append([0, 0, th - 1, tw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
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
                                f"for task 137eaa0f"
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
                    f"Failed to build a complete episode for task 137eaa0f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"137eaa0f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
