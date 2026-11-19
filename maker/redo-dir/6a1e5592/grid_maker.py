"""
ARC Task: 6a1e5592 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:                       # local fallback
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}

VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]          # 1 is reserved for the answer
    barc, bgc, objc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"barc": barc, "bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, barc, bgc, objc, rot=None) -> dict:
    from random import randint, choice

    def unifint(lb, ub, bounds):
        a, b = bounds
        return randint(a + int((b - a) * lb), a + int((b - a) * ub))

    def nb8(ij):
        i, j = ij
        return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)}

    def nb4(ij):
        i, j = ij
        return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

    def norm(cells):
        mi = min(i for i, j in cells)
        mj = min(j for i, j in cells)
        return frozenset((i - mi, j - mj) for i, j in cells)

    if rot is None:
        rot = choice((0, 1, 2, 3))
    if rot in (1, 3) and (max_w < 9 or max_h < 5):
        rot = 0
    if rot in (1, 3):
        hub = max(9, min(30, max_w))
        wub = max(5, min(30, max_h))
    else:
        hub = max(9, min(30, max_h))
        wub = max(5, min(30, max_w))

    while True:
        h = unifint(diff_lb, diff_ub, (9, hub))
        w = unifint(diff_lb, diff_ub, (5, wub))
        barh = randint(3, h // 3)
        maxobjh = h - barh - 1
        nobjs = unifint(diff_lb, diff_ub, (1, max(1, w // 3)))
        gi = [[barc] * w for _ in range(barh)] + [[bgc] * w for _ in range(h - barh)]
        go = [row[:] for row in gi]
        barinds = {(i, j) for i in range(barh) for j in range(w)}
        iinds = {(i, j) for i in range(barh, h) for j in range(w)}
        oinds = {(i, j) for i in range(h) for j in range(w)}
        placopts = list(range(1, w - 1))
        forbmarkers = set()
        onecells = set()
        fieldcells = set()
        destcells = set()
        tr = 0
        succ = 0
        maxtr = 10 * nobjs
        while tr < maxtr and succ < nobjs:
            tr += 1
            oh = randint(1, maxobjh)
            ow = randint(1, max(1, min(4, w // 2)))
            bounds = {(i, j) for i in range(oh) for j in range(ow)}
            ncells = randint(1, oh * ow)
            obj = {(0, randint(0, ow - 1))}
            for _ in range(ncells - 1):
                cand = set()
                for cc in obj:
                    cand |= nb4(cc)
                cand = (cand & bounds) - obj
                if not cand:
                    break
                obj.add(choice(sorted(cand)))
            obj = set(norm(obj))
            oh = max(i for i, j in obj) + 1
            ow = max(j for i, j in obj) + 1
            markerh = randint(1, min(oh, barh - 1))
            markpartn = norm({ij for ij in obj if ij[0] < markerh})
            invalid = False
            for k in range(1, markerh + 1):
                if norm({ij for ij in markpartn if ij[0] < k}) in forbmarkers:
                    invalid = True
            if invalid:
                continue
            for k in range(1, markerh + 1):
                forbmarkers.add(norm({ij for ij in markpartn if ij[0] < k}))
            popt = set(placopts)
            plc = [jj for jj in placopts if set(range(jj, jj + ow + 1)) <= popt]
            if not plc:
                continue
            jloc = choice(plc)
            iloc = barh - markerh
            oplcd = {(i + iloc, j + jloc) for i, j in obj}
            if oplcd <= oinds and not (oplcd & fieldcells):
                icands = [ij for ij in iinds if ij[0] <= h - oh and ij[1] <= w - ow]
                if not icands:
                    continue
                loc = choice(sorted(icands))
                iplcd = {(i + loc[0], j + loc[1]) for i, j in obj}
                if iplcd <= iinds and not (iplcd & destcells):
                    succ += 1
                    nbs = set()
                    for cc in iplcd:
                        nbs |= nb8(cc)
                    iinds = (iinds - iplcd) - nbs
                    oinds = oinds - oplcd
                    for (i, j) in iplcd:
                        gi[i][j] = objc
                    for (i, j) in (oplcd & barinds):
                        gi[i][j] = bgc
                    for (i, j) in oplcd:
                        go[i][j] = 1
                    fieldcells |= iplcd
                    destcells |= oplcd
                    onecells |= oplcd
                    jm = {j for (i, j) in onecells}
                    bad = jm | {j - 1 for j in jm} | {j + 1 for j in jm}
                    placopts = sorted(set(placopts) - bad)
            if len(placopts) == 0:
                break
        if succ >= 1:
            break

    def rotk(g, k):
        a = [row[:] for row in g]
        for _ in range(k % 4):
            a = [list(r) for r in zip(*a[::-1])]
        return a

    gi = rotk(gi, rot)
    go = rotk(go, rot)
    return {"input": tuple(tuple(r) for r in gi), "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    A = np.asarray(I, dtype=int)
    B = np.asarray(O, dtype=int)
    h, w = A.shape
    ops, sels = [], []

    if np.array_equal(A, B):
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    def bar_top(G):
        """barh if G holds the solid bar along its TOP edge, else None."""
        gh, gw = G.shape
        c = int(G[0, 0])
        if not np.all(G[0, :] == c):
            return None
        bh = 0
        while bh < gh and G[bh, 0] == c:
            bh += 1
        if bh < 3 or bh >= gh:
            return None
        if not np.all(G[:bh, gw - 1] == c):
            return None
        if np.any(G[bh:, :] == c):
            return None
        return bh

    # ---- which edge holds the bar? pick the mirror that brings it to the top
    if bar_top(A) is not None:
        kind = "i"
        N = A.copy()
    elif bar_top(np.flipud(A)) is not None:
        kind = "h"                       # hmirror
        N = np.flipud(A).copy()
    elif bar_top(A.T) is not None:
        kind = "d"                       # dmirror (main diagonal)
        N = A.T.copy()
    elif bar_top(np.rot90(A.T, 2)) is not None:
        kind = "c"                       # cmirror (anti diagonal)
        N = np.rot90(A.T, 2).copy()
    else:
        kind = "i"
        N = A.copy()

    def mirror_ops(G, k):
        """ARCLE ops that actually PERFORM mirror `k` on grid G (whole-grid selections)."""
        gh, gw = G.shape
        if k == "i":
            return [], []
        if k == "h":
            return [27], [[0, 0, gh - 1, gw - 1]]          # whole grid: up <-> down
        sq = max(gh, gw)
        P = np.zeros((sq, sq), dtype=int)
        P[:gh, :gw] = G
        M = np.zeros((sq, sq), dtype=bool)
        M[:gh, :gw] = True
        target = G.T if k == "d" else np.rot90(G.T, 2)
        pairs = [(24, 27), (26, 24), (25, 26), (27, 25)] if k == "d" \
            else [(24, 26), (27, 24), (25, 27), (26, 25)]

        def ap(X, op):
            if op == 24:
                return np.rot90(X, 1)
            if op == 25:
                return np.rot90(X, 3)
            if op == 26:
                return np.fliplr(X)
            return np.flipud(X)

        best = None
        for a, b in pairs:
            X1 = ap(P, a)
            X2 = ap(X1, b)
            Mm = ap(ap(M, a), b)
            idx = np.argwhere(Mm)
            r0, c0 = idx.min(0)
            r1, c1 = idx.max(0)
            if (r1 - r0 + 1, c1 - c0 + 1) != (gw, gh):
                continue
            if not np.array_equal(X2[r0:r1 + 1, c0:c1 + 1], target):
                continue
            oo, ss = [], []
            if gh != gw:
                oo.append(33)
                ss.append([0, 0, sq - 1, sq - 1])              # square canvas
            oo.append(a)
            ss.append([0, 0, sq - 1, sq - 1])                  # whole square region
            oo.append(b)
            ss.append([0, 0, sq - 1, sq - 1])                  # whole square region
            if gh != gw:
                oo.append(33)
                ss.append([int(r0), int(c0), gw - 1, gh - 1])  # crop back
            if best is None:
                best = (oo, ss)
            if not np.array_equal(X1, P) and not np.array_equal(X2, X1):
                return oo, ss                                  # both halves visibly act
        return best if best is not None else ([], [])

    o1, s1 = mirror_ops(A, kind)                # bring the bar to the top
    ops += o1
    sels += s1

    H, W = N.shape
    bh = bar_top(N)
    barc = int(N[0, 0])
    bar_others = sorted(set(int(v) for v in N[:bh, :].flatten()) - {barc})
    lower = set(int(v) for v in N[bh:, :].flatten())
    if bar_others:
        bgc = bar_others[0]
    else:
        vals, cnts = np.unique(N[bh:, :], return_counts=True)
        bgc = int(vals[int(np.argmax(cnts))])
    objcs = sorted(lower - {bgc})

    # ---- the objects lying loose in the field
    comps = []
    if objcs:
        objc = objcs[0]
        seen = set()
        for r in range(bh, H):
            for c in range(W):
                if N[r, c] == objc and (r, c) not in seen:
                    stack = [(r, c)]
                    seen.add((r, c))
                    comp = []
                    while stack:
                        cr, cc = stack.pop()
                        comp.append((cr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = cr + dr, cc + dc
                                if bh <= nr < H and 0 <= nc < W and (nr, nc) not in seen \
                                        and N[nr, nc] == objc:
                                    seen.add((nr, nc))
                                    stack.append((nr, nc))
                    comps.append(sorted(comp))
    comps.sort(key=lambda cm: (cm[0][0], cm[0][1]))

    # ---- the bites taken out of the bar
    bar_holes = frozenset((r, c) for r in range(bh) for c in range(W) if N[r, c] == bgc)

    # every way an object's top rows could fill a bite
    cand_lists = []
    for comp in comps:
        mi = min(r for r, c in comp)
        mj = min(c for r, c in comp)
        own = set(comp)
        shape = [(r - mi, c - mj) for r, c in comp]
        oh = max(r for r, c in shape) + 1
        lst = []
        for m in range(1, min(oh, bh - 1) + 1):
            head = [(r, c) for r, c in shape if r < m]
            iloc = bh - m
            hc = sorted(set(c for r, c in head))
            if not hc:
                continue
            for jloc in range(-hc[0], W - hc[-1]):
                placed = frozenset((r + iloc, c + jloc) for r, c in head)
                if not placed <= bar_holes:
                    continue
                full = frozenset((r + iloc, c + jloc) for r, c in shape)
                if not all(0 <= r < H and 0 <= c < W for r, c in full):
                    continue
                if not all(N[r, c] == bgc or (r, c) in own for r, c in full):
                    continue        # it must come to rest on empty ground
                plugs = True        # and it must plug the bite completely
                for (r, c) in placed:
                    for (nr, nc) in ((r - 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < bh and 0 <= nc < W and (nr, nc) not in full \
                                and (nr, nc) in bar_holes:
                            plugs = False
                if plugs:
                    lst.append((placed, full))
        cand_lists.append(lst)

    order = sorted(range(len(cand_lists)), key=lambda i: len(cand_lists[i]))

    def solve(k, used, chosen):
        if k == len(order):
            return list(chosen) if used == bar_holes else None
        for placed, full in cand_lists[order[k]]:
            if placed & used:
                continue
            chosen.append(full)
            res = solve(k + 1, used | placed, chosen)
            if res is not None:
                return res
            chosen.pop()
        return None

    def to_norm(G):
        if kind == "i":
            return G
        if kind == "h":
            return np.flipud(G)
        if kind == "d":
            return G.T
        return np.rot90(G.T, 2)

    dests = solve(0, frozenset(), []) if comps else []
    Bn = to_norm(B)
    target = set((r, c) for r in range(H) for c in range(W) if Bn[r, c] == 1)
    if dests is not None and comps:
        got = set()
        for f in dests:
            got |= set(f)
        if got != target:
            dests = None
    if dests is None:
        # safety net (does not trigger on generated data)
        seen = set()
        dests = []
        for r in range(H):
            for c in range(W):
                if (r, c) in target and (r, c) not in seen:
                    stack = [(r, c)]
                    seen.add((r, c))
                    comp = []
                    while stack:
                        cr, cc = stack.pop()
                        comp.append((cr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = cr + dr, cc + dc
                                if (nr, nc) in target and (nr, nc) not in seen:
                                    seen.add((nr, nc))
                                    stack.append((nr, nc))
                    dests.append(frozenset(comp))

    # ---- wipe every object out of the field (all of them first: one shape's resting
    #      place can sit on another object's old footprint)
    for comp in comps:
        ops.append(bgc)
        sels.append(sel_of(comp))

    # ---- draw each shape, in colour 1, hanging from the bite it belongs to
    for full in sorted(dests, key=lambda f: (min(c for r, c in f), min(r for r, c in f))):
        ops.append(1)
        sels.append(sel_of(sorted(full)))

    Nf = N.copy()
    for comp in comps:
        for (r, c) in comp:
            Nf[r, c] = bgc
    for full in dests:
        for (r, c) in full:
            Nf[r, c] = 1
    o2, s2 = mirror_ops(Nf, kind)               # put the bar back on its own edge
    ops += o2
    sels += s2

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6a1e5592"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6a1e5592"
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
                                f"for task 6a1e5592"
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
                    f"Failed to build a complete episode for task 6a1e5592 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6a1e5592-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
