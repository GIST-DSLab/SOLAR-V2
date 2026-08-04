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
def sample_colors(num_examples=None) -> dict:
    import random
    cols = list(range(10))
    bgc, sqc, linc, cpcol, nbhcol = random.sample(cols, 5)

    VARIANTS = [{"has_ring": True}, {"has_ring": False}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "sqc": sqc, "linc": linc, "cpcol": cpcol,
            "nbhcol": nbhcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w,
             bgc=1, sqc=8, linc=3, cpcol=4, nbhcol=6, has_ring=None) -> dict:
    import random

    def _unifint(lb, ub):
        if ub < lb:
            ub = lb
        a = lb + int((ub - lb) * diff_lb)
        b = lb + int((ub - lb) * diff_ub)
        if b < a:
            a, b = b, a
        a = max(lb, min(ub, a))
        b = max(lb, min(ub, b))
        return random.randint(a, b)

    if has_ring is None:
        has_ring = random.choice([True, False])

    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    NEIGH8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    hlo = min(15, max_h)
    wlo = min(15, max_w)
    h = _unifint(hlo, max_h)
    w = _unifint(wlo, max_w)
    h = max(12, min(h, max_h))
    w = max(12, min(w, max_w))

    for _attempt in range(40):
        nspikes = random.randint(1, 4)
        spikes = random.sample(DIRS, nspikes)
        lns_rel = set()
        for d in spikes:
            lns_rel.add((d[0], d[1]))
            lns_rel.add((2 * d[0], 2 * d[1]))
        obj_rel = {(0, 0): cpcol}
        for cell in lns_rel:
            obj_rel[cell] = linc
        if has_ring:
            for cell in NEIGH8:
                if cell not in lns_rel:
                    obj_rel[cell] = nbhcol

        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]

        loci = random.randint(0, h - 5)
        locj = random.randint(0, w - 5)
        kctr = (loci + 2, locj + 2)
        keycells = set()
        for (dr, dc), col in obj_rel.items():
            r, c = kctr[0] + dr, kctr[1] + dc
            gi[r][c] = col
            keycells.add((r, c))

        blocked = set()
        for (r, c) in keycells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    blocked.add((r + dr, c + dc))
        inds = set((r, c) for r in range(h) for c in range(w)
                   if (r, c) not in blocked)

        numsq = _unifint(1, max(1, (h * w) // 100))
        succ = 0
        tr = 0
        maxtr = 10 * numsq + 10
        while succ < numsq and tr < maxtr:
            tr += 1
            gh = random.randint(5, h // 2 + 1)
            gw = random.randint(5, w // 2 + 1)
            cands = [ij for ij in inds if ij[0] <= h - gh and ij[1] <= w - gw]
            if not cands:
                continue
            loc = random.choice(sorted(cands))
            bck = set((loc[0] + i, loc[1] + j) for i in range(gh) for j in range(gw))
            if not bck <= inds:
                continue

            ginds = set((i, j) for i in range(gh) for j in range(gw))
            g1 = [[sqc] * gw for _ in range(gh)]
            g2 = [[sqc] * gw for _ in range(gh)]
            lincells = set()
            noccs = _unifint(1, max(1, (gh * gw) // 25))
            succ2 = 0
            tr2 = 0
            maxtr2 = 5 * noccs + 5
            while succ2 < noccs and tr2 < maxtr2:
                tr2 += 1
                cands2 = [ij for ij in ginds if ij[0] <= gh - 5 and ij[1] <= gw - 5]
                if not cands2:
                    break
                loc2 = random.choice(sorted(cands2))
                ctr = (loc2[0] + 2, loc2[1] + 2)
                lns2 = set()
                for d in spikes:
                    rr, cc = ctr[0] + d[0], ctr[1] + d[1]
                    while 0 <= rr < gh and 0 <= cc < gw:
                        lns2.add((rr, cc))
                        rr += d[0]
                        cc += d[1]
                plcd2i = set((ctr[0] + dr, ctr[1] + dc) for (dr, dc) in obj_rel)
                if plcd2i <= ginds and lns2 <= (ginds | lincells) and len(lns2 - plcd2i) > 0:
                    succ2 += 1
                    nbrs = set()
                    for (r, c) in plcd2i:
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nbrs.add((r + dr, c + dc))
                    ginds = ((ginds - plcd2i) - nbrs) - lns2
                    g1[ctr[0]][ctr[1]] = cpcol
                    for (dr, dc), col in obj_rel.items():
                        g2[ctr[0] + dr][ctr[1] + dc] = col
                    for (r, c) in lns2:
                        g2[r][c] = linc
                    lincells |= lns2

            if succ2 > 0:
                succ += 1
                ob = set((loc[0] + i, loc[1] + j)
                         for i in range(-1, gh + 1) for j in range(-1, gw + 1))
                inds -= ob
                for i in range(gh):
                    for j in range(gw):
                        gi[loc[0] + i][loc[1] + j] = g1[i][j]
                        go[loc[0] + i][loc[1] + j] = g2[i][j]

        if succ > 0:
            return {'input': tuple(tuple(row) for row in gi),
                    'output': tuple(tuple(row) for row in go)}

    return {'input': tuple(tuple(row) for row in gi),
            'output': tuple(tuple(row) for row in go)}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def components(grid, bg):
        seen = np.zeros((h, w), dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != bg and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    cells = []
                    while stack:
                        rr, cc = stack.pop()
                        cells.append((rr, cc))
                        for dr, dc in DIRS:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] \
                                    and grid[nr, nc] != bg:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
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
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if (r1 - r0 + 1) < 5 or (c1 - c0 + 1) < 5:
                return None
            if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
                return None
        return comps

    # --- background: the canvas colour the key object and the rectangles sit on ---
    counts = Counter(I.flatten().tolist())
    bgc = counts.most_common(1)[0][0]
    comps = None
    for cand, _n in counts.most_common():
        got = bg_ok(cand)
        if got is not None:
            bgc = cand
            comps = got
            break
    if comps is None:
        comps = components(I, bgc)
        comps.sort(key=len)

    ops, sels = [], []
    G = I.copy()

    def emit(color, cells):
        cells = [(int(r), int(c)) for r, c in cells if 0 <= r < h and 0 <= c < w]
        if not cells:
            return
        if all(G[r, c] == color for r, c in cells):
            return                      # would change nothing -> not an action
        for r, c in cells:
            G[r, c] = color
        ops.append(int(color))
        sels.append(sel_of(cells))

    if not comps:
        ops.append(34)
        sels.append(sel_of([(r, c) for r in range(ho) for c in range(wo)]))
        return ops, sels

    key = comps[0]
    squares = comps[1:]
    keyset = set(key)
    keycol = {(r, c): int(I[r, c]) for (r, c) in key}

    # --- the marker colour: the odd pixel inside each rectangle ---
    square_info = []
    marker_colors = Counter()
    for sq in squares:
        rs = [r for r, _ in sq]
        cs = [c for _, c in sq]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        cnt = Counter(int(I[r, c]) for (r, c) in sq)
        sqcol = cnt.most_common(1)[0][0]
        markers = sorted([(r, c) for (r, c) in sq if int(I[r, c]) != sqcol])
        for m in markers:
            marker_colors[int(I[m[0], m[1]])] += 1
        square_info.append((r0, c0, r1, c1, sqcol, markers))
    square_info.sort(key=lambda s: (s[0], s[1]))
    cpcol = marker_colors.most_common(1)[0][0] if marker_colors else None

    # --- centre of the key object: the hub the arms radiate from ---
    def center_ok(cr, cc):
        for (r, c) in keyset:
            dr, dc = r - cr, c - cc
            m = max(abs(dr), abs(dc))
            if m > 2:
                return False
            if m == 2:
                if dr != 0 and dc != 0:
                    return False
                if (cr + (dr // 2 if dr else 0), cc + (dc // 2 if dc else 0)) not in keyset:
                    return False
        return True

    best, bestscore = None, -1
    for (r, c) in key:
        if cpcol is not None and keycol[(r, c)] != cpcol:
            continue
        if not center_ok(r, c):
            continue
        score = sum(1 for (rr, cc) in key if max(abs(rr - r), abs(cc - c)) == 1)
        if score > bestscore:
            bestscore, best = score, (r, c)
    if best is None:
        for (r, c) in key:
            if center_ok(r, c):
                score = sum(1 for (rr, cc) in key if max(abs(rr - r), abs(cc - c)) == 1)
                if score > bestscore:
                    bestscore, best = score, (r, c)
    if best is None:
        rs = [r for r, _ in key]
        cs = [c for _, c in key]
        best = ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
    cr, cc = best

    # --- arms (which directions, what colour) and the surrounding ring ---
    spikes = []
    for d in DIRS:
        if (cr + d[0], cc + d[1]) in keyset and (cr + 2 * d[0], cc + 2 * d[1]) in keyset:
            spikes.append(d)
    linc = None
    if spikes:
        d = spikes[0]
        linc = keycol[(cr + 2 * d[0], cc + 2 * d[1])]

    ring = []
    spikeset = set(spikes)
    for (r, c) in key:
        dr, dc = r - cr, c - cc
        m = max(abs(dr), abs(dc))
        if m == 0 or m >= 2:
            continue
        if (dr, dc) in spikeset:
            continue
        ring.append(((dr, dc), keycol[(r, c)]))

    # --- stamp the template on every marker, then grow its arms to the rectangle edge ---
    for (r0, c0, r1, c1, sqcol, markers) in square_info:
        for (mr, mc) in markers:
            bycol = {}
            for (dr, dc), col in ring:
                rr, ccc = mr + dr, mc + dc
                if r0 <= rr <= r1 and c0 <= ccc <= c1:
                    bycol.setdefault(col, []).append((rr, ccc))
            for col in sorted(bycol):
                emit(col, bycol[col])
            if linc is None:
                continue
            for d in spikes:
                line = []
                rr, ccc = mr + d[0], mc + d[1]
                while r0 <= rr <= r1 and c0 <= ccc <= c1:
                    line.append((rr, ccc))
                    rr += d[0]
                    ccc += d[1]
                emit(linc, line)

    # --- the key object has done its job: wipe it back to background ---
    emit(bgc, key)

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(ho) for c in range(wo)]))
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
