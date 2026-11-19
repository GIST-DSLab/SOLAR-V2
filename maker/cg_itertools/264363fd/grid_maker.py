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
from collections import Counter
import numpy as np

def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc, linc = random.sample(cols, 3)
    rem = [c for c in cols if c not in (bgc, sqc, linc)]
    return {"bgc": bgc, "sqc": sqc, "linc": linc,
            "cpcol": random.choice(rem), "nbhcol": random.choice(rem)}

def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, linc, cpcol, nbhcol) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))
    def nbrs(p):
        r, c = p
        return {(r + i, c + j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)}
    def sh(cells, o):
        return {(r + o[0], c + o[1]) for r, c in cells}
    def shoot(start, d, n=40):
        return {(start[0] + k * d[0], start[1] + k * d[1]) for k in range(n)}
    cp = (2, 2)
    neighs = nbrs(cp)
    o1 = frozenset(sh({(0, 1), (-1, 1)}, (1, 1)))
    o2 = frozenset(sh({(1, 0), (1, -1)}, (1, 1)))
    o3 = frozenset(sh({(2, 1), (3, 1)}, (1, 1)))
    o4 = frozenset(sh({(1, 2), (1, 3)}, (1, 1)))
    mpr = {o1: (-1, 0), o2: (0, -1), o3: (1, 0), o4: (0, 1)}
    h = unifint(diff_lb, diff_ub, (15, max(15, max_h)))
    w = unifint(diff_lb, diff_ub, (15, max(15, max_w)))
    spikes = random.sample([o1, o2, o3, o4], random.randint(1, 4))
    lns = set().union(*spikes)
    obj = {cp: cpcol}
    for p in lns: obj[p] = linc
    for p in neighs - lns: obj[p] = nbhcol
    loc = (random.randint(0, h - 5), random.randint(0, w - 5))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for (r, c), v in obj.items():
        gi[r + loc[0]][c + loc[1]] = v
    placed = sh(obj.keys(), loc)
    inds = {(r, c) for r in range(h) for c in range(w) if gi[r][c] == bgc}
    for p in placed: inds -= nbrs(p)
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 100))
    succ = tr = 0
    while succ < numsq and tr < 10 * numsq:
        tr += 1
        gh = random.randint(5, h // 2 + 1); gw = random.randint(5, w // 2 + 1)
        cands = [ij for ij in inds if ij[0] <= h - gh and ij[1] <= w - gw]
        if not cands: continue
        L = random.choice(sorted(cands))
        g1 = [[sqc] * gw for _ in range(gh)]
        g2 = [[sqc] * gw for _ in range(gh)]
        full = {(r, c) for r in range(gh) for c in range(gw)}
        ginds = set(full)
        bck = sh(full, L)
        if not bck <= inds: continue
        noccs = unifint(diff_lb, diff_ub, (1, (gh * gw) // 25))
        s2 = t2 = 0
        while s2 < noccs and t2 < 5 * noccs:
            t2 += 1
            c2 = [ij for ij in ginds if ij[0] <= gh - 5 and ij[1] <= gw - 5]
            if not c2: break
            l2 = random.choice(sorted(c2))
            lns2 = set()
            for sp in spikes:
                d = mpr[sp]
                lns2 |= shoot((cp[0] + l2[0] + d[0], cp[1] + l2[1] + d[1]), d)
            lns2 &= full
            pl = {(r + l2[0], c + l2[1]): v for (r, c), v in obj.items()}
            pli = set(pl)
            lincs = {(r, c) for r, c in full if g2[r][c] == linc}
            if pli <= ginds and lns2 <= (ginds | lincs) and lns2 - pli:
                s2 += 1
                nb = set()
                for p in pli: nb |= nbrs(p)
                ginds = ginds - pli - nb - lns2
                g1[cp[0] + l2[0]][cp[1] + l2[1]] = cpcol
                for (r, c), v in pl.items(): g2[r][c] = v
                for r, c in lns2: g2[r][c] = linc
        if s2 > 0:
            succ += 1
            ob = set()
            for r in range(L[0] - 1, L[0] + gh + 1):
                for c in range(L[1] - 1, L[1] + gw + 1):
                    ob.add((r, c))
            inds = inds - ob
            for r in range(gh):
                for c in range(gw):
                    gi[r + L[0]][c + L[1]] = g1[r][c]
                    go[r + L[0]][c + L[1]] = g2[r][c]
    return {"input": gi, "output": go}

def derive_operations(I, O):
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def components(bg):
        seen = np.zeros((h, w), dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if I[r, c] != bg and not seen[r, c]:
                    stack = [(r, c)]; seen[r, c] = True; cells = []
                    while stack:
                        rr, cc = stack.pop(); cells.append((rr, cc))
                        for dr, dc in D4:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] != bg:
                                seen[nr, nc] = True; stack.append((nr, nc))
                    comps.append(cells)
        return comps

    def bg_ok(bg):
        comps = components(bg)
        if len(comps) < 2: return None
        comps.sort(key=len)
        if len(comps[0]) > 13: return None
        for comp in comps[1:]:
            rs = [r for r, _ in comp]; cs = [c for _, c in comp]
            hh = max(rs) - min(rs) + 1; ww = max(cs) - min(cs) + 1
            if hh < 5 or ww < 5 or len(comp) != hh * ww: return None
        return comps

    # background: the canvas the key object and the rectangles sit on
    counts = Counter(I.flatten().tolist())
    bgc = counts.most_common(1)[0][0]
    comps = None
    for cand, _n in counts.most_common():
        got = bg_ok(cand)
        if got is not None:
            bgc, comps = cand, got
            break
    if comps is None:
        comps = sorted(components(bgc), key=len)

    ops, sels = [], []
    G = I.copy()

    def emit(color, cells):
        cells = [(int(r), int(c)) for r, c in cells if 0 <= r < h and 0 <= c < w]
        if not cells or all(G[r, c] == color for r, c in cells):
            return
        for r, c in cells: G[r, c] = color
        ops.append(int(color)); sels.append(sel_of(cells))

    full = [(r, c) for r in range(ho) for c in range(wo)]
    if not comps:
        return [34], [sel_of(full)]

    key = comps[0]
    keyset = set(key)
    keycol = {(r, c): int(I[r, c]) for (r, c) in key}

    # rectangles and their marker pixels
    rects = []
    mcols = Counter()
    for sq in comps[1:]:
        rs = [r for r, _ in sq]; cs = [c for _, c in sq]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        sqcol = Counter(int(I[r, c]) for r, c in sq).most_common(1)[0][0]
        marks = sorted((r, c) for r, c in sq if int(I[r, c]) != sqcol)
        for m in marks: mcols[int(I[m])] += 1
        rects.append((r0, c0, r1, c1, marks))
    rects.sort(key=lambda s: (s[0], s[1]))
    cpcol = mcols.most_common(1)[0][0] if mcols else None

    # hub of the key: the cell the 8 rays leave from
    def center_ok(cr, cc):
        for (r, c) in keyset:
            dr, dc = r - cr, c - cc
            m = max(abs(dr), abs(dc))
            if m > 2: return False
            if m == 2:
                if dr and dc: return False
                if (cr + (dr // 2 if dr else 0), cc + (dc // 2 if dc else 0)) not in keyset:
                    return False
        return True

    def pick(require_col):
        best, bs = None, -1
        for (r, c) in key:
            if require_col and cpcol is not None and keycol[(r, c)] != cpcol: continue
            if not center_ok(r, c): continue
            s = sum(1 for (rr, cc) in key if max(abs(rr - r), abs(cc - c)) == 1)
            if s > bs: bs, best = s, (r, c)
        return best
    best = pick(True) or pick(False)
    if best is None:
        rs = [r for r, _ in key]; cs = [c for _, c in key]
        best = ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
    cr, cc = best

    # the key read as 8 rays: each direction has the colour of its first cell;
    # a ray continues (to the rectangle edge) when the key has a 2nd cell that way
    rays = []
    for e in (-1, 0, 1):
        for f in (-1, 0, 1):
            if e == 0 and f == 0: continue
            p1 = (cr + e, cc + f)
            if p1 not in keyset: continue
            rays.append(((e, f), keycol[p1], (cr + 2 * e, cc + 2 * f) in keyset))

    # cast each ray, direction by direction, from every marker
    for (e, f), col, long_ in rays:
        for (r0, c0, r1, c1, marks) in rects:
            for (mr, mc) in marks:
                line = []
                rr, ccc = mr + e, mc + f
                while r0 <= rr <= r1 and c0 <= ccc <= c1:
                    line.append((rr, ccc))
                    if not long_: break
                    rr += e; ccc += f
                emit(col, line)

    # the key has served as the legend: wipe it to background
    emit(bgc, key)

    ops.append(34); sels.append(sel_of(full))
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
