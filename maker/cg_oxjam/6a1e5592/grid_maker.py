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
from collections import Counter

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    barc, bgc, objc = random.sample(cols, 3)
    return {"barc": barc, "bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, barc=None, bgc=None, objc=None) -> dict:
    if barc is None or bgc is None or objc is None:
        barc, bgc, objc = sample([c for c in range(10) if c != 1], 3)

    def n_solutions(gi, h, w, barh, objs):
        # how many ways can the objects plug every hole of the bar (stop at 2)?
        holes = frozenset((r, c) for r in range(barh) for c in range(w) if gi[r][c] == bgc)
        cand = []
        for obj in objs:
            nobj = normalize(obj)
            ow = width(nobj)
            out = []
            for top in range(barh):
                for left in range(w - ow + 1):
                    sh = shift(nobj, (top, left))
                    if any(r >= h for r, _ in sh):
                        continue
                    inbar = frozenset(ij for ij in sh if ij[0] < barh)
                    if not inbar or not inbar <= holes:
                        continue
                    bad = any(0 <= r + da < barh and 0 <= c + db < w and (r + da, c + db) not in sh
                              and gi[r + da][c + db] == bgc
                              for r, c in inbar for da, db in ((-1, 0), (0, 1), (0, -1)))
                    if not bad:
                        out.append(inbar)
            cand.append(out)
        cnt = [0]

        def rec(i, used):
            if cnt[0] >= 2:
                return
            if i == len(cand):
                if used == holes:
                    cnt[0] += 1
                return
            for ib in cand[i]:
                if not (ib & used):
                    rec(i + 1, used | ib)
        rec(0, frozenset())
        return cnt[0]

    rotf = choice((identity, rot90, rot180, rot270))
    if rotf in (rot90, rot270):
        hub, wub = max_w, max_h
    else:
        hub, wub = max_h, max_w
    hub = max(9, min(30, hub))
    wub = max(5, min(30, wub))
    for _attempt in range(100):
        h = unifint(diff_lb, diff_ub, (9, hub))
        w = unifint(diff_lb, diff_ub, (5, wub))
        barh = randint(3, h // 3)
        maxobjh = h - barh - 1
        nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
        c1 = canvas(barc, (barh, w))
        c2 = canvas(bgc, (h - barh, w))
        gi = vconcat(c1, c2)
        go = tuple(e for e in gi)
        tr = 0
        succ = 0
        maxtr = 10 * nobjs
        placopts = interval(1, w - 1, 1)
        iinds = ofcolor(gi, bgc)
        oinds = asindices(go)
        barinds = ofcolor(gi, barc)
        forbmarkers = set()
        placed_objs = []
        while tr < maxtr and succ < nobjs:
            tr += 1
            oh = randint(1, maxobjh)
            ow = randint(1, min(4, w // 2))
            bounds = asindices(canvas(-1, (oh, ow)))
            ncells = randint(1, oh * ow)
            sp = choice(totuple(connect((0, 0), (0, ow - 1))))
            obj = {sp}
            for k in range(ncells - 1):
                obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
            obj = normalize(obj)
            oh, ow = shape(obj)
            markerh = randint(1, min(oh, barh - 1))
            markpart = sfilter(obj, lambda ij: ij[0] < markerh)
            markpartn = normalize(markpart)
            isinvalid = False
            for k in range(1, markerh + 1):
                if normalize(sfilter(markpartn, lambda ij: ij[0] < k)) in forbmarkers:
                    isinvalid = True
            if isinvalid:
                continue
            for k in range(1, markerh + 1):
                forbmarkers.add(normalize(sfilter(markpartn, lambda ij: ij[0] < k)))
            placoptcands = sfilter(placopts, lambda jj: set(interval(jj, jj + ow + 1, 1)).issubset(set(placopts)))
            if len(placoptcands) == 0:
                continue
            jloc = choice(placoptcands)
            iloc = barh - markerh
            oplcd = shift(obj, (iloc, jloc))
            if oplcd.issubset(oinds):
                icands = sfilter(iinds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
                if len(icands) == 0:
                    continue
                loc = choice(totuple(icands))
                iplcd = shift(obj, loc)
                if iplcd.issubset(iinds):
                    succ += 1
                    iinds = (iinds - iplcd) - mapply(neighbors, iplcd)
                    oinds = (oinds - oplcd)
                    gi = fill(gi, objc, iplcd)
                    gi = fill(gi, bgc, oplcd & barinds)
                    go = fill(go, 1, oplcd)
                    placed_objs.append(iplcd)
                    jm = apply(last, ofcolor(go, 1))
                    placopts = sorted(difference(placopts, jm | apply(decrement, jm) | apply(increment, jm)))
            if len(placopts) == 0:
                break
        # the hole-to-object assignment must be unambiguous
        if placed_objs and n_solutions(gi, h, w, barh, placed_objs) == 1:
            break
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (measured from I only): a solid bar of `barc` lines one edge; background-coloured
    notches inside the bar are the top rows of the loose objects in the body.  Each object
    is painted 1 and slid (unit Moves, one grab) until its top rows plug its notch exactly.
    The golf reading agrees with this one (same unit: each loose object; same quantity
    fixed: its shape, shifted so it fills a notch without touching the bar elsewhere), so
    the decomposition is unchanged.  Only the bookkeeping differs: background repair is
    done once, after every object has travelled, on the start-footprint cells that are
    visibly still empty (0) at that moment.
    """
    from maker.sel_helpers import sel_of

    Ia = np.asarray(I, dtype=int)
    h, w = Ia.shape
    ops, sels = [], []
    submit_sel = [0, 0, h - 1, w - 1]

    def finish():
        ops.append(34)
        sels.append(submit_sel)
        return ops, sels

    # ---- 1. bar band, canonical frame with bar on top ----
    idx = np.arange(h * w).reshape(h, w)
    found = None
    for k in range(4):
        A = np.rot90(Ia, k)
        M = np.rot90(idx, k)
        hh, ww = A.shape
        c0 = int(A[0, 0])
        if not bool((A[0] == c0).all()):
            continue
        rows_with = [r for r in range(hh) if bool((A[r] == c0).any())]
        barh = len(rows_with)
        if rows_with != list(range(barh)):
            continue
        if barh < 3 or 3 * barh > hh:
            continue
        band_other = sorted(set(A[:barh].flatten().tolist()) - {c0})
        if len(band_other) > 1:
            continue
        found = (A, M, hh, ww, barh, c0, band_other)
        break
    if found is None:
        return finish()
    A, M, hh, ww, barh, barc, band_other = found

    if band_other:
        bgc = int(band_other[0])
    else:
        bgc = int(Counter(A[barh:].flatten().tolist()).most_common(1)[0][0])
    body_cols = sorted(set(A[barh:].flatten().tolist()) - {bgc})
    if len(body_cols) != 1:
        return finish()
    objc = int(body_cols[0])

    N = {(r, c) for r in range(barh) for c in range(ww) if int(A[r, c]) == bgc}
    if not N:
        return finish()

    # ---- 2. loose objects (4-connected) ----
    pool = {(r, c) for r in range(barh, hh) for c in range(ww) if int(A[r, c]) == objc}
    objs = []
    while pool:
        s = pool.pop()
        comp = {s}
        stack = [s]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (nr, nc) in pool:
                    pool.discard((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        objs.append(comp)
    if not objs:
        return finish()

    # ---- 3. exact cover of notch cells by object tops ----
    def build_candidates(restrict):
        out = []
        for obj in objs:
            rs = [r for r, _ in obj]
            cs = [c for _, c in obj]
            r0, c0 = min(rs), min(cs)
            oh = max(rs) - r0 + 1
            ow = max(cs) - c0 + 1
            norm = {(r - r0, c - c0) for r, c in obj}
            cands = []
            for mk in range(1, min(oh, barh - 1) + 1):
                marker = {(r, c) for r, c in norm if r < mk}
                top = barh - mk
                if top + oh > hh:
                    continue
                jrange = range(1, max(1, ww - ow - 1)) if restrict else range(0, ww - ow + 1)
                for jl in jrange:
                    placed = frozenset((top + r, jl + c) for r, c in marker)
                    if placed <= N:
                        full = frozenset((top + r, jl + c) for r, c in norm)
                        cands.append((placed, full, top - r0, jl - c0))
            out.append(cands)
        return out

    def cover(cand_lists):
        n = len(cand_lists)
        order = sorted(range(n), key=lambda i: len(cand_lists[i]))
        sol = [None] * n

        def rec(i, usedN, usedF):
            if i == n:
                return usedN == N
            oi = order[i]
            for placed, full, dr, dc in cand_lists[oi]:
                if placed & usedN or full & usedF:
                    continue
                sol[oi] = (dr, dc)
                if rec(i + 1, usedN | placed, usedF | full):
                    return True
            sol[oi] = None
            return False

        return sol if rec(0, frozenset(), frozenset()) else None

    cand_lists = build_candidates(True)
    sol = cover(cand_lists)
    if sol is None:
        cand_lists = build_candidates(False)
        sol = cover(cand_lists)
    if sol is None:
        sol = [None] * len(objs)
        usedN, usedF = set(), set()
        for i, cands in enumerate(cand_lists):
            for placed, full, dr, dc in cands:
                if (placed & usedN) or (full & usedF):
                    continue
                sol[i] = (dr, dc)
                usedN |= set(placed)
                usedF |= set(full)
                break

    def to_orig(rc):
        v = int(M[rc[0], rc[1]])
        return (v // w, v % w)

    items = []
    for i, obj in enumerate(objs):
        if sol[i] is None:
            continue
        dr, dc = sol[i]
        src_can = sorted(obj)
        dst_can = [(r + dr, c + dc) for r, c in src_can]
        items.append({
            "src_can": set(src_can), "dst_can": set(dst_can),
            "src": [to_orig(p) for p in src_can],
            "dst": [to_orig(p) for p in dst_can],
        })
    if not items:
        return finish()

    # ---- 4. plan the slides.  Each leg is one grab + unit Moves.  An object may only be
    #      grabbed while nothing else lies on it, and a leg may not end on another object.
    #      Normally every object goes straight to its slot (L- or Z-shaped route, whichever
    #      keeps every Move visible); if two objects block each other's slots, one is first
    #      parked on open background and finished later with a fresh grab.
    n_it = len(items)
    srcs = [items[i]["src"] for i in range(n_it)]
    goal = [(items[i]["dst"][0][0] - srcs[i][0][0], items[i]["dst"][0][1] - srcs[i][0][1])
            for i in range(n_it)]

    def cells_at(i, off):
        return [(r + off[0], c + off[1]) for r, c in srcs[i]]

    def leg_steps(frm, to, vfirst, split=None):
        dr, dc = to[0] - frm[0], to[1] - frm[1]
        v = [(21 if dr > 0 else 20, (1 if dr > 0 else -1, 0))] * abs(dr)
        hz = [(22 if dc > 0 else 23, (0, 1 if dc > 0 else -1))] * abs(dc)
        a, b = (v, hz) if vfirst else (hz, v)
        if split is None:
            return a + b
        return a[:split] + b + a[split:]      # Z-route: part of the first axis, the other, the rest

    def routes(frm, to):
        dr, dc = abs(to[0] - frm[0]), abs(to[1] - frm[1])
        out = [(True, None), (False, None)]
        if dr and dc:
            out += [(True, k) for k in range(1, dr)] + [(False, k) for k in range(1, dc)]
        return out

    def sim_leg(G, cur, steps, paint):
        out = []
        G0 = G
        if paint:
            G0 = G.copy()
            for p in cur:
                G0[p] = 1
            if (G0 == G).all():
                return None
            out.append(G0)
        bg = G0.copy()
        for p in cur:
            bg[p] = 0
        prev = G0
        pos = list(cur)
        for _, (a, b) in steps:
            pos = [(r + a, c + b) for r, c in pos]
            Gn = bg.copy()
            for r, c in pos:
                if 0 <= r < h and 0 <= c < w:
                    Gn[r, c] = 1
            if (Gn == prev).all():
                return None
            out.append(Gn)
            prev = Gn
        return out

    def occupied(offs, excl):
        occ = set()
        for j in range(n_it):
            if j != excl:
                occ |= set(cells_at(j, offs[j]))
        return occ

    result = []
    budget = [3000]
    max_parks = [0]

    def dfs(G, offs, painted, seen, plan, parks):
        if result or budget[0] <= 0:
            return
        budget[0] -= 1
        todo = [i for i in range(n_it) if offs[i] != goal[i]]
        if not todo:
            result.append(list(plan))
            return
        for allow_park in (False, True):
            if allow_park and parks >= max_parks[0]:
                return
            for i in todo:
                occ = occupied(offs, i)
                if allow_park:
                    dst_others = set()
                    for j in range(n_it):
                        if j != i:
                            dst_others |= set(cells_at(j, goal[j]))
                    cur_set = set(cells_at(i, offs[i]))
                    targets = []
                    for orr in range(-h, h):
                        for occ_c in range(-w, w):
                            t = (orr, occ_c)
                            if t == offs[i] or t == goal[i]:
                                continue
                            cs = cells_at(i, t)
                            if any(not (0 <= r < h and 0 <= c < w) for r, c in cs):
                                continue
                            if set(cs) & (occ | dst_others):
                                continue
                            if any(int(G[r, c]) != bgc for r, c in cs if (r, c) not in cur_set):
                                continue      # park only on open background
                            targets.append(t)
                    targets.sort(key=lambda t: abs(t[0] - offs[i][0]) + abs(t[1] - offs[i][1]))
                    targets = targets[:12]
                else:
                    if set(cells_at(i, goal[i])) & occ:
                        continue
                    targets = [goal[i]]
                for t in targets:
                    for vfirst, split in routes(offs[i], t):
                        steps = leg_steps(offs[i], t, vfirst, split)
                        grids = sim_leg(G, cells_at(i, offs[i]), steps, not painted[i])
                        if grids is None:
                            continue
                        keys = [g.tobytes() for g in grids]
                        if len(set(keys)) != len(keys) or any(k in seen for k in keys):
                            continue
                        no = list(offs); no[i] = t
                        npnt = list(painted); npnt[i] = True
                        plan.append((i, cells_at(i, offs[i]), steps, not painted[i]))
                        dfs(grids[-1], no, npnt, seen | set(keys), plan, parks + allow_park)
                        plan.pop()
                        if result:
                            return
            if result:
                return

    for mp in range(0, n_it + 1):          # fewest detours first
        max_parks[0] = mp
        budget[0] = 3000
        dfs(Ia.copy(), [(0, 0)] * n_it, [False] * n_it, {Ia.tobytes()}, [], 0)
        if result:
            break
    if result:
        plan = result[0]
    else:
        plan = [(i, srcs[i], leg_steps((0, 0), goal[i], True), True) for i in range(n_it)]

    # ---- emit: paint 1 (first touch), then one grab + empty-selection Moves per leg ----
    G = Ia.copy()
    for i, cur, steps, paint in plan:
        if paint:
            ops.append(1)
            sels.append(sel_of(cur))
        for k, (op, _) in enumerate(steps):
            ops.append(op)
            sels.append(sel_of(cur) if k == 0 else sel_of([]))
        grids = sim_leg(G, cur, steps, paint)
        if grids:
            G = grids[-1]

    # ---- 5. restore background where the slides left lifted footprints empty ----
    if bgc != 0:
        start_cells = set()
        for _, cur, steps, _ in plan:          # every footprint a grab lifted off
            if steps:
                start_cells |= set(cur)
        empty = sorted(p for p in start_cells if G[p] == 0)   # visible: still 0 right now
        if empty:
            ops.append(bgc)
            sels.append(sel_of(empty))

    return finish()


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
