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
    Rule (measured from I only): a solid bar of `barc` lies along one edge; bgc-coloured
    notches in it are the top rows of the loose objects in the body.  Every object turns
    colour 1 and slides (toward the bar) until its top rows exactly fill its notch.
    Reading (shared with the golf entry): the loose objects are placed jointly as one
    assignment of objects to notches, so they are recoloured together -- one Color1 on
    every object-coloured cell -- and then each slides home (grab once, then
    empty-selection Moves).  A final Color(bgc) restores the footprints the objects were
    picked up from.  Nothing is read from O except the submit size.
    """
    import heapq
    from collections import Counter
    from maker.sel_helpers import sel_of

    Ia = np.asarray(I, dtype=int)
    h, w = Ia.shape
    ops, sels = [], []
    submit_sel = [0, 0, h - 1, w - 1]

    def finish():
        ops.append(34)
        sels.append(submit_sel)
        return ops, sels

    # ---- 1. bar band, canonical frame with the bar on top ----
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
        if bool((A[barh:] == c0).any()):
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
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in pool:
                    pool.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        objs.append(comp)

    # ---- 3. match each object to its notch (exact cover) ----
    def build_candidates(restrict):
        out = []
        for obj in objs:
            r0 = min(r for r, _ in obj); c0 = min(c for _, c in obj)
            oh = max(r for r, _ in obj) - r0 + 1
            ow = max(c for _, c in obj) - c0 + 1
            norm = {(r - r0, c - c0) for r, c in obj}
            cands = []
            for mk in range(1, min(oh, barh - 1) + 1):
                marker = {(r, c) for r, c in norm if r < mk}
                top = barh - mk
                if not marker or top + oh > hh:
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
                if (placed & usedN) or (full & usedF):
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
                usedN |= set(placed); usedF |= set(full)
                break

    def to_orig(p):
        v = int(M[p[0], p[1]])
        return (v // w, v % w)

    items = []
    for i, obj in enumerate(objs):
        if sol[i] is None:
            continue
        dr, dc = sol[i]
        src_can = sorted(obj)
        src = [to_orig(p) for p in src_can]
        dst = [to_orig((r + dr, c + dc)) for r, c in src_can]
        items.append((src, dst[0][0] - src[0][0], dst[0][1] - src[0][1]))
    if not items:
        return finish()

    # ---- 4./5. recolour to 1 and slide every object into its notch ----
    # All loose objects are ONE colour class, recoloured together, then each slides home:
    # grab once, then empty-selection Moves.  A segment slides one object along one axis,
    # fully to its slot, part way, or briefly aside (to clear a landing spot another object
    # needs).  The plan is checked in Python so every unit Move visibly changes the grid and
    # no object is grabbed while another sits on it.  If no such plan exists, fall back to
    # sliding each object in its own colour and recolouring it to 1 once it has landed.
    all_obj = sorted(to_orig(p) for o in objs for p in o)
    srcs = [set(src) for src, _, _ in items]
    dsts = [{(r + Dr, c + Dc) for r, c in src} for src, Dr, Dc in items]
    P = Ia.copy()                                  # rule-predicted end state (from I only)
    for p in all_obj:
        P[p] = bgc
    for d in dsts:
        for p in d:
            P[p] = 1
    STEP = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}
    allidx = list(range(len(items)))
    moving = [i for i in allidx if items[i][1] or items[i][2]]
    hole_all = sorted(set().union(*[srcs[k] for k in moving]) - set().union(*dsts)) if moving else []

    def slide(g, grab, cells, op, k):
        # simulate ARCLE.  grab=None: pick up `cells` (snapshot the grid with them zeroed);
        # grab=(snap, obj): keep gliding the object already held.  None if a step is invisible.
        if grab is None:
            snap = g.copy()
            obj = {p: int(g[p]) for p in cells}
            for p in obj:
                snap[p] = 0
        else:
            snap, obj = grab
        dr, dc = STEP[op]
        cur = g
        for _ in range(k):
            obj = {(r + dr, c + dc): v for (r, c), v in obj.items()}
            ng = snap.copy()
            for (r, c), v in obj.items():
                if 0 <= r < h and 0 <= c < w and v != 0:
                    ng[r, c] = v
            if np.array_equal(ng, cur):
                return None
            cur = ng
        return cur, (snap, obj)

    def search(G0, per_object):
        # best-first search over segments; a state is (positions, held object)
        dst_all = set().union(*dsts) if dsts else set()

        def legs_left(pos):
            return sum((pos[i][1][0] != 0) + (pos[i][1][1] != 0) for i in moving)

        pos0 = {i: (list(items[i][0]), (items[i][1], items[i][2])) for i in allidx}
        start = (G0, pos0, None, frozenset(), [])
        tie = 0
        heap = [(legs_left(pos0), 0, tie, start)]
        seen = set()
        expansions = 0
        while heap and expansions < 6000:
            _, gcost, _, (g, pos, held, feet, path) = heapq.heappop(heap)
            last = held[0] if held else None
            key = (tuple(pos[i][1] for i in moving), last)
            if key in seen:
                continue
            seen.add(key)
            expansions += 1
            if all(pos[i][1] == (0, 0) for i in moving):
                hole = sorted(feet - dst_all) if bgc != 0 else []
                g2 = g.copy()
                for p in hole:
                    g2[p] = bgc
                if np.array_equal(g2, P):
                    return path, hole
                continue
            for i in moving:
                cells, (Rr, Rc) = pos[i]
                others, slots = set(), set()
                for j in moving:
                    if j != i:
                        others |= set(pos[j][0])
                        if pos[j][1] != (0, 0):
                            slots |= dsts[j]
                cont = held is not None and held[0] == i
                if not cont and any(p not in dst_all and int(P[p]) != bgc for p in cells):
                    continue          # picking it up here would leave a hole off the background
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                room = {20: min(rs), 21: h - 1 - max(rs), 22: w - 1 - max(cs), 23: min(cs)}
                for op in (20, 21, 22, 23):
                    dr, dc = STEP[op]
                    ax = 0 if dr else 1
                    D = Rr if ax == 0 else Rc
                    sg = dr + dc
                    toward = D != 0 and (D > 0) == (sg > 0)
                    if toward:
                        ks = range(abs(D), 0, -1)
                    else:                     # a detour: step aside to let another pass
                        ks = range(1, room[op] + 1)
                    for k in ks:
                        ncells = [(r + dr * k, c + dc * k) for r, c in cells]
                        end = set(ncells)
                        if end & others:
                            continue
                        partial = (not toward) or k < abs(D)
                        if partial and (end & slots):
                            continue          # a stop short of the slot only to clear someone's slot
                        res = slide(g, (held[1], held[2]) if cont else None, cells, op, k)
                        if res is None:
                            continue
                        g2, (snap, obj) = res
                        nfeet = feet if cont else feet | frozenset(cells)
                        nrem = (Rr - dr * k, Rc - dc * k)
                        nheld = (i, snap, obj)
                        step = [("move", None if cont else cells, op, k)]
                        if per_object and nrem == (0, 0):     # landed: now it turns 1
                            g2 = g2.copy()
                            for p in ncells:
                                g2[p] = 1
                            step.append(("color", ncells))
                            nheld = None                      # a Color op ends the glide
                        npos = dict(pos)
                        npos[i] = (ncells, nrem)
                        ng = gcost + (0 if cont else 1) + (0 if not partial else (1 if toward else 3))
                        tie += 1
                        heapq.heappush(heap, (ng + legs_left(npos), ng, tie,
                                              (g2, npos, nheld, nfeet, path + step)))
        return None

    G_all = Ia.copy()
    for p in all_obj:
        G_all[p] = 1
    res = search(G_all, False)
    if res is not None:
        ops.append(1); sels.append(sel_of(all_obj))        # every loose object turns 1
    else:
        G_one = Ia.copy()
        for i in allidx:
            if i not in moving:
                for p in items[i][0]:
                    G_one[p] = 1
        res = search(G_one, True)
        still = [i for i in allidx if i not in moving]
        if still:                                            # objects already in their slot
            ops.append(1); sels.append(sel_of(sorted(p for i in still for p in items[i][0])))
        if res is None:                                      # last resort: one object at a time
            steps = []
            for i in moving:
                cells, Dr, Dc = list(items[i][0]), items[i][1], items[i][2]
                first = True
                for ax, D in ((0, Dr), (1, Dc)):
                    if D:
                        op = (21 if D > 0 else 20) if ax == 0 else (22 if D > 0 else 23)
                        steps.append(("move", cells if first else None, op, abs(D)))
                        first = False
                        dr, dc = STEP[op]
                        cells = [(r + dr * abs(D), c + dc * abs(D)) for r, c in cells]
                steps.append(("color", cells))
            res = (steps, hole_all if bgc != 0 else [])
    steps, hole = res
    for st in steps:
        if st[0] == "color":
            ops.append(1); sels.append(sel_of(st[1]))       # the landed object turns 1
            continue
        _, cells, op, k = st
        for q in range(k):
            ops.append(op)
            sels.append(sel_of(cells) if (q == 0 and cells is not None) else sel_of([]))

    # ---- 6. every footprint an object was picked up from and that nothing now covers
    #         reads 0: give it back the background ----
    if hole:
        ops.append(int(bgc)); sels.append(sel_of(hole))
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
