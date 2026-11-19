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
    Rule (measured from I only): a solid bar of `barc` lies along one edge; bgc-coloured
    notches in it are the top rows of the loose objects in the body.  Every object turns
    colour 1 and slides (toward the bar) until its top rows exactly fill its notch.
    Ops: one Color1 on all object cells, then unit Moves where objects sharing a direction
    ride together (a group is re-grabbed only when one of its members arrives), then one
    Color(bgc) on the vacated footprint cells ARCLE left at 0.
    """
    from maker.sel_helpers import sel_of
    from itertools import permutations

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

    # ---- 4. recolour every loose object at once: all cells of the object colour ----
    all_obj = [to_orig(p) for o in objs for p in o]
    ops.append(1); sels.append(sel_of(all_obj))
    G0 = Ia.copy()
    for p in all_obj:
        G0[p] = 1

    # rule-predicted end state (from I only) to validate a move plan in Python
    P = Ia.copy()
    for p in all_obj:
        P[p] = bgc
    for src, Dr, Dc in items:
        for r, c in src:
            P[r + Dr, c + Dc] = 1

    STEP = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}

    def axis_op(axis, sgn):
        if axis == 0:
            return 21 if sgn > 0 else 20
        return 22 if sgn > 0 else 23

    def plan_grouped(phases):
        # phases: list of (axis, sign).  Returns list of (op, grab_set_or_None).
        pos = [list(src) for src, _, _ in items]
        rem = [[Dr, Dc] for _, Dr, Dc in items]
        seq = []
        grabbed = None
        for axis, sgn in phases:
            while True:
                grp = tuple(i for i in range(len(items)) if rem[i][axis] * sgn > 0)
                if not grp:
                    break
                op = axis_op(axis, sgn)
                if grp != grabbed:
                    seq.append((op, [p for i in grp for p in pos[i]]))
                    grabbed = grp
                else:
                    seq.append((op, None))
                d = STEP[op]
                for i in grp:
                    pos[i] = [(r + d[0], c + d[1]) for r, c in pos[i]]
                    rem[i][axis] -= sgn
        return seq

    def run_seq(state, seq):
        # advance a simulator state over seq; None if some op leaves the grid unchanged
        g, snap, obj, foot = state
        foot = set(foot)
        for op, grab in seq:
            if grab is not None:
                snap = g.copy()
                obj = {}
                for p in grab:
                    if g[p] != 0:
                        obj[p] = int(g[p])
                    snap[p] = 0
                foot |= set(grab)
            d = STEP[op]
            obj = {(r + d[0], c + d[1]): v for (r, c), v in obj.items()}
            ng = snap.copy()
            for (r, c), v in obj.items():
                if 0 <= r < h and 0 <= c < w:
                    ng[r, c] = v
            if np.array_equal(ng, g):
                return None
            g = ng
        return (g, snap, obj, foot)

    def finish_state(state):
        g = state[0].copy()
        hole = sorted(p for p in state[3] if g[p] == 0) if bgc != 0 else []
        for p in hole:
            g[p] = bgc
        return (g, hole) if np.array_equal(g, P) else None

    init = (G0.copy(), None, None, set())
    chosen = None
    # (a) objects sharing a direction ride together
    for seq in sorted((plan_grouped(list(ph)) for ph in
                       permutations([(0, 1), (0, -1), (1, 1), (1, -1)])), key=len):
        st = run_seq(init, seq)
        res = finish_state(st) if st is not None else None
        if res is not None:
            chosen = (seq, res[1])
            break
    # (b) one object-leg (full travel along one axis) at a time, interleaved by search
    if chosen is None:
        n_it = len(items)
        tgt = [[(r + Dr, c + Dc) for r, c in src] for src, Dr, Dc in items]
        budget = [30000]
        seen = set()

        def dfs(state, pos, grabbed, acc):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            key = (state[0].tobytes(), tuple(tuple(sorted(x)) for x in pos), grabbed)
            if key in seen:
                return None
            seen.add(key)
            remv = [((tgt[i][0][0] - pos[i][0][0]), (tgt[i][0][1] - pos[i][0][1])) for i in range(n_it)]
            if all(rv == (0, 0) for rv in remv):
                res = finish_state(state)
                return (acc, res[1]) if res is not None else None
            g = state[0]
            moves = []
            for i in range(n_it):
                for ax in (0, 1):
                    D = remv[i][ax]
                    if D == 0:
                        continue
                    for k in range(abs(D), 0, -1):
                        moves.append((0 if i == grabbed else 1, abs(D) - k, i, ax, k, 1 if D > 0 else -1))
            moves.sort()
            for _, _, i, ax, k, sg in moves:
                if grabbed != i:
                    others = {p for j in range(n_it) if j != i for p in pos[j]}
                    if any(g[p] != 1 for p in pos[i]) or (set(pos[i]) & others):
                        continue
                op = axis_op(ax, sg)
                seq = [(op, (pos[i] if grabbed != i else None) if q == 0 else None) for q in range(k)]
                st = run_seq(state, seq)
                if st is None:
                    continue
                dd = STEP[op]
                npos = list(pos)
                npos[i] = [(r + dd[0] * k, c + dd[1] * k) for r, c in pos[i]]
                out = dfs(st, npos, i, acc + seq)
                if out is not None:
                    return out
            return None

        chosen = dfs(init, [list(src) for src, _, _ in items], None, [])
    if chosen is None:
        seq = []
        for i, (src, Dr, Dc) in enumerate(items):
            first = True
            cur = list(src)
            for ax, D in ((0, Dr), (1, Dc)):
                for _ in range(abs(D)):
                    op = axis_op(ax, 1 if D > 0 else -1)
                    seq.append((op, cur if first else None))
                    first = False
                    dd = STEP[op]
                    cur = [(r + dd[0], c + dd[1]) for r, c in cur]
        st = run_seq(init, seq)
        chosen = (seq, sorted(p for p in st[3] if st[0][p] == 0) if (st and bgc != 0) else [])
    seq, hole = chosen
    for op, grab in seq:
        ops.append(op)
        sels.append(sel_of(grab) if grab is not None else sel_of([]))
    if hole:
        ops.append(int(bgc)); sels.append(sel_of(hole))   # vacated footprints -> background
    return finish()





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
