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
    # colour 1 is reserved for the answer; the three roles are fixed per episode
    cols = [c for c in range(10) if c != 1]
    barc, bgc, objc = random.sample(cols, 3)
    return {"barc": barc, "bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, barc=None, bgc=None, objc=None) -> dict:
    if barc is None or bgc is None or objc is None:
        barc, bgc, objc = sample([c for c in range(10) if c != 1], 3)

    def n_placements(gi, h, w, barh, objs):
        # number of joint placements (stop at 2): every notch covered exactly by the
        # objects' top rows, objects flush against the bar, lanes not touching
        holes = frozenset((r, c) for r in range(barh) for c in range(w) if gi[r][c] == bgc)
        cand = []
        for obj in objs:
            nobj = normalize(obj)
            oh, ow = shape(nobj)
            out = []
            for mk in range(1, min(oh, barh - 1) + 1):
                marker = {(r, c) for r, c in nobj if r < mk}
                top = barh - mk
                if top + oh > h:
                    continue
                for jl in range(0, w - ow + 1):
                    pl = frozenset((top + r, jl + c) for r, c in marker)
                    if pl <= holes:
                        out.append((pl, jl, jl + ow - 1))
            cand.append(out)
        cnt = [0]

        def rec(i, used, spans):
            if cnt[0] >= 2:
                return
            if i == len(cand):
                if used == holes:
                    cnt[0] += 1
                return
            for pl, a, b in cand[i]:
                if pl & used:
                    continue
                if any(not (b + 1 < s0 or e0 + 1 < a) for s0, e0 in spans):
                    continue
                rec(i + 1, used | pl, spans + [(a, b)])
        rec(0, frozenset(), [])
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
        # the notch-to-object assignment must be unambiguous so the episode is learnable
        if placed_objs and n_placements(gi, h, w, barh, placed_objs) == 1:
            break
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (all measured from I): a solid bar of `barc` lines one edge; bgc-coloured
    notches in it are the tops of the loose `objc` objects lying in the body.  Read as a
    JOINT placement (as the golf entry does): find the one way every object can sit,
    flush against the bar, so that together their top rows fill every notch exactly and
    no two placed objects touch side by side.  Each object turns colour 1 as it is
    picked up and slides -- first along the bar to its slot's lane, then toward the bar
    into the slot.  Last, the footprints the objects vacated (left at 0 by ARCLE's
    grab) go back to the background in one Color(bgc).
    """
    from maker.sel_helpers import sel_of

    Ia = np.asarray(I, dtype=int)
    h, w = Ia.shape
    ops, sels = [], []

    def finish():
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # ---- 1. bar band; canonical frame with the bar on top ----
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
        if rows_with != list(range(barh)) or barh < 3 or 3 * barh > hh:
            continue
        if bool((A[barh:] == c0).any()):
            continue
        band_other = sorted(set(A[:barh].flatten().tolist()) - {c0})
        if len(band_other) > 1:
            continue
        found = (k, A, M, hh, ww, barh, c0, band_other)
        break
    if found is None:
        return finish()
    k, A, M, hh, ww, barh, barc, band_other = found
    if band_other:
        bgc = int(band_other[0])
    else:
        bgc = int(Counter(A[barh:].flatten().tolist()).most_common(1)[0][0])
    body_cols = sorted(set(A[barh:].flatten().tolist()) - {bgc})
    if len(body_cols) != 1:
        return finish()
    objc = int(body_cols[0])
    N = frozenset((r, c) for r in range(barh) for c in range(ww) if int(A[r, c]) == bgc)
    if not N:
        return finish()

    # ---- 2. loose objects (4-connected) ----
    pool = {(r, c) for r in range(barh, hh) for c in range(ww) if int(A[r, c]) == objc}
    objs = []
    while pool:
        s = pool.pop()
        comp = {s}
        st = [s]
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in pool:
                    pool.discard(nb)
                    comp.add(nb)
                    st.append(nb)
        objs.append(comp)

    # ---- 3. joint placement: exact cover of the notches, lanes not touching ----
    def candidates(restrict):
        out = []
        for obj in objs:
            r0 = min(r for r, _ in obj); c0 = min(c for _, c in obj)
            oh = max(r for r, _ in obj) - r0 + 1
            ow = max(c for _, c in obj) - c0 + 1
            norm = {(r - r0, c - c0) for r, c in obj}
            cs = []
            for mk in range(1, min(oh, barh - 1) + 1):
                marker = {(r, c) for r, c in norm if r < mk}
                top = barh - mk
                if not marker or top + oh > hh:
                    continue
                jr = range(1, max(1, ww - ow - 1)) if restrict else range(0, ww - ow + 1)
                for jl in jr:
                    placed = frozenset((top + r, jl + c) for r, c in marker)
                    if placed <= N:
                        cs.append((placed, jl, jl + ow - 1, top - r0, jl - c0))
            out.append(cs)
        return out

    def cover(cl, gap):
        n = len(cl)
        order = sorted(range(n), key=lambda i: len(cl[i]))
        sol = [None] * n

        def rec(i, used, spans):
            if i == n:
                return used == N
            oi = order[i]
            for placed, a, b, dr, dc in cl[oi]:
                if placed & used:
                    continue
                if any(not (b + gap < s0 or e0 + gap < a) for s0, e0 in spans):
                    continue
                sol[oi] = (dr, dc)
                if rec(i + 1, used | placed, spans + [(a, b)]):
                    return True
            sol[oi] = None
            return False
        return sol if rec(0, frozenset(), []) else None

    sol = None
    for restrict, gap in ((True, 1), (False, 1), (True, 0), (False, 0)):
        sol = cover(candidates(restrict), gap)
        if sol is not None:
            break
    if sol is None:
        return finish()

    def to_orig(p):
        v = int(M[p[0], p[1]])
        return (v // w, v % w)

    items = []
    for obj, (dr, dc) in zip(objs, sol):
        src = [to_orig(p) for p in sorted(obj)]
        dst = [to_orig((r + dr, c + dc)) for r, c in sorted(obj)]
        items.append((src, dst[0][0] - src[0][0], dst[0][1] - src[0][1]))

    # canonical "toward the bar" is vertical; in the original frame it is the
    # row axis when k is even, the column axis when k is odd.
    toward_axis = 0 if k % 2 == 0 else 1
    along_axis = 1 - toward_axis

    # ---- 4. plan: each object turns colour 1 when it is picked up, then slides ----
    G0 = Ia.copy()

    def axis_op(axis, sgn):
        if axis == 0:
            return 21 if sgn > 0 else 20
        return 22 if sgn > 0 else 23

    n = len(items)

    # rule-predicted end state (from I alone) used to double-check a plan in Python
    P = Ia.copy()
    for src, _, _ in items:
        for p in src:
            P[p] = bgc
    for src, Dr, Dc in items:
        for r, c in src:
            P[r + Dr, c + Dc] = 1

    srcs = [src for src, _, _ in items]
    goal = tuple((Dr, Dc) for _, Dr, Dc in items)
    lim = []
    for src in srcs:
        rs = [r for r, _ in src]; cs = [c for _, c in src]
        lim.append(((-min(rs), h - 1 - max(rs)), (-min(cs), w - 1 - max(cs))))

    def cells_at(i, off):
        return [(r + off[0], c + off[1]) for r, c in srcs[i]]

    def in_hist(hist, key):
        while hist is not None:
            parent, keys = hist
            if key in keys:
                return True
            hist = parent
        return False

    def run_leg(node, i, ax, newoff):
        # replay one leg exactly as ARCLE would; None if any step re-creates a grid
        # that was already visible earlier on this trajectory
        g, snap, hist, painted, gcells, last, st = node
        cur = cells_at(i, st[i])
        seq = []
        keys = set()
        g = g.copy()
        if i not in painted:                     # picked up -> answer colour
            seq.append((1, list(cur)))
            for p in cur:
                g[p] = 1
            keys.add(g.tobytes())
            painted = painted | {i}
            last = None
        if last != i:
            snap = g.copy()
            for p in cur:
                snap[p] = 0
            gcells = gcells | frozenset(cur)
        D = newoff[ax] - st[i][ax]
        sg = 1 if D > 0 else -1
        for q in range(abs(D)):
            seq.append((axis_op(ax, sg), list(cur) if (q == 0 and last != i) else None))
            cur = [(r + (sg if ax == 0 else 0), c + (sg if ax == 1 else 0)) for r, c in cur]
            ng = snap.copy()
            for p in cur:
                ng[p] = 1
            key = ng.tobytes()
            if key in keys or in_hist(hist, key):
                return None
            keys.add(key)
            g = ng
        nst = st[:i] + (newoff,) + st[i + 1:]
        return (g, snap, (hist, frozenset(keys)), painted, gcells, i, nst), seq

    def close(node):
        # every cell an object was picked up from that nothing covers now reads 0
        g, snap, hist, painted, gcells, last, st = node
        g = g.copy()
        seq = []
        hole = sorted(p for p in gcells if g[p] == 0)
        if bgc != 0 and hole:
            seq.append((bgc, hole))
            for p in hole:
                g[p] = bgc
            if in_hist(hist, g.tobytes()):
                return None
        return seq if np.array_equal(g, P) else None

    # Best-first search over "legs" (one object's slide along one axis).  An object
    # may not come to rest on another object's current cells -- that object's later
    # grab would swallow them -- and no unit step may leave the visible grid in a
    # state it already showed (e.g. a thin object gliding through a placed block).
    # Direct legs to the slot come first (along the bar, then toward it); when slots
    # and sources block each other in a cycle, an object is parked out of the way.
    import heapq

    def heur(st):
        return sum((st[i][0] != goal[i][0]) + (st[i][1] != goal[i][1]) for i in range(n))

    slot_cells = []
    for i in range(n):
        slot_cells.append({p for j in range(n) if j != i for p in cells_at(j, goal[j])})
    start = tuple((0, 0) for _ in range(n))
    root = (G0.copy(), None, (None, frozenset([G0.tobytes()])), frozenset(), frozenset(), None, start)
    cnt = 0
    heap = [(heur(start), 0, cnt, root, [])]
    visited = set()
    plan = None
    expansions = 0
    while heap and expansions < 30000:
        _, gcost, _, node, seq = heapq.heappop(heap)
        st, last = node[6], node[5]
        if st == goal:
            tail = close(node)
            if tail is not None:
                plan = seq + tail
                break
            continue
        vkey = (node[0].tobytes(), last)
        if vkey in visited:
            continue
        visited.add(vkey)
        expansions += 1
        occ = [set(cells_at(i, st[i])) for i in range(n)]
        for i in range(n):
            others = set().union(*[occ[j] for j in range(n) if j != i])
            for ax in (along_axis, toward_axis):
                # direct leg to the slot's coordinate, plus a few parking spots:
                # the nearest offsets on either side that clear every other
                # object's current cells and every other object's slot
                vals = [goal[i][ax]]
                for dirn in (1, -1):
                    v = st[i][ax] + dirn
                    found_n = 0
                    while lim[i][ax][0] <= v <= lim[i][ax][1] and found_n < 2:
                        trial = (v, st[i][1]) if ax == 0 else (st[i][0], v)
                        tc = set(cells_at(i, trial))
                        if v != goal[i][ax] and not (tc & others) and not (tc & slot_cells[i]):
                            vals.append(v)
                            found_n += 1
                        v += dirn
                for v in vals:
                    if v == st[i][ax]:
                        continue
                    newoff = (v, st[i][1]) if ax == 0 else (st[i][0], v)
                    if set(cells_at(i, newoff)) & others:
                        continue
                    res = run_leg(node, i, ax, newoff)
                    if res is None:
                        continue
                    child, legseq = res
                    step_cost = 1 + (0 if v == goal[i][ax] else 2) + (0 if last == i else 0.1) \
                        + (0 if ax == along_axis or st[i][along_axis] == goal[i][along_axis] else 0.05)
                    cnt += 1
                    heapq.heappush(heap, (gcost + step_cost + heur(child[6]), gcost + step_cost,
                                          cnt, child, seq + legseq))
    if plan is None:
        return finish()

    for op, cells in plan:
        ops.append(op)
        sels.append(sel_of(cells) if cells is not None else sel_of([]))
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
