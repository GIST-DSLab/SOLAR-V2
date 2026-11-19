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

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    barc, bgc, objc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"barc": barc, "bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, barc=None, bgc=None, objc=None, rot=None) -> dict:
    if barc is None or bgc is None or objc is None:
        barc, bgc, objc = sample([c for c in range(10) if c != 1], 3)
    if rot is None:
        rot = random.choice(ROTS)

    def n_solutions(gi, h, w, barh, objs):
        # how many ways can the objects plug every notch of the bar (stop at 2)?
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
                    if inbar and inbar <= holes:
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

    rotf = (identity, rot90, rot180, rot270)[rot]
    if rot in (1, 3):
        hub, wub = max_w, max_h
    else:
        hub, wub = max_h, max_w
    hub = max(9, min(30, hub))
    wub = max(5, min(30, wub))
    for _attempt in range(200):
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
        # the notch-to-object assignment must be unambiguous
        if placed_objs and n_solutions(gi, h, w, barh, placed_objs) == 1:
            break
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (measured from I only): a solid bar of `barc` lies along one edge; the bgc-coloured
    notches in it are the top rows of the loose `objc` objects in the body. Every object turns
    colour 1 and slides toward the bar until its leading rows exactly plug its notch.
    Ops (larger strokes than per-object recolor/repair):
      * ONE Color1 on all cells of the object colour,
      * per displacement, ONE grab of the object(s) sharing it, then empty-selection unit Moves,
      * ONE Color(bgc) on the body cells the moves left at 0 (the vacated footprints).
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
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in pool:
                    pool.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        objs.append(comp)

    # ---- 3. match each object to its notch (exact cover of the notch cells) ----
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

    # ---- 4. group objects by displacement (original frame) ----
    groups = {}
    for i, obj in enumerate(objs):
        if sol[i] is None:
            continue
        src_can = sorted(obj)
        src = [to_orig(p) for p in src_can]
        d0 = to_orig((src_can[0][0] + sol[i][0], src_can[0][1] + sol[i][1]))
        D = (d0[0] - src[0][0], d0[1] - src[0][1])
        groups.setdefault(D, []).extend(src)
    if not groups:
        return finish()
    items = [(list(cells), D) for D, cells in groups.items()]

    # ---- 5. one Color1 on every cell of the object colour ----
    all_obj = [to_orig(p) for o in objs for p in o]
    ops.append(1); sels.append(sel_of(all_obj))

    # ---- 6. one grab per group, then empty-selection unit Moves ----
    # Python-side replica of ARCLE's grab/move, used only to pick an object order and,
    # per group, vertical-first or horizontal-first so that every unit Move changes the grid.
    # If groups block each other in a cycle, a group may already be partly covered by an
    # earlier group that landed on its cells: grab only its uncovered cells, and paint where
    # the uncarried part lands with one Color1 afterwards.
    G0 = Ia.copy()
    for p in all_obj:
        G0[p] = 1

    def legs(D, vfirst):
        Dr, Dc = D
        v = (abs(Dr), 21 if Dr > 0 else 20)
        hz = (abs(Dc), 22 if Dc > 0 else 23)
        return [v, hz] if vfirst else [hz, v]

    STEP = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}

    def simulate(order):
        g = G0.copy()
        landed, missed, seq = set(), [], []
        clean = True
        for i in order:
            cells, D = items[i]
            grab = [p for p in cells if p not in landed]
            missed_i = [(r + D[0], c + D[1]) for r, c in cells if (r, c) in landed]
            best = None
            if grab and D != (0, 0):
                for vfirst in (True, False):
                    bg = g.copy()
                    obj = {}
                    for q in grab:
                        if g[q] != 0:
                            obj[q] = int(g[q])
                        bg[q] = 0
                    cur, steps, ok = g, [], True
                    first = True
                    for n, op in legs(D, vfirst):
                        for _ in range(n):
                            d = STEP[op]
                            obj = {(r + d[0], c + d[1]): v for (r, c), v in obj.items()}
                            ng = bg.copy()
                            for (r, c), v in obj.items():
                                if 0 <= r < h and 0 <= c < w:
                                    ng[r, c] = v
                            if np.array_equal(ng, cur):
                                ok = False
                            cur = ng
                            steps.append((op, grab if first else []))
                            first = False
                    if best is None or (ok and not best[2]):
                        best = (cur, steps, ok)
                    if ok:
                        break
                g = best[0]
                seq += best[1]
                clean = clean and best[2]
            missed += missed_i
            landed |= {(r + D[0], c + D[1]) for r, c in cells}
        return seq, landed, missed, clean

    # topological preference: a group moves before any group whose destination covers it
    rem = list(range(len(items)))
    topo = []
    while rem:
        pick = None
        for i in rem:
            dst = {(r + items[i][1][0], c + items[i][1][1]) for r, c in items[i][0]}
            if all(not (dst & set(items[j][0])) for j in rem if j != i):
                pick = i
                break
        if pick is None:
            pick = rem[0]
        topo.append(pick)
        rem.remove(pick)
    cand_orders = [topo]
    if len(items) <= 6:
        from itertools import permutations
        cand_orders += [list(o) for o in permutations(range(len(items))) if list(o) != topo]
    else:
        rng = random.Random(0)
        for _ in range(300):
            o = list(range(len(items))); rng.shuffle(o); cand_orders.append(o)
    chosen = None
    for o in cand_orders:
        res = simulate(o)
        if res[3] and not res[2]:
            chosen = res
            break
        if res[3] and chosen is None:
            chosen = res
    if chosen is None:
        chosen = simulate(topo)
    seq, landed, missed, _ = chosen
    for op, grab in seq:
        ops.append(op)
        sels.append(sel_of(grab))

    # ---- 7. vacated footprints: the body cells left at 0 by the grabs -> bgc ----
    if bgc != 0:
        hole = sorted(set(all_obj) - landed)          # source cells nothing landed on
        if hole:
            ops.append(int(bgc)); sels.append(sel_of(hole))
    if missed:
        ops.append(1); sels.append(sel_of(missed))   # the part a covered group could not carry
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
