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

ROTS = ['identity', 'rot90', 'rot180', 'rot270']


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    objc = random.choice([c for c in cols if c != 0])  # 0 cannot be grabbed by Move
    barc, bgc = random.sample([c for c in cols if c != objc], 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append({"rot": random.choice(ex)})
    return {"barc": barc, "bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, barc, bgc, objc, rot=None) -> dict:
    if rot is None:
        rot = choice(ROTS)
    rotf = {'identity': identity, 'rot90': rot90, 'rot180': rot180, 'rot270': rot270}[rot]
    swaps = rot in ('rot90', 'rot270')
    hub = max_w if swaps else max_h
    wub = max_h if swaps else max_w
    h = unifint(diff_lb, diff_ub, (9, max(9, hub)))
    w = unifint(diff_lb, diff_ub, (5, max(5, wub)))
    barh = randint(3, h // 3)
    maxobjh = h - barh - 1
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, w // 3)))
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
    while tr < maxtr and succ < nobjs:
        tr += 1
        oh = randint(1, maxobjh)
        ow = randint(1, min(4, w // 2))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = randint(1, oh * ow)
        sp = choice(totuple(connect((0, 0), (0, ow - 1))))
        obj = {sp}
        for k in range(ncells - 1):
            cands = totuple((bounds - obj) & mapply(dneighbors, obj))
            if len(cands) == 0:
                break
            obj.add(choice(cands))
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
                jm = apply(last, ofcolor(go, 1))
                placopts = sorted(difference(placopts, jm | apply(decrement, jm) | apply(increment, jm)))
        if len(placopts) == 0:
            break
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Reading: a solid bar along one edge has notches (background-coloured cells) on its
    inner side.  Each loose puzzle piece is translated (no rotation) until it plugs a
    notch exactly; finally every piece colour cell turns blue (1).
    Ops: for each piece, one grab-Move then empty Moves (along the bar, then into it,
    or the reverse when that is the only way every step stays visible; a piece that
    sits on another's slot is first parked on open background).  Then one Color(bgc)
    on the footprints the grabs left at 0, then one FloodFill1 per seated piece.
    Pieces drawn in colour 0 cannot be carried by Move, so they turn blue first.
    Nothing is read from O.
    """
    import numpy as np
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

    N = {(r, c) for r in range(barh) for c in range(ww) if int(A[r, c]) == bgc}
    if not N:
        return finish()

    cells = {(r, c) for r in range(barh, hh) for c in range(ww) if int(A[r, c]) == objc}
    objs = []
    pool = set(cells)
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
                if not marker:
                    continue
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
        items.append({
            "src_can": set(src_can),
            "dst_can": {(r + dr, c + dc) for r, c in src_can},
            "src": [to_orig(p) for p in src_can],
            "dst": [to_orig((r + dr, c + dc)) for r, c in src_can],
        })

    # ---- 5. plan the slides (plain-Python simulation of ARCLE's grab/move) ----
    # A depth-first search over which piece to slide next (and which leg first) so
    # that every unit Move visibly changes the grid and no grid state repeats.
    toward_is_row = (k % 2 == 0)        # canonical "toward the bar" axis in original coords

    def legs(Dr, Dc, lateral_first):
        v = [(21 if Dr > 0 else 20, (1 if Dr > 0 else -1, 0))] * abs(Dr)
        hz = [(22 if Dc > 0 else 23, (0, 1 if Dc > 0 else -1))] * abs(Dc)
        lat, tow = (hz, v) if toward_is_row else (v, hz)
        return lat + tow if lateral_first else tow + lat

    def simulate(G, cells, steps, seen, strict):
        bg = G.copy()
        colours = [(r, c, int(G[r, c])) for r, c in cells]
        for r, c in cells:
            bg[r, c] = 0
        dr = dc = 0
        states = []
        local = set()
        for _, (a, b2) in steps:
            dr += a
            dc += b2
            ng = bg.copy()
            for r, c, v in colours:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and v != 0:
                    ng[rr, cc] = v
            key = ng.tobytes()
            if strict and (key in seen or key in local):
                return None
            local.add(key)
            states.append(key)
            last = ng
        return last, states

    items_n = len(items)
    dst_of = {i: sorted(items[i]["dst"]) for i in range(items_n)}
    budget = [4000]

    def options(G, cur, rem):
        def occupied_by_others(i):
            out = set()
            for j in rem:
                if j != i:
                    out |= set(cur[j])
            return out
        opts = []
        for i in rem:
            if set(dst_of[i]) & occupied_by_others(i):
                continue
            Dr = dst_of[i][0][0] - cur[i][0][0]
            Dc = dst_of[i][0][1] - cur[i][0][1]
            for lf in (True, False):
                opts.append((i, cur[i], dst_of[i], legs(Dr, Dc, lf)))
        if opts:
            return opts
        # deadlock: each piece lies on another's slot -- park one on open background
        for i in rem:
            blocked = set()
            for j in rem:
                if j != i:
                    blocked |= set(dst_of[j]) | set(cur[j])
            base = cur[i]
            bset = set(base)
            offs = sorted(((dr, dc) for dr in range(-h, h + 1) for dc in range(-w, w + 1)
                           if (dr, dc) != (0, 0)), key=lambda t: (abs(t[0]) + abs(t[1]), t))
            cnt = 0
            for dr, dc in offs:
                tgt = [(r + dr, c + dc) for r, c in base]
                if not all(0 <= r < h and 0 <= c < w for r, c in tgt):
                    continue
                if any(p in blocked for p in tgt):
                    continue
                if not all(int(G[r, c]) == bgc or (r, c) in bset for r, c in tgt):
                    continue
                for lf in (True, False):
                    opts.append((i, base, tgt, legs(dr, dc, lf)))
                cnt += 1
                if cnt >= 3:
                    break
        return opts

    def search(G, cur, rem, seen, strict):
        if not rem:
            return []
        budget[0] -= 1
        if budget[0] < 0:
            return None
        for i, base, tgt, stp in options(G, cur, rem):
            res = simulate(G, base, stp, seen, strict)
            if res is None:
                continue
            NG, states = res
            ncur = dict(cur)
            ncur[i] = tgt
            nrem = [j for j in rem if j != i] if tgt == dst_of[i] else list(rem)
            sub = search(NG, ncur, nrem, seen | set(states), strict)
            if sub is not None:
                return [(base, stp, NG)] + sub
            if not strict:
                return None
        return None

    # Move only carries non-zero cells: pieces drawn in colour 0 must turn blue
    # before they can travel (otherwise they turn blue once seated, step 7).
    G0 = Ia.copy()
    if objc == 0:
        for i in range(items_n):
            ops.append(1)
            sels.append(sel_of(items[i]["src"]))
            for r, c in items[i]["src"]:
                G0[r, c] = 1

    cur0 = {i: sorted(items[i]["src"]) for i in range(items_n)}
    rem0 = [i for i in range(items_n) if cur0[i] != dst_of[i]]
    plan = search(G0.copy(), cur0, rem0, {Ia.tobytes(), G0.tobytes()}, True)
    if plan is None:
        budget[0] = 4000
        plan = search(G0.copy(), cur0, rem0, {G0.tobytes()}, False) or []

    G = G0.copy()
    grabbed = set()                     # every footprint a grab zeroed in the snapshot
    for base, stp, NG in plan:
        for n_, (op, _) in enumerate(stp):
            ops.append(op)
            sels.append(sel_of(base) if n_ == 0 else sel_of([]))
        grabbed.update(base)
        G = NG

    # ---- 6. the footprints the grabs left at 0 go back to background ----
    if bgc != 0:
        holes = sorted(p for p in grabbed if int(G[p[0], p[1]]) == 0)
        if holes:
            ops.append(int(bgc))
            sels.append(sel_of(holes))
            for r, c in holes:
                G[r, c] = bgc

    # ---- 7. every seated piece turns blue: FloodFill1 seeded on the piece ----
    if objc not in (0, 1):
        for i in range(len(items)):
            sr, sc = dst_of[i][0]
            if int(G[sr, sc]) != objc:
                continue
            ops.append(11)
            sels.append([sr, sc, 0, 0])
            stack = [(sr, sc)]
            G[sr, sc] = 1
            while stack:
                r, c = stack.pop()
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < h and 0 <= nc < w and int(G[nr, nc]) == objc:
                        G[nr, nc] = 1
                        stack.append((nr, nc))
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
