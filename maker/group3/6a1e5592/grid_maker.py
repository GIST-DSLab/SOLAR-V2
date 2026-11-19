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
    n_ex = num_examples if num_examples else 3
    rots = [0, 1, 2, 3]
    if n_ex >= len(rots):
        examples = [{"rot": r} for r in rots]
        examples += [{"rot": random.choice(rots)} for _ in range(n_ex - len(rots))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(rots, n_ex)]
    plan = [dict(e) for e in examples] + [dict(random.choice(examples))]
    return {"barc": barc, "bgc": bgc, "objc": objc, "instance_plan": plan}


def _canon(I):
    """Find the rotation that puts the solid bar band on top. Returns
    (A, M, barh, barc, bgc) or None. M maps canonical cells to flat original indices."""
    Ia = np.asarray(I, dtype=int)
    h, w = Ia.shape
    idx = np.arange(h * w).reshape(h, w)
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
        other = sorted(set(A[:barh].flatten().tolist()) - {c0})
        if len(other) > 1:
            continue
        if other:
            bgc = int(other[0])
        else:
            bgc = int(Counter(A[barh:].flatten().tolist()).most_common(1)[0][0])
        return A, M, barh, c0, bgc
    return None


def _plan(I):
    """Return (bgc, objc, items) where items = list of (src_cells, dst_cells) in
    original coordinates, ordered so nothing lands on a shape still to move."""
    Ia = np.asarray(I, dtype=int)
    h, w = Ia.shape
    found = _canon(Ia)
    if found is None:
        return None
    A, M, barh, barc, bgc = found
    hh, ww = A.shape
    body_cols = sorted(set(A[barh:].flatten().tolist()) - {bgc})
    if len(body_cols) != 1:
        return None
    objc = int(body_cols[0])
    N = {(r, c) for r in range(barh) for c in range(ww) if int(A[r, c]) == bgc}
    if not N:
        return None
    pool = {(r, c) for r in range(barh, hh) for c in range(ww) if int(A[r, c]) == objc}
    objs = []
    while pool:
        s = pool.pop()
        comp, stack = {s}, [s]
        while stack:
            r, c = stack.pop()
            for p in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if p in pool:
                    pool.discard(p); comp.add(p); stack.append(p)
        objs.append(comp)

    def cands_for(restrict):
        out = []
        for obj in objs:
            r0 = min(r for r, _ in obj); c0 = min(c for _, c in obj)
            norm = {(r - r0, c - c0) for r, c in obj}
            oh = max(r for r, _ in norm) + 1; ow = max(c for _, c in norm) + 1
            cl = []
            for mk in range(1, min(oh, barh - 1) + 1):
                marker = {(r, c) for r, c in norm if r < mk}
                top = barh - mk
                if top + oh > hh:
                    continue
                jr = range(1, max(1, ww - ow - 1)) if restrict else range(0, ww - ow + 1)
                for jl in jr:
                    placed = frozenset((top + r, jl + c) for r, c in marker)
                    if placed <= N:
                        full = frozenset((top + r, jl + c) for r, c in norm)
                        cl.append((placed, full, top - r0, jl - c0))
            out.append(cl)
        return out

    def cover(cl):
        n = len(cl)
        order = sorted(range(n), key=lambda i: len(cl[i]))
        sol = [None] * n

        def rec(i, uN, uF):
            if i == n:
                return uN == N
            oi = order[i]
            for placed, full, dr, dc in cl[oi]:
                if placed & uN or full & uF:
                    continue
                sol[oi] = (dr, dc)
                if rec(i + 1, uN | placed, uF | full):
                    return True
            sol[oi] = None
            return False
        return sol if rec(0, frozenset(), frozenset()) else None

    sol = cover(cands_for(True))
    if sol is None:
        sol = cover(cands_for(False))
    if sol is None:
        return None

    def to_orig(p):
        v = int(M[p[0], p[1]])
        return (v // w, v % w)

    items = []
    for obj, (dr, dc) in zip(objs, sol):
        src_can = sorted(obj)
        src = [to_orig(p) for p in src_can]
        dst = [to_orig((r + dr, c + dc)) for r, c in src_can]
        items.append((src, dst))
    rem = list(range(len(items)))
    ordered = []
    while rem:
        pick = next((i for i in rem
                     if all(not (set(items[i][1]) & set(items[j][0])) for j in rem if j != i)),
                    rem[0])
        ordered.append(items[pick]); rem.remove(pick)
    return bgc, objc, ordered


def _predict(I):
    P = _plan(I)
    if P is None:
        return None
    bgc, objc, items = P
    G = np.array(I, dtype=int)
    for src, _ in items:
        for r, c in src:
            G[r, c] = bgc
    for _, dst in items:
        for r, c in dst:
            G[r, c] = 1
    return G


def generate(diff_lb, diff_ub, max_h, max_w, barc, bgc, objc, rot=None, **kwargs) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    for _ in range(300):
        hlim, wlim = (max_w, max_h) if rot % 2 else (max_h, max_w)
        hlim = min(30, hlim); wlim = min(30, wlim)
        if hlim < 9 or wlim < 5:
            hlim, wlim = max(hlim, 9), max(wlim, 5)
        h = unifint(diff_lb, diff_ub, (9, hlim))
        w = unifint(diff_lb, diff_ub, (5, wlim))
        barh = randint(3, h // 3)
        maxobjh = h - barh - 1
        nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
        gi = vconcat(canvas(barc, (barh, w)), canvas(bgc, (h - barh, w)))
        go = tuple(e for e in gi)
        tr = 0; succ = 0; maxtr = 10 * nobjs
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
                obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
            obj = normalize(obj)
            oh, ow = shape(obj)
            markerh = randint(1, min(oh, barh - 1))
            markpartn = normalize(sfilter(obj, lambda ij: ij[0] < markerh))
            if any(normalize(sfilter(markpartn, lambda ij: ij[0] < k)) in forbmarkers
                   for k in range(1, markerh + 1)):
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
        if succ == 0:
            continue
        gi_a = np.rot90(np.array(gi, dtype=int), rot)
        go_a = np.rot90(np.array(go, dtype=int), rot)
        if gi_a.shape[0] > max_h or gi_a.shape[1] > max_w:
            continue
        pred = _predict(gi_a)
        if pred is None or not np.array_equal(pred, go_a):
            continue
        return {"input": gi_a.tolist(), "output": go_a.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O):
    """Rule: every loose shape slides into the notch cut for its head in the bar
    and is drawn in colour 1.
    Unit of work: the recolouring is done once for the whole colour (every cell of
    the loose-shape colour turns to 1 in a single stroke); then the shapes slide one
    by one with Move chains; the vacated footprints are cleaned up together at the
    end in one stroke."""
    Ia = np.asarray(I, dtype=int)
    h, w = Ia.shape
    P = _plan(Ia)
    if P is None:
        return [34], [[0, 0, h - 1, w - 1]]
    bgc, objc, items = P

    # 1. one stroke for the colour: all cells of the loose shapes' colour become 1
    base = Ia.copy()
    cells = [(r, c) for r in range(h) for c in range(w) if int(Ia[r, c]) == objc]
    for r, c in cells:
        base[r, c] = 1
    head_ops, head_sels = [1], [sel_of(cells)]

    def slide(work, visited, src, dst):
        """Unit-step route (vertical/horizontal legs interleaved as needed) so each
        step visibly changes the grid and never revisits an earlier picture."""
        Dr = dst[0][0] - src[0][0]
        Dc = dst[0][1] - src[0][1]
        snap = work.copy()
        for r, c in src:
            snap[r, c] = 0

        def frame(cur):
            f = snap.copy()
            for r, c in cur:
                f[r, c] = 1
            return f

        budget = [3000]

        def dfs(cur, up, side, path, seen):
            if up == 0 and side == 0:
                return path
            budget[0] -= 1
            if budget[0] < 0:
                return None
            legs = []
            if up:
                legs.append((1 if up > 0 else -1, 0))
            if side:
                legs.append((0, 1 if side > 0 else -1))
            for sr, sc in legs:
                nxt = [(r + sr, c + sc) for r, c in cur]
                f = frame(nxt)
                key = f.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                res = dfs(nxt, up - sr, side - sc, path + [(sr, sc, nxt, f)], seen)
                if res is not None:
                    return res
                seen.discard(key)
            return None

        return dfs(list(src), Dr, Dc, [], set(visited))

    def emit(order, strict):
        work = base.copy()
        visited = {Ia.tobytes(), work.tobytes()}
        ops, sels = list(head_ops), list(head_sels)
        landed, vacated = set(), set()
        for i in order:
            src, dst = items[i]
            route = slide(work, visited, src, dst)
            if route is None:
                return None
            held = False
            cur = list(src)
            for sr, sc, nxt, f in route:
                ops.append((21 if sr > 0 else 20) if sr else (22 if sc > 0 else 23))
                sels.append(sel_of([]) if held else sel_of(cur))   # grab once, then continue
                held = True
                cur, work = nxt, f
                visited.add(work.tobytes())
            # a slotted shape whose cells were carried off with this one gets them back
            torn = sorted(p for p in (landed & set(src)) if int(work[p]) == 0)
            if torn:
                if strict:
                    return None
                ops.append(1); sels.append(sel_of(torn))
                for p in torn:
                    work[p] = 1
                visited.add(work.tobytes())
            landed |= set(dst)
            vacated |= set(src)
        # 3. clean every footprint the shapes left behind (it reads 0 after the grabs)
        if bgc != 0:
            hole = sorted(p for p in vacated if int(work[p]) == 0)
            if hole:
                ops.append(bgc); sels.append(sel_of(hole))
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # 2. choose the order the shapes slide in: a shape goes before any shape whose
    # footprint it would land on, and before shapes it would have to pass through
    n = len(items)
    tries = [0]

    def search(prefix, rest, strict):
        if not rest:
            return emit(prefix, strict)
        tries[0] += 1
        if tries[0] > 200:
            return None
        pref = sorted(rest, key=lambda i: sum(
            len(set(items[i][1]) & set(items[j][0])) for j in rest if j != i))
        for i in pref:
            res = search(prefix + [i], [j for j in rest if j != i], strict)
            if res is not None:
                return res
        return None

    for strict in (True, False):
        tries[0] = 0
        res = search([], list(range(n)), strict)
        if res is not None:
            return res
    return emit(list(range(n)), False) or ([34], [[0, 0, h - 1, w - 1]])


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
