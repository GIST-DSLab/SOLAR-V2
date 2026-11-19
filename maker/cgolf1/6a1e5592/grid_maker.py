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
    res = None
    for _ in range(6):
        res = _generate_once(diff_lb, diff_ub, max_h, max_w, barc, bgc, objc)
        I = np.array(res['input']); O = np.array(res['output'])
        if not (I != O).any():
            continue
        try:
            ops, _ = derive_operations(I, O)
        except Exception:
            continue
        if len(ops) > 1:          # pieces can be walked into their notches
            return res
    return res


def _generate_once(diff_lb, diff_ub, max_h, max_w, barc, bgc, objc) -> dict:
    rotf = choice((identity, rot90, rot180, rot270))
    if rotf in (rot90, rot270):
        hub, wub = max_w, max_h
    else:
        hub, wub = max_h, max_w
    hub = max(9, min(30, hub))
    wub = max(5, min(30, wub))
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
                jm = apply(last, ofcolor(go, 1))
                placopts = sorted(difference(placopts, jm | apply(decrement, jm) | apply(increment, jm)))
        if len(placopts) == 0:
            break
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Reading (the maker and the golf entry agree on it): a solid bar lies along one edge,
    and the background-coloured notches cut into it are the imprints of the loose pieces.
    Each piece is rigid. It turns colour 1 and is translated so that its leading rows
    exactly plug one notch (an exact cover of the notch cells). The rule is the same in
    both; where the golf reading differs is the motion. A piece is placed by lining it up
    with its notch, so it first slides ALONG the bar until it sits under its notch and
    then is pushed straight INTO the bar. Ops: Color1 on the piece, unit Moves along the
    bar, unit Moves toward the bar (one grab, then empty selections), and at the end one
    Color(bgc) on every grabbed footprint cell that ARCLE left at 0. The piece order is
    searched (simulating ARCLE) so that no piece comes to rest on a piece that has not
    moved yet and no op leaves the grid in a state it has already been in. Everything is
    measured from I.
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

    # ---- 1. locate the bar; canonical frame has the bar on top ----
    idx = np.arange(h * w).reshape(h, w)
    found = None
    for k in range(4):
        A = np.rot90(Ia, k)
        Mi = np.rot90(idx, k)
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
        found = (k, A, Mi, hh, ww, barh, c0, band_other)
        break
    if found is None:
        return finish()
    k, A, Mi, hh, ww, barh, barc, band_other = found
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

    # ---- 2. loose pieces (4-connected) ----
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
    if not objs:
        return finish()

    # ---- 3. assign every piece to a notch (exact cover of notch cells) ----
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
        v = int(Mi[p[0], p[1]])
        return (v // w, v % w)

    items = []
    for i, obj in enumerate(objs):
        if sol[i] is None:
            continue
        dr, dc = sol[i]
        src_can = sorted(obj)
        dst_can = [(r + dr, c + dc) for r, c in src_can]
        items.append({"src_can": set(src_can), "dst_can": set(dst_can),
                      "src": [to_orig(p) for p in src_can],
                      "dst": [to_orig(p) for p in dst_can]})
    if not items:
        return finish()

    # In the original frame the bar is on top/bottom for k in {0,2}: lateral = columns.
    LAT = 1 if k in (0, 2) else 0
    STEP = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}
    n = len(items)
    D = [(it["dst"][0][0] - it["src"][0][0], it["dst"][0][1] - it["src"][0][1]) for it in items]

    def leg_op(ax, v):
        return (21 if v > 0 else 20) if ax == 0 else (22 if v > 0 else 23)

    # Plan: each piece is recoloured, then slid along the bar (lateral leg), then pushed
    # into the bar (toward leg).  Search the order of pieces/legs so that no piece comes
    # to rest on a piece that has not moved yet, and every op visibly changes the grid.
    # Simulation mirrors ARCLE: one grab snapshots the grid with the piece zeroed.
    def sim_leg(G, snap, cur, op, steps, grab, seen):
        out = []
        d = STEP[op]
        for q in range(steps):
            if q == 0 and grab:
                snap = G.copy()
                for p in cur:
                    snap[p] = 0
            cur = [(r + d[0], c + d[1]) for r, c in cur]
            G = snap.copy()
            for p in cur:
                G[p] = 1
            key = G.tobytes()
            if key in seen:
                return None
            seen = seen | {key}
            out.append(op)
        return G, snap, cur, out, seen

    budget = [200000]

    def dfs(G, snap, pos, legs, active, seen, acc, colored, depth=0):
        # legs[i]: list of remaining (axis, delta); active: index of grabbed piece
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        if depth > 3 * n + 6:
            return None
        key = (tuple(tuple(sorted(p)) for p in pos), tuple(tuple(l) for l in legs), active)
        if key in visited:
            return None
        visited.add(key)
        if all(not l for l in legs):
            return acc, G
        cand = []
        for i in range(n):
            if not legs[i]:
                continue
            options = [legs[i]] if len(legs[i]) < 2 else [legs[i], legs[i][::-1]]
            for oi, lg in enumerate(options):
                ax, v = lg[0]
                sg = 1 if v > 0 else -1
                act = 0 if i == active else 1
                # a whole leg first; part of a leg only to break a mutual blockage
                cand.append((0, act, oi, i, ax, v, lg[1:]))
                for part in (range(abs(v) - 1, 0, -1) if allow_part else ()):
                    cand.append((1, act, oi, i, ax, sg * part, [(ax, v - sg * part)] + lg[1:]))
        if allow_detour:
            # step aside (any straight direction) to a resting spot clear of every other
            # piece and every other slot, then head for the slot afresh
            for i in range(n):
                if not legs[i]:
                    continue
                tgt = items[i]["dst"][0]
                for ax in (0, 1):
                    for sg in (1, -1):
                        for part in range(1, max(h, w)):
                            end0 = [(r + (sg * part if ax == 0 else 0),
                                     c + (sg * part if ax == 1 else 0)) for r, c in pos[i]]
                            if any(not (0 <= r < h and 0 <= c < w) for r, c in end0):
                                break
                            rr = (tgt[0] - end0[0][0], tgt[1] - end0[0][1])
                            rest = [(a, rr[a]) for a in (LAT, 1 - LAT) if rr[a]]
                            cand.append((2, 0 if i == active else 1, part, i, ax, sg * part, rest))
        cand.sort(key=lambda t: t[:3])
        for _, _, _, i, ax, v, rest in cand:
            others = {p for j in range(n) if j != i for p in pos[j]}
            d = STEP[leg_op(ax, v)]
            end = [(r + d[0] * abs(v), c + d[1] * abs(v)) for r, c in pos[i]]
            if set(end) & others:
                continue
            if rest and (rest[0][0] == ax or allow_detour) and (set(end) & dst_of_others[i]):
                continue          # a part-way stop must park clear of every other slot
            G1, snap1, seen1, pre = G, snap, seen, []
            fresh = (i != active)
            if fresh and i in colored:
                # re-grab only where the piece rests on open background (outside the bar)
                if any(G[p] != 1 for p in pos[i]) or any(p not in body for p in pos[i]):
                    continue
            elif fresh:
                if any(G[p] != objc for p in pos[i]):
                    continue
                G1 = G.copy()
                for p in pos[i]:
                    G1[p] = 1
                if G1.tobytes() in seen1:
                    continue
                seen1 = seen1 | {G1.tobytes()}
                pre = [("color", list(pos[i]))]
            res = sim_leg(G1, snap1, pos[i], leg_op(ax, v), abs(v), fresh, seen1)
            if res is None:
                continue
            G2, snap2, cur2, mops, seen2 = res
            step_list = pre + [(m, None) for m in mops]
            if fresh:
                step_list[len(pre)] = (mops[0], list(pos[i]))   # first Move grabs the piece
            npos = list(pos); npos[i] = cur2
            nlegs = list(legs); nlegs[i] = rest
            out = dfs(G2, snap2, npos, nlegs, i, seen2, acc + step_list, colored | {i}, depth + 1)
            if out is not None:
                return out
        return None

    dst_of_others = [{p for j, it in enumerate(items) if j != i for p in it["dst"]}
                     for i in range(n)]
    body = {to_orig((r, c)) for r in range(barh, hh) for c in range(ww)}
    legs0 = []
    for i in range(n):
        lg = []
        for ax in (LAT, 1 - LAT):
            if D[i][ax]:
                lg.append((ax, D[i][ax]))
        legs0.append(lg)
    # pieces that need no move still turn colour 1
    pos0 = [list(it["src"]) for it in items]
    G0 = Ia.copy()
    seen0 = {G0.tobytes()}
    pre_ops = []
    for i in range(n):
        if not legs0[i]:
            for p in pos0[i]:
                G0[p] = 1
            seen0.add(G0.tobytes())
            pre_ops.append(("color", pos0[i]))
    res = None
    for allow_part, allow_detour in ((False, False), (True, False), (True, True)):
        visited = set()
        budget = [200000]
        res = dfs(G0, None, pos0, legs0, None, seen0, [], frozenset())
        if res is not None:
            break
    if res is None:
        return finish()
    plan, G = res
    for op, cells in pre_ops + plan:
        if op == "color":
            ops.append(1); sels.append(sel_of(cells))       # this piece becomes colour 1
        else:
            ops.append(op); sels.append(sel_of(cells) if cells is not None else sel_of([]))
    # every grabbed footprint cell that a move left at 0 (no piece covers it now) -> bgc
    grabbed = set()
    for op, cells in plan:
        if op != "color" and cells is not None:
            grabbed |= set(cells)
    hole = sorted(p for p in grabbed if G[p] == 0)
    if bgc != 0 and hole:
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
