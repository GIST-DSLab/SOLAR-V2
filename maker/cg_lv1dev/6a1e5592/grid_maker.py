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

# Reading of the task (shared by the maker and the code-golf entry, which agree):
# a solid bar runs along one edge, and its background-coloured notches are the
# top rows of the loose objects lying in the body. Each object is matched to the
# notch it exactly fills (an exact cover of the notch cells), turns colour 1, and
# slides into place. The golf entry also tries every object at every notch
# position and keeps the assignment that fills all notches, so the units, the
# order and the fixed quantities are the same. Nothing changes in the reading.
# The changes are in how the trajectory is laid out: the object order and each
# object's axis order are chosen so that no op leaves the visible grid unchanged.
# That happens, for example, when the objects are colour 0 or when an object
# glides over a slot already filled with 1s. Cyclic slot dependencies are broken
# by one detour to a free background spot. The emptied footprints are cleaned
# up in one stroke at the end.


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    barc, bgc, objc = random.sample(cols, 3)
    return {"barc": barc, "bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, barc=None, bgc=None, objc=None) -> dict:
    if barc is None or bgc is None or objc is None:
        barc, bgc, objc = sample([c for c in range(10) if c != 1], 3)
    h = unifint(diff_lb, diff_ub, (9, max(9, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
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
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    Ia = np.asarray(I, dtype=int)
    h, w = Ia.shape
    ops, sels = [], []
    submit_sel = [0, 0, h - 1, w - 1]

    def finish():
        ops.append(34)
        sels.append(submit_sel)
        return ops, sels

    # ---- 1. find the bar band; work in a canonical frame with the bar on top ----
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

    # ---- 2. the loose objects (4-connected, as the generator builds them) ----
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
    objs.sort(key=lambda o: min(o))

    # ---- 3. match every object to its notch (exact cover of the notch cells) ----
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

    sol = cover(build_candidates(True))
    if sol is None:
        sol = cover(build_candidates(False))
    if sol is None:
        return finish()

    def to_orig(rc):
        v = int(M[rc[0], rc[1]])
        return (v // w, v % w)

    items = []
    for i, obj in enumerate(objs):
        dr, dc = sol[i]
        src_can = sorted(obj)
        dst_can = [(r + dr, c + dc) for r, c in src_can]
        items.append({"src_can": set(src_can), "dst_can": set(dst_can),
                      "src": [to_orig(p) for p in src_can],
                      "dst": [to_orig(p) for p in dst_can]})

    # ---- 4. plan the slides ----
    # Python-side model of ARCLE's grab-then-glide Move, used only to pick an
    # order whose intermediate grids never repeat an earlier visible grid.
    def plan_moves(src, Dr, Dc, rows_first):
        legs = [(Dr, (21, 20), 0), (Dc, (22, 23), 1)]
        if not rows_first:
            legs = legs[::-1]
        seq = []
        for D, (pos_op, neg_op), axis in legs:
            if not D:
                continue
            op = pos_op if D > 0 else neg_op
            st = 1 if D > 0 else -1
            for _ in range(abs(D)):
                seq.append((op, axis, st))
        return seq

    def simulate(start, src, seq, hole):
        g = start.copy()
        for r, c in src:
            g[r, c] = 1
        states = [g.tobytes()]
        snap = g.copy()
        for r, c in src:
            snap[r, c] = 0
        cur = list(src)
        for op, axis, st in seq:
            cur = [(r + st, c) if axis == 0 else (r, c + st) for r, c in cur]
            g = snap.copy()
            for r, c in cur:
                if 0 <= r < h and 0 <= c < w:
                    g[r, c] = 1
            states.append(g.tobytes())
        return g, states

    # Choose an object order and, per object, an axis order so that every op
    # visibly changes the grid and no grid state is ever revisited.  An object
    # never lands on cells another object has yet to leave; if the slots form a
    # cycle, one object first slides to a free parking spot on the background.
    dsts = {i: tuple(it["dst"]) for i, it in enumerate(items)}
    budget = [6000]

    def offs(src, tgt):
        return tgt[0][0] - src[0][0], tgt[0][1] - src[0][1]

    def try_leg(grid, seen, i, src, tgt, rows_first):
        Dr, Dc = offs(src, tgt)
        hole = sorted(set(src) - set(tgt))
        seq = plan_moves(src, Dr, Dc, rows_first)
        g2, states = simulate(grid, src, seq, hole)
        if all(int(grid[r, c]) == 1 for r, c in src):
            states = states[1:]                  # already recoloured: no Color op here
        if len(set(states)) != len(states) or (set(states) & seen):
            return None
        return seq, g2, states

    failed = set()

    def search(grid, seen, pos, parked_once, allow_park):
        remaining = sorted(pos)
        if not remaining:
            return []
        key = (grid.tobytes(), tuple((j, tuple(pos[j])) for j in remaining), bool(parked_once))
        if key in failed:
            return None
        res = search_inner(grid, seen, pos, parked_once, allow_park, remaining)
        if res is None and budget[0] > 0:
            failed.add(key)
        return res

    def search_inner(grid, seen, pos, parked_once, allow_park, remaining):
        for i in remaining:
            src = pos[i]
            if any(set(dsts[i]) & set(pos[j]) for j in remaining if j != i):
                continue
            for rows_first in (True, False):
                if budget[0] <= 0:
                    return None
                budget[0] -= 1
                leg = try_leg(grid, seen, i, src, list(dsts[i]), rows_first)
                if leg is None:
                    continue
                seq, g2, states = leg
                pos2 = {j: p for j, p in pos.items() if j != i}
                rest = search(g2, seen | set(states), pos2, parked_once, allow_park)
                if rest is not None:
                    return [(src, seq)] + rest
        # cyclic slots: park one object on free background, then carry on
        if not allow_park or parked_once:        # at most one detour per plan
            return None
        for i in remaining:
            src = pos[i]
            Dr, Dc = offs(src, list(dsts[i]))
            others = set()
            for j in remaining:
                if j != i:
                    others |= set(pos[j]) | set(dsts[j])
            cands = []
            for pr in range(-h, h + 1):
                for pc in range(-w, w + 1):
                    if (pr, pc) == (0, 0) or (pr, pc) == (Dr, Dc):
                        continue
                    cells_p = [(r + pr, c + pc) for r, c in src]
                    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells_p):
                        continue
                    if any((r, c) not in set(src) and int(grid[r, c]) != bgc for r, c in cells_p):
                        continue
                    if set(cells_p) & others or set(cells_p) & set(dsts[i]):
                        continue
                    cost = abs(pr) + abs(pc) + abs(Dr - pr) + abs(Dc - pc)
                    cands.append((cost, pr, pc, cells_p))
            cands.sort(key=lambda t: t[:3])
            for cost, pr, pc, cells_p in cands[:40]:
                for rows_first in (True, False):
                    if budget[0] <= 0:
                        return None
                    budget[0] -= 1
                    leg = try_leg(grid, seen, i, src, cells_p, rows_first)
                    if leg is None:
                        continue
                    seq, g2, states = leg
                    pos2 = dict(pos)
                    pos2[i] = cells_p
                    rest = search(g2, seen | set(states), pos2, parked_once | {i}, allow_park)
                    if rest is not None:
                        return [(src, seq)] + rest
        return None

    start_pos = {i: list(it["src"]) for i, it in enumerate(items)}
    order = search(Ia.copy(), {Ia.tobytes()}, start_pos, frozenset(), False)
    if order is None:
        budget[0] = 20000
        failed.clear()
        order = search(Ia.copy(), {Ia.tobytes()}, start_pos, frozenset(), True)
    if order is None:                            # fallback: plain order, rows first
        order = []
        for i, it in enumerate(items):
            Dr, Dc = offs(it["src"], it["dst"])
            order.append((list(it["src"]), plan_moves(it["src"], Dr, Dc, True)))

    # ---- 5. emit ops: recolour, grab once, glide with empty selections ----
    orig_srcs = {frozenset(it["src"]) for it in items}
    grid = Ia.copy()
    vacated = set()
    for src, seq in order:
        cur = list(src)
        if frozenset(src) in orig_srcs:          # first time this object is touched
            ops.append(1)                        # the loose object takes the answer colour
            sels.append(sel_of(cur))
        first = True
        for op, axis, st in seq:                 # grab once, then glide with empty selections
            ops.append(op)
            sels.append(sel_of(cur) if first else sel_of([]))
            first = False
            cur = [(r + st, c) if axis == 0 else (r, c + st) for r, c in cur]
        grid, _ = simulate(grid, src, seq, None)
        vacated |= set(src)
    # ARCLE leaves every vacated footprint empty (0); give the empty cells
    # the body background in one cleanup stroke.
    empty = sorted(p for p in vacated if int(grid[p[0], p[1]]) == 0)
    if bgc != 0 and empty:
        ops.append(int(bgc))
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
