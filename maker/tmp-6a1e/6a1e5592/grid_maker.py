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
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # --- find the bar side.  The bar's outer edge line is one solid colour; rotating so that
    # edge is on top, the bar is the run of rows touching that colour, and what lies past it
    # holds the background plus the loose objects (>= 2 colours).  A solid background edge
    # fails that test: past its "bar" there is only bar colour.
    edges = {0: I[0, :], 1: I[:, -1], 2: I[-1, :], 3: I[:, 0]}  # np.rot90(I, k) puts edge k on top
    best = None
    for k, v in edges.items():
        if len(set(v.tolist())) != 1:
            continue
        X = int(v[0])
        Nk = np.rot90(I, k)
        bh = 0
        while bh < Nk.shape[0] and np.any(Nk[bh] == X):
            bh += 1
        rest_cols = set(Nk[bh:].flatten().tolist())
        holes_c = set(Nk[:bh].flatten().tolist()) - {X}
        score = (len(rest_cols) >= 2 and X not in rest_cols, len(holes_c) == 1)
        if best is None or score > best[0]:
            best = (score, k, X, bh, holes_c)
    _, k, barc, barh, holes_c = best
    N = np.rot90(I, k)
    idx = np.rot90(np.arange(hi * wi).reshape(hi, wi), k)
    nh, nw = N.shape
    to_orig = lambda r, c: divmod(int(idx[r, c]), wi)
    if len(holes_c) == 1:
        bgc = int(next(iter(holes_c)))            # the holes in the bar show the background
    else:
        bgc = Counter(N[barh:].flatten().tolist()).most_common(1)[0][0]

    # objects: 4-connected components of non-bar, non-bgc cells
    mask = (N != barc) & (N != bgc)
    seen = np.zeros_like(mask)
    objs = []
    for r in range(nh):
        for c in range(nw):
            if mask[r, c] and not seen[r, c]:
                st = [(r, c)]; seen[r, c] = True; comp = []
                while st:
                    a, b = st.pop(); comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < nh and 0 <= y < nw and mask[x, y] and not seen[x, y]:
                            seen[x, y] = True; st.append((x, y))
                objs.append(comp)

    holes = {(r, c) for r in range(barh) for c in range(nw) if N[r, c] == bgc}

    def fits(obj):
        r0 = min(r for r, _ in obj)
        c0 = min(c for _, c in obj)
        nobj = [(r - r0, c - c0) for r, c in obj]
        ow = max(c for _, c in nobj) + 1
        out = []
        for top in range(0, barh):
            for left in range(0, nw - ow + 1):
                sh = {(r + top, c + left) for r, c in nobj}
                if any(r >= nh for r, _ in sh):
                    continue
                inbar = frozenset((r, c) for r, c in sh if r < barh)
                if not inbar or not inbar <= holes:
                    continue
                ok = True
                for r, c in inbar:
                    for da, db in ((-1, 0), (0, 1), (0, -1)):
                        x, y = r + da, c + db
                        if 0 <= x < barh and 0 <= y < nw and (x, y) not in sh and N[x, y] == bgc:
                            ok = False
                if ok:
                    out.append(((top - r0, left - c0), inbar))
        return out

    # every hole must be plugged by exactly one object: solve the assignment
    cand = [fits(o) for o in objs]
    order = sorted(range(len(objs)), key=lambda i: len(cand[i]))
    assign = {}

    def solve(t, used):
        if t == len(order):
            return used == holes
        i = order[t]
        for d, inbar in cand[i]:
            if inbar & used:
                continue
            assign[i] = d
            if solve(t + 1, used | inbar):
                return True
        assign.pop(i, None)
        return False

    if not solve(0, frozenset()):
        assign = {i: cand[i][0][0] for i in range(len(objs)) if cand[i]}

    plans = []
    for i, obj in enumerate(objs):
        if i not in assign:
            continue
        d = assign[i]
        src = [to_orig(r, c) for r, c in obj]
        dst = [to_orig(r + d[0], c + d[1]) for r, c in obj]
        dr = dst[0][0] - src[0][0]
        dc = dst[0][1] - src[0][1]
        plans.append((src, dst, dr, dc))

    D = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}

    def slide(G, cells, steps):
        """Simulate an ARCLE grab-and-slide; return final grid and whether every step changed it."""
        vals = {p: G[p] for p in cells}
        bg = G.copy()
        for p in cells:
            bg[p] = 0
        cur = G
        off = [0, 0]
        ok = True
        for op in steps:
            off[0] += D[op][0]; off[1] += D[op][1]
            nxt = bg.copy()
            for (r, c), v in vals.items():
                if v != 0:
                    nxt[r + off[0], c + off[1]] = v
            if np.array_equal(nxt, cur):
                ok = False
            cur = nxt
        return cur, ok

    def route(cur, dst, vfirst):
        dr = dst[0][0] - cur[0][0]
        dc = dst[0][1] - cur[0][1]
        vs = [(20 if dr < 0 else 21)] * abs(dr)
        hs = [(23 if dc < 0 else 22)] * abs(dc)
        return vs + hs if vfirst else hs + vs

    def shifted(cells, steps):
        r = sum(D[o][0] for o in steps)
        c = sum(D[o][1] for o in steps)
        return [(x + r, y + c) for x, y in cells]

    def schedule(plan_list):
        clean = [True]
        G = I.copy()
        ops, sels = [], []
        # state per object: [current cells, destination cells, recoloured yet?]
        pending = [[list(src), list(dst), False] for src, dst, _, _ in plan_list]
        placed = set()

        def do_slide(st, steps, final):
            nonlocal G
            cur, dst, pre = st
            if not pre and any(G[x] == 0 for x in cur):
                # colour-0 object is invisible to Move: recolour it to 1 in place first
                ops.append(1); sels.append(sel_of(cur))
                for x in cur:
                    G[x] = 1
                st[2] = pre = True
            for i, op in enumerate(steps):
                ops.append(op)
                sels.append(sel_of(cur) if i == 0 else sel_of([]))   # grab once, then continue
            G, _ = slide(G, cur, steps)
            new = shifted(cur, steps)
            hole = sorted(set(cur) - set(new))
            if bgc != 0 and hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))
                for x in hole:
                    G[x] = bgc
            st[0] = new
            if final:
                placed.update(new)
                if not pre:
                    ops.append(1); sels.append(sel_of(new))          # slotted object becomes colour 1
                    for x in new:
                        G[x] = 1

        def occupied(excl):
            o = set()
            for q in pending:
                if q is not excl:
                    o |= set(q[0])
            return o

        def visible(st, steps):
            G2 = G.copy()
            if not st[2] and any(G2[x] == 0 for x in st[0]):
                for x in st[0]:
                    G2[x] = 1
            return slide(G2, st[0], steps)[1]

        while pending:
            done = False
            for st in pending:
                occ = occupied(st)
                if set(st[1]) & occ:
                    continue
                for vf in (True, False):
                    steps = route(st[0], st[1], vf)
                    if visible(st, steps):
                        pending.remove(st)
                        do_slide(st, steps, True)
                        done = True
                        break
                if done:
                    break
            if done:
                continue
            # blocked (objects sit in each other's slots): park one object aside first
            offs = sorted(((x, y) for x in range(-hi + 1, hi) for y in range(-wi + 1, wi) if (x, y) != (0, 0)),
                          key=lambda t: (abs(t[0]) + abs(t[1]), t))
            for st in pending:
                occ = occupied(st)
                others_dst = set()
                for q in pending:
                    if q is not st:
                        others_dst |= set(q[1])
                if not (set(st[0]) & others_dst):
                    continue
                own = set(st[0])
                for x0, y0 in offs:
                    mid = [(x + x0, y + y0) for x, y in st[0]]
                    if any(not (0 <= x < hi and 0 <= y < wi) for x, y in mid):
                        continue
                    ms = set(mid)
                    if ms & occ or ms & others_dst or ms & placed:
                        continue
                    if any(G[p] != bgc for p in ms - own):
                        continue
                    for vf in (True, False):
                        steps = route(st[0], mid, vf)
                        if visible(st, steps):
                            do_slide(st, steps, False)
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            if done:
                continue
            clean[0] = False
            st = pending.pop(0)
            do_slide(st, route(st[0], st[1], True), True)
        return ops, sels, clean[0]

    best = None
    rng = random.Random(0)
    for attempt in range(30):
        pl = list(plans)
        if attempt:
            rng.shuffle(pl)
        res = schedule(pl)
        if best is None:
            best = res
        if res[2]:
            best = res
            break
    ops, sels = list(best[0]), list(best[1])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
    return ops, sels



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
