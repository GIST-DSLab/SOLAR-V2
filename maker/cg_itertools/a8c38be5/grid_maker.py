"""
ARC Task: a8c38be5 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, sqc)]
    ccols = random.sample(remcols, random.randint(1, 8))
    return {"bgc": bgc, "sqc": sqc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or sqc is None:
        bgc, sqc = sample(cols, 2)
    if not ccols:
        remcols = remove(bgc, remove(sqc, cols))
        ccols = sample(remcols, unifint(diff_lb, diff_ub, (1, 8)))
    ccols = tuple(ccols)
    goh = unifint(diff_lb, diff_ub, (9, max(9, min(20, max_h - 4))))
    gow = unifint(diff_lb, diff_ub, (9, max(9, min(20, max_w - 4))))
    h = unifint(diff_lb, diff_ub, (goh + 4, max(goh + 4, max_h)))
    w = unifint(diff_lb, diff_ub, (gow + 4, max(gow + 4, max_w)))
    go = canvas(sqc, (goh, gow))
    go = fill(go, bgc, box(asindices(go)))
    loci1 = randint(2, goh - 7)
    loci2 = randint(loci1 + 4, goh - 3)
    locj1 = randint(2, gow - 7)
    locj2 = randint(locj1 + 4, gow - 3)
    fs = hfrontier((loci1, 0)) | hfrontier((loci2, 0)) | vfrontier((0, locj1)) | vfrontier((0, locj2))
    go = fill(go, sqc, fs)
    go = fill(go, bgc, {((loci1 + loci2) // 2, 1)})
    go = fill(go, bgc, {((loci1 + loci2) // 2, gow - 2)})
    go = fill(go, bgc, {(1, (locj1 + locj2) // 2)})
    go = fill(go, bgc, {(goh - 2, (locj1 + locj2) // 2)})
    objs = objects(go, T, F, T)
    objs = merge(set(recolor(choice(ccols), obj) for obj in objs))
    go = paint(go, objs)
    gi = go
    hdelt = h - goh
    hdelt1 = randint(1, hdelt - 3)
    hdelt2 = randint(1, hdelt - hdelt1 - 2)
    hdelt3 = randint(1, hdelt - hdelt1 - hdelt2 - 1)
    hdelt4 = hdelt - hdelt1 - hdelt2 - hdelt3
    wdelt = w - gow
    wdelt1 = randint(1, wdelt - 3)
    wdelt2 = randint(1, wdelt - wdelt1 - 2)
    wdelt3 = randint(1, wdelt - wdelt1 - wdelt2 - 1)
    wdelt4 = wdelt - wdelt1 - wdelt2 - wdelt3
    gi = gi[:loci2] + repeat(repeat(bgc, gow), hdelt2) + gi[loci2:]
    gi = gi[:loci1 + 1] + repeat(repeat(bgc, gow), hdelt3) + gi[loci1 + 1:]
    gi = repeat(repeat(bgc, gow), hdelt1) + gi + repeat(repeat(bgc, gow), hdelt4)
    gi = dmirror(gi)
    gi = gi[:locj2] + repeat(repeat(bgc, h), wdelt2) + gi[locj2:]
    gi = gi[:locj1 + 1] + repeat(repeat(bgc, h), wdelt3) + gi[locj1 + 1:]
    gi = repeat(repeat(bgc, h), wdelt1) + gi + repeat(repeat(bgc, h), wdelt4)
    gi = dmirror(gi)
    nswitcheroos = unifint(diff_lb, diff_ub, (0, 10))
    if choice((True, False)):
        gi = gi[loci1 + hdelt1 + 1:] + gi[:loci1 + hdelt1 + 1]
    if choice((True, False)):
        gi = dmirror(gi)
        gi = gi[locj1 + wdelt1 + 1:] + gi[:locj1 + wdelt1 + 1]
        gi = dmirror(gi)
    for k in range(nswitcheroos):
        o = asobject(gi)
        tmpc = canvas(bgc, (h + 12, w + 12))
        tmpc = paint(tmpc, shift(o, (6, 6)))
        objs = objects(tmpc, F, T, T)
        objs = apply(rbind(shift, (-6, -6)), objs)
        mpr = dict()
        for obj in objs:
            mpr.setdefault(shape(obj), []).append(obj)
        if max([len(x) for x in mpr.values()]) == 1:
            break
        ress = [(kk, v) for kk, v in mpr.items() if len(v) > 1]
        res, abc = choice(ress)
        a, b = sample(abc, 2)
        ap = shift(normalize(a), ulcorner(b))
        bp = shift(normalize(b), ulcorner(a))
        gi = paint(gi, ap | bp)
    nshifts = unifint(diff_lb, diff_ub, (0, 30))
    for k in range(nshifts):
        o = asobject(gi)
        tmpc = canvas(bgc, (h + 12, w + 12))
        tmpc = paint(tmpc, shift(o, (6, 6)))
        objs = objects(tmpc, F, F, T)
        objs = apply(rbind(shift, (-6, -6)), objs)
        objs = sfilter(objs, compose(flip, rbind(bordering, gi)))
        if len(objs) == 0:
            break
        obj = choice(totuple(objs))
        direc1 = (randint(-1, 1), randint(-1, 1))
        direc2 = position({(h // 2, w // 2)}, {center(obj)})
        direc = choice((direc1, direc2))
        gi = fill(gi, bgc, obj)
        gi = paint(gi, shift(obj, direc))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    for fn in sample(mfs, choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    # keep only instances whose eight framed panels survived the scrambling as
    # intact, non-overlapping rectangles, so the panel reading reproduces the output
    GI = np.asarray(gi, dtype=int)
    GO = np.asarray(go, dtype=int)
    cnt = Counter(GI.flatten().tolist())
    if cnt.most_common(1)[0][0] != bgc or cnt.most_common(2)[1][1] == cnt[bgc]:
        raise ValueError("background not majority")
    plan = derive_operations(GI, None, _plan_only=True)
    if plan is None:
        raise ValueError("panels not recoverable")
    pbg, fillc, row_h, col_w, P = plan
    if fillc != sqc or (sum(row_h), sum(col_w)) != GO.shape:
        raise ValueError("bad layout")
    rects = list(P.values())
    for a in range(len(rects)):
        for b in range(a + 1, len(rects)):
            ar, ac, ah, aw = rects[a]; br, bc, bh, bw = rects[b]
            if ar < br + bh and br < ar + ah and ac < bc + bw and bc < ac + aw:
                raise ValueError("panels overlap")
    A = np.full(GO.shape, fillc)
    ro = [0, row_h[0], row_h[0] + row_h[1]]
    co = [0, col_w[0], col_w[0] + col_w[1]]
    for (i, j), (r0, c0, ph, pw) in P.items():
        A[ro[i]:ro[i] + ph, co[j]:co[j] + pw] = GI[r0:r0 + ph, c0:c0 + pw]
    if not (A == GO).all():
        raise ValueError("panels corrupted")
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None, _plan_only=False):
    # Reading (golf view): the input is nine rectangular panels of a framed
    # picture scattered on the background -- four corner panels, four edge
    # panels and a plain centre. The output is those panels butted together in
    # a 3x3 layout. The unit is the WHOLE panel rectangle (frame line plus its
    # strip of interior), not just the coloured frame stroke: each panel is
    # slid rigidly into its slot, the plain centre is laid in the interior
    # colour, and the canvas is cut down to the assembled picture.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def find_plan(G):
        cnt = Counter(G.flatten().tolist())
        bgc = cnt.most_common(1)[0][0]
        rest = [c for c in cnt if c != bgc]
        if not rest:
            return None
        fillc = max(rest, key=lambda c: cnt[c])
        seen = np.zeros((hi, wi), bool)
        found = {}
        for r in range(hi):
            for c in range(wi):
                col = G[r, c]
                if col in (bgc, fillc) or seen[r, c]:
                    continue
                st = [(r, c)]; seen[r, c] = True; cells = []
                while st:
                    y, x = st.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and G[ny, nx] == col:
                            seen[ny, nx] = True; st.append((ny, nx))
                rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
                r0, c0 = min(rs), min(cs)
                h, w = max(rs) - r0 + 1, max(cs) - c0 + 1
                rel = {(y - r0, x - c0) for y, x in cells}
                top = {(0, k) for k in range(w)}; bot = {(h - 1, k) for k in range(w)}
                lef = {(k, 0) for k in range(h)}; rig = {(k, w - 1) for k in range(h)}
                if h >= 2 and w >= 2 and rel == top | lef: name = (0, 0)
                elif h >= 2 and w >= 2 and rel == top | rig: name = (0, 2)
                elif h >= 2 and w >= 2 and rel == bot | lef: name = (2, 0)
                elif h >= 2 and w >= 2 and rel == bot | rig: name = (2, 2)
                elif h == 2 and w >= 3 and top <= rel and len(rel) == w + 1: name = (0, 1)
                elif h == 2 and w >= 3 and bot <= rel and len(rel) == w + 1: name = (2, 1)
                elif w == 2 and h >= 3 and lef <= rel and len(rel) == h + 1: name = (1, 0)
                elif w == 2 and h >= 3 and rig <= rel and len(rel) == h + 1: name = (1, 2)
                else:
                    continue
                if name in found:
                    return None
                found[name] = (r0, c0, h, w)
        if len(found) != 8:
            return None
        P = {}
        # corner panel = frame corner stroke + one row and one column of interior
        for (i, j) in ((0, 0), (0, 2), (2, 0), (2, 2)):
            r0, c0, h, w = found[(i, j)]
            P[(i, j)] = (r0 - (1 if i == 2 else 0), c0 - (1 if j == 2 else 0), h + 1, w + 1)
        band_h = [P[(0, 0)][2], None, P[(2, 0)][2]]
        band_w = [P[(0, 0)][3], None, P[(0, 2)][3]]
        if P[(0, 2)][2] != band_h[0] or P[(2, 2)][2] != band_h[2]:
            return None
        if P[(2, 0)][3] != band_w[0] or P[(2, 2)][3] != band_w[2]:
            return None
        r0, c0, h, w = found[(0, 1)]; P[(0, 1)] = (r0, c0, band_h[0], w)
        r0, c0, h, w = found[(2, 1)]; P[(2, 1)] = (r0 + h - band_h[2], c0, band_h[2], w)
        r0, c0, h, w = found[(1, 0)]; P[(1, 0)] = (r0, c0, h, band_w[0])
        r0, c0, h, w = found[(1, 2)]; P[(1, 2)] = (r0, c0 + w - band_w[2], h, band_w[2])
        for (r0, c0, h, w) in P.values():
            if r0 < 0 or c0 < 0 or r0 + h > hi or c0 + w > wi:
                return None
        if P[(0, 1)][3] != P[(2, 1)][3] or P[(1, 0)][2] != P[(1, 2)][2]:
            return None
        row_h = [band_h[0], P[(1, 0)][2], band_h[2]]
        col_w = [band_w[0], P[(0, 1)][3], band_w[2]]
        return bgc, fillc, row_h, col_w, P

    plan = find_plan(I)
    if _plan_only:
        return plan
    bgc, fillc, row_h, col_w, P = plan
    ho, wo = sum(row_h), sum(col_w)
    ro = [0, row_h[0], row_h[0] + row_h[1]]
    co = [0, col_w[0], col_w[0] + col_w[1]]
    dest = {k: (ro[k[0]], co[k[1]]) for k in P}

    def overlap(a, b):
        ar, ac, ah, aw = a; br, bc, bh, bw = b
        return ar < br + bh and br < ar + ah and ac < bc + bw and bc < ac + aw

    ops, sels = [], []
    G = I.copy()                      # python mirror of the working grid
    slots = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    pending = list(slots)
    deferred = []                     # panels whose source gets covered before they can slide

    def fix_zero_cells(k):
        # colour-0 cells of a panel are transparent to Move/Paste; write them
        r0, c0, h, w = P[k]
        dr, dc = dest[k]
        zs = [(dr + y, dc + x) for y in range(h) for x in range(w)
              if I[r0 + y, c0 + x] == 0 and G[dr + y, dc + x] != 0]
        if zs:
            for (y, x) in zs:
                G[y, x] = 0
            ops.append(0); sels.append(sel_of(zs))

    while pending or deferred:
        if pending:
            def blockers(cand):
                drect = (dest[cand][0], dest[cand][1], P[cand][2], P[cand][3])
                return [o for o in pending if o != cand and overlap(drect, P[o])]
            k = min(pending, key=lambda cnd: len(blockers(cnd)))
            for o in blockers(k):
                # o's slot-taker lands on it first: o will be rebuilt from the input
                pending.remove(o); deferred.append(o)
            pending.remove(k)
            slide = True
        else:
            k = deferred.pop(0)
            slide = False
        r0, c0, h, w = P[k]
        dr, dc = dest[k]
        patch = I[r0:r0 + h, c0:c0 + w]
        plan_steps = None
        if slide:
            if (dr, dc) == (r0, c0):
                continue
            vert = [(20 if dr < r0 else 21)] * abs(dr - r0)
            horz = [(23 if dc < c0 else 22)] * abs(dc - c0)
            bg = G.copy(); bg[r0:r0 + h, c0:c0 + w] = 0     # what ARCLE snapshots on the grab
            for steps in (vert + horz, horz + vert):
                # every unit step must visibly change the grid
                prev = G; y, x = r0, c0; ok = True
                for op in steps:
                    y += {20: -1, 21: 1}.get(op, 0); x += {23: -1, 22: 1}.get(op, 0)
                    cur = bg.copy()
                    reg = cur[y:y + h, x:x + w]
                    reg[patch != 0] = patch[patch != 0]
                    if (cur == prev).all():
                        ok = False
                        break
                    prev = cur
                if ok:
                    plan_steps = steps
                    break
        if plan_steps is not None:
            # slide the whole panel rectangle: grab once, then empty selections
            for t, op in enumerate(plan_steps):
                ops.append(op)
                sels.append([r0, c0, h - 1, w - 1] if t == 0 else sel_of([]))  # full panel rectangle
            G = bg
            reg = G[dr:dr + h, dc:dc + w]
            reg[patch != 0] = patch[patch != 0]
        else:
            # panel cannot be slid cleanly: reproduce it from the input
            before = G.copy()
            reg = G[dr:dr + h, dc:dc + w]
            reg[patch != 0] = patch[patch != 0]
            if not (G == before).all():
                ops.append(28); sels.append([r0, c0, h - 1, w - 1])  # whole panel rectangle in the input
                ops.append(30); sels.append([dr, dc, 0, 0])
        fix_zero_cells(k)

    # plain centre panel: lay the interior colour over its slot
    cr, cc, ch, cw = ro[1], co[1], row_h[1], col_w[1]
    if (G[cr:cr + ch, cc:cc + cw] != fillc).any():
        G[cr:cr + ch, cc:cc + cw] = fillc
        ops.append(int(fillc)); sels.append([cr, cc, ch - 1, cw - 1])  # full centre rectangle

    # cut the canvas down to the assembled picture
    if (hi, wi) != (ho, wo):
        ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
    return ops, sels


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
                        f"num_examples+1 ({num_examples + 1}) for task a8c38be5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a8c38be5"
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
                                f"for task a8c38be5"
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
                    f"Failed to build a complete episode for task a8c38be5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a8c38be5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
