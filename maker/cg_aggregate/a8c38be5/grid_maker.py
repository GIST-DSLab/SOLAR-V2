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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or sqc is None:
        bgc, sqc = sample(cols, 2)
    # the final mirror/rotation may transpose the grid, so bound both sides
    max_h = max_w = min(max_h, max_w, 30)
    goh = unifint(diff_lb, diff_ub, (9, min(20, max_h - 4)))
    gow = unifint(diff_lb, diff_ub, (9, min(20, max_w - 4)))
    h = unifint(diff_lb, diff_ub, (goh + 4, max_h))
    w = unifint(diff_lb, diff_ub, (gow + 4, max_w))
    remcols = remove(bgc, remove(sqc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    go = canvas(sqc, (goh, gow))
    go = fill(go, bgc, box(asindices(go)))
    loci1 = randint(2, goh - 7)
    loci2 = randint(loci1 + 4, goh - 3)
    locj1 = randint(2, gow - 7)
    locj2 = randint(locj1 + 4, gow - 3)
    f1 = hfrontier((loci1, 0))
    f2 = hfrontier((loci2, 0))
    f3 = vfrontier((0, locj1))
    f4 = vfrontier((0, locj2))
    fs = f1 | f2 | f3 | f4
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
            shp = shape(obj)
            if shp in mpr:
                mpr[shp].append(obj)
            else:
                mpr[shp] = [obj]
        if max([len(x) for x in mpr.values()]) == 1:
            break
        ress = [(kk, v) for kk, v in mpr.items() if len(v) > 1]
        res, abc = choice(ress)
        a, b = sample(abc, 2)
        ulca = ulcorner(a)
        ulcb = ulcorner(b)
        ap = shift(normalize(a), ulcb)
        bp = shift(normalize(b), ulca)
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
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    # Reading (the golf entry's): the output is a 3x3 jigsaw of solid rectangular
    # PIECES (4 corners, 4 edges, 1 plain middle) cut apart in I. Each whole piece
    # (its coloured edge strip + its interior) is the unit; it is slid with Move
    # into its slot of the assembled frame, then the canvas is cropped to the frame.
    # Nothing here reads O; bgc/interior colours come from the demonstrations.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # ---- background / interior colours: read from the demonstrations -------
    # The assembled frame (an example's output) never contains the background,
    # and its interior colour is the majority colour of that frame.
    bgc = sqc = None
    if examples:
        bg_votes, sq_votes = Counter(), Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            gone = set(ei.flatten().tolist()) - set(eo.flatten().tolist())
            if len(gone) == 1:
                bg_votes[gone.pop()] += 1
            sq_votes[Counter(eo.flatten().tolist()).most_common(1)[0][0]] += 1
        if bg_votes:
            bgc = bg_votes.most_common(1)[0][0]
        if sq_votes:
            sqc = sq_votes.most_common(1)[0][0]
    cnt = Counter(I.flatten().tolist())
    if bgc is None or bgc not in cnt:
        bgc = cnt.most_common(1)[0][0]
    if sqc is None or sqc == bgc or sqc not in cnt:
        sqc = max([c for c in cnt if c != bgc], key=lambda c: cnt[c])
    marker_colors = [c for c in cnt if c not in (bgc, sqc)]

    def comps(color):
        seen = np.zeros((hi, wi), bool)
        out = []
        for r in range(hi):
            for c in range(wi):
                if I[r, c] == color and not seen[r, c]:
                    st = [(r, c)]
                    seen[r, c] = True
                    cells = []
                    while st:
                        y, x = st.pop()
                        cells.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and I[ny, nx] == color and not seen[ny, nx]:
                                seen[ny, nx] = True
                                st.append((ny, nx))
                    out.append(cells)
        return out

    def classify(rel, h, w):
        top = {(0, c) for c in range(w)}
        bot = {(h - 1, c) for c in range(w)}
        left = {(r, 0) for r in range(h)}
        right = {(r, w - 1) for r in range(h)}
        ul, ur = (0, 0) in rel, (0, w - 1) in rel
        ll, lr = (h - 1, 0) in rel, (h - 1, w - 1) in rel
        r0n = sum(1 for (r, c) in rel if r == 0)
        rHn = sum(1 for (r, c) in rel if r == h - 1)
        c0n = sum(1 for (r, c) in rel if c == 0)
        cWn = sum(1 for (r, c) in rel if c == w - 1)
        if rel == top | left:
            return 'TL'
        if rel == top | right:
            return 'TR'
        if rel == left | bot:
            return 'BL'
        if rel == bot | right:
            return 'BR'
        if top <= rel and not ll and not lr and rHn == 1:
            return 'topT'
        if bot <= rel and not ul and not ur and r0n == 1:
            return 'botT'
        if left <= rel and not ur and not lr and cWn == 1:
            return 'leftT'
        if right <= rel and not ul and not ll and c0n == 1:
            return 'rightT'
        return None

    # ---- the coloured edge strip of each frame piece -------------------------
    slot = {}
    for col in marker_colors:
        for cells in comps(col):
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            r0, c0 = min(rs), min(cs)
            h = max(rs) - r0 + 1
            w = max(cs) - c0 + 1
            rel = frozenset((r - r0, c - c0) for r, c in cells)
            name = classify(rel, h, w)
            if name and name not in slot:
                slot[name] = (col, r0, c0, h, w, rel)

    def H(n):
        return slot[n][3]

    def W(n):
        return slot[n][4]

    # three row bands / three column bands of the assembled frame
    hT, hM, hB = H('TL') + 1, H('leftT'), H('BL') + 1
    wL, wM, wR = W('TL') + 1, W('topT'), W('TR') + 1
    ho, wo = hT + hM + hB, wL + wM + wR

    # ---- each piece = a solid rectangle: its strip + interior colour ---------
    # (source top-left in I, size, and slot top-left inside the assembled frame)
    pieces = {}

    def add_piece(n, sr, sc, ph, pw, fr, fc):
        pieces[n] = (sr, sc, ph, pw, fr, fc)
    s = slot
    add_piece('TL', s['TL'][1], s['TL'][2], hT, wL, 0, 0)
    add_piece('TR', s['TR'][1], s['TR'][2] + W('TR') - wR, hT, wR, 0, wL + wM)
    add_piece('BL', s['BL'][1] + H('BL') - hB, s['BL'][2], hB, wL, hT + hM, 0)
    add_piece('BR', s['BR'][1] + H('BR') - hB, s['BR'][2] + W('BR') - wR, hB, wR, hT + hM, wL + wM)
    add_piece('topT', s['topT'][1], s['topT'][2], hT, wM, 0, wL)
    add_piece('botT', s['botT'][1] + H('botT') - hB, s['botT'][2], hB, wM, hT + hM, wL)
    add_piece('leftT', s['leftT'][1], s['leftT'][2], hM, wL, hT, 0)
    add_piece('rightT', s['rightT'][1], s['rightT'][2] + W('rightT') - wR, hM, wR, hT, wL + wM)

    def rect(r, c, h, w):
        return [(y, x) for y in range(r, r + h) for x in range(c, c + w)]

    def intact(n):
        sr, sc, ph, pw = pieces[n][:4]
        if sr < 0 or sc < 0 or sr + ph > hi or sc + pw > wi:
            return False
        col, mr, mc, _, _, rel = slot[n]
        mk = {(mr + a, mc + b) for a, b in rel}
        for (y, x) in rect(sr, sc, ph, pw):
            want = col if (y, x) in mk else sqc
            if I[y, x] != want:
                return False
        return True

    ok = all(intact(n) for n in pieces)

    # the plain middle piece: a solid interior-coloured hM x wM block lying
    # outside every edge piece
    if ok:
        taken = set()
        for n in pieces:
            sr, sc, ph, pw = pieces[n][:4]
            taken.update(rect(sr, sc, ph, pw))
        for r in range(hi - hM + 1):
            for c in range(wi - wM + 1):
                cells = rect(r, c, hM, wM)
                if all(I[y, x] == sqc for y, x in cells) and not any(p in taken for p in cells):
                    pieces['mid'] = (r, c, hM, wM, hT, wL)
                    break
            if 'mid' in pieces:
                break
    has_zero = (sqc == 0) or any(slot[n][0] == 0 for n in slot)

    ops, sels = [], []

    # ---- choose where on the canvas to assemble the frame --------------------
    VSTEP = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}

    def simulate(R0, C0, order):
        """Replay the slides in Python; None if some step would be invisible."""
        G = I.copy()
        out = []          # (op, sel)
        moved = set()
        placed = {}
        for n in order:
            sr, sc, ph, pw, fr, fc = pieces[n]
            if (sr, sc) == (R0 + fr, C0 + fc):
                placed[n] = {(y, x): int(I[y, x]) for y, x in rect(sr, sc, ph, pw)}
        for n in order:
            sr, sc, ph, pw, fr, fc = pieces[n]
            dr0, dc0 = R0 + fr, C0 + fc
            src = rect(sr, sc, ph, pw)
            srcset = set(src)
            # what the slide actually carries is the piece as it stands NOW; a
            # cell it shared in I with an already-moved neighbour is gone (0)
            want = {(y - sr, x - sc): int(I[y, x]) for y, x in src}
            content = {(y - sr, x - sc): int(G[y, x]) for y, x in src}
            if all(v == 0 for v in want.values()):
                continue                      # nothing a Move could carry
            if all(v == 0 for v in content.values()):
                return None                   # piece was wiped before its turn
            dr, dc = dr0 - sr, dc0 - sc
            if dr == 0 and dc == 0:
                moved.add(n)
                continue
            pre = []
            G1 = G.copy()
            if has_zero and bgc != 0:
                # clear the slot first so the piece's 0-cells land on 0
                clear = [p for p in rect(dr0, dc0, ph, pw) if p not in srcset and G1[p] != 0]
                if clear:
                    pre.append((0, sel_of(clear)))
                    for p in clear:
                        G1[p] = 0
            vs = [21 if dr > 0 else 20] * abs(dr)
            hs = [22 if dc > 0 else 23] * abs(dc)
            done = None
            for steps in (vs + hs, hs + vs):
                snap = G1.copy()
                for p in srcset:
                    snap[p] = 0
                prev = G1
                cy, cx = sr, sc
                good = True
                for op in steps:
                    cy += VSTEP[op][0]
                    cx += VSTEP[op][1]
                    g = snap.copy()
                    for (u, v), val in content.items():
                        if val != 0:
                            g[cy + u, cx + v] = val
                    if np.array_equal(g, prev):
                        good = False
                        break
                    prev = g
                if good:
                    done = (steps, prev)
                    break
            if done is None:
                if n == 'mid':
                    continue                  # middle block will be painted
                return None
            steps, G = done
            out.extend(pre)
            # one grab of the whole piece, then empty selections continue the slide
            for k, op in enumerate(steps):
                out.append((op, sel_of(src) if k == 0 else sel_of([])))
            moved.add(n)
            placed[n] = {(dr0 + u, dc0 + v): val for (u, v), val in want.items()}
            # patch cells of the pieces already in the frame that this slide
            # disturbed (cells shared with / covered by a neighbour in I)
            fix = {}
            for m in placed:
                for p, val in placed[m].items():
                    if G[p] != val:
                        fix.setdefault(val, []).append(p)
            for val, cells in sorted(fix.items()):
                out.append((val, sel_of(cells)))
                for p in cells:
                    G[p] = val
        if 'mid' not in moved:
            # the middle slot is plain interior colour: paint it
            mcells = [p for p in rect(R0 + hT, C0 + wL, hM, wM) if G[p] != sqc]
            if mcells:
                out.append((sqc, sel_of(mcells)))
        return out

    route = None
    if ok and ho <= hi and wo <= wi:
        names = list(pieces)
        cands = []
        for R0 in range(hi - ho + 1):
            for C0 in range(wi - wo + 1):
                src = {n: set(rect(*pieces[n][:4])) for n in names}
                dst = {n: set(rect(R0 + pieces[n][4], C0 + pieces[n][5],
                                   pieces[n][2], pieces[n][3])) for n in names}
                # a piece may only land once every other piece under its slot has left
                deps = {n: {m for m in names if m != n and dst[n] & src[m]} for n in names}
                order, done = [], set()
                viol = 0
                while len(order) < len(names):
                    ready = [n for n in names if n not in done and deps[n] <= done]
                    if ready:
                        for n in ready:
                            order.append(n); done.add(n)
                        continue
                    # cycle: land the piece that covers the fewest unmoved cells
                    n = min((n for n in names if n not in done),
                            key=lambda n: sum(len(dst[n] & src[m]) for m in deps[n] - done))
                    viol += sum(len(dst[n] & src[m]) for m in deps[n] - done)
                    order.append(n); done.add(n)
                cost = (viol, sum(abs(R0 + pieces[n][4] - pieces[n][0]) +
                                  abs(C0 + pieces[n][5] - pieces[n][1]) for n in names))
                cands.append((cost, R0, C0, order))
        cands.sort(key=lambda t: (t[0], t[1], t[2]))
        cands = cands[:200]
        for cost, R0, C0, order in cands:
            seq = simulate(R0, C0, order)
            if seq is not None:
                route = (R0, C0, seq)
                break

    if route is not None:
        R0, C0, seq = route
        for op, sl in seq:
            ops.append(op); sels.append(sl)
        if not (R0 == 0 and C0 == 0 and ho == hi and wo == wi):
            # cut the canvas down to the assembled frame (full rectangle intended)
            ops.append(33); sels.append([R0, C0, ho - 1, wo - 1])
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # ---- fallback: pieces damaged in I -> rebuild from their edge strips ------
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    ops.append(sqc); sels.append([0, 0, ho - 1, wo - 1])
    for n in ['TL', 'TR', 'BR', 'BL', 'topT', 'rightT', 'botT', 'leftT']:
        col, r0, c0, h, w, rel = slot[n]
        sr, sc, ph, pw, fr, fc = pieces[n]
        offr, offc = r0 - sr, c0 - sc
        cells = [(fr + offr + a, fc + offc + b) for a, b in rel]
        ops.append(int(col)); sels.append(sel_of(cells))
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
