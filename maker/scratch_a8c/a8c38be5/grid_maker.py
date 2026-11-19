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
import numpy as np
import random
from random import randint, choice, sample
from collections import Counter
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sqc": sqc}


def _assemble_plan(I):
    """Locate the eight framed panels of I from their frame-line pieces.

    Returns (bgc, fill, ho, wo, panels) where panels maps a 3x3 slot (i, j)
    (centre excluded) to (r0, c0, h, w) — the panel rectangle in I — and the
    slot origins are computed from the panel sizes.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    fill = max([c for c in cnt if c != bgc], key=lambda c: cnt[c])
    seen = np.zeros((hi, wi), bool)
    found = {}
    for r in range(hi):
        for c in range(wi):
            col = I[r, c]
            if col in (bgc, fill) or seen[r, c]:
                continue
            st = [(r, c)]; seen[r, c] = True; cells = []
            while st:
                y, x = st.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True; st.append((ny, nx))
            rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
            r0, c0 = min(rs), min(cs)
            h, w = max(rs) - r0 + 1, max(cs) - c0 + 1
            rel = {(y - r0, x - c0) for y, x in cells}
            top = {(0, k) for k in range(w)}; bot = {(h - 1, k) for k in range(w)}
            lef = {(k, 0) for k in range(h)}; rig = {(k, w - 1) for k in range(h)}
            def n_on(side):
                return len(rel & side)
            if rel == top | lef: name = (0, 0)
            elif rel == top | rig: name = (0, 2)
            elif rel == bot | lef: name = (2, 0)
            elif rel == bot | rig: name = (2, 2)
            elif top <= rel and len(rel) == w + 1 and h == 2: name = (0, 1)
            elif bot <= rel and len(rel) == w + 1 and h == 2: name = (2, 1)
            elif lef <= rel and len(rel) == h + 1 and w == 2: name = (1, 0)
            elif rig <= rel and len(rel) == h + 1 and w == 2: name = (1, 2)
            else: continue
            found[name] = (r0, c0, h, w)
    if len(found) != 8:
        return None
    # corner panels: the frame line plus one row/column of interior inside it
    P = {}
    for (i, j) in ((0, 0), (0, 2), (2, 0), (2, 2)):
        r0, c0, h, w = found[(i, j)]
        P[(i, j)] = (r0 - (1 if i == 2 else 0), c0 - (1 if j == 2 else 0), h + 1, w + 1)
    band_h = [P[(0, 0)][2], None, P[(2, 0)][2]]
    band_w = [P[(0, 0)][3], None, P[(0, 2)][3]]
    r0, c0, h, w = found[(0, 1)]; P[(0, 1)] = (r0, c0, band_h[0], w)
    r0, c0, h, w = found[(2, 1)]; P[(2, 1)] = (r0 + h - band_h[2], c0, band_h[2], w)
    r0, c0, h, w = found[(1, 0)]; P[(1, 0)] = (r0, c0, h, band_w[0])
    r0, c0, h, w = found[(1, 2)]; P[(1, 2)] = (r0, c0 + w - band_w[2], h, band_w[2])
    for (r0, c0, h, w) in P.values():
        if r0 < 0 or c0 < 0 or r0 + h > hi or c0 + w > wi:
            return None
    row_h = [band_h[0], P[(1, 0)][2], band_h[2]]
    col_w = [band_w[0], P[(0, 1)][3], band_w[2]]
    return bgc, fill, row_h, col_w, P


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, **color_kwargs) -> dict:
    cols = interval(0, 10, 1)
    gh_hi = min(20, max_h - 4)
    gw_hi = min(20, max_w - 4)
    if gh_hi < 9:
        gh_hi = 9
    if gw_hi < 9:
        gw_hi = 9
    goh = unifint(diff_lb, diff_ub, (9, gh_hi))
    gow = unifint(diff_lb, diff_ub, (9, gw_hi))
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
    # keep only instances whose frame panels survived the scrambling intact
    # (the rule — and a solver — reads each panel as one solid rectangle)
    GI = np.asarray(gi, dtype=int)
    cnt = Counter(GI.flatten().tolist())
    if cnt.most_common(1)[0][0] != bgc or cnt.most_common(2)[1][1] == cnt[bgc]:
        raise ValueError("background not majority")
    plan = _assemble_plan(GI)
    if plan is None:
        raise ValueError("panels not recoverable")
    _, fillc, row_h, col_w, P = plan
    GO = np.asarray(go, dtype=int)
    if fillc != sqc or (sum(row_h), sum(col_w)) != GO.shape:
        raise ValueError("bad layout")
    A = np.full(GO.shape, fillc)
    ro = [0, row_h[0], row_h[0] + row_h[1]]; co = [0, col_w[0], col_w[0] + col_w[1]]
    for (i, j), (r0, c0, h, w) in P.items():
        A[ro[i]:ro[i] + h, co[j]:co[j] + w] = GI[r0:r0 + h, c0:c0 + w]
    if not (A == GO).all():
        raise ValueError("panels corrupted")
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    # Reading: the input holds nine framed panels scattered on the background —
    # four corners, four edge panels, and a plain centre.  The output is the
    # panels butted together in a 3x3 layout.  Each framed panel is ONE unit:
    # its whole rectangle (frame line + interior) is copied from the input and
    # pasted into its slot; the plain centre is the interior base colour.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc, fill, row_h, col_w, P = _assemble_plan(I)
    ho, wo = sum(row_h), sum(col_w)
    row_off = [0, row_h[0], row_h[0] + row_h[1]]
    col_off = [0, col_w[0], col_w[0] + col_w[1]]

    ops, sels = [], []
    G = np.zeros((30, 30), dtype=int)
    G[:hi, :wi] = I

    # 1. cut the canvas to the size of the assembled frame (sum of panel sizes)
    sub = G[:ho, :wo].copy()
    G = np.zeros((30, 30), dtype=int); G[:ho, :wo] = sub
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    # 2. lay the interior colour over the whole canvas (this is the plain centre)
    if (G[:ho, :wo] != fill).any():
        G[:ho, :wo] = fill
        ops.append(int(fill)); sels.append([0, 0, ho - 1, wo - 1])

    # 3. panel by panel, in reading order of the 3x3 layout: copy the panel's
    #    rectangle from the input and paste it into its slot
    for i in range(3):
        for j in range(3):
            if (i, j) == (1, 1):
                continue
            r0, c0, h, w = P[(i, j)]
            dr, dc = row_off[i], col_off[j]
            src = I[r0:r0 + h, c0:c0 + w]
            before = G.copy()
            reg = G[dr:dr + h, dc:dc + w]
            reg[src != 0] = src[src != 0]
            if not (G == before).all():
                ops.append(28); sels.append([r0, c0, h - 1, w - 1])  # the whole panel rectangle
                ops.append(30); sels.append([dr, dc, 0, 0])
            # colour 0 does not travel through Copy/Paste: write the panel's
            # 0-cells onto the cells the copy landed on
            zs = [(dr + y, dc + x) for y in range(h) for x in range(w)
                  if src[y, x] == 0 and G[dr + y, dc + x] != 0]
            if zs:
                for (y, x) in zs:
                    G[y, x] = 0
                ops.append(0); sels.append(sel_of(zs))

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
