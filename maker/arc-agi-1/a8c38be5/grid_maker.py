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
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    non_bg = [c for c in cnt if c != bgc]
    sqc = max(non_bg, key=lambda c: cnt[c])
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

    markers = []
    for col in marker_colors:
        for cells in comps(col):
            markers.append((col, cells))

    def classify(rel, h, w):
        top = {(0, c) for c in range(w)}
        bot = {(h - 1, c) for c in range(w)}
        left = {(r, 0) for r in range(h)}
        right = {(r, w - 1) for r in range(h)}
        ul = (0, 0) in rel
        ur = (0, w - 1) in rel
        ll = (h - 1, 0) in rel
        lr = (h - 1, w - 1) in rel
        ft = top <= rel
        fb = bot <= rel
        fl = left <= rel
        fr = right <= rel
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
        if ft and not ll and not lr and rHn == 1:
            return 'topT'
        if fb and not ul and not ur and r0n == 1:
            return 'botT'
        if fl and not ur and not lr and cWn == 1:
            return 'leftT'
        if fr and not ul and not ll and c0n == 1:
            return 'rightT'
        return None

    slot = {}
    for col, cells in markers:
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

    # canvas dims measured from the marker pieces (matches verifier x95 / x101)
    ho = H('TL') + H('BL') + H('leftT') + 2
    wo = W('TL') + W('TR') + W('topT') + 2

    # interior/base color read from I just inside the top-left bracket (verifier x104)
    tl_r0, tl_c0 = slot['TL'][1], slot['TL'][2]
    fill_color = int(I[tl_r0 + 1, tl_c0 + 1])

    dest = {
        'TL': (0, 0),
        'TR': (0, wo - W('TR')),
        'BL': (ho - H('BL'), 0),
        'BR': (ho - H('BR'), wo - W('BR')),
        'topT': (0, W('TL') + 1),
        'botT': (ho - H('botT'), W('BL') + 1),
        'leftT': (H('TL') + 1, 0),
        'rightT': (H('TR') + 1, wo - W('rightT')),
    }

    ops = []
    sels = []

    # shrink working canvas to output size (crops top-left of I; fully repainted next)
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])
    # lay the interior base over the whole canvas
    ops.append(fill_color)
    sels.append([0, 0, ho - 1, wo - 1])

    # draw each marker piece at its measured destination
    order = ['TL', 'TR', 'BR', 'BL', 'topT', 'rightT', 'botT', 'leftT']
    for n in order:
        col, r0, c0, h, w, rel = slot[n]
        dr, dc = dest[n]
        cells = [(dr + rr, dc + cc) for (rr, cc) in rel]
        ops.append(int(col))
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
