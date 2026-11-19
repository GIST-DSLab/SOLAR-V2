"""
ARC Task: 36d67576 (RE-ARC) — LLM-generated grid_maker
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
    bgc, mainc, markerc = random.sample(cols, 3)
    remcols = [c for c in cols if c not in (bgc, mainc, markerc)]
    ncols = random.randint(1, len(remcols))
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, mainc, markerc, ccols) -> dict:
    while True:
        h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
        gi = canvas(bgc, (h, w))
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if choice((True, False)):
            oh, ow = ow, oh
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (4, len(bounds)))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        if oh > h or ow > w:
            continue
        ntocompc = unifint(diff_lb, diff_ub, (1, max(1, ncells - 3)))
        markercell = choice(totuple(obj))
        remobj = remove(markercell, obj)
        markercellobj = {(markerc, markercell)}
        ntocompc = min(ntocompc, len(remobj))
        tocompc = set(sample(totuple(remobj), ntocompc))
        mainpart = (obj - {markercell}) - tocompc
        if len(mainpart) == 0:
            continue
        mainpartobj = recolor(mainc, mainpart)
        tocompcobj = {(choice(ccols), ij) for ij in tocompc}
        obj = tocompcobj | mainpartobj | markercellobj
        smobj = mainpartobj | markercellobj
        smobjn = normalize(smobj)
        isfakesymm = False
        for symmf in [dmirror, cmirror, hmirror, vmirror]:
            if symmf(smobjn) == smobjn and symmf(obj) != obj:
                isfakesymm = True
                break
        if isfakesymm:
            continue
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        plcd = shift(obj, (loci, locj))
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (asindices(gi) - plcdi) - mapply(neighbors, plcdi)
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        succ = 0
        tr = 0
        maxtr = noccs * 5
        go = tuple(e for e in gi)
        while tr < maxtr and succ < noccs:
            tr += 1
            mf1 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf2 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf = compose(mf1, mf2)
            outobj = normalize(mf(obj))
            inobj = sfilter(outobj, lambda cij: cij[0] in [mainc, markerc])
            ohh, oww = shape(outobj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - ohh and ij[1] <= w - oww)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            outobjp = shift(outobj, loc)
            inobjp = shift(inobj, loc)
            outobjpi = toindices(outobjp)
            if outobjpi.issubset(inds):
                succ += 1
                inds = (inds - outobjpi) - mapply(neighbors, outobjpi)
                gi = paint(gi, inobjp)
                go = paint(go, outobjp)
        break
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (read from I alone): the largest object is the template (main colour,
    marker cell, completion colours).  Every other object is the main+marker
    part of a mirrored/rotated copy of it; complete each one.
    Strokes: CopyI the template's bounding rectangle once and Paste it onto an
    occurrence's rectangle (then FlipH/FlipV the rectangle when the copy is a
    mirror image); once one occurrence of an orientation is complete, CopyO it
    and Paste it onto the others of that orientation.  Diagonal orientations
    (no flip reaches them) get their first copy painted colour-by-colour.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    submit = [0, 0, O.shape[0] - 1, O.shape[1] - 1]
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                a, b = stack.pop(); cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bgc:
                            seen[na, nb] = True; stack.append((na, nb))
            comps.append(cells)
    if len(comps) < 2:
        ops.append(34); sels.append(submit); return ops, sels

    tidx = max(range(len(comps)), key=lambda k: len(comps[k]))
    tcells = comps[tidx]
    sub_palette = set()
    for k, cc in enumerate(comps):
        if k != tidx:
            for (r, c) in cc:
                sub_palette.add(int(I[r, c]))
    template = frozenset((int(I[r, c]), (r, c)) for (r, c) in tcells)
    if not any(v in sub_palette for v, _ in template):
        ops.append(34); sels.append(submit); return ops, sels
    tr0 = min(r for r, _ in tcells); tc0 = min(c for _, c in tcells)
    th = max(r for r, _ in tcells) - tr0 + 1; tw = max(c for _, c in tcells) - tc0 + 1
    tset = set(tcells)
    src_clean = all(I[r, c] == bgc or (r, c) in tset
                    for r in range(tr0, tr0 + th) for c in range(tc0, tc0 + tw))
    comp_cols = {v for v, _ in template if v not in sub_palette}
    tmpl_cols = {v for v, _ in template}
    copy_ok = not (bgc != 0 and 0 in comp_cols)          # 0 never pastes
    flip_ok = copy_ok and src_clean and not (bgc != 0 and 0 in tmpl_cols)

    def norm(obj):
        mi = min(p[0] for _, p in obj); mj = min(p[1] for _, p in obj)
        return frozenset((v, (p[0] - mi, p[1] - mj)) for v, p in obj)
    f_id = lambda o: norm(o)
    f_d = lambda o: norm(frozenset((v, (j, i)) for v, (i, j) in o))
    f_c = lambda o: norm(frozenset((v, (-j, -i)) for v, (i, j) in o))
    f_h = lambda o: norm(frozenset((v, (-i, j)) for v, (i, j) in o))
    f_v = lambda o: norm(frozenset((v, (i, -j)) for v, (i, j) in o))
    funcs = [f_id, f_d, f_c, f_h, f_v]
    base = norm(template)
    flip_of = {base: [], f_v(base): [26], f_h(base): [27], f_h(f_v(base)): [26, 27]}
    variants = []
    for f1 in funcs:
        for f2 in funcs:
            t = f1(f2(base))
            if t not in variants:
                variants.append(t)

    # occurrences: (loc, variant index, full-object cells placed, sub cells placed)
    occs = []
    for vi, var in enumerate(variants):
        sub = [(v, p) for v, p in var if v in sub_palette]
        if not sub:
            continue
        sr = min(p[0] for _, p in sub); sc = min(p[1] for _, p in sub)
        maxa = max(p[0] for _, p in sub) - sr; maxb = max(p[1] for _, p in sub) - sc
        for i in range(h - maxa):
            for j in range(w - maxb):
                if all(I[i + p[0] - sr, j + p[1] - sc] == v for v, p in sub):
                    full = [(v, (i + p[0] - sr, j + p[1] - sc)) for v, p in var]
                    subp = {(i + p[0] - sr, j + p[1] - sc) for _, p in sub}
                    occs.append(((i, j), vi, full, subp))

    G = I.copy()
    clip = [None]

    def color_route(full):
        bycol = {}
        for v, (rr, cc) in full:
            if 0 <= rr < h and 0 <= cc < w and G[rr, cc] != v:
                bycol.setdefault(v, []).append((rr, cc))
        for v in sorted(bycol):
            ops.append(int(v)); sels.append(sel_of(bycol[v]))
            for (rr, cc) in bycol[v]:
                G[rr, cc] = v

    def color_cost(full):
        return len({v for v, (rr, cc) in full
                    if 0 <= rr < h and 0 <= cc < w and G[rr, cc] != v})

    def bbox(full):
        r0 = min(p[0] for _, p in full); c0 = min(p[1] for _, p in full)
        r1 = max(p[0] for _, p in full); c1 = max(p[1] for _, p in full)
        return r0, c0, r1 - r0, c1 - c0

    def clean(full, subp):
        r0, c0, dh, dw = bbox(full)
        if r0 < 0 or c0 < 0 or r0 + dh >= h or c0 + dw >= w:
            return False
        return all(G[r, c] == bgc or (r, c) in subp
                   for r in range(r0, r0 + dh + 1) for c in range(c0, c0 + dw + 1))

    def land(full):           # the pasted rectangle now holds exactly the full object
        r0, c0, dh, dw = bbox(full)
        G[r0:r0 + dh + 1, c0:c0 + dw + 1] = bgc
        for v, (rr, cc) in full:
            G[rr, cc] = v

    def intact(rect, full):   # rectangle still holds exactly that finished object
        r0, c0, dh, dw = rect
        want = {p: v for v, p in full}
        return all(G[r, c] == want.get((r, c), bgc)
                   for r in range(r0, r0 + dh + 1) for c in range(c0, c0 + dw + 1))

    occs.sort(key=lambda t: (t[0], t[2]))   # raster order of the occurrences
    tmpl_key = variants.index(base)
    done = {}                               # orientation -> a finished rectangle
    for _, vi, full, subp in occs:
        var = variants[vi]
        cc = color_cost(full)
        if cc == 0:
            continue
        if not (copy_ok and clean(full, subp)):
            color_route(full); continue
        r0, c0, dh, dw = bbox(full)
        rect = [r0, c0, dh, dw]            # the occurrence's full rectangle (clean)
        if clip[0] == vi:                   # clipboard already holds this orientation
            ops.append(30); sels.append([r0, c0, 0, 0])
            land(full); done[vi] = (rect, full); continue
        if vi in done and 2 <= cc and intact(*done[vi]):   # CopyO a finished copy
            ops.append(29); sels.append(list(done[vi][0])); clip[0] = vi
            ops.append(30); sels.append([r0, c0, 0, 0])
            land(full); done[vi] = (rect, full); continue
        if var in flip_of and (vi == tmpl_key or flip_ok) and src_clean:
            fl = flip_of[var]
            cost = (0 if clip[0] == tmpl_key else 1) + 1 + len(fl) * (2 if bgc == 0 else 1)
            if cost <= cc:
                if bgc == 0:
                    for f in fl:            # bring the partial into template orientation
                        ops.append(f); sels.append(rect)
                if clip[0] != tmpl_key:
                    ops.append(28); sels.append([tr0, tc0, th - 1, tw - 1])  # CopyI template rect
                    clip[0] = tmpl_key
                ops.append(30); sels.append([r0, c0, 0, 0])
                for f in fl:                # mirror the pasted copy into place
                    ops.append(f); sels.append(rect)
                land(full); done[vi] = (rect, full); continue
        color_route(full); done[vi] = (rect, full)

    ops.append(34); sels.append(submit)
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
                        f"num_examples+1 ({num_examples + 1}) for task 36d67576"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 36d67576"
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
                                f"for task 36d67576"
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
                    f"Failed to build a complete episode for task 36d67576 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"36d67576-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
