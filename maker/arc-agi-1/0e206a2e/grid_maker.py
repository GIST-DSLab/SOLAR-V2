"""
ARC Task: 0e206a2e (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, acol, bcol, ccol, Dcol = sample(cols, 5) if False else __import__('random').sample(cols, 5)
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "ccol": ccol, "Dcol": Dcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol, **kwargs) -> dict:
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    nsrcs = unifint(diff_lb, diff_ub, (1, max(1, min(h, w) // 5)))
    srcs = []
    abclist = []
    maxtrforsrc = 5 * nsrcs
    trforsrc = 0
    srcsucc = 0
    while trforsrc < maxtrforsrc and srcsucc < nsrcs:
        trforsrc += 1
        objsize = unifint(diff_lb, diff_ub, (5, 20))
        bb = asindices(canvas(-1, (7, 7)))
        sp = choice(totuple(bb))
        bb = remove(sp, bb)
        shp = {sp}
        for k in range(objsize - 1):
            cands = totuple((bb - shp) & mapply(dneighbors, shp))
            if len(cands) == 0:
                break
            shp.add(choice(cands))
        while 1 in shape(shp):
            cands = totuple((bb - shp) & mapply(dneighbors, shp))
            if len(cands) == 0:
                break
            shp.add(choice(cands))
        guard = 0
        while (len(set([x - y for x, y in shp])) == 1 or len(set([x + y for x, y in shp])) == 1) and guard < 50:
            guard += 1
            cands = totuple((bb - shp) & mapply(dneighbors, shp))
            if len(cands) == 0:
                break
            shp.add(choice(cands))
        if len(shp) < 5 or 1 in shape(shp):
            continue
        shp = normalize(shp)
        shp = list(shp)
        shuffle(shp)
        a, b, c = shp[:3]
        guard = 0
        while (1 in shape({a, b, c}) or (len(set([x - y for x, y in {a, b, c}])) == 1 or len(set([x + y for x, y in {a, b, c}])) == 1)) and guard < 200:
            guard += 1
            shuffle(shp)
            a, b, c = shp[:3]
        if 1 in shape({a, b, c}):
            continue
        if sorted(shape({a, b, c})) in abclist:
            continue
        D = shp[3:]
        markers = {(acol, a), (bcol, b), (ccol, c)}
        obj = markers | {(Dcol, ij) for ij in D}
        obj = frozenset(obj)
        opts = sfilter(inds, lambda ij: shift(set(shp), ij).issubset(inds))
        if len(opts) == 0:
            continue
        loc = choice(totuple(opts))
        srcsucc += 1
        gi = paint(gi, shift(obj, loc))
        shpplcd = shift(set(shp), loc)
        go = fill(go, -1, shpplcd)
        inds = (inds - shpplcd) - mapply(neighbors, shpplcd)
        srcs.append((obj, markers))
        abclist.append(sorted(shape({a, b, c})))
    if len(srcs) == 0:
        return generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol)
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 30)))
    maxtrials = 10 * num
    tr = 0
    succ = 0
    while succ < num and tr < maxtrials:
        mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
        fn = choice(mfs)
        gi = fn(gi)
        go = fn(go)
        aigo = asindices(go)
        fullinds = ofcolor(go, bgc) - mapply(neighbors, aigo - ofcolor(go, bgc))
        obj, markers = choice(srcs)
        shp = toindices(obj)
        if len(fullinds) == 0:
            break
        loctr = choice(totuple(fullinds))
        xx = shift(shp, loctr)
        if xx.issubset(fullinds):
            succ += 1
            gi = paint(gi, shift(markers, loctr))
            go = paint(go, shift(obj, loctr))
        tr += 1
    go = replace(go, -1, bgc)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (measured from I only):
      * 4-connected non-bg components with exactly FOUR colours are TEMPLATES.
        Each template = one body colour (the majority colour inside it) plus a
        triple of three single-cell markers.
      * Every template is ERASED (covered with the background).
      * Elsewhere the grid holds bare marker TRIPLES.  For each of the 8 dihedral
        orientations of the grid, wherever a template's marker triple occurs, the
        template's body is drawn back around it.
    O is never inspected.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    ops, sels = [], []

    # ---- 1. connected components of non-background cells (4-connectivity) ----
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bgc:
                            seen[na, nb] = True
                            stack.append((na, nb))
                comps.append(sorted(cells))

    # ---- 2. templates = components with exactly 4 distinct colours ----
    templates = []
    for cells in comps:
        colors = [int(I[r, c]) for r, c in cells]
        if len(set(colors)) != 4:
            continue
        body_col = int(Counter(colors).most_common(1)[0][0])
        markers = [(int(I[r, c]), (r, c)) for r, c in cells if int(I[r, c]) != body_col]
        body = [(r, c) for r, c in cells if int(I[r, c]) == body_col]
        if len(markers) == 3 and len(body) >= 2:
            templates.append((cells, body_col, markers, body))

    g = I.copy()

    # ---- 3. erase every template (one Color-bgc op per template object) ----
    for cells, body_col, markers, body in templates:
        ops.append(bgc)
        sels.append(sel_of(cells))
        for r, c in cells:
            g[r, c] = bgc

    # ---- 4. templates normalized w.r.t. their marker-triple upper-left corner --
    tmpl = []
    for cells, body_col, markers, body in templates:
        mr = min(p[0] for _, p in markers)
        mc = min(p[1] for _, p in markers)
        mrel = [(v, (p[0] - mr, p[1] - mc)) for v, p in markers]
        brel = [(r - mr, c - mc) for r, c in body]
        mh = max(d[0] for _, d in mrel) + 1
        mw = max(d[1] for _, d in mrel) + 1
        tmpl.append((body_col, mrel, brel, mh, mw))

    # ---- 5. the 8 dihedral views, in the task's canonical order ----
    TRANSFORMS = [
        (lambda a: a,                              lambda r, c, H, W: (r, c)),                 # identity
        (lambda a: a.T,                            lambda r, c, H, W: (c, r)),                 # dmirror
        (lambda a: np.flipud(np.fliplr(a)).T,      lambda r, c, H, W: (H - 1 - c, W - 1 - r)), # cmirror
        (lambda a: np.flipud(a),                   lambda r, c, H, W: (H - 1 - r, c)),         # hmirror
        (lambda a: np.fliplr(a),                   lambda r, c, H, W: (r, W - 1 - c)),         # vmirror
        (lambda a: np.rot90(a, 1),                 lambda r, c, H, W: (c, W - 1 - r)),         # rot270
        (lambda a: np.rot90(a, 2),                 lambda r, c, H, W: (H - 1 - r, W - 1 - c)), # rot180
        (lambda a: np.rot90(a, 3),                 lambda r, c, H, W: (H - 1 - c, r)),         # rot90
    ]

    for fwd, invp in TRANSFORMS:
        Tg = fwd(g)
        th, tw = Tg.shape
        placements = []   # occurrences are all found on the pre-paint view
        for body_col, mrel, brel, mh, mw in tmpl:
            for i in range(th - mh + 1):
                for j in range(tw - mw + 1):
                    ok = True
                    for v, (dr, dc) in mrel:
                        if int(Tg[i + dr, j + dc]) != v:
                            ok = False
                            break
                    if not ok:
                        continue
                    cells = [(i + dr, j + dc) for dr, dc in brel
                             if 0 <= i + dr < th and 0 <= j + dc < tw]
                    if cells:
                        placements.append((body_col, cells))
        for body_col, cells in placements:
            todo = sorted({invp(r, c, h, w) for r, c in cells})
            todo = [p for p in todo if int(g[p[0], p[1]]) != body_col]
            if not todo:
                continue
            ops.append(int(body_col))
            sels.append(sel_of(todo))
            for r, c in todo:
                g[r, c] = body_col

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 0e206a2e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0e206a2e"
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
                                f"for task 0e206a2e"
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
                    f"Failed to build a complete episode for task 0e206a2e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0e206a2e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
