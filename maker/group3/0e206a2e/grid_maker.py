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
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, acol, bcol, ccol, Dcol = random.sample(range(10), 5)
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "ccol": ccol, "Dcol": Dcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol, **kwargs) -> dict:
    from random import choice, shuffle
    from dsl import (canvas, asindices, remove, dneighbors, mapply, normalize,
                     shape, totuple, sfilter, shift, paint, fill, neighbors, ofcolor,
                     identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270,
                     toindices, replace)
    from utils import unifint
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    nsrcs = unifint(diff_lb, diff_ub, (1, min(h, w) // 5))
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
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while 1 in shape(shp):
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while len(set([x - y for x, y in shp])) == 1 or len(set([x + y for x, y in shp])) == 1:
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        shp = normalize(shp)
        shp = list(shp)
        shuffle(shp)
        a, b, c = shp[:3]
        while 1 in shape({a, b, c}) or (len(set([x - y for x, y in {a, b, c}])) == 1 or len(set([x + y for x, y in {a, b, c}])) == 1):
            shuffle(shp)
            a, b, c = shp[:3]
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
    if not srcs:
        raise ValueError("no template placed")
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 30))
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
    if succ == 0:
        raise ValueError("no copy placed")
    go = replace(go, -1, bgc)
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    """
    Rule (from I only): 4-connected components with exactly four colours are
    TEMPLATES (one body colour + three single-cell markers).  Templates vanish;
    wherever a template's marker triple appears elsewhere (in any of the 8
    dihedral orientations) the template body is rebuilt around it.

    Unit of work here is the COLOUR LAYER, not the object:
      1. one stroke erases the whole template layer (all template cells at once),
      2. one stroke per body colour paints every rebuilt body of that colour.
    O is never inspected.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # background: the colour every demonstration input is drawn on
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ei, _ in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])

    # ---- 4-connected non-background components ----
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
                comps.append(cells)

    templates = []
    for cells in comps:
        colors = [int(I[r, c]) for r, c in cells]
        if len(set(colors)) != 4:
            continue
        body_col = int(Counter(colors).most_common(1)[0][0])
        markers = [(int(I[r, c]), (r, c)) for r, c in cells if int(I[r, c]) != body_col]
        body = [(r, c) for r, c in cells if int(I[r, c]) == body_col]
        if len(markers) == 3 and len(body) >= 1:
            templates.append((cells, body_col, markers, body))

    ops, sels = [], []
    g = I.copy()

    # ---- 1. erase the whole template layer in one stroke ----
    tcells = sorted({p for t in templates for p in t[0]})
    if tcells:
        ops.append(bgc)
        sels.append(sel_of(tcells))
        for r, c in tcells:
            g[r, c] = bgc

    # ---- 2. locate every marker triple in every dihedral view ----
    tmpl = []
    for cells, body_col, markers, body in templates:
        mr = min(p[0] for _, p in markers)
        mc = min(p[1] for _, p in markers)
        mrel = [(v, (p[0] - mr, p[1] - mc)) for v, p in markers]
        brel = [(r - mr, c - mc) for r, c in body]
        mh = max(d[0] for _, d in mrel) + 1
        mw = max(d[1] for _, d in mrel) + 1
        tmpl.append((body_col, mrel, brel, mh, mw))

    TRANSFORMS = [
        (lambda a: a,                         lambda r, c, H, W: (r, c)),
        (lambda a: a.T,                       lambda r, c, H, W: (c, r)),
        (lambda a: np.flipud(np.fliplr(a)).T, lambda r, c, H, W: (H - 1 - c, W - 1 - r)),
        (lambda a: np.flipud(a),              lambda r, c, H, W: (H - 1 - r, c)),
        (lambda a: np.fliplr(a),              lambda r, c, H, W: (r, W - 1 - c)),
        (lambda a: np.rot90(a, 1),            lambda r, c, H, W: (c, W - 1 - r)),
        (lambda a: np.rot90(a, 2),            lambda r, c, H, W: (H - 1 - r, W - 1 - c)),
        (lambda a: np.rot90(a, 3),            lambda r, c, H, W: (H - 1 - c, r)),
    ]

    # rebuilt body cells grouped by body colour (planning only, from I + rule)
    layer = {}
    plan = g.copy()
    for fwd, invp in TRANSFORMS:
        Tg = fwd(plan)
        th, tw = Tg.shape
        found = []
        for body_col, mrel, brel, mh, mw in tmpl:
            for i in range(th - mh + 1):
                for j in range(tw - mw + 1):
                    if all(int(Tg[i + dr, j + dc]) == v for v, (dr, dc) in mrel):
                        cells = [invp(i + dr, j + dc, h, w) for dr, dc in brel
                                 if 0 <= i + dr < th and 0 <= j + dc < tw]
                        found.append((body_col, cells))
        for body_col, cells in found:
            for r, c in cells:
                plan[r, c] = body_col
                layer.setdefault(body_col, set()).add((r, c))

    # ---- 3. paint each body-colour layer in one stroke ----
    for body_col in sorted(layer):
        todo = sorted(p for p in layer[body_col] if int(g[p[0], p[1]]) != body_col)
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
