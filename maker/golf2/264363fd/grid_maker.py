"""
ARC Task: 264363fd (RE-ARC) — LLM-generated grid_maker
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

try:
    from dsl import *
except Exception:
    pass
try:
    from utils import *
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc, linc = random.sample(cols, 3)
    remcols = [c for c in cols if c not in (bgc, sqc, linc)]
    cpcol = random.choice(remcols)
    nbhcol = random.choice(remcols)
    return {"bgc": bgc, "sqc": sqc, "linc": linc, "cpcol": cpcol, "nbhcol": nbhcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, linc, cpcol, nbhcol) -> dict:
    cp = (2, 2)
    neighs = neighbors(cp)
    o1 = shift(frozenset({(0, 1), (-1, 1)}), (1, 1))
    o2 = shift(frozenset({(1, 0), (1, -1)}), (1, 1))
    o3 = shift(frozenset({(2, 1), (3, 1)}), (1, 1))
    o4 = shift(frozenset({(1, 2), (1, 3)}), (1, 1))
    mpr = {o1: (-1, 0), o2: (0, -1), o3: (1, 0), o4: (0, 1)}
    h = unifint(diff_lb, diff_ub, (15, max(15, max_h)))
    w = unifint(diff_lb, diff_ub, (15, max(15, max_w)))
    nspikes = randint(1, 4)
    spikes = sample((o1, o2, o3, o4), nspikes)
    lns = merge(set(spikes))
    obj = {(cpcol, cp)} | recolor(linc, lns) | recolor(nbhcol, neighs - lns)
    loci = randint(0, h - 5)
    locj = randint(0, w - 5)
    loc = (loci, locj)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = paint(gi, shift(obj, loc))
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 100))
    succ = 0
    tr = 0
    maxtr = 10 * numsq
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(shift(obj, loc)))
    while succ < numsq and tr < maxtr:
        tr += 1
        gh = randint(5, h // 2 + 1)
        gw = randint(5, w // 2 + 1)
        cands = sfilter(inds, lambda ij: ij[0] <= h - gh and ij[1] <= w - gw)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        g1 = canvas(sqc, (gh, gw))
        g2 = canvas(sqc, (gh, gw))
        ginds = asindices(g1)
        gindsfull = asindices(g1)
        bck = shift(ginds, loc)
        if bck.issubset(inds):
            noccs = unifint(diff_lb, diff_ub, (1, (gh * gw) // 25))
            succ2 = 0
            tr2 = 0
            maxtr2 = 5 * noccs
            while succ2 < noccs and tr2 < maxtr2:
                tr2 += 1
                cands2 = sfilter(ginds, lambda ij: ij[0] <= gh - 5 and ij[1] <= gw - 5)
                if len(cands2) == 0:
                    break
                loc2 = choice(totuple(cands2))
                lns2 = merge(frozenset({shoot(add(cp, add(loc2, mpr[spike])), mpr[spike]) for spike in spikes}))
                lns2 = lns2 & gindsfull
                plcd2 = shift(obj, loc2)
                plcd2i = toindices(plcd2)
                if plcd2i.issubset(ginds) and lns2.issubset(ginds | ofcolor(g2, linc)) and len(lns2 - plcd2i) > 0:
                    succ2 += 1
                    ginds = ((ginds - plcd2i) - mapply(neighbors, plcd2i)) - lns2
                    g1 = fill(g1, cpcol, {add(cp, loc2)})
                    g2 = paint(g2, plcd2)
                    g2 = fill(g2, linc, lns2)
            if succ2 > 0:
                succ += 1
                inds = (inds - bck) - outbox(bck)
                objfull1 = shift(asobject(g1), loc)
                objfull2 = shift(asobject(g2), loc)
                gi = paint(gi, objfull1)
                go = paint(go, objfull2)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    def comps(bg):
        seen = np.zeros((h, w), bool)
        out = []
        for r in range(h):
            for c in range(w):
                if I[r, c] != bg and not seen[r, c]:
                    st = [(r, c)]
                    seen[r, c] = True
                    cs = []
                    while st:
                        a, b = st.pop()
                        cs.append((a, b))
                        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            x, y = a + da, b + db
                            if 0 <= x < h and 0 <= y < w and not seen[x, y] and I[x, y] != bg:
                                seen[x, y] = True
                                st.append((x, y))
                    out.append(cs)
        return out

    def isrect(cs):
        rs = [r for r, _ in cs]
        cc = [c for _, c in cs]
        H = max(rs) - min(rs) + 1
        W = max(cc) - min(cc) + 1
        return H >= 5 and W >= 5 and H * W == len(cs)

    full = sel_of([(r, c) for r in range(h) for c in range(w)])

    # background: the colour under which there are only full rectangles + one small key object
    bgc = None
    key = None
    rects = []
    for cand, _ in Counter(I.flatten().tolist()).most_common():
        cl = comps(cand)
        non = [x for x in cl if not isrect(x)]
        if len(non) == 1 and len(non[0]) <= 13:
            bgc, key, rects = cand, non[0], [x for x in cl if isrect(x)]
            break
    if key is None:
        return [34], [full]

    # key centre: centre of the cells with >1 orthogonal neighbour inside the key (3x3 core)
    ks = set(key)
    core = [p for p in key
            if sum((p[0] + a, p[1] + b) in ks for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))) > 1]
    cr = (min(r for r, _ in core) + max(r for r, _ in core)) // 2
    cc = (min(c for _, c in core) + max(c for _, c in core)) // 2
    stamp = {(r - cr, c - cc): int(I[r, c]) for r, c in key}
    dirs = [(dr // 2, dc // 2) for (dr, dc) in stamp if max(abs(dr), abs(dc)) == 2]
    lincs = [stamp[(2 * a, 2 * b)] for a, b in dirs]

    # per colour: every stamp around each marker + the spike lines out to the rectangle edge
    paint_cells = {}
    for rc in rects:
        rs = [r for r, _ in rc]
        cs = [c for _, c in rc]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        sq = Counter(int(I[r, c]) for r, c in rc).most_common(1)[0][0]
        for mr, mc in rc:
            if I[mr, mc] == sq:
                continue
            for (dr, dc), col in stamp.items():
                paint_cells.setdefault(col, set()).add((mr + dr, mc + dc))
            for (a, b), col in zip(dirs, lincs):
                k = 3
                while r0 <= mr + a * k <= r1 and c0 <= mc + b * k <= c1:
                    paint_cells[col].add((mr + a * k, mc + b * k))
                    k += 1

    ops, sels = [], []
    G = I.copy()
    # 1. erase the key object
    ops.append(int(bgc))
    sels.append(sel_of(sorted(key)))
    for p in key:
        G[p] = bgc
    # 2. one Color op per stamp colour (lines first)
    order = ([lincs[0]] if lincs else []) + [c for c in paint_cells if not lincs or c != lincs[0]]
    for col in order:
        cells = sorted(p for p in paint_cells[col] if 0 <= p[0] < h and 0 <= p[1] < w)
        if cells and any(G[p] != col for p in cells):
            for p in cells:
                G[p] = col
            ops.append(int(col))
            sels.append(sel_of(cells))
    ops.append(34)
    sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 264363fd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 264363fd"
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
                                f"for task 264363fd"
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
                    f"Failed to build a complete episode for task 264363fd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"264363fd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
