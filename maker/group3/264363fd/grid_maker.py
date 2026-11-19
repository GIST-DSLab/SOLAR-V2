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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc, linc = random.sample(cols, 3)
    remcols = [c for c in cols if c not in (bgc, sqc, linc)]
    cpcol = random.choice(remcols)
    nbhcol = random.choice(remcols)
    return {"bgc": bgc, "sqc": sqc, "linc": linc, "cpcol": cpcol, "nbhcol": nbhcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, linc=None,
             cpcol=None, nbhcol=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or sqc is None or linc is None:
        bgc, sqc, linc = sample(cols, 3)
    remcols = difference(cols, (bgc, sqc, linc))
    if cpcol is None:
        cpcol = choice(remcols)
    if nbhcol is None:
        nbhcol = choice(remcols)
    cp = (2, 2)
    neighs = neighbors(cp)
    o1 = shift(frozenset({(0, 1), (-1, 1)}), (1, 1))
    o2 = shift(frozenset({(1, 0), (1, -1)}), (1, 1))
    o3 = shift(frozenset({(2, 1), (3, 1)}), (1, 1))
    o4 = shift(frozenset({(1, 2), (1, 3)}), (1, 1))
    mpr = {o1: (-1, 0), o2: (0, -1), o3: (1, 0), o4: (0, 1)}
    h = unifint(diff_lb, diff_ub, (15, max(15, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (15, max(15, min(30, max_w))))
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
    if succ == 0:
        raise ValueError("no rectangle placed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in sorted(set(cells))]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ALL8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    def components(bg):
        seen = np.zeros((h, w), dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if I[r, c] != bg and not seen[r, c]:
                    seen[r, c] = True
                    st, cells = [(r, c)], []
                    while st:
                        a, b = st.pop()
                        cells.append((a, b))
                        for dr, dc in DIRS:
                            na, nb = a + dr, b + dc
                            if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bg:
                                seen[na, nb] = True
                                st.append((na, nb))
                    comps.append(cells)
        return comps

    def is_rect(comp):
        rs = [r for r, _ in comp]; cs = [c for _, c in comp]
        hh, ww = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        return hh >= 5 and ww >= 5 and len(comp) == hh * ww

    def has_hub(comp):
        s = set(comp)
        return any(all((r + a, c + b) in s for a, b in ALL8) for r, c in comp)

    # background: the canvas colour on which one small key glyph and the
    # solid rectangles sit as separate objects
    counts = Counter(I.flatten().tolist())
    bgc, comps = None, None
    for cand, _ in counts.most_common():
        cs_ = components(cand)
        if not cs_:
            continue
        cs_.sort(key=len)
        if len(cs_[0]) <= 13 and has_hub(cs_[0]) and all(is_rect(c) for c in cs_[1:]):
            bgc, comps = cand, cs_
            break
    if bgc is None:
        bgc = counts.most_common(1)[0][0]
        comps = sorted(components(bgc), key=len)

    key = comps[0]
    keyset = set(key)
    rects = comps[1:]

    # hub of the key: the only key cell whose 8 neighbours all belong to the key
    hub = next((r, c) for r, c in key if all((r + a, c + b) in keyset for a, b in ALL8))
    cr, cc = hub
    spikes = [d for d in DIRS if (cr + 2 * d[0], cc + 2 * d[1]) in keyset]
    linc = int(I[cr + 2 * spikes[0][0], cc + 2 * spikes[0][1]]) if spikes else None
    ring = [((a, b), int(I[cr + a, cc + b])) for a, b in ALL8 if (a, b) not in spikes]

    # markers: odd pixels inside each rectangle
    rect_info = []
    for comp in rects:
        rs = [r for r, _ in comp]; cs = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        sqcol = Counter(int(I[r, c]) for r, c in comp).most_common(1)[0][0]
        markers = [(r, c) for r, c in comp if int(I[r, c]) != sqcol]
        rect_info.append((r0, c0, r1, c1, markers))

    ops, sels = [], []
    G = I.copy()

    def emit(color, cells):
        cells = sorted(set(cells))
        if not cells or all(G[r, c] == color for r, c in cells):
            return
        for r, c in cells:
            G[r, c] = color
        ops.append(int(color))
        sels.append(sel_of(cells))

    # Unit of work = one colour across the WHOLE grid.
    # Pass 1: every ring colour of the key, stamped around every marker at once.
    ring_by_col = {}
    for (r0, c0, r1, c1, markers) in rect_info:
        for (mr, mc) in markers:
            for (a, b), col in ring:
                rr, cc2 = mr + a, mc + b
                if r0 <= rr <= r1 and c0 <= cc2 <= c1:
                    ring_by_col.setdefault(col, []).append((rr, cc2))
    for col in sorted(ring_by_col):
        emit(col, ring_by_col[col])

    # Pass 2: all arm rays of every marker, shot to their rectangle's border, in one stroke.
    if linc is not None:
        rays = []
        for (r0, c0, r1, c1, markers) in rect_info:
            for (mr, mc) in markers:
                for d in spikes:
                    rr, cc2 = mr + d[0], mc + d[1]
                    while r0 <= rr <= r1 and c0 <= cc2 <= c1:
                        rays.append((rr, cc2))
                        rr += d[0]; cc2 += d[1]
        emit(linc, rays)

    # Pass 3: the key glyph has served its purpose; wipe it to background.
    emit(bgc, key)

    # whole grid, a full rectangle
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
