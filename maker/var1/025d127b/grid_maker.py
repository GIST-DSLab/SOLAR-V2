"""
ARC Task: 025d127b (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. episode colours
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """The rule here is purely positional (the piece reaching furthest right
    stays, everything else of that shape slides one cell right), so only the
    canvas colour really has to be pinned.  The palette the shapes are drawn
    from is fixed too, so the whole episode looks like one puzzle."""
    cols = list(range(10))
    bgc = random.choice(cols)
    rest = [c for c in cols if c != bgc]
    ccols = random.sample(rest, random.randint(1, 9))
    return {"bgc": bgc, "ccols": ccols}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if ccols is None:
        remcols = remove(bgc, cols)
        ccols = list(random.sample(list(remcols), random.randint(1, 9)))
    ccols = list(ccols)

    hlim = max(5, min(30, int(max_h)))
    wlim = max(5, min(30, int(max_w)))

    gi = canvas(bgc, (5, 5))
    go = canvas(bgc, (5, 5))

    for _attempt in range(24):
        h = unifint(diff_lb, diff_ub, (5, hlim))
        w = unifint(diff_lb, diff_ub, (5, wlim))
        numcols = unifint(diff_lb, diff_ub, (1, len(ccols)))
        icols = random.sample(ccols, numcols)
        nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 20))

        succ = 0
        tr = 0
        maxtr = 5 * nobjs
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        inds = asindices(gi)
        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = random.randint(3, 6)
            ow = random.randint(3, 6)
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = random.choice(totuple(cands))
            topl = connect((0, 0), (0, ow - 1))
            leftl = connect((1, 0), (oh - 2, oh - 3))
            rightl = connect((1, ow), (oh - 2, ow + oh - 3))
            botl = connect((oh - 1, oh - 2), (oh - 1, oh - 3 + ow))
            inobj = topl | leftl | rightl | botl
            outobj = shift(topl, (0, 1)) | botl | shift(leftl, (0, 1)) | \
                connect((1, ow + 1), (oh - 3, ow + oh - 3)) | {(oh - 2, ow + oh - 3)}
            outobj = sfilter(outobj, lambda ij: ij[1] <= rightmost(inobj))
            fullobj = inobj | outobj
            inobj = shift(inobj, loc)
            outobj = shift(outobj, loc)
            fullobj = shift(fullobj, loc)
            if fullobj.issubset(inds):
                inds = (inds - fullobj) - mapply(neighbors, fullobj)
                succ += 1
                col = random.choice(icols)
                gi = fill(gi, col, inobj)
                go = fill(go, col, outobj)
        if succ > 0:
            break

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. trajectory
# ----------------------------------------------------------------------------
def derive_operations(I, O, examples=None):
    """Rule (read off the inputs, never off O):

    every drawn shape is an outline made of several 4-connected pieces; the one
    piece that reaches furthest right stays exactly where it is, and every other
    piece of that shape slides ONE cell to the right.

    The rule is fixed, the route is not.  Three things a person actually chooses
    are read off the input grid itself -- which order to work through the shapes,
    how coarsely to grab each shape (all the sliding cells at once / piece by
    piece / row by row), and whether to wipe each vacated cell as they go or
    sweep them all up at the end.  Same grid -> same route; the variety comes
    from the grids differing.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # ---- background: the colour the canvas is painted with.  It is the same
    #      colour in every demonstration input, so pool them when given.
    counts = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            try:
                ex_in = ex["input"] if isinstance(ex, dict) else ex[0]
                counts.update(np.asarray(ex_in, dtype=int).flatten().tolist())
            except Exception:
                pass
    bgc = int(counts.most_common(1)[0][0])

    # ---- 4-connected same-colour pieces of the foreground -------------------
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or int(I[r, c]) == bgc:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                            and int(I[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": col, "cells": sorted(cells)})

    n = len(comps)
    if n == 0:
        return [34], [[0, 0, ho - 1, wo - 1]]

    # ---- pieces that touch diagonally and share a colour are one shape -------
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    owner = {}
    for i, cp in enumerate(comps):
        for cell in cp["cells"]:
            owner[cell] = i
    for i, cp in enumerate(comps):
        for (r, c) in cp["cells"]:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    j = owner.get((r + dr, c + dc))
                    if j is not None and j != i and comps[j]["color"] == cp["color"]:
                        union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    shapes = []
    for _, idxs in groups.items():
        # the anchor piece: the one reaching furthest right -> it does NOT move
        keeper = max(idxs, key=lambda i: (max(c for _, c in comps[i]["cells"]),
                                          max(r for r, _ in comps[i]["cells"]),
                                          min(c for _, c in comps[i]["cells"])))
        cells = sorted(cell for i in idxs for cell in comps[i]["cells"])
        shapes.append({
            "color": comps[idxs[0]]["color"],
            "cells": cells,
            "anchor": min(cells),
            "left": min(c for _, c in cells),
            "pieces": [comps[i]["cells"] for i in idxs if i != keeper],
        })

    nshapes = len(shapes)
    reading = sorted(shapes, key=lambda s: s["anchor"])
    first = reading[0]["anchor"]
    csum = sum(s["color"] for s in shapes)

    # ---- route choices, all read off the input grid --------------------------
    # which way to sweep: a wide grid invites a left-to-right sweep, a tall one
    # a top-to-bottom sweep, a square one gets done colour by colour.
    if wi > hi:
        order = sorted(shapes, key=lambda s: (s["left"], s["anchor"]))
    elif hi > wi:
        order = list(reading)
    else:
        order, done = [], []
        for s in reading:
            if s["color"] not in done:
                done.append(s["color"])
        for col in done:
            order.extend([s for s in reading if s["color"] == col])
    # and whether to start at the near end or the far end of that sweep
    backwards = ((first[0] + first[1] + nshapes) % 2 == 1)
    if backwards:
        order = order[::-1]

    # how big a bite to take out of each shape
    gran = (bgc + csum + nshapes) % 3          # 0 whole, 1 piece by piece, 2 row by row
    # tidy the vacated cells as you go, or sweep them all up at the very end
    tidy_end = ((bgc * 3 + csum) % 2 == 1)

    def units_of(shape):
        movers = sorted({cell for p in shape["pieces"] for cell in p})
        if not movers:
            return []
        if gran == 0:
            grps = [movers]
        elif gran == 1:
            grps = [sorted(p) for p in shape["pieces"]]
        else:
            byrow = {}
            for (r, c) in movers:
                byrow.setdefault(r, []).append((r, c))
            grps = [sorted(v) for v in byrow.values()]
        grps.sort(key=lambda g: (min(r for r, _ in g), min(c for _, c in g)))
        if backwards:
            grps.reverse()
        return grps

    # ---- emit ----------------------------------------------------------------
    ops, sels = [], []
    G = I.copy()                      # the grid as it stands, step by step
    vacated = []                      # for the tidy-up-at-the-end route

    for shape in order:
        col = shape["color"]
        for src in units_of(shape):
            dst = [(r, c + 1) for (r, c) in src if 0 <= c + 1 < wi]
            if col != 0:
                # slide this bite of the shape one cell right
                ops.append(22)
                sels.append(sel_of(src))
                for (r, c) in src:
                    G[r, c] = 0       # ARCLE zeroes the cells it grabbed
                for (r, c) in dst:
                    G[r, c] = col
            else:
                # colour 0 is invisible to ARCLE's object buffer -> draw it
                fresh = [p for p in dst if int(G[p[0], p[1]]) != 0]
                if fresh:
                    ops.append(0)
                    sels.append(sel_of(fresh))
                    for (r, c) in fresh:
                        G[r, c] = 0
            hole = sorted(set(src) - set(dst))       # what this bite left behind
            if tidy_end:
                vacated.extend(hole)
            else:
                need = [p for p in hole if int(G[p[0], p[1]]) != bgc]
                if need:
                    ops.append(bgc)
                    sels.append(sel_of(need))
                    for (r, c) in need:
                        G[r, c] = bgc

    if tidy_end:
        need = [p for p in sorted(set(vacated)) if int(G[p[0], p[1]]) != bgc]
        if need:
            ops.append(bgc)
            sels.append(sel_of(need))
            for (r, c) in need:
                G[r, c] = bgc

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])   # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 025d127b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 025d127b"
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
                                f"for task 025d127b"
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
                    f"Failed to build a complete episode for task 025d127b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"025d127b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
