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
from collections import Counter, deque

import numpy as np

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. colors: bgc is fixed for the whole episode; the object palette is fixed too
#    (the rule itself is colour-blind, but a stable palette keeps the episode
#    coherent -- and colour 0 is allowed to be an ordinary picture colour).
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC 025d127b with h/w caps and externally supplied colours)
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols) -> dict:
    hcap = max(5, min(30, int(max_h)))
    wcap = max(5, min(30, int(max_w)))
    ccols = list(ccols) if ccols else [c for c in range(10) if c != bgc][:1]

    while True:
        h = unifint(diff_lb, diff_ub, (5, hcap))
        w = unifint(diff_lb, diff_ub, (5, wcap))
        nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
        succ = 0
        tr = 0
        maxtr = 5 * nobjs
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        inds = asindices(gi)
        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = randint(3, 6)
            ow = randint(3, 6)
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
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
                col = choice(ccols)
                gi = fill(gi, col, inobj)
                go = fill(go, col, outobj)
        if succ > 0:
            return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
#
#    Rule (read off the examples): every same-colour 8-connected figure is made
#    of several 4-connected strokes (a top bar, loose diagonal pips, a bottom
#    bar).  The stroke that reaches furthest right is the anchor and stays put;
#    every other stroke of that figure slides one cell to the right.
#
#    The ROUTE (which figure first, which stroke first, when the vacated cells
#    get repainted) is chosen from measurements of the INPUT only, so the same
#    picture always yields the same route while different pictures are solved
#    in visibly different orders.
# ----------------------------------------------------------------------------
def _example_input(ex):
    if isinstance(ex, dict):
        return ex.get("input")
    return ex[0]


def _infer_bgc(I, examples=None):
    cnt = Counter(I.flatten().tolist())
    if examples:
        votes = []
        for ex in examples:
            gi = _example_input(ex)
            if gi is None:
                continue
            gi = np.asarray(gi, dtype=int)
            votes.append(Counter(gi.flatten().tolist()).most_common(1)[0][0])
        if votes and len(set(votes)) == 1 and votes[0] in cnt:
            return int(votes[0])
    return int(cnt.most_common(1)[0][0])


def _figures(I, bgc):
    """same-colour 8-connected figures"""
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    figs = []
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
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                and int(I[nr, nc]) == col:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            figs.append({"color": col, "cells": sorted(cells)})
    return figs


def _strokes(cells):
    """4-connected strokes inside one figure"""
    pool = set(cells)
    strokes = []
    while pool:
        start = min(pool)
        pool.discard(start)
        q = deque([start])
        part = []
        while q:
            rr, cc = q.popleft()
            part.append((rr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (rr + dr, cc + dc)
                if nb in pool:
                    pool.discard(nb)
                    q.append(nb)
        part.sort()
        strokes.append({"cells": part,
                        "right": max(c for _, c in part),
                        "top": part[0]})
    return strokes


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    bgc = _infer_bgc(I, examples)
    figs = _figures(I, bgc)

    ops, sels = [], []

    if not figs:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])   # full-grid rectangle: submit
        return ops, sels

    # ---- route choices, measured from the input picture ---------------------
    n_fig = len(figs)
    n_cells = sum(len(f["cells"]) for f in figs)
    anchor0 = min(f["cells"][0] for f in figs)        # top-left-most painted cell
    palette = sorted({f["color"] for f in figs})

    order_mode = (anchor0[0] + anchor0[1] + n_fig) % 4      # which figure first
    stroke_mode = (n_cells + palette[0]) % 3                # which stroke first
    repair_mode = (n_fig + len(palette) + bgc) % 3          # when to patch holes

    if order_mode == 0:                      # reading order, top-left figure first
        figs.sort(key=lambda f: f["cells"][0])
    elif order_mode == 1:                    # backwards, bottom-right figure first
        figs.sort(key=lambda f: f["cells"][0], reverse=True)
    elif order_mode == 2:                    # sweep the columns left to right
        figs.sort(key=lambda f: (f["cells"][0][1], f["cells"][0][0]))
    else:                                    # biggest figure first
        figs.sort(key=lambda f: (-len(f["cells"]), f["cells"][0]))

    deferred = []          # vacated cells still waiting for one bgc repaint

    for fig in figs:
        col = fig["color"]
        strokes = _strokes(fig["cells"])
        # the stroke reaching furthest right is the anchor -> it does not move
        keeper = max(strokes, key=lambda s: (s["right"], len(s["cells"]),
                                             -s["top"][0], -s["top"][1]))
        movers = [s for s in strokes if s is not keeper]

        if stroke_mode == 0:                 # top stroke first
            movers.sort(key=lambda s: s["top"])
        elif stroke_mode == 1:               # bottom stroke first
            movers.sort(key=lambda s: s["top"], reverse=True)
        else:                                # furthest from the anchor first
            movers.sort(key=lambda s: (s["right"], s["top"]))

        fig_holes = []
        for s in movers:
            cells = s["cells"]
            dst = [(r, c + 1) for (r, c) in cells]
            hole = sorted(set(cells) - set(dst))       # the column it vacates

            if col != 0:
                ops.append(22)                          # slide the stroke right
                sels.append(sel_of(cells))
            else:
                # colour 0 is "nothing there" to ARCLE's object buffer, so this
                # stroke cannot be carried: put it down where it lands.
                fresh = sorted(set(dst) - set(cells))
                if fresh:
                    ops.append(0)
                    sels.append(sel_of(fresh))

            if hole and bgc != 0:
                if repair_mode == 0:
                    ops.append(bgc)                     # patch straight away
                    sels.append(sel_of(hole))
                else:
                    fig_holes.extend(hole)

        if fig_holes and bgc != 0:
            if repair_mode == 1:
                ops.append(bgc)                         # one patch per figure
                sels.append(sel_of(sorted(set(fig_holes))))
            else:
                deferred.extend(fig_holes)

    if deferred and bgc != 0:
        ops.append(bgc)                                 # one patch for the lot
        sels.append(sel_of(sorted(set(deferred))))

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
