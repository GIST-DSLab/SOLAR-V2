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
# 1. Episode-level colors
#    The rule ("every 4-connected piece of an object except the right-most one
#    slides one cell to the right") does not depend on which colors are used,
#    only on background vs. foreground.  Still, fix bgc AND the foreground
#    palette once per episode so every instance shares one color scheme.
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    remcols = random.sample(remcols, len(remcols))     # fixed shuffled palette
    return {"bgc": bgc, "ccols": remcols}


# ----------------------------------------------------------------------------
# 2. Generator (RE-ARC 025d127b) with 30 -> max_h/max_w and injected colors
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        ccols = [c for c in cols if c != bgc]

    mh = max(5, int(max_h))
    mw = max(5, int(max_w))

    h = unifint(diff_lb, diff_ub, (5, mh))
    w = unifint(diff_lb, diff_ub, (5, mw))

    numcols = unifint(diff_lb, diff_ub, (1, len(ccols)))
    ccols_used = list(ccols)[:numcols]

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
            col = choice(ccols_used)
            gi = fill(gi, col, inobj)
            go = fill(go, col, outobj)
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
#
# Rule (measured from I, matches the verifier exactly):
#   * an "object" = diagonally-connected same-color blob (the parallelogram
#     outline the generator draws: top line, two diagonals, bottom line).
#   * split that object into its 4-CONNECTED same-color pieces.
#   * the piece that reaches the largest column (bottom line + the last cell of
#     the right diagonal, which sits directly above it) stays where it is.
#   * EVERY other piece slides exactly one cell to the RIGHT.
#
#   Piece sizes/counts vary with the object's oh/ow (3..6 each), so everything
#   below is measured from I, never hardcoded.
#
#   A slide is emitted as a real MoveR on the piece's own cells, then the one
#   vacated column of that piece is repaired with a single Color(bgc).
#   Exception: ARCLE's object buffer keeps only NON-ZERO cells, so a piece whose
#   color is 0 cannot be carried by Move at all; such a piece is drawn at its
#   new place with Color0 and its old trace cleared with Color(bgc).
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background = the color the generator paints the canvas with; objects cover
    # at most ~25% of the grid here, so the majority color is reliably bgc.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- 4-connected same-color pieces of the foreground -------------------
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
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

    # ---- group pieces into whole objects (same color + diagonal touch) ------
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

    objects = []
    for _, idxs in groups.items():
        anchor = min(min(comps[i]["cells"]) for i in idxs)
        objects.append((anchor, idxs))
    objects.sort()                                  # reading order of objects

    ops, sels = [], []

    for _, idxs in objects:
        # the anchor piece: the one extending furthest right -> it does NOT move
        keeper = max(idxs, key=lambda i: (max(c for _, c in comps[i]["cells"]),
                                          -min(r for r, _ in comps[i]["cells"])))
        movers = [i for i in idxs if i != keeper]
        movers.sort(key=lambda i: min(comps[i]["cells"]))

        for i in movers:
            cells = comps[i]["cells"]
            col = comps[i]["color"]
            dst = [(r, c + 1) for (r, c) in cells]
            hole = sorted(set(cells) - set(dst))     # the column it vacates

            if col != 0:
                # slide this piece one cell right
                ops.append(22)
                sels.append(sel_of(cells))
                if bgc != 0 and hole:
                    ops.append(bgc)
                    sels.append(sel_of(hole))
            else:
                # color 0 is invisible to ARCLE's object buffer -> draw it
                fresh = sorted(set(dst) - set(cells))
                if fresh:
                    ops.append(0)
                    sels.append(sel_of(fresh))
                if hole:
                    ops.append(bgc)
                    sels.append(sel_of(hole))

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
