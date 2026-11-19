"""
ARC Task: 1f642eb9 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    """Episode-level colour roles.

    bgc  : canvas background
    sqc  : the solid rectangle's colour
    ccols: palette the border markers are drawn from.
           0 is deliberately kept out of this palette: a marker is slid onto the
           rectangle with ARCLE Move ops, and ARCLE's object buffer keeps only
           NONZERO cells of a selection, so a 0-valued marker could not be
           grabbed/moved at all.  The rule itself is colour-independent, so
           restricting the marker palette costs nothing.
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    ccols = [c for c in cols if c not in (bgc, sqc) and c != 0]
    return {"bgc": bgc, "sqc": sqc, "ccols": ccols}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, sqc: int, ccols) -> dict:
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    ih = unifint(diff_lb, diff_ub, (2, min(h - 4, 2 * (h // 3))))
    iw = unifint(diff_lb, diff_ub, (2, min(w - 4, 2 * (w // 3))))
    loci = randint(2, h - ih - 2)
    locj = randint(2, w - iw - 2)
    ncells = 2 * ih + 2 * iw - 4
    # keep at least a few markers (so several sides are demonstrated) and cap the
    # count so the Move-based trajectory stays a sane length
    lob = min(4, ncells)
    upb = max(lob, min(ncells, 24))
    numcells = unifint(diff_lb, diff_ub, (lob, upb))
    outs = []
    ins = []
    c1 = choice((True, False))
    c2 = choice((True, False))
    c3 = choice((True, False))
    c4 = choice((True, False))
    for a in range(loci + (not c1), loci + ih - (not c2)):
        outs.append((a, 0))
        ins.append((a, locj))
    for a in range(loci + (not c3), loci + ih - (not c4)):
        outs.append((a, w - 1))
        ins.append((a, locj + iw - 1))
    for b in range(locj + c1, locj + iw - (c3)):
        outs.append((0, b))
        ins.append((loci, b))
    for b in range(locj + (c2), locj + iw - (c4)):
        outs.append((h - 1, b))
        ins.append((loci + ih - 1, b))
    inds = interval(0, ncells, 1)
    locs = sample(inds, numcells)
    pal = list(ccols)
    numc = unifint(diff_lb, diff_ub, (1, min(8, len(pal))))
    used = sample(pal, numc)
    outs = [e for j, e in enumerate(outs) if j in locs]
    ins = [e for j, e in enumerate(ins) if j in locs]
    cv = canvas(bgc, (h, w))
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    gi = fill(cv, sqc, bd)
    seq = [choice(used) for k in range(numcells)]
    for cc, loc in zip(seq, outs):
        gi = fill(gi, cc, {loc})
    go = tuple(e for e in gi)
    for cc, loc in zip(seq, ins):
        go = fill(go, cc, {loc})
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (read off I alone):

    A solid rectangle sits in the middle of the canvas and never moves.  Every
    other coloured cell sits on the grid's outer border, aligned with the
    rectangle, and SLIDES straight towards it until it lands on the rectangle's
    facing edge (it ends up on top of that edge cell, not one short of it).
    The border cell itself stays lit as well, so after each slide the vacated
    origin is re-lit with the same colour.

    Expressed as: one grab + a chain of unit Moves per marker (ARCLE restores the
    path from the grab-time snapshot, so nothing along the way is repainted),
    then a single Color op putting the marker back at its origin.
    """
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background: the grid corners are always canvas (markers live strictly
    # inside the borders' middle stretch, the rectangle is fully interior)
    bgc = int(I[0, 0])

    # --- locate the solid rectangle: largest single-colour 4-connected
    #     component that exactly fills its bounding box -------------------
    seen = np.zeros((hi, wi), dtype=bool)
    best = None
    for r0 in range(hi):
        for c0 in range(wi):
            if seen[r0, c0] or I[r0, c0] == bgc:
                continue
            col = int(I[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                            and int(I[nr, nc]) == col:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            rt, rb = min(rs), max(rs)
            cl, cr = min(cs), max(cs)
            if len(comp) != (rb - rt + 1) * (cr - cl + 1) or len(comp) < 4:
                continue
            if rt == 0 or cl == 0 or rb == hi - 1 or cr == wi - 1:
                continue  # the rectangle is always interior
            if best is None or len(comp) > best[0]:
                best = (len(comp), rt, rb, cl, cr)

    if best is None:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    _, rt, rb, cl, cr = best

    # --- the markers: every non-background cell outside the rectangle -----
    top, bottom, left, right = [], [], [], []
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v == bgc:
                continue
            if rt <= r <= rb and cl <= c <= cr:
                continue
            if cl <= c <= cr and r < rt:
                top.append((r, c, v))
            elif cl <= c <= cr and r > rb:
                bottom.append((r, c, v))
            elif rt <= r <= rb and c < cl:
                left.append((r, c, v))
            elif rt <= r <= rb and c > cr:
                right.append((r, c, v))

    top.sort(key=lambda t: t[1])
    bottom.sort(key=lambda t: t[1])
    left.sort(key=lambda t: t[0])
    right.sort(key=lambda t: t[0])

    plan = []
    for (r, c, v) in top:      # slide down onto the rectangle's top edge
        plan.append((r, c, v, 21, 1, 0, rt - r))
    for (r, c, v) in left:     # slide right onto the left edge
        plan.append((r, c, v, 22, 0, 1, cl - c))
    for (r, c, v) in right:    # slide left onto the right edge
        plan.append((r, c, v, 23, 0, -1, c - cr))
    for (r, c, v) in bottom:   # slide up onto the bottom edge
        plan.append((r, c, v, 20, -1, 0, r - rb))

    for (r, c, v, mop, dr, dc, steps) in plan:
        if steps <= 0:
            continue
        if v == 0:
            # a 0-valued cell cannot be grabbed as an ARCLE object; paint the
            # landing cell directly (never produced by this generator)
            ops.append(0)
            sels.append(sel_of([(r + dr * steps, c + dc * steps)]))
            continue
        # first Move GRABS this single-cell marker (snapshot of the rest of the
        # grid is taken here); every further Move carries an EMPTY selection so
        # ARCLE keeps the same object grabbed and restores the path behind it
        ops.append(mop)
        sels.append(sel_of([(r, c)]))
        for _ in range(steps - 1):
            ops.append(mop)
            sels.append(sel_of([]))
        # the only cell left at 0 is the marker's own origin, which the task
        # keeps lit in its original colour
        ops.append(v)
        sels.append(sel_of([(r, c)]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])  # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 1f642eb9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1f642eb9"
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
                                f"for task 1f642eb9"
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
                    f"Failed to build a complete episode for task 1f642eb9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1f642eb9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
