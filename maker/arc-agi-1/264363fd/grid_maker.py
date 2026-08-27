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
# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
def sample_colors(num_examples=None) -> dict:
    """bgc / sqc / linc are three distinct colors (generator: sample(cols, 3));
    cpcol and nbhcol are each drawn from the remaining colors and MAY coincide."""
    import random
    cols = list(range(10))
    bgc, sqc, linc = random.sample(cols, 3)
    remcols = [c for c in cols if c not in (bgc, sqc, linc)]
    cpcol = random.choice(remcols)
    nbhcol = random.choice(remcols)
    return {"bgc": bgc, "sqc": sqc, "linc": linc,
            "cpcol": cpcol, "nbhcol": nbhcol}


def generate(diff_lb, diff_ub, max_h, max_w,
             bgc=1, sqc=8, linc=3, cpcol=4, nbhcol=6) -> dict:
    """Faithful port of generate_264363fd: colors are injected, the hardcoded
    30 bounds become max_h / max_w, and degenerate draws (no square placed, or
    a grid whose background is not the dominant / largest-bbox color, which
    would break the reference rule's own object detection) are re-rolled."""
    from random import randint, choice, sample

    if max_h < 11 or max_w < 11:
        raise ValueError("grid too small for task 264363fd")

    cp = (2, 2)
    neighs = neighbors(cp)
    o1 = shift(frozenset({(0, 1), (-1, 1)}), (1, 1))
    o2 = shift(frozenset({(1, 0), (1, -1)}), (1, 1))
    o3 = shift(frozenset({(2, 1), (3, 1)}), (1, 1))
    o4 = shift(frozenset({(1, 2), (1, 3)}), (1, 1))
    mpr = {o1: (-1, 0), o2: (0, -1), o3: (1, 0), o4: (0, 1)}
    hbounds = (min(15, max_h), max_h)
    wbounds = (min(15, max_w), max_w)

    for _attempt in range(60):
        h = unifint(diff_lb, diff_ub, hbounds)
        w = unifint(diff_lb, diff_ub, wbounds)
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
        numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 100)))
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
            loc2p = choice(totuple(cands))
            g1 = canvas(sqc, (gh, gw))
            g2 = canvas(sqc, (gh, gw))
            ginds = asindices(g1)
            gindsfull = asindices(g1)
            bck = shift(ginds, loc2p)
            if bck.issubset(inds):
                noccs = unifint(diff_lb, diff_ub, (1, max(1, (gh * gw) // 25)))
                succ2 = 0
                tr2 = 0
                maxtr2 = 5 * noccs
                while succ2 < noccs and tr2 < maxtr2:
                    tr2 += 1
                    cands2 = sfilter(ginds, lambda ij: ij[0] <= gh - 5 and ij[1] <= gw - 5)
                    if len(cands2) == 0:
                        break
                    loc2 = choice(totuple(cands2))
                    lns2 = merge(frozenset({
                        shoot(add(cp, add(loc2, mpr[spike])), mpr[spike]) for spike in spikes
                    }))
                    lns2 = lns2 & gindsfull
                    plcd2 = shift(obj, loc2)
                    plcd2i = toindices(plcd2)
                    if plcd2i.issubset(ginds) and lns2.issubset(ginds | ofcolor(g2, linc)) \
                            and len(lns2 - plcd2i) > 0:
                        succ2 += 1
                        ginds = ((ginds - plcd2i) - mapply(neighbors, plcd2i)) - lns2
                        g1 = fill(g1, cpcol, {add(cp, loc2)})
                        g2 = paint(g2, plcd2)
                        g2 = fill(g2, linc, lns2)
                if succ2 > 0:
                    succ += 1
                    inds = (inds - bck) - outbox(bck)
                    objfull1 = shift(asobject(g1), loc2p)
                    objfull2 = shift(asobject(g2), loc2p)
                    gi = paint(gi, objfull1)
                    go = paint(go, objfull2)
        if succ == 0:
            continue
        # the rule identifies the background as the most common color AND as the
        # color of the widest-bounding-box mono object; re-roll if a freak draw
        # of squares would outvote it
        if mostcolor(gi) != bgc:
            continue
        if mostcolor(argmax(objects(gi, T, F, F), fork(multiply, height, width))) != bgc:
            continue
        return {'input': gi, 'output': go}
    raise ValueError("could not build a valid 264363fd instance")


def derive_operations(I, O):
    """The legend (smallest non-background blob) is a dot with a halo and 1-4
    two-cell spikes.  Every legend-colored dot inside a plain square grows the
    same halo, and each spike direction is shot out as a ray to the square's
    border.  The legend itself is then wiped."""
    import numpy as np
    from collections import deque
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            uniq = sorted({(int(r), int(c)) for r, c in cells})
            return {"cells": [[r, c] for r, c in uniq]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    def comps(pred, same_color):
        seen = np.zeros((hi, wi), dtype=bool)
        out = []
        for r in range(hi):
            for c in range(wi):
                if seen[r, c] or not pred(r, c):
                    continue
                col = I[r, c]
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and pred(ny, nx):
                            if (not same_color) or I[ny, nx] == col:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                out.append(cells)
        return out

    # background = color of the mono component with the largest bounding box
    best, bgc = -1, int(I[0, 0])
    for cells in comps(lambda r, c: True, True):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        if area > best:
            best, bgc = area, int(I[cells[0][0], cells[0][1]])

    blobs = comps(lambda r, c: I[r, c] != bgc, False)
    blobs.sort(key=len)
    legend = blobs[0]                      # smallest blob = the legend key
    squares = sorted(blobs[1:], key=lambda cs: (min(p[0] for p in cs), min(p[1] for p in cs)))

    # legend anatomy: 3x3 core (cells with >=2 orthogonal partners) + spike tips
    lset = set(legend)
    core = [p for p in legend
            if sum(((p[0] + dy, p[1] + dx) in lset)
                   for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))) > 1]
    cr = min(p[0] for p in core) + (max(p[0] for p in core) - min(p[0] for p in core) + 1) // 2
    cc = min(p[1] for p in core) + (max(p[1] for p in core) - min(p[1] for p in core) + 1) // 2
    cpcol = int(I[cr, cc])
    nbhcol = int(I[cr - 1, cc - 1])        # a diagonal halo cell is never a spike
    tips = [p for p in legend if p not in core]
    linc = int(I[tips[0][0], tips[0][1]])
    dirs = [((p[0] - cr) // 2, (p[1] - cc) // 2) for p in tips]
    dirs = [d for d in ((-1, 0), (1, 0), (0, -1), (0, 1)) if d in dirs]

    ops, sels = [], []

    for cells in squares:
        r0 = min(p[0] for p in cells); r1 = max(p[0] for p in cells)
        c0 = min(p[1] for p in cells); c1 = max(p[1] for p in cells)
        marks = sorted(p for p in cells if I[p[0], p[1]] == cpcol)
        for (mr, mc) in marks:
            halo = [(mr + dy, mc + dx)
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
            halo = [(r, c) for r, c in halo if r0 <= r <= r1 and c0 <= c <= c1]
            ops.append(nbhcol); sels.append(sel_of(halo))
            for dy, dx in dirs:
                ray, r, c = [], mr + dy, mc + dx
                while r0 <= r <= r1 and c0 <= c <= c1:
                    ray.append((r, c)); r += dy; c += dx
                if ray:
                    ops.append(linc); sels.append(sel_of(ray))

    ops.append(bgc); sels.append(sel_of(legend))   # the key has done its job
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
