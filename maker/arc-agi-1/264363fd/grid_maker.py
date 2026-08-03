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
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # Roles: bgc = canvas, sqc = the plain squares, and the three template colours
    # (cpcol centre, nbhcol ring, linc spikes/rays). The rule is "stamp the template on
    # every cpcol marker and shoot its spike rays", so every colour role must be stable
    # across the episode. The three template colours are kept non-zero and distinct so the
    # 3x3 core is an unambiguous, fully-opaque stamp.
    nz = list(range(1, 10))
    linc, cpcol, nbhcol = random.sample(nz, 3)
    rest = [c for c in range(10) if c not in (linc, cpcol, nbhcol)]
    bgc, sqc = random.sample(rest, 2)
    return {"bgc": bgc, "sqc": sqc, "linc": linc, "cpcol": cpcol, "nbhcol": nbhcol}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, sqc: int, linc: int, cpcol: int, nbhcol: int) -> dict:
    cp = (2, 2)
    neighs = neighbors(cp)
    o1 = shift(frozenset({(0, 1), (-1, 1)}), (1, 1))
    o2 = shift(frozenset({(1, 0), (1, -1)}), (1, 1))
    o3 = shift(frozenset({(2, 1), (3, 1)}), (1, 1))
    o4 = shift(frozenset({(1, 2), (1, 3)}), (1, 1))
    mpr = {o1: (-1, 0), o2: (0, -1), o3: (1, 0), o4: (0, 1)}
    h = unifint(diff_lb, diff_ub, (min(15, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(15, max_w), max_w))
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
        gh = randint(5, max(5, h // 2 + 1))
        gw = randint(5, max(5, w // 2 + 1))
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


def derive_operations(I, O):
    """
    I holds one small multi-coloured 'template' (a cpcol centre, an nbhcol ring, and
    1..4 linc spikes sticking out) plus several plain squares, each carrying one or more
    lone cpcol markers.

    O: the template is gone, and on every marker the template's 3x3 core is stamped and
    each spike is shot outward as a linc ray until it hits the square's border.

    Plan: clipboard <- template core, erase the template, then per square, per marker:
    Paste the core, then draw one ray per spike direction.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    D4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # --- background = colour of the single-colour region with the largest bounding box ---
    seen = np.zeros((hi, wi), bool)
    best_area, bgc = -1, int(I[0, 0])
    for r0 in range(hi):
        for c0 in range(wi):
            if seen[r0, c0]:
                continue
            col = int(I[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            rmin = rmax = r0
            cmin = cmax = c0
            while stack:
                r, c = stack.pop()
                if r < rmin: rmin = r
                if r > rmax: rmax = r
                if c < cmin: cmin = c
                if c > cmax: cmax = c
                for dr, dc in D4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            area = (rmax - rmin + 1) * (cmax - cmin + 1)
            if area > best_area:
                best_area, bgc = area, col

    # --- non-background components: the squares + the one template ---
    seen2 = np.zeros((hi, wi), bool)
    comps = []
    for r0 in range(hi):
        for c0 in range(wi):
            if seen2[r0, c0] or I[r0, c0] == bgc:
                continue
            stack = [(r0, c0)]
            seen2[r0, c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in D4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen2[nr, nc] and I[nr, nc] != bgc:
                        seen2[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)

    if not comps:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # the template is by far the smallest component (<=17 cells vs >=25 for any square)
    ti = min(range(len(comps)), key=lambda k: len(comps[k]))
    tmpl = comps[ti]

    tcols = Counter(int(I[r, c]) for r, c in tmpl)
    cp_candidates = [col for col, n in tcols.items() if n == 1]
    cpcol = cp_candidates[0]
    cr, cc = [(r, c) for r, c in tmpl if int(I[r, c]) == cpcol][0]
    nbhcol = int(I[cr - 1, cc - 1])                     # a ring corner is never a spike
    rest = [col for col in tcols if col not in (cpcol, nbhcol)]
    linc = rest[0] if rest else nbhcol

    spikes = []
    for dr, dc in D4:
        rr, ccx = cr + 2 * dr, cc + 2 * dc
        if 0 <= rr < hi and 0 <= ccx < wi and int(I[rr, ccx]) == linc:
            spikes.append((dr, dc))

    # --- gather the squares and their markers ---
    squares = []
    for k, comp in enumerate(comps):
        if k == ti:
            continue
        markers = sorted([(r, c) for r, c in comp if int(I[r, c]) == cpcol])
        if not markers:
            continue
        rs = [x[0] for x in comp]; cs = [x[1] for x in comp]
        squares.append((min(rs), min(cs), max(rs), max(cs), markers))

    # 1. the template's 3x3 core is the stamp
    if squares:
        ops.append(28); sels.append([cr - 1, cc - 1, 2, 2])

    # 2. the template itself disappears: its core, then each spike tip
    ops.append(int(bgc)); sels.append([cr - 1, cc - 1, 2, 2])
    for dr, dc in spikes:
        ops.append(int(bgc)); sels.append([cr + 2 * dr, cc + 2 * dc, 0, 0])

    # 3. one square at a time, one marker at a time: stamp, then shoot its rays
    for (r0, c0, r1, c1, markers) in squares:
        for (mr, mc) in markers:
            ops.append(30); sels.append([mr - 1, mc - 1, 0, 0])
            for dr, dc in spikes:
                if dr == -1:
                    if mr - 2 >= r0:
                        ops.append(int(linc)); sels.append([r0, mc, (mr - 2) - r0, 0])
                elif dr == 1:
                    if mr + 2 <= r1:
                        ops.append(int(linc)); sels.append([mr + 2, mc, r1 - (mr + 2), 0])
                elif dc == -1:
                    if mc - 2 >= c0:
                        ops.append(int(linc)); sels.append([mr, c0, 0, (mc - 2) - c0])
                else:
                    if mc + 2 <= c1:
                        ops.append(int(linc)); sels.append([mr, mc + 2, 0, c1 - (mc + 2)])

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
