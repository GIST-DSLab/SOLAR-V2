"""
ARC Task: e76a88a6 (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. colours: bgc (canvas) and dmyc (the colour the blank copies are drawn in)
#    are the only two colour roles the rule depends on, so they are fixed for
#    the whole episode.  The multi-colour palette of the template is whatever
#    is left over, which is already determined by (bgc, dmyc).
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dmyc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dmyc": dmyc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, dmyc=None) -> dict:
    cols = interval(0, 10, 1)

    lo_h, hi_h = min(8, max_h), max(min(8, max_h), max_h)
    lo_w, hi_w = min(8, max_w), max(min(8, max_w), max_w)
    h = unifint(diff_lb, diff_ub, (lo_h, hi_h))
    w = unifint(diff_lb, diff_ub, (lo_w, hi_w))

    objh = unifint(diff_lb, diff_ub, (2, max(2, min(5, h))))
    objw = unifint(diff_lb, diff_ub, (2, max(2, min(5, w))))
    bounds = asindices(canvas(0, (objh, objw)))
    shp = {choice(totuple(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, max(2, len(bounds) - 2)))
    for j in range(nc):
        cands = totuple((bounds - shp) & mapply(dneighbors, shp))
        if len(cands) == 0:
            break
        ij = choice(cands)
        shp.add(ij)
    shp = normalize(shp)

    if bgc is None:
        bgc = choice(cols)
    remcols = remove(bgc, cols)
    if dmyc is None:
        dmyc = choice(remcols)
    remcols = remove(dmyc, remcols)

    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    shpp = shift(shp, (loci, locj))
    numco = unifint(diff_lb, diff_ub, (2, min(8, len(remcols))))
    colll = sample(remcols, numco)
    shppc = frozenset({(choice(colll), ij) for ij in shpp})
    while numcolors(shppc) == 1:
        shppc = frozenset({(choice(colll), ij) for ij in shpp})
    shppcn = normalize(shppc)

    gi = canvas(bgc, (h, w))
    gi = paint(gi, shppc)
    go = tuple(e for e in gi)

    ub = ((h * w) / (oh * ow)) // 2
    ub = max(1, ub)
    numlocs = unifint(diff_lb, diff_ub, (1, ub))
    cnt = 0
    fails = 0
    maxfails = 5 * numlocs
    idns = (asindices(gi) - shpp) - mapply(dneighbors, shpp)
    idns = sfilter(idns, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
    while cnt < numlocs and fails < maxfails:
        if len(idns) == 0:
            break
        loc = choice(totuple(idns))
        plcd = shift(shppcn, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(idns):
            go = paint(go, plcd)
            gi = fill(gi, dmyc, plcdi)
            cnt += 1
            idns = (idns - plcdi) - mapply(dneighbors, plcdi)
        else:
            fails += 1

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. the rule: one object is painted in several colours (the template); every
#    other object has the SAME shape but is drawn in a single dummy colour.
#    Each of those blanks has to be repainted with the template's colouring.
#
#    The route (which blank first, which way to sweep, stamp the template or
#    redraw it, colour by colour or object by object) is chosen from measurable
#    facts of the INPUT only -- the template's position, the number of blanks,
#    the number of colours in the template -- so one instance always yields one
#    route, and different instances yield different routes.  O is never read.
# ----------------------------------------------------------------------------
def derive_operations(I, O=None, examples=None):
    import numpy as np
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def submit_only():
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # --- 4-connected non-<bg> components -------------------------------------
    def comps_of(bg):
        seen = [[False] * w for _ in range(h)]
        out = []
        for r0 in range(h):
            for c0 in range(w):
                if int(I[r0, c0]) != bg and not seen[r0][c0]:
                    stack = [(r0, c0)]
                    seen[r0][c0] = True
                    cells = []
                    while stack:
                        rr, cc = stack.pop()
                        cells.append((rr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] \
                                    and int(I[nr, nc]) != bg:
                                seen[nr][nc] = True
                                stack.append((nr, nc))
                    out.append(sorted(cells))
        return out

    def ncolors_of(cells):
        return len({int(I[r, c]) for r, c in cells})

    def nshape(cells):
        r0 = min(r for r, c in cells)
        c0 = min(c for r, c in cells)
        return frozenset((r - r0, c - c0) for r, c in cells)

    # --- background: the colour for which the picture decomposes into exactly
    #     one multi-coloured object plus same-shaped mono-coloured copies ------
    counts = {}
    for r in range(h):
        for c in range(w):
            v = int(I[r, c])
            counts[v] = counts.get(v, 0) + 1
    cand_order = sorted(counts, key=lambda v: (-counts[v], v))

    tmpl = None
    dests = None
    for cand in cand_order:
        comps = comps_of(cand)
        if not comps:
            continue
        multi = [k for k in comps if ncolors_of(k) > 1]
        mono = [k for k in comps if ncolors_of(k) == 1]
        if len(multi) != 1 or not mono:
            continue
        tsh = nshape(multi[0])
        if len({int(I[k[0][0], k[0][1]]) for k in mono}) != 1:
            continue
        if any(nshape(k) != tsh for k in mono):
            continue
        tmpl, dests = multi[0], mono
        break

    if tmpl is None:                       # fallback: majority colour is bg
        comps = comps_of(cand_order[0])
        if not comps:
            return submit_only()
        tmpl = max(comps, key=ncolors_of)
        if ncolors_of(tmpl) < 2:
            return submit_only()
        dests = [k for k in comps if k is not tmpl]
    if not dests:
        return submit_only()               # nothing to fill in

    tr = min(r for r, c in tmpl)
    tc = min(c for r, c in tmpl)
    oh = max(r for r, c in tmpl) - tr + 1
    ow = max(c for r, c in tmpl) - tc + 1
    pattern = {(r - tr, c - tc): int(I[r, c]) for r, c in tmpl}

    origins = [(min(r for r, c in d), min(c for r, c in d)) for d in dests]
    ncopies = len(origins)
    pal = {}
    for col in pattern.values():
        pal[col] = pal.get(col, 0) + 1
    ncols_t = len(pal)

    # --- which blank first / which way to sweep (read off the template's seat)
    downward = (2 * tr + oh) <= h          # template up top -> work downwards
    rightward = (2 * tc + ow) <= w         # template on the left -> go right
    rows_first = (h >= w)                  # sweep along the grid's long axis
    if ncopies <= 3:                       # only a few: start with the closest
        origins.sort(key=lambda o: (max(abs(o[0] - tr), abs(o[1] - tc)), o[0], o[1]))
    elif rows_first:
        origins.sort(key=lambda o: (o[0] if downward else -o[0],
                                    o[1] if rightward else -o[1]))
    else:
        origins.sort(key=lambda o: (o[1] if rightward else -o[1],
                                    o[0] if downward else -o[0]))

    # --- stamp the template, redraw it blank by blank, or colour by colour ---
    pick = (ncopies + ncols_t + tr + tc) % 3
    if ncopies == 1 and pick == 0:         # copying a shape used once is silly
        pick = 1 + ((tr + tc) % 2)

    if (ncols_t + tc) % 2 == 0:
        col_order = sorted(pal)                                  # light to dark
    else:
        col_order = sorted(pal, key=lambda c: (-pal[c], c))      # biggest first

    sim = I.copy()

    if pick == 0:
        # take the template once, stamp it on every blank
        # (bbox selection intended: CopyI grabs the whole template rectangle)
        ops.append(28)
        sels.append([tr, tc, oh - 1, ow - 1])
        clip = I[tr:tr + oh, tc:tc + ow].copy()

        target = I.copy()
        for (dr0, dc0) in origins:
            for (ddr, ddc), col in pattern.items():
                target[dr0 + ddr, dc0 + ddc] = col

        for (dr0, dc0) in origins:
            ops.append(30)
            sels.append(sel_of([(dr0, dc0)]))
            for i in range(oh):
                for j in range(ow):
                    v = int(clip[i, j])
                    if v != 0 and dr0 + i < h and dc0 + j < w:
                        sim[dr0 + i, dc0 + j] = v
            # colour 0 never travels with a Paste -> lay those cells by hand
            zeros = sorted((dr0 + ddr, dc0 + ddc)
                           for (ddr, ddc), col in pattern.items()
                           if col == 0 and int(sim[dr0 + ddr, dc0 + ddc]) != 0)
            if zeros:
                ops.append(0)
                sels.append(sel_of(zeros))
                for rr, cc in zeros:
                    sim[rr, cc] = 0

        # a stamp's rectangle can clip a neighbouring object sitting in one of
        # the shape's holes -- put those objects' cells back
        fix = {}
        for r in range(h):
            for c in range(w):
                if int(sim[r, c]) != int(target[r, c]):
                    fix.setdefault(int(target[r, c]), []).append((r, c))
        for col in sorted(fix):
            cells = sorted(fix[col])
            ops.append(int(col))
            sels.append(sel_of(cells))
            for rr, cc in cells:
                sim[rr, cc] = col

    elif pick == 1:
        # finish one blank completely before moving to the next
        for (dr0, dc0) in origins:
            buckets = {}
            for (ddr, ddc), col in pattern.items():
                rr, cc = dr0 + ddr, dc0 + ddc
                if 0 <= rr < h and 0 <= cc < w and int(sim[rr, cc]) != col:
                    buckets.setdefault(col, []).append((rr, cc))
            for col in col_order:
                if col not in buckets:
                    continue
                cells = sorted(buckets[col])
                ops.append(int(col))
                sels.append(sel_of(cells))
                for rr, cc in cells:
                    sim[rr, cc] = col

    else:
        # one colour at a time, across every blank
        for col in col_order:
            cells = []
            for (dr0, dc0) in origins:
                for (ddr, ddc), pcol in sorted(pattern.items()):
                    if pcol != col:
                        continue
                    rr, cc = dr0 + ddr, dc0 + ddc
                    if 0 <= rr < h and 0 <= cc < w and int(sim[rr, cc]) != col:
                        cells.append((rr, cc))
            if cells:
                ops.append(int(col))
                sels.append(sel_of(cells))
                for rr, cc in cells:
                    sim[rr, cc] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task e76a88a6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e76a88a6"
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
                                f"for task e76a88a6"
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
                    f"Failed to build a complete episode for task e76a88a6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e76a88a6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
