"""
ARC Task: 321b1fc6 (RE-ARC) — LLM-generated grid_maker
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
    """Roles: the canvas background and the colour that marks the blank copies.
    The template's own palette is re-drawn per instance (the rule is colour-agnostic
    about it) which is what lets colour 0 show up as ordinary picture colour."""
    cols = list(range(10))
    bgc = random.choice(cols)
    dmyc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dmyc": dmyc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dmyc) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(8, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(8, max_w), max_w))
    objh = unifint(diff_lb, diff_ub, (2, max(2, min(5, h))))
    objw = unifint(diff_lb, diff_ub, (2, max(2, min(5, w))))
    bounds = asindices(canvas(0, (objh, objw)))
    shp = {random.choice(totuple(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, max(2, len(bounds) - 2)))
    for j in range(nc):
        cands = totuple((bounds - frozenset(shp)) & mapply(dneighbors, frozenset(shp)))
        if len(cands) == 0:
            break
        shp.add(random.choice(cands))
    shp = normalize(frozenset(shp))
    remcols = remove(bgc, cols)
    remcols = remove(dmyc, remcols)
    oh, ow = shape(shp)
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    shpp = shift(shp, (loci, locj))
    numco = unifint(diff_lb, diff_ub, (2, 8))
    colll = random.sample(list(remcols), numco)
    shppc = frozenset({(random.choice(colll), ij) for ij in shpp})
    while numcolors(shppc) == 1:
        shppc = frozenset({(random.choice(colll), ij) for ij in shpp})
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
        loc = random.choice(totuple(idns))
        plcd = shift(shppcn, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(idns):
            go = paint(go, plcd)
            gi = fill(gi, dmyc, plcdi)
            cnt += 1
            idns = (idns - plcdi) - mapply(dneighbors, plcdi)
        else:
            fails += 1
    go = fill(go, bgc, shpp)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Rule (read off I alone): one component is multi-coloured -- the template.
    Every other component is a blank copy of the same shape. Give each blank copy
    the template's colours, then wipe the template itself back to background.

    The ROUTE (which copy first, stamp vs. paint, base-coat vs. straight, erase the
    original first or last) is chosen from a signature of the input, so the same
    puzzle always gets the same route and different puzzles get different ones.
    O is never consulted."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    def comps_of(bg):
        seen = [[False] * w for _ in range(h)]
        out = []
        for r in range(h):
            for c in range(w):
                if int(I[r][c]) != bg and not seen[r][c]:
                    st = [(r, c)]
                    seen[r][c] = True
                    cells = []
                    while st:
                        rr, cc = st.pop()
                        cells.append((rr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and (not seen[nr][nc]) and int(I[nr][nc]) != bg:
                                seen[nr][nc] = True
                                st.append((nr, nc))
                    out.append(sorted(cells))
        return out

    def norm(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    def ncols(cells):
        return len({int(I[r][c]) for r, c in cells})

    # --- identify background, the multicoloured template, and the blank copies -------------
    bgc = None
    tmpl = None
    copies = []
    for cand, _ in Counter(I.flatten().tolist()).most_common():
        cs = comps_of(cand)
        multi = [k for k in cs if ncols(k) > 1]
        if len(multi) != 1:
            continue
        t = multi[0]
        rest = [k for k in cs if k is not t]
        tn = norm(t)
        if any(ncols(k) != 1 or norm(k) != tn for k in rest):
            continue
        if len({int(I[k[0][0]][k[0][1]]) for k in rest}) > 1:
            continue
        bgc, tmpl, copies = cand, t, rest
        break
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
        cs = comps_of(bgc)
        tmpl = max(cs, key=ncols)
        tn = norm(tmpl)
        copies = [k for k in cs if k is not tmpl and norm(k) == tn]
    bgc = int(bgc)

    tr = min(r for r, _ in tmpl)
    tc = min(c for _, c in tmpl)
    th = max(r for r, _ in tmpl) - tr + 1
    tw = max(c for _, c in tmpl) - tc + 1
    rel = {(r - tr, c - tc): int(I[r][c]) for r, c in tmpl}
    holes = [(i, j) for i in range(th) for j in range(tw) if (i, j) not in rel]
    zero_rel = sorted(k for k, v in rel.items() if v == 0)
    # the stamp's bounding rectangle may only be copied wholesale if its gaps are pure background
    clip_clean = all(int(I[tr + i][tc + j]) == bgc for i, j in holes)

    pal = sorted(set(rel.values()))
    cnt_by_col = Counter(rel.values())
    base = min(pal, key=lambda v: (-cnt_by_col[v], v))
    dmy = int(I[copies[0][0][0]][copies[0][0][1]]) if copies else bgc

    # --- route signature: everything in it is visible in the input -------------------------
    sig = (len(copies) * 31 + tr * 7 + tc * 13 + th * 17 + tw * 19
           + len(rel) * 37 + len(pal) * 23 + sum(pal) * 11 + bgc * 29 + dmy * 41)
    strategy = sig % 4            # 0 stamp the first one around, 1 draw each, 2 base-coat, 3 by colour
    order_mode = (sig // 4) % 6   # which copy to deal with first / which way to sweep
    col_mode = (sig // 24) % 3    # in which order the colours get laid down
    erase_first = ((sig // 72) % 2) == 1   # clear the original before or after the copies

    def ul(cell_list):
        return (min(r for r, _ in cell_list), min(c for _, c in cell_list))

    def copy_key(cell_list):
        cr, cc = ul(cell_list)
        if order_mode == 0:
            return (cr, cc)            # top-left down
        if order_mode == 1:
            return (-cr, -cc)          # bottom-right up
        if order_mode == 2:
            return (cc, cr)            # left-hand column across
        if order_mode == 3:
            return (-cc, -cr)          # right-hand column back
        d = abs(cr - tr) + abs(cc - tc)
        if order_mode == 4:
            return (d, cr, cc)         # outwards from the template
        return (-d, cr, cc)            # far ones first

    ordered = sorted(copies, key=copy_key)

    def pal_order(colours):
        if col_mode == 0:
            return sorted(colours)
        if col_mode == 1:
            return sorted(colours, reverse=True)
        return sorted(colours, key=lambda v: (-cnt_by_col[v], v))

    G = I.copy()
    ops, sels = [], []
    copied = False

    def emit_color(v, cells):
        v = int(v)
        cells = sorted({(r, c) for r, c in cells if int(G[r][c]) != v})
        if cells:
            ops.append(v)
            sels.append(sel_of(cells))
            for r, c in cells:
                G[r][c] = v

    def stamp_copy(cr, cc):
        nonlocal copied
        if not copied:
            # bbox selection is intentional: the whole stamp rectangle is copied from the INPUT
            ops.append(28)
            sels.append([tr, tc, th - 1, tw - 1])
            copied = True
        ops.append(30)
        sels.append([cr, cc, 0, 0])
        for (i, j), v in rel.items():
            if v != 0:
                G[cr + i][cc + j] = v
        for i, j in holes:
            hv = int(I[tr + i][tc + j])
            if hv != 0:
                G[cr + i][cc + j] = hv
        # Paste carries no 0s - the stamp's black pixels are put down by hand, if it has any
        if zero_rel:
            emit_color(0, [(cr + i, cc + j) for i, j in zero_rel])

    def draw_copy(cr, cc):
        for v in pal_order(pal):
            emit_color(v, [(cr + i, cc + j) for (i, j), vv in rel.items() if vv == v])

    def basecoat_copy(cr, cc):
        # lay the shape's dominant colour over the whole blank, then draw the rest on top
        emit_color(base, [(cr + i, cc + j) for (i, j) in rel])
        for v in pal_order([v for v in pal if v != base]):
            emit_color(v, [(cr + i, cc + j) for (i, j), vv in rel.items() if vv == v])

    def erase_template():
        emit_color(bgc, [(r, c) for r, c in sorted(tmpl)])

    if erase_first:
        erase_template()

    if strategy == 3:
        # one colour at a time, across every copy at once
        for v in pal_order(pal):
            cells = []
            for cell_list in ordered:
                cr, cc = ul(cell_list)
                cells += [(cr + i, cc + j) for (i, j), vv in rel.items() if vv == v]
            emit_color(v, cells)
    else:
        for cell_list in ordered:
            cr, cc = ul(cell_list)
            # a bbox paste is only safe when nothing foreign sits in the stamp's gaps here
            safe = clip_clean and all(int(G[cr + i][cc + j]) == bgc for i, j in holes)
            if strategy == 0 and safe:
                stamp_copy(cr, cc)
            elif strategy == 2:
                basecoat_copy(cr, cc)
            else:
                draw_copy(cr, cc)

    if not erase_first:
        erase_template()

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
                        f"num_examples+1 ({num_examples + 1}) for task 321b1fc6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 321b1fc6"
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
                                f"for task 321b1fc6"
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
                    f"Failed to build a complete episode for task 321b1fc6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"321b1fc6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
