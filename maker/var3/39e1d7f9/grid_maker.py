"""
ARC Task: 39e1d7f9 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    """Background, separator-line colour and marker colour are episode-wide roles.

    The colours of the template's body cells are NOT fixed: the rule only copies whatever
    colours it finds, so they may (and do) differ from instance to instance.
    """
    cols = list(range(10))
    bgc, linc, dotc = random.sample(cols, 3)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, linc: int, dotc: int) -> dict:
    cols = interval(0, 10, 1)
    hub = max(5, min(10, (max_h - 1) // 2))
    wub = max(5, min(10, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (5, hub))
    w = unifint(diff_lb, diff_ub, (5, wub))
    remcols = difference(cols, (bgc, linc, dotc))
    gi = canvas(bgc, (h, w))
    loci = randint(1, h - 2)
    locj = randint(1, w - 2)
    if h == 5:
        loci = choice((1, h - 2))
    if w == 5:
        locj = choice((1, w - 2))
    npix = unifint(diff_lb, diff_ub, (1, 8))
    ncols = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, ncols)
    candsss = neighbors((loci, locj))
    pixs = {(loci, locj)}
    for k in range(npix):
        cands = totuple((mapply(dneighbors, pixs) & candsss) - pixs)
        if len(cands) == 0:
            break
        pixs.add(choice(cands))
    pixs = totuple(remove((loci, locj), pixs))
    obj = {(choice(ccols), ij) for ij in pixs}
    gi = fill(gi, dotc, {(loci, locj)})
    gi = paint(gi, obj)
    go = tuple(e for e in gi)
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(pixs) + 1))))
    succ = 0
    tr = 0
    maxtr = 6 * noccs
    inds = ofcolor(gi, bgc) - mapply(dneighbors, neighbors((loci, locj)))
    objn = shift(obj, (-loci, -locj))
    triedandfailed = set()
    while (tr < maxtr and succ < noccs) or succ == 0:
        lopcands = totuple(inds - triedandfailed)
        if len(lopcands) == 0:
            break
        tr += 1
        loci, locj = choice(lopcands)
        plcd = shift(objn, (loci, locj))
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            inds = inds - (plcdi | {(loci, locj)})
            succ += 1
            gi = fill(gi, dotc, {(loci, locj)})
            go = fill(go, dotc, {(loci, locj)})
            go = paint(go, plcd)
        else:
            triedandfailed.add((loci, locj))
    hfac = unifint(diff_lb, diff_ub, (1, max(1, (max_h - h + 1) // h)))
    wfac = unifint(diff_lb, diff_ub, (1, max(1, (max_w - w + 1) // w)))
    fullh = hfac * h + h - 1
    fullw = wfac * w + w - 1
    gi2 = canvas(linc, (fullh, fullw))
    go2 = canvas(linc, (fullh, fullw))
    bd = asindices(canvas(-1, (hfac, wfac)))
    for a in range(h):
        for b in range(w):
            c = gi[a][b]
            gi2 = fill(gi2, c, shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    for a in range(h):
        for b in range(w):
            c = go[a][b]
            go2 = fill(go2, c, shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    gi, go = gi2, go2
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Copy the multi-coloured model block-shape onto every lone marker block.

    Everything (block-grid pitch, background, marker colour, which block-region is the
    model, where its marker sits, where the copies go) is read from I alone.

    Two honest routes exist and the input decides which one is taken, the way two people
    would differ: with a lot of blocks to place it pays to copy the model once and stamp
    it (CopyI + Paste per marker, then put down what a Paste cannot carry); with only a
    handful it is quicker to read the shape off and paint it by hand, one colour at a
    time.  The sweep order is also read off the input: scan along the block grid's longer
    side, starting from the side the model itself sits on and working away from it.
    (No background base is laid: at every destination the shape's footprint is already
    background apart from the marker itself, so a bgc fill there would paint cells the
    colour they already hold.)
    """
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in cells]}

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    submit_sel = [0, 0, hi - 1, wi - 1]          # whole grid: exactly this rectangle

    # ---- 1. block-grid geometry: separator lines are the constant rows / columns ----
    sep_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    sep_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    if not sep_rows or not sep_cols or sep_rows[0] == 0 or sep_cols[0] == 0:
        return [34], [submit_sel]
    hfac, wfac = sep_rows[0], sep_cols[0]        # block size in pixels
    hs, ws = hfac + 1, wfac + 1                  # block pitch (block + separator)
    h, w = (hi + 1) // hs, (wi + 1) // ws
    if h < 1 or w < 1:
        return [34], [submit_sel]

    g = [[int(I[a * hs, b * ws]) for b in range(w)] for a in range(h)]   # one value per block
    bgc = Counter([g[a][b] for a in range(h) for b in range(w)]).most_common(1)[0][0]

    def blk(a, b):                               # one block: exactly this rectangle
        return [a * hs, b * ws, hfac - 1, wfac - 1]

    def blk_cells(a, b):
        return [(a * hs + i, b * ws + j) for i in range(hfac) for j in range(wfac)]

    # ---- 2. the model shape and the lone markers ----
    seen = [[False] * w for _ in range(h)]
    comps = []
    for a in range(h):
        for b in range(w):
            if g[a][b] != bgc and not seen[a][b]:
                stack, cur = [(a, b)], []
                seen[a][b] = True
                while stack:
                    r, c = stack.pop()
                    cur.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] != bgc:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
                comps.append(cur)
    if len(comps) < 2:
        return [34], [submit_sel]

    ti = max(range(len(comps)),
             key=lambda k: (len({g[r][c] for r, c in comps[k]}), len(comps[k])))
    template = comps[ti]
    other_cells = [rc for k, cm in enumerate(comps) if k != ti for rc in cm]
    if not other_cells:
        return [34], [submit_sel]
    dotc = Counter([g[r][c] for r, c in other_cells]).most_common(1)[0][0]

    dot_cells = [(r, c) for (r, c) in template if g[r][c] == dotc]
    if not dot_cells:
        return [34], [submit_sel]
    dot_r, dot_c = min(dot_cells)                       # the model's own marker block

    dests = sorted([rc for rc in other_cells if g[rc[0]][rc[1]] == dotc])
    if not dests:
        return [34], [submit_sel]

    rel = [(r - dot_r, c - dot_c, g[r][c]) for (r, c) in template]   # shape around the marker

    # ---- 3. sweep read off the input: along the longer side, starting at the model ----
    row_major = (w >= h)
    if row_major:
        rev = (2 * dot_r >= h - 1)
        key = lambda rc: (rc[0], rc[1])
    else:
        rev = (2 * dot_c >= w - 1)
        key = lambda rc: (rc[1], rc[0])
    dests.sort(key=key, reverse=rev)
    rel.sort(key=lambda t: key((t[0], t[1])), reverse=rev)

    # intended result, predicted from I only
    sim = [row[:] for row in g]
    want = [row[:] for row in g]
    for (dr, dc) in dests:
        for (ar, ac, col) in rel:
            r, c = dr + ar, dc + ac
            if 0 <= r < h and 0 <= c < w:
                want[r][c] = col

    def draw_by_hand():
        """One copy at a time, one colour of that copy at a time."""
        for (dr, dc) in dests:
            groups, index = [], {}
            for (ar, ac, col) in rel:
                r, c = dr + ar, dc + ac
                if not (0 <= r < h and 0 <= c < w):
                    continue
                if sim[r][c] == col:                     # marker block already holds it
                    continue
                if col not in index:
                    index[col] = len(groups)
                    groups.append((col, []))
                groups[index[col]][1].append((r, c))
            for col, cells in groups:
                if len(cells) == 1:
                    a, b = cells[0]
                    ops.append(col); sels.append(blk(a, b))          # one block: exact rect
                else:
                    pix = []
                    for (a, b) in cells:
                        pix.extend(blk_cells(a, b))
                    ops.append(col); sels.append(sel_of(pix))        # same-colour blocks, no separators
                for (a, b) in cells:
                    sim[a][b] = col

    handwork = len(dests) * max(0, len(rel) - 1)         # blocks that would be placed by hand

    if handwork > 6:
        # ---- stamp route: copy the model once, press it onto every marker ----
        r0 = min(r for r, _ in template); c0 = min(c for _, c in template)
        r1 = max(r for r, _ in template); c1 = max(c for _, c in template)
        th, tw = r1 - r0 + 1, c1 - c0 + 1
        off_r, off_c = dot_r - r0, dot_c - c0

        plan = []
        for (dr, dc) in dests:
            orr, occ = dr - off_r, dc - off_c
            eff = False
            for i in range(th):
                for j in range(tw):
                    v = g[r0 + i][c0 + j]
                    if v == 0:                            # colour 0 never travels with a Paste
                        continue
                    r, c = orr + i, occ + j
                    if 0 <= r < h and 0 <= c < w and sim[r][c] != v:
                        sim[r][c] = v
                        eff = True
            plan.append(((orr, occ), eff))

        if any(eff for _, eff in plan):
            # the model's whole rectangle, blocks and separators alike: an exact rectangle
            ops.append(28); sels.append([r0 * hs, c0 * ws, th * hs - 2, tw * ws - 2])
            for (orr, occ), eff in plan:
                if eff:
                    ops.append(30); sels.append([orr * hs, occ * ws, 0, 0])   # paste origin
            # put down what the Paste could not carry (colour-0 blocks) and restore any
            # earlier copy a later rectangle laid background over - all measured from I
            cells_order = sorted([(i, j) for i in range(th) for j in range(tw)],
                                 key=key, reverse=rev)
            for (orr, occ), _ in plan:
                for (i, j) in cells_order:
                    r, c = orr + i, occ + j
                    if not (0 <= r < h and 0 <= c < w):
                        continue
                    if sim[r][c] != want[r][c]:
                        col = want[r][c]
                        ops.append(col); sels.append(blk(r, c))       # one block: exact rect
                        sim[r][c] = col
        else:
            sim = [row[:] for row in g]
            draw_by_hand()
    else:
        draw_by_hand()

    ops.append(34); sels.append(submit_sel)
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
                        f"num_examples+1 ({num_examples + 1}) for task 39e1d7f9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 39e1d7f9"
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
                                f"for task 39e1d7f9"
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
                    f"Failed to build a complete episode for task 39e1d7f9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"39e1d7f9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
