"""
ARC Task: 6d0160f0 (RE-ARC) — LLM-generated grid_maker
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
import math
import random
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # background colour and frontier-line colour are the two structural colours
    # of this task; the object colours are irrelevant to the rule.
    cols = [c for c in range(10) if c != 4]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc=None, linc=None) -> dict:
    cols = difference(interval(0, 10, 1), (4,))
    if bgc is None or linc is None:
        bgc, linc = sample(cols, 2)

    # full grid side for a block side k is k*k + k - 1  ->  clamp k to the canvas
    hub = 2
    for cand in range(2, 6):
        if cand * cand + cand - 1 <= max_h:
            hub = cand
    wub = 2
    for cand in range(2, 6):
        if cand * cand + cand - 1 <= max_w:
            wub = cand

    h = unifint(diff_lb, diff_ub, (2, hub))
    w = unifint(diff_lb, diff_ub, (2, wub))
    nh, nw = h, w
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    for iloc in range(h, fullh, h + 1):
        gi = fill(gi, linc, hfrontier((iloc, 0)))
    for jloc in range(w, fullw, w + 1):
        gi = fill(gi, linc, vfrontier((0, jloc)))
    noccs = unifint(diff_lb, diff_ub, (1, h * w))
    denseinds = asindices(canvas(-1, (h, w)))
    sparseinds = {(a * (h + 1), b * (w + 1)) for a, b in denseinds}
    locs = sample(totuple(sparseinds), noccs)
    trgtl = choice(locs)
    remlocs = remove(trgtl, locs)
    ntrgt = unifint(diff_lb, diff_ub, (1, (h * w - 1)))
    place = choice(totuple(denseinds))
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, ncols)
    candss = totuple(remove(place, denseinds))
    trgrem = sample(candss, ntrgt)
    trgrem = {(choice(ccols), ij) for ij in trgrem}
    trgtobj = {(4, place)} | trgrem
    go = paint(gi, shift(sfilter(trgtobj, lambda cij: cij[0] != linc), multiply(place, increment((h, w)))))
    gi = paint(gi, shift(trgtobj, trgtl))
    toleaveout = ccols
    for rl in remlocs:
        tlo = choice(totuple(ccols))
        ncells = unifint(diff_lb, diff_ub, (1, h * w - 1))
        inds = sample(totuple(denseinds), ncells)
        obj = {(choice(remove(tlo, ccols) if len(ccols) > 1 else ccols), ij) for ij in inds}
        toleaveout = remove(tlo, toleaveout)
        gi = paint(gi, shift(obj, rl))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Rule: frontier lines cut the grid into nb x nb blocks.  Exactly one cell
    holds a 4; its block is the marked tile, and the 4's position INSIDE that
    block names the block the tile belongs in.  Slide the tile there (line-
    coloured cells of the tile read as background) and clear every other block.

    Only I (and, for the fill colour, the worked examples) is read: handed a
    different answer grid this returns the same operations.  Which leg of the
    slide goes first, whether the board is tidied before or after the slide, and
    in what order/grouping the blocks are cleared are all fixed by features of
    the input, so instances differ but one instance is always solved the same way.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    G = I.copy()
    ops, sels = [], []

    # ---------- the lattice ----------------------------------------------
    # generator: k cells per block and k blocks per side, so side = k*k + k - 1
    def side(n, uniform):
        k = int(round((-1.0 + math.sqrt(4.0 * n + 5.0)) / 2.0))
        if k >= 1 and k * k + k - 1 == n:
            return k
        for i, u in enumerate(uniform):
            if u:
                return i
        return n

    row_uniform = [len(set(I[r].tolist())) == 1 for r in range(hi)]
    col_uniform = [len(set(I[:, c].tolist())) == 1 for c in range(wi)]
    bh = side(hi, row_uniform)
    bw = side(wi, col_uniform)
    linc = int(I[bh, 0])
    rstarts = list(range(0, hi, bh + 1))
    cstarts = list(range(0, wi, bw + 1))
    nbr, nbc = len(rstarts), len(cstarts)

    # ---------- the colour everything else becomes ------------------------
    bgc = None
    if examples:
        cnt = Counter()
        for pair in examples:
            g = pair['output'] if isinstance(pair, dict) else pair[1]
            cnt.update(np.asarray(g, dtype=int).flatten().tolist())
        if cnt:
            bgc = int(cnt.most_common(1)[0][0])
    if bgc is None:
        per_block = []
        for br in range(nbr):
            for bc in range(nbc):
                r0, c0 = rstarts[br], cstarts[bc]
                per_block.append([int(I[r, c]) for r in range(r0, r0 + bh)
                                  for c in range(c0, c0 + bw)])
        tot = Counter()
        for vals in per_block:
            tot.update(vals)
        sets = [set(v) for v in per_block]
        everywhere = [c for c in tot if all(c in s for s in sets)]
        pool = everywhere if everywhere else list(tot)
        bgc = int(max(pool, key=lambda c: tot[c]))

    # ---------- the marked tile and where it belongs ----------------------
    p4 = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) == 4][0]
    sbr, sbc = p4[0] // (bh + 1), p4[1] // (bw + 1)
    dbr, dbc = p4[0] % (bh + 1), p4[1] % (bw + 1)
    sr, sc = rstarts[sbr], cstarts[sbc]
    tr, tc = rstarts[dbr], cstarts[dbc]
    dr, dc = tr - sr, tc - sc
    moving = (dr != 0 or dc != 0)
    src_cells = [(r, c) for r in range(sr, sr + bh) for c in range(sc, sc + bw)]
    dst_cells = [(r, c) for r in range(tr, tr + bh) for c in range(tc, tc + bw)]

    # what the destination block must hold: the tile itself, its line-coloured
    # cells reading as background
    want = {}
    for i in range(bh):
        for j in range(bw):
            v = int(I[sr + i, sc + j])
            want[(tr + i, tc + j)] = bgc if v == linc else v

    # ---------- route style, read off the input ---------------------------
    vertical_first = abs(dr) >= abs(dc)          # longer leg of the slide first
    clear_first = (dr < 0) or (dr == 0 and dc < 0)   # travelling back: tidy first
    sweep_mode = (dbr + dbc) % 3                 # 0 row bands, 1 column bands, 2 rings
    sweep_reverse = ((sbr + sbc) % 2 == 1)       # which end of the sweep to start at

    # ---------- primitives -------------------------------------------------
    def color_cells(cells, col):
        col = int(col)
        cells = [(r, c) for (r, c) in cells if int(G[r, c]) != col]
        if not cells:
            return
        ops.append(col)
        sels.append(sel_of(cells))
        for (r, c) in cells:
            G[r, c] = col

    def slide(cells, legs):
        # one grab, then empty selections: ARCLE keeps the tile grabbed and
        # restores everything it glides over
        base = G.copy()
        vals = []
        for (r, c) in cells:
            vals.append(((r, c), int(G[r, c])))
            base[r, c] = 0
        first = True
        orow = ocol = 0
        for (op, count, stepr, stepc) in legs:
            for _ in range(count):
                ops.append(op)
                sels.append(sel_of(list(cells)) if first else sel_of([]))
                first = False
                orow += stepr
                ocol += stepc
        G[:, :] = base
        for (r, c), v in vals:
            if v != 0:
                rr, cc = r + orow, c + ocol
                if 0 <= rr < hi and 0 <= cc < wi:
                    G[rr, cc] = v

    def block_cells(b):
        r0, c0 = rstarts[b[0]], cstarts[b[1]]
        return [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]

    def sweep(blocks):
        if not blocks:
            return
        if sweep_mode == 0:
            grp = lambda b: b[0]
            key = lambda b: (b[0], b[1])
        elif sweep_mode == 1:
            grp = lambda b: b[1]
            key = lambda b: (b[1], b[0])
        else:
            grp = lambda b: abs(b[0] - sbr) + abs(b[1] - sbc)
            key = lambda b: (abs(b[0] - sbr) + abs(b[1] - sbc), b[0], b[1])
        order = sorted(blocks, key=key, reverse=sweep_reverse)
        pend = [b for b in order
                if any(int(G[r, c]) != bgc for (r, c) in block_cells(b))]
        n = len(pend)
        if n == 0:
            return
        if n <= 2:                      # a couple of blocks: one at a time
            for b in pend:
                color_cells(block_cells(b), bgc)
        elif n <= 6:                    # a handful: band by band
            cells, cur = [], None
            for b in pend:
                g = grp(b)
                if cur is not None and g != cur:
                    color_cells(cells, bgc)
                    cells = []
                cur = g
                cells.extend(block_cells(b))
            if cells:
                color_cells(cells, bgc)
        else:                           # most of the board: one wipe
            cells = []
            for b in pend:
                cells.extend(block_cells(b))
            color_cells(cells, bgc)

    # ---------- the route --------------------------------------------------
    others = [(br, bc) for br in range(nbr) for bc in range(nbc)
              if (br, bc) != (sbr, sbc) and (br, bc) != (dbr, dbc)]

    # a base coat in the destination block, laid only where the tile cannot
    # carry the cell itself (Move is transparent to 0-valued cells)
    prewipe = moving and any(
        int(I[sr + i, sc + j]) == 0
        and want[(tr + i, tc + j)] == bgc
        and int(G[tr + i, tc + j]) != bgc
        for i in range(bh) for j in range(bw))

    if clear_first:
        sweep(others + ([(dbr, dbc)] if prewipe else []))
    elif prewipe:
        color_cells(dst_cells, bgc)

    if moving:
        legs = []
        if dr:
            legs.append((21 if dr > 0 else 20, abs(dr), 1 if dr > 0 else -1, 0))
        if dc:
            legs.append((22 if dc > 0 else 23, abs(dc), 0, 1 if dc > 0 else -1))
        if len(legs) == 2 and not vertical_first:
            legs.reverse()
        slide(src_cells, legs)          # the whole tile travels, background included
        if clear_first:
            color_cells(src_cells, bgc)  # the hole the grab left behind

    # touch up the tile: line-coloured cells become background, cells Move
    # could not carry get painted
    fixes = {}
    for cell in dst_cells:
        v = want[cell]
        if int(G[cell]) != v:
            fixes.setdefault(v, []).append(cell)
    for v in sorted(fixes, key=lambda k: min(fixes[k]), reverse=sweep_reverse):
        color_cells(fixes[v], v)

    if not clear_first:
        sweep(others + ([(sbr, sbc)] if moving else []))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # whole-grid rectangle for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 6d0160f0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6d0160f0"
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
                                f"for task 6d0160f0"
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
                    f"Failed to build a complete episode for task 6d0160f0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6d0160f0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
