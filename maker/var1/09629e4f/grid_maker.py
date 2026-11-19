"""
ARC Task: 09629e4f (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    barcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "barcol": barcol}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, barcol: int) -> dict:
    cols = interval(0, 10, 1)
    # full grid is h*h + h - 1 tall and w*w + w - 1 wide -> cap h, w by max_h / max_w
    hcap = 2
    for k in range(2, 6):
        if k * k + k - 1 <= max_h:
            hcap = k
    wcap = 2
    for k in range(2, 6):
        if k * k + k - 1 <= max_w:
            wcap = k
    h = unifint(diff_lb, diff_ub, (2, hcap))
    w = unifint(diff_lb, diff_ub, (2, wcap))
    nrows, ncolumns = h, w
    remcols = remove(bgc, cols)
    remcols = remove(barcol, remcols)
    ncols = unifint(diff_lb, diff_ub, (2, min(7, (h * w) - 2)))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    fullh, fullw = h * nrows + nrows - 1, w * ncolumns + ncolumns - 1
    gi = canvas(barcol, (fullh, fullw))
    locs = totuple(product(interval(0, fullh, h + 1), interval(0, fullw, w + 1)))
    trgloc = choice(locs)
    remlocs = remove(trgloc, locs)
    colssf = sample(remcols, ncols)
    colsss = remove(choice(colssf), colssf)
    trgssf = sample(inds, ncols - 1)
    gi = fill(gi, bgc, shift(inds, trgloc))
    for ij, cl in zip(trgssf, colsss):
        gi = fill(gi, cl, {add(trgloc, ij)})
    for rl in remlocs:
        trgss = sample(inds, ncols)
        tmpg = tuple(e for e in c)
        for ij, cl in zip(trgss, colssf):
            tmpg = fill(tmpg, cl, {ij})
        gi = paint(gi, shift(asobject(tmpg), rl))
    go = canvas(bgc, (fullh, fullw))
    go = fill(go, barcol, ofcolor(gi, barcol))
    for ij, cl in zip(trgssf, colsss):
        go = fill(go, cl, shift(inds, multiply(ij, (h + 1, w + 1))))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (read from I alone):
      The grid is a lattice of nrows x ncolumns tiles of size h x w, separated by
      bar lines.  Exactly one tile carries fewer distinct colours than the others -
      that is the key tile.  Every tile is blanked to the background, and each
      coloured pixel at (i, j) inside the key tile turns the WHOLE tile (i, j) of
      the lattice that colour.

    The ROUTE (which tiles first, which direction, base-then-draw vs. draw
    straight on, stamping a blank tile vs. painting each) is chosen from features
    of the input itself, so the same input always yields the same route while
    different inputs are solved differently.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- lattice geometry -------------------------------------------------
    uni_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    uni_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    if uni_rows:
        barcol = int(I[uni_rows[0], 0])
    else:
        barcol = int(I[0, uni_cols[0]])
    bar_rows = [r for r in uni_rows if int(I[r, 0]) == barcol]
    bar_cols = [c for c in uni_cols if int(I[0, c]) == barcol]
    h = bar_rows[0] if bar_rows else hi
    w = bar_cols[0] if bar_cols else wi
    nrows = (hi + 1) // (h + 1)
    ncolumns = (wi + 1) // (w + 1)

    bgc = int(Counter([v for v in I.flatten().tolist() if v != barcol]).most_common(1)[0][0])

    def origin(i, j):
        return i * (h + 1), j * (w + 1)

    def cells(i, j):
        r0, c0 = origin(i, j)
        return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]

    blocks = [(i, j) for i in range(nrows) for j in range(ncolumns)]

    # --- key tile = fewest distinct non-background colours -----------------
    ti, tj, fewest = 0, 0, None
    for (i, j) in blocks:
        r0, c0 = origin(i, j)
        ncol = len(set(I[r0:r0 + h, c0:c0 + w].flatten().tolist()) - {bgc})
        if fewest is None or ncol < fewest:
            fewest, ti, tj = ncol, i, j
    tr0, tc0 = origin(ti, tj)

    pattern = {}
    for i in range(min(h, nrows)):
        for j in range(min(w, ncolumns)):
            cl = int(I[tr0 + i, tc0 + j])
            if cl != bgc:
                pattern[(i, j)] = cl
    blanks = [b for b in blocks if b not in pattern]

    # --- route choice, derived from the input ------------------------------
    style = 3 * barcol + 5 * bgc + 7 * ti + 11 * tj + nrows + ncolumns
    base_mode = style % 5           # 0 wipe-all / 1 wipe-bands / 2 wipe-tiles / 3 direct / 4 stamp
    order_mode = (style // 5) % 5   # sweep direction
    color_order = (style // 25) % 3 # which coloured tile first
    if base_mode == 4 and (bgc == 0 or len(blanks) < 2):
        base_mode = style % 3       # stamping needs a non-zero blank tile to copy

    def sweep(seq):
        seq = list(seq)
        if order_mode == 0:                                     # row by row
            return sorted(seq)
        if order_mode == 1:                                     # column by column
            return sorted(seq, key=lambda b: (b[1], b[0]))
        if order_mode == 2:                                     # serpentine rows
            return sorted(seq, key=lambda b: (b[0], b[1] if b[0] % 2 == 0 else -b[1]))
        if order_mode == 3:                                     # from the far corner back
            return sorted(seq, key=lambda b: (-b[0], -b[1]))
        return sorted(seq, key=lambda b: (max(abs(b[0] - ti), abs(b[1] - tj)), b[0], b[1]))

    def coloured():
        items = list(pattern.items())
        if color_order == 0:                                    # as read in the key tile
            return sorted(items, key=lambda kv: kv[0])
        if color_order == 1:                                    # by colour
            return sorted(items, key=lambda kv: (kv[1], kv[0]))
        return sorted(items, key=lambda kv: (abs(kv[0][0] - ti) + abs(kv[0][1] - tj), kv[0]))

    ops, sels = [], []

    def paint_cells(color, cell_list):
        ops.append(int(color))
        sels.append(sel_of(cell_list))

    if base_mode == 0:
        # one sweep of background over every tile, then the coloured tiles on top
        allcells = []
        for b in sweep(blocks):
            allcells.extend(cells(*b))
        paint_cells(bgc, allcells)
        for (i, j), cl in coloured():
            paint_cells(cl, cells(i, j))

    elif base_mode == 1:
        # background laid band by band (whole lattice rows / columns), then colours
        if order_mode in (1, 3):
            band_ids = list(range(ncolumns))
            if order_mode == 3:
                band_ids.reverse()
            for j in band_ids:
                cs = []
                for i in range(nrows):
                    cs.extend(cells(i, j))
                paint_cells(bgc, cs)
        else:
            band_ids = list(range(nrows))
            if order_mode == 4:
                band_ids.sort(key=lambda i: (abs(i - ti), i))
            for i in band_ids:
                cs = []
                for j in range(ncolumns):
                    cs.extend(cells(i, j))
                paint_cells(bgc, cs)
        for (i, j), cl in coloured():
            paint_cells(cl, cells(i, j))

    elif base_mode == 2:
        # blank each tile in turn, then drop the colours on top
        for b in sweep(blocks):
            paint_cells(bgc, cells(*b))
        for (i, j), cl in coloured():
            paint_cells(cl, cells(i, j))

    elif base_mode == 3:
        # straight onto the grid: every tile painted once, with its final colour
        for b in sweep(blocks):
            paint_cells(pattern.get(b, bgc), cells(*b))

    else:
        # make one blank tile, stamp it over the other blank ones, paint the rest
        order = sweep(blanks)
        first = order[0]
        paint_cells(bgc, cells(*first))
        ops.append(29)                       # CopyO the freshly blanked tile
        sels.append(sel_of(cells(*first)))   # exactly that full h x w rectangle
        for b in order[1:]:
            rr, cc = origin(*b)
            ops.append(30)
            sels.append(sel_of([(rr, cc)]))  # paste origin
        for (i, j), cl in coloured():
            paint_cells(cl, cells(i, j))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])      # full grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 09629e4f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 09629e4f"
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
                                f"for task 09629e4f"
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
                    f"Failed to build a complete episode for task 09629e4f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"09629e4f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
