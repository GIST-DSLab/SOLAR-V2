"""
ARC Task: 5c0a986e (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}


def sample_colors(num_examples=None) -> dict:
    # The generator hardcodes the two object colours 1 and 2 (and their two ray
    # directions), so the only randomly sampled colour is the background, drawn
    # from 0..9 minus {1, 2}.  bgc == 0 is legal and stays legal here: the route
    # only ever paints colour-1 / colour-2 ray cells with explicit Color ops and
    # never Copies/Pastes/Moves anything, so no cell that might hold 0 is carried
    # by an operation that would read 0 as "nothing there".
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    hcap = max(5, min(int(max_h), 30))
    wcap = max(5, min(int(max_w), 30))
    res = None
    for _attempt in range(30):
        h = unifint(diff_lb, diff_ub, (5, hcap))
        w = unifint(diff_lb, diff_ub, (5, wcap))
        nobjs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 10)))
        # every instance shows BOTH ray directions: the first two squares that land
        # take colour 1 and colour 2 (in random order), the rest are free.
        colseq = [1, 2] + [choice((1, 2)) for _ in range(max(0, nobjs - 2))]
        if choice((0, 1)):
            colseq[0], colseq[1] = colseq[1], colseq[0]
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        tr = 0
        maxtr = 5 * nobjs
        succ = 0
        placed = set()
        inds = asindices(gi)
        fullinds = asindices(gi)
        while succ < nobjs and tr < maxtr:
            tr += 1
            cands = sfilter(inds, lambda ij: 0 < ij[0] <= h - 3 and 0 < ij[1] <= w - 3)
            if len(cands) == 0:
                break
            loc = choice(totuple(cands))
            col = colseq[succ]
            sq = {(loc), add(loc, (0, 1)), add(loc, (1, 0)), add(loc, (1, 1))}
            if col == 1:
                obj = sq | (shoot(loc, (-1, -1)) & fullinds)
            else:
                obj = sq | (shoot(loc, (1, 1)) & fullinds)
            if obj.issubset(inds):
                succ += 1
                placed.add(col)
                inds = (inds - obj) - mapply(dneighbors, sq)
                gi = fill(gi, col, sq)
                go = fill(go, col, obj)
        res = {'input': gi, 'output': go}
        if placed == {1, 2}:
            break
    return res


# ── helpers for derive_operations ────────────────────────────────────────────

_DEFAULT_DIRS = {1: (-1, -1), 2: (1, 1)}


def _squares(G, cols):
    """Every isolated 2x2 block of an object colour -> (r, c, col) at its top-left.
    The generator keeps squares apart (dneighbors excluded), so a 2x2 run of one
    object colour is always exactly one square."""
    G = np.asarray(G, dtype=int)
    h, w = G.shape
    out = []
    for r in range(h - 1):
        for c in range(w - 1):
            col = int(G[r, c])
            if col not in cols:
                continue
            if G[r, c + 1] != col or G[r + 1, c] != col or G[r + 1, c + 1] != col:
                continue
            if r > 0 and G[r - 1, c] == col:
                continue
            if c > 0 and G[r, c - 1] == col:
                continue
            out.append((r, c, col))
    return out


def _infer_dirs(examples):
    """Which diagonal each square colour shoots along. A lone input never says it,
    so it is read off the worked examples (never off O)."""
    dirs = {}
    if not examples:
        return dict(_DEFAULT_DIRS)
    votes = {}
    for pair in examples:
        try:
            Ie = np.asarray(pair[0], dtype=int)
            Oe = np.asarray(pair[1], dtype=int)
        except Exception:
            continue
        if Ie.shape != Oe.shape:
            continue
        flat = Ie.flatten().tolist()
        bg = int(Counter(flat).most_common(1)[0][0])
        cols = set(int(v) for v in flat) - {bg}
        h, w = Ie.shape
        for (r, c, col) in _squares(Ie, cols):
            for d in ((-1, -1), (1, 1), (-1, 1), (1, -1)):
                # first cell of that diagonal beyond the 2x2 block
                rr = r + (d[0] if d[0] < 0 else 1 + d[0])
                cc = c + (d[1] if d[1] < 0 else 1 + d[1])
                if 0 <= rr < h and 0 <= cc < w and Ie[rr, cc] != col and Oe[rr, cc] == col:
                    votes.setdefault(col, Counter())[d] += 1
    for col, cnt in votes.items():
        dirs[col] = cnt.most_common(1)[0][0]
    for col, d in _DEFAULT_DIRS.items():
        dirs.setdefault(col, d)
    return dirs


def _ray(r, c, d, h, w):
    """Cells of the diagonal a 2x2 square at (r, c) shoots along d, beyond the square."""
    dr, dc = d
    cr = r + (0 if dr < 0 else 1)
    cc = c + (0 if dc < 0 else 1)
    cells = []
    while True:
        cr += dr
        cc += dc
        if not (0 <= cr < h and 0 <= cc < w):
            break
        cells.append((cr, cc))
    return cells


def derive_operations(I, O, examples=None):
    """
    Rule: every 2x2 square shoots a diagonal ray of its own colour out of one
    corner — colour 1 up-left from its top-left corner, colour 2 down-right from
    its bottom-right corner. Which corner goes with which colour is a convention
    the input alone does not fix, so it is read from the worked examples.

    The rule is the same every time; the route through it is not. Which square is
    dealt with first, whether a ray is laid down in one stroke / one cell at a
    time / together with every other ray of its colour, and whether a cell-by-cell
    ray is inked outward from the square or inward from the grid edge are all
    decided by the picture itself (its size, how many squares of each colour, where
    the top-left one sits, how much diagonal there is to draw). Same input -> same
    route; the variety comes from the instances differing. Nothing here looks at O.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    dirs = _infer_dirs(examples)
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    marker_cols = set(col for col in dirs if col != bgc)

    objs = []
    for (r, c, col) in _squares(I, marker_cols):
        # rays never collide with another object, so every ray cell is background;
        # anything already holding the colour is dropped so no op is a no-op
        cells = [(rr, cc) for rr, cc in _ray(r, c, dirs[col], h, w) if int(I[rr, cc]) != col]
        objs.append({"r": r, "c": c, "col": col, "ray": cells})

    ops, sels = [], []
    if not objs:
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])   # full-grid rectangle: submit
        return ops, sels

    # ── the route, fixed by the input picture ────────────────────────────────
    n_by_col = Counter(o["col"] for o in objs)
    total_cells = sum(len(o["ray"]) for o in objs)
    first = min(objs, key=lambda o: (o["r"], o["c"]))     # top-left-most square
    style = (h * w + 3 * n_by_col[1] + 5 * n_by_col[2]
             + 7 * (first["r"] + first["c"]) + 11 * first["col"] + total_cells)

    lead = first["col"]
    colour_rank = {lead: 0}
    for i, col in enumerate([c for c in sorted(n_by_col) if c != lead]):
        colour_rank[col] = i + 1

    orders = [
        lambda o: (o["r"], o["c"]),                         # top row down
        lambda o: (-o["r"], -o["c"]),                       # bottom row up
        lambda o: (o["c"], o["r"]),                         # column by column
        lambda o: (colour_rank[o["col"]], o["r"], o["c"]),  # one colour, then the other
        lambda o: (-len(o["ray"]), o["r"], o["c"]),         # longest diagonals first
        lambda o: (o["r"] + o["c"], o["r"]),                # nearest the top-left corner first
    ]
    order = orders[style % len(orders)]
    mode = (style // len(orders)) % 3          # 0 stroke per ray, 1 cell by cell, 2 per colour
    if mode == 1 and total_cells > 48:
        mode = 0                               # too much diagonal to ink cell by cell
    if mode == 2 and len(objs) > 12:
        mode = 0                               # too many squares to sweep in one stroke
    inward = bool((style // (len(orders) * 3)) % 2)

    objs.sort(key=order)

    if mode == 2:
        # every diagonal of one colour in a single stroke, colour by colour
        seen = []
        for o in objs:
            if o["col"] not in seen:
                seen.append(o["col"])
        for col in seen:
            cells = [p for o in objs if o["col"] == col for p in o["ray"]]
            if cells:
                ops.append(int(col))           # Color<col> over that colour's whole ray set
                sels.append(sel_of(cells))
    else:
        for o in objs:
            cells = list(o["ray"])
            if not cells:
                continue
            if mode == 0:
                ops.append(int(o["col"]))      # Color<col>: this square's whole ray in one stroke
                sels.append(sel_of(cells))
            else:
                if inward:
                    cells = cells[::-1]        # drawn from the grid edge back to the square
                for cell in cells:
                    ops.append(int(o["col"]))  # Color<col>: one cell of the ray
                    sels.append(sel_of([cell]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])          # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 5c0a986e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5c0a986e"
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
                                f"for task 5c0a986e"
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
                    f"Failed to build a complete episode for task 5c0a986e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5c0a986e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
