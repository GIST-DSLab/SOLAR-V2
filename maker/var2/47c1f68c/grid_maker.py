"""
ARC Task: 47c1f68c (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

import numpy as np

from maker.sel_helpers import sel_of

SCF_VARIANTS = ["identity", "hmirror", "vmirror", "both"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc, objc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(SCF_VARIANTS):
        examples = [{"scf": v} for v in SCF_VARIANTS]
        examples += [{"scf": random.choice(SCF_VARIANTS)}
                     for _ in range(n_ex - len(SCF_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [{"scf": v} for v in random.sample(SCF_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, objc, scf=None) -> dict:
    if scf is None:
        scf = random.choice(SCF_VARIANTS)

    def unif(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    hmax = max(2, min(14, (max_h - 1) // 2))
    wmax = max(2, min(14, (max_w - 1) // 2))
    h = unif(diff_lb, diff_ub, (2, hmax))
    w = unif(diff_lb, diff_ub, (2, wmax))
    nc = unif(diff_lb, diff_ub, (1, h * w - 1))
    nc = max(1, min(h * w - 1, nc))

    allcells = [(r, c) for r in range(h) for c in range(w)]
    bx = set(allcells)
    start = random.choice(allcells)
    obj = {start}
    for _ in range(nc - 1):
        dns = set()
        for (r, c) in obj:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        dns.add((r + dr, c + dc))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj.add(ch)
        bx.discard(ch)

    objgrid = [[objc if (r, c) in obj else bgc for c in range(w)] for r in range(h)]
    blank = [[bgc] * w for _ in range(h)]
    top = [objgrid[r] + [linc] + blank[r] for r in range(h)]
    mid = [[linc] * (2 * w + 1)]
    bot = [blank[r] + [linc] + blank[r] for r in range(h)]
    gi = top + mid + bot

    P = [[linc if (r, c) in obj else bgc for c in range(w)] for r in range(h)]
    tophalf = [P[r] + P[r][::-1] for r in range(h)]
    go = tophalf + tophalf[::-1]

    if scf in ("hmirror", "both"):
        gi = gi[::-1]
        go = go[::-1]
    if scf in ("vmirror", "both"):
        gi = [row[::-1] for row in gi]
        go = [row[::-1] for row in go]

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


# ── the rule ─────────────────────────────────────────────────────────────────
# The input is four h x w panels separated by a cross of the line colour; one
# panel holds a shape, the other three are blank.  The answer is that panel
# alone, blown up to a 2h x 2w mosaic of itself and its three mirrors, with the
# shape repainted in the line colour.  The panel keeps the corner it occupied
# in the input, so quadrant (i, j) of the answer is the panel mirrored
# left-right iff j differs from the panel's column and up-down iff i differs
# from the panel's row.
#
# Which ROUTE gets there is a free choice, and different solvers make it
# differently.  Every choice below is read off the input (the corner the panel
# sits in, how many cells the shape has, how many pieces it falls into, the
# panel's proportions), so one instance always draws the same route while the
# routes differ from instance to instance:
#   * the shape is recoloured where it sits, or after the panel is cut out;
#   * it is recoloured piece by piece with FloodFill, or in one Color stroke;
#   * the mosaic is assembled quadrant by quadrant, or by mirroring a whole
#     strip at a time, or by laying the background down and drawing the shape
#     into each quadrant by hand;
#   * and the quadrants / pieces are visited in different sweep orders.
# Nothing here reads O: hand this function a different answer grid and it
# returns the same operations.

_QUAD_ORDERS = [
    [(0, 0), (0, 1), (1, 0), (1, 1)],   # row by row
    [(0, 0), (1, 0), (0, 1), (1, 1)],   # column by column
    [(0, 0), (0, 1), (1, 1), (1, 0)],   # clockwise round the mosaic
    [(0, 0), (1, 0), (1, 1), (0, 1)],   # anticlockwise round the mosaic
]


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # ── read the board: panel size, colours, which panel holds the shape ──────
    h, w = (hi - 1) // 2, (wi - 1) // 2
    linc = int(I[h, w])                          # the crossing of the two dividers
    origins = [((0, 0), (0, 0)), ((0, 1), (0, w + 1)),
               ((1, 0), (h + 1, 0)), ((1, 1), (h + 1, w + 1))]
    bgc, home, horigin = None, None, (0, 0)
    for (ij, (qr, qc)) in origins:
        vals = set(I[qr:qr + h, qc:qc + w].flatten().tolist())
        if len(vals) == 1:                       # a blank panel shows the background
            if bgc is None:
                bgc = int(next(iter(vals)))
        else:
            home, horigin = ij, (qr, qc)         # the one panel holding the shape
    if home is None:
        home, horigin = (0, 0), (0, 0)
    i0, j0 = home
    r0, c0 = horigin
    P = I[r0:r0 + h, c0:c0 + w].copy()
    if bgc is None:
        bgc = int(Counter(P.flatten().tolist()).most_common(1)[0][0])
    objc = bgc
    for v in P.flatten().tolist():
        if int(v) != bgc:
            objc = int(v)
            break
    cells = [(r, c) for r in range(h) for c in range(w) if int(P[r, c]) == objc]

    # ── the shape's 4-connected pieces, swept from the panel's outer corner ───
    sr = -1 if i0 == 1 else 1
    sc = -1 if j0 == 1 else 1
    todo, seen, pieces = set(cells), set(), []
    for cell in sorted(cells, key=lambda rc: (sr * rc[0], sc * rc[1])):
        if cell in seen:
            continue
        seen.add(cell)
        q = deque([cell])
        while q:
            y, x = q.popleft()
            for nb in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if nb in todo and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        pieces.append(cell)                      # the cell the sweep enters it by

    # ── the route, read off the input ────────────────────────────────────────
    n_cell, n_piece = len(cells), len(pieces)
    route = (n_cell + n_piece) % 3               # 0 quadrants, 1 strips, 2 by hand
    recolour_first = (n_cell % 2 == 0)           # recolour in place, then cut out
    piecewise = (n_piece <= 4)                   # few pieces -> fill one by one
    order = _QUAD_ORDERS[(2 * i0 + j0 + n_piece) % 4]

    Q = P.copy()
    Q[Q == objc] = linc                          # the panel once it is recoloured

    ops, sels = [], []

    def recolour(off_r, off_c):
        if piecewise:
            for seed in pieces:                  # one FloodFill per piece
                ops.append(10 + linc)
                sels.append([off_r + seed[0], off_c + seed[1], 0, 0])
        else:                                    # one stroke over the whole shape
            ops.append(linc)
            sels.append(sel_of([(off_r + r, off_c + c) for r, c in cells]))

    def flip_into(rr, cc, cur, fh, fv, hh, ww):
        """mirror the hh x ww block at (rr, cc) -- that whole rectangle -- in place"""
        if fh:
            nxt = np.fliplr(cur)
            if not np.array_equal(nxt, cur):     # skip a mirror that changes nothing
                ops.append(26)
                sels.append([rr, cc, hh - 1, ww - 1])
            cur = nxt
        if fv:
            nxt = np.flipud(cur)
            if not np.array_equal(nxt, cur):
                ops.append(27)
                sels.append([rr, cc, hh - 1, ww - 1])
            cur = nxt
        return cur

    if recolour_first:
        recolour(r0, c0)
    ops.append(33)                          # cut out the panel (that full rectangle)
    sels.append([r0, c0, h - 1, w - 1])
    if not recolour_first:
        recolour(0, 0)

    if route == 0:
        # ── quadrant by quadrant: one stamp of the panel per corner ───────────
        if w % 2 == 0:                      # take the copy before or after growing
            ops.append(29)
            sels.append([0, 0, h - 1, w - 1])
            ops.append(33)
            sels.append([0, 0, 2 * h - 1, 2 * w - 1])
        else:
            ops.append(33)
            sels.append([0, 0, 2 * h - 1, 2 * w - 1])
            ops.append(29)
            sels.append([0, 0, h - 1, w - 1])
        for (i, j) in order:
            rr, cc = i * h, j * w
            if (i, j) != (0, 0):            # (0,0) already holds the panel
                ops.append(30)
                sels.append([rr, cc, 0, 0])
            flip_into(rr, cc, Q.copy(), j != j0, i != i0, h, w)
    elif route == 1:
        # ── strips: mirror the panel once, then mirror the whole strip ────────
        A = flip_into(0, 0, Q.copy(), j0 == 1, i0 == 1, h, w)   # corner quadrant
        if w >= h:                          # a wide panel: sideways first
            ops.append(33)
            sels.append([0, 0, h - 1, 2 * w - 1])               # widen the canvas
            ops.append(29)
            sels.append([0, 0, h - 1, w - 1])
            ops.append(30)
            sels.append([0, w, 0, 0])
            flip_into(0, w, A.copy(), True, False, h, w)
            top = np.hstack([A, np.fliplr(A)])
            ops.append(33)
            sels.append([0, 0, 2 * h - 1, 2 * w - 1])           # heighten it
            ops.append(29)
            sels.append([0, 0, h - 1, 2 * w - 1])
            ops.append(30)
            sels.append([h, 0, 0, 0])
            flip_into(h, 0, top, False, True, h, 2 * w)
        else:                               # a tall panel: downwards first
            ops.append(33)
            sels.append([0, 0, 2 * h - 1, w - 1])
            ops.append(29)
            sels.append([0, 0, h - 1, w - 1])
            ops.append(30)
            sels.append([h, 0, 0, 0])
            flip_into(h, 0, A.copy(), False, True, h, w)
            left = np.vstack([A, np.flipud(A)])
            ops.append(33)
            sels.append([0, 0, 2 * h - 1, 2 * w - 1])
            ops.append(29)
            sels.append([0, 0, 2 * h - 1, w - 1])
            ops.append(30)
            sels.append([0, w, 0, 0])
            flip_into(0, w, left, True, False, 2 * h, w)
    else:
        # ── drawn by hand: background down, then the shape into each quadrant ─
        # this is the route that must survive linc == 0: the shape is laid down
        # with an explicit Color(0) over the background it sits on.
        flip_into(0, 0, Q.copy(), j0 == 1, i0 == 1, h, w)       # corner quadrant
        ops.append(33)
        sels.append([0, 0, 2 * h - 1, 2 * w - 1])
        for (i, j) in order:
            if (i, j) == (0, 0):
                continue
            rr, cc = i * h, j * w
            if bgc != 0:                    # fresh canvas is already 0 = nothing
                ops.append(bgc)             # lay this quadrant's background
                sels.append([rr, cc, h - 1, w - 1])
            shape = [(rr + (r if i == i0 else h - 1 - r),
                      cc + (c if j == j0 else w - 1 - c)) for r, c in cells]
            ops.append(linc)                # draw the mirrored shape on top
            sels.append(sel_of(shape))

    ops.append(34)
    sels.append([0, 0, 2 * h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
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
                                f"for task 47c1f68c"
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
                    f"Failed to build a complete episode for task 47c1f68c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"47c1f68c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
