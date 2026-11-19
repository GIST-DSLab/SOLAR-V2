"""
ARC Task: 10fcaaa3 (RE-ARC) — LLM-generated grid_maker
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
    # generator: cols = 0..9 without 8 ; bgc = choice(cols)
    # the rule is "diagonal halo of ANY non-background cell", so object colours
    # matter only by presence — only the background must be fixed per episode.
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = remove(8, interval(0, 10, 1))
    # output is 2h x 2w -> keep it inside the canvas limits
    hmax = max(2, min(15, max_h // 2))
    wmax = max(2, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 6)))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = remove(bgc, cols)
    ncols = min(ncols, len(remcols))
    ccols = sample(remcols, ncols)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    locs = frozenset(sample(totuple(inds), ncells))
    obj = frozenset({(choice(ccols), ij) for ij in locs})
    gi = paint(c, obj)
    go = hconcat(gi, gi)
    go = vconcat(go, go)
    fullocs = locs | shift(locs, (0, w)) | shift(locs, (h, 0)) | shift(locs, (h, w))
    nbhs = mapply(ineighbors, fullocs)
    topaint = nbhs & ofcolor(go, bgc)
    go = fill(go, 8, topaint)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule, read off I alone: the tile is repeated 2x2, then every background cell
    that touches a non-background cell DIAGONALLY is marked 8.

    The rule is fixed; the ROUTE through it is read off the input, so two
    different inputs get solved the way two different people would solve them
    (and the same input always the same way):

      * where to start     <- the corner the marks cluster around (centroid of I)
      * which way to sweep <- the tile's long axis (rows for a wide tile,
                              columns for a tall one; a square tile follows the
                              direction the marks are spread out in)
      * how to tile        <- double along the long axis first, then copy the
                              doubled band across in one go; a square tile has no
                              long axis, so its three empty quadrants are stamped
                              one at a time
      * how to mark        <- how many colours are in play: one colour -> finish a
                              quadrant before starting the next; two -> finish a
                              pixel (all four of its copies) before the next;
                              three or more -> work through one colour at a time
      * stroke size        <- how many marked pixels there are: an odd count is
                              drawn X-at-a-time, an even one cell-at-a-time
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = 2 * hi, 2 * wi

    # background: the canvas colour the generator paints before dropping at most
    # (h*w)//6 cells on it, so it is always the strict majority of I.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    srcs = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    n = len(srcs)
    colours = sorted({int(I[r, c]) for r, c in srcs})
    k = len(colours)

    ops, sels = [], []

    # ---- 1. grow the canvas to the 2h x 2w tiled size --------------------------
    # (bbox selection: the intended cells ARE exactly this full rectangle)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    # the top-left tile still holds I: its non-zero cells were copied and its zero
    # cells are zero in the freshly cleared canvas too.

    # ---- 2. replicate the tile over the other three quadrants ------------------
    if wi > hi:
        # wide tile: double it sideways, then carry the doubled band downwards
        ops.append(28); sels.append([0, 0, hi - 1, wi - 1])     # CopyI the tile
        ops.append(30); sels.append([0, wi, 0, 0])              # paste to its right
        ops.append(29); sels.append([0, 0, hi - 1, wo - 1])     # CopyO the top band
        ops.append(30); sels.append([hi, 0, 0, 0])              # paste it below
    elif hi > wi:
        # tall tile: double it downwards, then carry the doubled band sideways
        ops.append(28); sels.append([0, 0, hi - 1, wi - 1])     # CopyI the tile
        ops.append(30); sels.append([hi, 0, 0, 0])              # paste below it
        ops.append(29); sels.append([0, 0, ho - 1, wi - 1])     # CopyO the left band
        ops.append(30); sels.append([0, wi, 0, 0])              # paste it to the right
    else:
        # square tile: no long axis to prefer, stamp each empty quadrant in turn
        ops.append(28); sels.append([0, 0, hi - 1, wi - 1])     # CopyI the tile

    # ---- where the solver starts and which way it sweeps ----------------------
    if n:
        mean_r = sum(r for r, _ in srcs) / float(n)
        mean_c = sum(c for _, c in srcs) / float(n)
    else:
        mean_r, mean_c = (hi - 1) / 2.0, (wi - 1) / 2.0
    top = mean_r <= (hi - 1) / 2.0          # marks cluster in the upper half?
    left = mean_c <= (wi - 1) / 2.0         # ... and in the left half?
    if wi != hi:
        row_major = wi > hi                 # sweep along the tile's long axis
    elif n:
        rspan = max(r for r, _ in srcs) - min(r for r, _ in srcs)
        cspan = max(c for _, c in srcs) - min(c for _, c in srcs)
        row_major = cspan >= rspan          # square tile: follow the marks' spread
    else:
        row_major = True

    def _oriented(a, b):
        return (a, b) if row_major else (b, a)

    def pkey(rc):
        r, c = rc
        return _oriented(r if top else hi - 1 - r, c if left else wi - 1 - c)

    sweep = sorted(srcs, key=pkey)

    offsets = [(0, 0), (0, wi), (hi, 0), (hi, wi)]

    def okey(o):
        return _oriented(o[0] if top else -o[0], o[1] if left else -o[1])

    tile_order = sorted(offsets, key=okey)

    if hi == wi:
        for (tr, tc) in tile_order:
            if (tr, tc) == (0, 0):
                continue
            ops.append(30); sels.append([tr, tc, 0, 0])         # paste at tile origin

    # ---- 3. mark the diagonal halo of every copied pixel -----------------------
    occ = set()
    for (r, c) in srcs:
        for (tr, tc) in offsets:
            occ.add((r + tr, c + tc))       # occupied: never a halo cell

    diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    diag_order = sorted(diag, key=lambda d: _oriented(d[0] if top else -d[0],
                                                      d[1] if left else -d[1]))

    marked = set()
    x_stroke = (n % 2 == 1)   # odd count -> lay the whole X in one stroke

    def mark(r, c, tr, tc):
        cells = []
        for (er, ec) in diag_order:
            pr, pc = r + tr + er, c + tc + ec
            if not (0 <= pr < ho and 0 <= pc < wo):
                continue
            if (pr, pc) in occ:            # a pixel sits here, it keeps its colour
                continue
            if (pr, pc) in marked:         # already an 8 — repainting would do nothing
                continue
            cells.append((pr, pc))
        if not cells:
            return
        if x_stroke:
            ops.append(8); sels.append(sel_of(cells))
            marked.update(cells)
        else:
            for cell in cells:
                ops.append(8); sels.append(sel_of([cell]))
                marked.add(cell)

    if k >= 3:
        # busy palette: deal with one colour at a time, the commonest first
        counts = Counter(int(I[r, c]) for r, c in srcs)
        for col in sorted(colours, key=lambda x: (-counts[x], x)):
            for (r, c) in sweep:
                if int(I[r, c]) != col:
                    continue
                for (tr, tc) in tile_order:
                    mark(r, c, tr, tc)
    elif k == 2:
        # a pixel at a time: finish its mark in all four copies before moving on
        for (r, c) in sweep:
            for (tr, tc) in tile_order:
                mark(r, c, tr, tc)
    else:
        # one colour: every quadrant looks the same, so finish one before the next
        for (tr, tc) in tile_order:
            for (r, c) in sweep:
                mark(r, c, tr, tc)

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 10fcaaa3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 10fcaaa3"
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
                                f"for task 10fcaaa3"
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
                    f"Failed to build a complete episode for task 10fcaaa3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"10fcaaa3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
