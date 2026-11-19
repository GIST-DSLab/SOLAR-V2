"""
ARC Task: 9d9215db (RE-ARC) — LLM-generated grid_maker
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
from utils import *
from dsl import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    hmax = max(5, min(14, (max_h - 1) // 2))
    wmax = max(5, min(14, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (5, hmax))
    w = unifint(diff_lb, diff_ub, (5, wmax))
    h = h * 2 + 1
    w = w * 2 + 1
    ub = min(h, w) // 4
    nrings = unifint(diff_lb, diff_ub, (1, ub))
    onlinesbase = tuple([(2 * k + 1, 2 * k + 1) for k in range(ub)])
    onlines = sample(onlinesbase, nrings)
    onlines = {(choice(remcols), ij) for ij in onlines}
    gi = canvas(bgc, (h, w))
    gi = paint(gi, onlines)
    linsbase = apply(rbind(add, (0, 2)), onlinesbase[:-1])
    nlines = unifint(diff_lb, diff_ub, (1, len(linsbase)))
    linesps = sample(linsbase, nlines)
    colors = [choice(remcols) for k in range(nlines)]
    dots = {(col, ij) for col, ij in zip(colors, linesps)}
    dots2 = {(col, ij[::-1]) for col, ij in zip(colors, linesps)}
    gi = paint(gi, dots | dots2)
    ff = lambda ij: ij[1] % 2 == 1
    ff2 = lambda ij: ij[0] % 2 == 1
    linesps2 = tuple(x[::-1] for x in linesps)
    lines = tuple(sfilter(connect(ij, (ij[0], w - ij[1] - 1)), ff) for ij in linesps)
    lines2 = tuple(sfilter(connect(ij, (h - ij[0] - 1, ij[1])), ff2) for ij in linesps2)
    lines = merge({recolor(col, l1 | l2) for col, (l1, l2) in zip(colors, zip(lines, lines2))})
    gobase = paint(gi, lines)
    go = paint(gobase, merge(fgpartition(vmirror(gobase))))
    go = paint(go, merge(fgpartition(hmirror(gobase))))
    go = paint(go, merge(fgpartition(vmirror(hmirror(gobase)))))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Reading (shared with the verifier's rot90-x4 loop and the golf entry's
    "each cell ORs with its mirror"): the picture is four-fold symmetric, so
    the corner cluster of marks is FIRST replicated into all four corners;
    each same-coloured mark pair {(a, a+2), (a+2, a)} then becomes a closed
    frame: every one of its four sides is one dotted line joining a mark to
    its own mirror image across the grid.

    Trajectory:
      1. CopyI the seed quadrant -> Paste in the horizontally opposite quadrant
         (+ Color0 for 0-coloured marks, which a copy never carries) -> FlipH.
      2. CopyO the finished half -> Paste in the opposite half (+ Color0) -> FlipV.
      3. For every mark pair, draw its frame: top, bottom, left, right sides,
         each side one Color op on the odd cells strictly between the two
         marks that now bound it.
    """
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ch, cw = (H - 1) // 2, (W - 1) // 2

    # every example input is drawn on the same canvas colour; the mark
    # cluster is tiny, so the dominant colour is the canvas
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    g = I.copy()

    fg = [(r, c) for r in range(H) for c in range(W) if I[r, c] != bgc]
    top = (max(r for r, _ in fg) < ch) if fg else True
    left = (max(c for _, c in fg) < cw) if fg else True

    def tor(rr):
        return rr if top else H - 1 - rr

    def toc(cc):
        return cc if left else W - 1 - cc

    def paste_block(src_r, src_c, hh, ww, dst_r, dst_c):
        """Simulate a transparent Paste; return the cells a 0 failed to carry."""
        src = g[src_r:src_r + hh, src_c:src_c + ww].copy()
        miss = []
        for i in range(hh):
            for j in range(ww):
                if src[i, j] != 0:
                    g[dst_r + i, dst_c + j] = src[i, j]
                elif g[dst_r + i, dst_c + j] != 0:
                    miss.append((dst_r + i, dst_c + j))
        return miss

    # ---- 1. replicate the seed corner across the centre column -------------
    qr = 0 if top else ch + 1             # seed quadrant rows  qr .. qr+ch-1
    qc = 0 if left else cw + 1            # seed quadrant cols  qc .. qc+cw-1
    oc = cw + 1 if left else 0            # opposite quadrant cols
    # full rectangle: the whole seed quadrant, background included
    before = g.copy()
    miss = paste_block(qr, qc, ch, cw, qr, oc)
    if not np.array_equal(before, g):   # skip a Paste that would change nothing
        ops.append(28); sels.append([qr, qc, ch - 1, cw - 1])      # CopyI
        ops.append(30); sels.append([qr, oc, 0, 0])                # Paste
    if miss:   # 0-coloured marks: the copied block's colour-0 cells
        ops.append(0); sels.append(sel_of(miss))
        for (r, c) in miss:
            g[r, c] = 0
    # full rectangle: mirror the pasted quadrant in place
    blk = g[qr:qr + ch, oc:oc + cw]
    if not np.array_equal(blk, np.fliplr(blk)):   # symmetric block: flip is a no-op
        ops.append(26); sels.append([qr, oc, ch - 1, cw - 1])      # FlipH
        g[qr:qr + ch, oc:oc + cw] = np.fliplr(blk)

    # ---- 2. replicate the marked half across the centre row ----------------
    orow = ch + 1 if top else 0
    # full rectangle: the whole marked half (both corners), background included
    before = g.copy()
    miss = paste_block(qr, 0, ch, W, orow, 0)
    if not np.array_equal(before, g):
        ops.append(29); sels.append([qr, 0, ch - 1, W - 1])        # CopyO
        ops.append(30); sels.append([orow, 0, 0, 0])               # Paste
    if miss:
        ops.append(0); sels.append(sel_of(miss))
        for (r, c) in miss:
            g[r, c] = 0
    blk = g[orow:orow + ch, :]
    if not np.array_equal(blk, np.flipud(blk)):
        ops.append(27); sels.append([orow, 0, ch - 1, W - 1])      # FlipV
        g[orow:orow + ch, :] = np.flipud(blk)

    # ---- 3. every mark pair closes into a dotted frame ---------------------
    ub = min(H, W) // 4
    for k in range(max(0, ub - 1)):
        a, b = 2 * k + 1, 2 * k + 3       # pair (a, b) & (b, a), seed-corner frame
        col = int(I[tor(a), toc(b)])
        if col == bgc:
            continue
        # frame sides in absolute coordinates: two rows, two columns
        ra, rb = a, H - 1 - a             # top / bottom side rows
        ca, cb = a, W - 1 - a             # left / right side columns
        sides = [
            [(ra, c) for c in range(b + 2, W - 1 - b, 2)],   # top side
            [(rb, c) for c in range(b + 2, W - 1 - b, 2)],   # bottom side
            [(r, ca) for r in range(b + 2, H - 1 - b, 2)],   # left side
            [(r, cb) for r in range(b + 2, H - 1 - b, 2)],   # right side
        ]
        for cells in sides:
            cells = [(r, c) for (r, c) in cells if g[r, c] != col]
            if not cells:
                continue
            ops.append(col); sels.append(sel_of(cells))
            for (r, c) in cells:
                g[r, c] = col

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
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
                                f"for task 9d9215db"
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
                    f"Failed to build a complete episode for task 9d9215db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9d9215db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
