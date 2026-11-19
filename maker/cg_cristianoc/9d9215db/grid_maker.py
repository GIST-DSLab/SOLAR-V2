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
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    # Only the background colour is a fixed role; mark colours are free per instance
    # (the rule copies and connects whatever colour a mark has).
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    lim = min(max_h, max_w)                      # grid may be rotated, so cap both sides
    hub = max(5, min(14, (lim - 1) // 2))
    h = _unifint(diff_lb, diff_ub, (5, hub)) * 2 + 1
    w = _unifint(diff_lb, diff_ub, (5, hub)) * 2 + 1
    remcols = [c for c in cols if c != bgc]
    ub = min(h, w) // 4
    nrings = _unifint(diff_lb, diff_ub, (1, ub))
    onlinesbase = [(2 * k + 1, 2 * k + 1) for k in range(ub)]
    onlines = random.sample(onlinesbase, nrings)
    gi = [[bgc] * w for _ in range(h)]
    for (i, j) in onlines:
        gi[i][j] = random.choice(remcols)
    linsbase = [(i, j + 2) for (i, j) in onlinesbase[:-1]]
    nlines = _unifint(diff_lb, diff_ub, (1, len(linsbase)))
    linesps = random.sample(linsbase, nlines)
    colors = [random.choice(remcols) for _ in range(nlines)]
    for col, (i, j) in zip(colors, linesps):
        gi[i][j] = col
        gi[j][i] = col
    gb = [row[:] for row in gi]
    for col, (i, j) in zip(colors, linesps):
        for c in range(j, w - j):
            if c % 2 == 1:
                gb[i][c] = col
        for r in range(j, h - j):
            if r % 2 == 1:
                gb[r][i] = col
    gb = np.array(gb)
    go = gb.copy()
    for m in (np.fliplr(gb), np.flipud(gb), np.fliplr(np.flipud(gb))):
        go = np.where(m != bgc, m, go)
    gi = np.array(gi)
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Reading (mirror first, then connect):
      1. The input is one quarter of a figure that is symmetric about both axes.
         First copy the MARKS into all four corners: mirror the seed column-half
         across the centre column (CopyI / Paste / FlipH), then mirror the
         now full-width row-half across the centre row (CopyO / Paste / FlipV).
         Marks coloured 0 do not travel through the clipboard, so they are
         painted where the copy landed.
      2. Each mark pair now has a partner of the same colour on the far side of
         the grid. Connect each pair by painting the odd-spaced cells strictly
         between the two partners: one row segment or one column segment at a time.
    O is never read. Every selection comes from I or from the grid as it stands.
    """
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ch, cw = (H - 1) // 2, (W - 1) // 2
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    ops, sels = [], []
    g = I.copy()

    fg = [(r, c) for r in range(H) for c in range(W) if I[r, c] != bgc]
    top = max(r for r, _ in fg) < ch if fg else True
    left = max(c for _, c in fg) < cw if fg else True

    def mirror_block(copy_op, r0, c0, bh, bw, r1, c1, flip_op):
        nonlocal g
        src_grid = I if copy_op == 28 else g
        src = src_grid[r0:r0 + bh, c0:c0 + bw].copy()
        after = g.copy()
        dst = after[r1:r1 + bh, c1:c1 + bw]
        dst[src != 0] = src[src != 0]
        if not np.array_equal(after, g):
            # full rectangle: the whole half-block, background included
            ops.append(copy_op); sels.append([r0, c0, bh - 1, bw - 1])
            ops.append(30); sels.append([r1, c1, 0, 0])
            g = after
        # 0-coloured marks never travel through the clipboard: paint them
        # on the cells the copy landed on
        rep = [(r1 + i, c1 + j) for i in range(bh) for j in range(bw)
               if src[i, j] == 0 and g[r1 + i, c1 + j] != 0]
        if rep:
            ops.append(0); sels.append(sel_of(rep))
            for (r, c) in rep:
                g[r, c] = 0
        blk = g[r1:r1 + bh, c1:c1 + bw]
        fl = np.fliplr(blk) if flip_op == 26 else np.flipud(blk)
        if not np.array_equal(fl, blk):
            # full rectangle: the pasted block mirrored in place
            ops.append(flip_op); sels.append([r1, c1, bh - 1, bw - 1])
            g[r1:r1 + bh, c1:c1 + bw] = fl

    # 1a. seed column-half -> other column-half (mirror about centre column)
    c_src = 0 if left else cw + 1
    c_dst = cw + 1 if left else 0
    mirror_block(28, 0, c_src, H, cw, 0, c_dst, 26)
    # 1b. seed row-half (now full width) -> other row-half (mirror about centre row)
    r_src = 0 if top else ch + 1
    r_dst = ch + 1 if top else 0
    mirror_block(29, r_src, 0, ch, W, r_dst, 0, 27)

    # 2. connect every mark with its mirrored partner
    def tor(rr):
        return rr if top else H - 1 - rr

    def toc(cc):
        return cc if left else W - 1 - cc

    ub = min(H, W) // 4
    for k in range(max(0, ub - 1)):
        a, b = 2 * k + 1, 2 * k + 3
        col = int(I[tor(a), toc(b)])
        if col == bgc:
            continue                              # this ring carries no pair
        segs = []
        for rr in (a, H - 1 - a):                 # row segments between partners
            segs.append([(tor(rr), toc(c)) for c in range(b + 2, W - 1 - b, 2)])
        for cc in (a, W - 1 - a):                 # column segments between partners
            segs.append([(tor(r), toc(cc)) for r in range(b + 2, H - 1 - b, 2)])
        for cells in segs:
            cells = [p for p in cells if g[p] != col]
            if cells:
                ops.append(col); sels.append(sel_of(cells))
                for p in cells:
                    g[p] = col

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
