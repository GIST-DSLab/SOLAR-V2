"""
ARC Task: e40b9e2f (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """Background is fixed for the whole episode (0), foreground palette is a
    fixed per-episode ordering of the remaining colors.  The rule itself is
    purely geometric (complete the square block to 4-fold rotational symmetry),
    so which fg colors appear does not change the rule - only the palette needs
    to stay consistent across the instances of an episode."""
    bgc = 0
    palette = [c for c in range(10) if c != bgc]
    random.shuffle(palette)
    return {"bgc": bgc, "palette": palette}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int = 0, palette=None) -> dict:
    cols = interval(0, 10, 1)
    hub = max(6, int(max_h))
    wub = max(6, int(max_w))
    h = unifint(diff_lb, diff_ub, (6, hub))
    w = unifint(diff_lb, diff_ub, (6, wub))
    d = unifint(diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = randint(0, h - d)
    locj = randint(0, w - d)
    loc = (loci, locj)
    remcols = remove(bgc, cols)
    if palette is None:
        palette = list(remcols)
        random.shuffle(palette)
    pal = [c for c in palette if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = tuple(pal[:numcols])
    subg = canvas(bgc, (d, d))
    inds = asindices(subg)
    if d % 2 == 0:
        q = sfilter(inds, lambda ij: ij[0] < d // 2 and ij[1] < d // 2)
        cp = {(d // 2 - 1, d // 2 - 1), (d // 2, d // 2 - 1),
              (d // 2 - 1, d // 2), (d // 2, d // 2)}
    else:
        q = sfilter(inds, lambda ij: ij[0] < d // 2 and ij[1] <= d // 2)
        cp = {(d // 2, d // 2)} | ineighbors((d // 2, d // 2))
    nrings = unifint(diff_lb, diff_ub, (1, max(1, (d - 2) // 2)))
    rings = set()
    for k in range(nrings):
        ring = box({(k, k), (d - k - 1, d - k - 1)})
        rings = rings | ring
    qin = q - rings
    qout = rings & q
    ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
    tailobjcells = sample(totuple(q), ntailobjcells)
    tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
    tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    while hmirror(tailobj) == tailobj and vmirror(tailobj) == tailobj:
        ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
        tailobjcells = sample(totuple(q), ntailobjcells)
        tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
        tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    for k in range(4):
        subg = paint(subg, tailobj)
        subg = rot90(subg)
    fxobj = recolor(choice(ccols), cp)
    subg = paint(subg, fxobj)
    subgi = subg
    subgo = tuple(e for e in subgi)
    subgi = fill(subgi, bgc, rings)
    nsplits = unifint(diff_lb, diff_ub, (1, 4))
    splits = [set() for k in range(nsplits)]
    for idx, cel in enumerate(tailobj):
        splits[idx % nsplits].add(cel)
    for jj in range(4):
        if jj < len(splits):
            subgi = paint(subgi, splits[jj])
        subgi = rot90(subgi)
    subgi = paint(subgi, fxobj)
    rotf = choice((identity, rot90, rot180, rot270))
    subgi = rotf(subgi)
    subgo = rotf(subgo)
    gi = paint(canvas(bgc, (h, w)), shift(asobject(subgi), loc))
    go = paint(canvas(bgc, (h, w)), shift(asobject(subgo), loc))
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def _closure(blk, bgc):
    """Union of the four rotations of the square block; None on color conflict."""
    out = np.full(blk.shape, bgc, dtype=int)
    for k in (0, 1, 2, 3):
        rb = np.rot90(blk, k=k)
        mask = rb != bgc
        if bool((mask & (out != bgc) & (out != rb)).any()):
            return None
        out[mask] = rb[mask]
    return out


def _center_ok(blk, d, bgc):
    """The generator always stamps a uniformly colored center marker
    (odd d: center + its 4 diagonal neighbours, even d: the central 2x2)."""
    m = d // 2
    if d % 2 == 1:
        cells = [(m, m), (m - 1, m - 1), (m - 1, m + 1), (m + 1, m - 1), (m + 1, m + 1)]
    else:
        cells = [(m - 1, m - 1), (m - 1, m), (m, m - 1), (m, m)]
    vals = {int(blk[i, j]) for i, j in cells}
    return len(vals) == 1 and bgc not in vals


def _find_square_block(I, O, bgc):
    """Locate the d x d square the pattern lives in, using only properties of I:
    it contains every fg cell, carries the center marker, its four rotations are
    color-consistent, the completed pattern touches all four of its sides, and the
    cells still missing sit in the outer ring band."""
    h, w = I.shape
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    if not fg:
        return None
    a = min(r for r, _ in fg)
    b = max(r for r, _ in fg)
    p = min(c for _, c in fg)
    q = max(c for _, c in fg)
    span = max(b - a + 1, q - p + 1)
    best = None
    for d in range(max(4, span), min(h, w) + 1):
        for r0 in range(max(0, b - d + 1), min(a, h - d) + 1):
            for c0 in range(max(0, q - d + 1), min(p, w - d) + 1):
                blk = I[r0:r0 + d, c0:c0 + d]
                if not _center_ok(blk, d, bgc):
                    continue
                cl = _closure(blk, bgc)
                if cl is None:
                    continue
                nz = np.argwhere(cl != bgc)
                if nz.size == 0:
                    continue
                if nz[:, 0].min() != 0 or nz[:, 1].min() != 0:
                    continue
                if nz[:, 0].max() != d - 1 or nz[:, 1].max() != d - 1:
                    continue
                miss = [(i, j) for i in range(d) for j in range(d)
                        if blk[i, j] == bgc and cl[i, j] != bgc]
                if not miss:
                    continue
                lim = max(1, (d - 2) // 2) - 1
                if any(min(i, j, d - 1 - i, d - 1 - j) > lim for i, j in miss):
                    continue
                if best is None:
                    best = (r0, c0, d)
                full = I.copy()
                full[r0:r0 + d, c0:c0 + d] = cl
                if O is not None and np.array_equal(full, O):
                    return (r0, c0, d)
    return best


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    ops, sels = [], []
    grid = I.copy()

    info = _find_square_block(I, O, bgc)
    if info is not None:
        r0, c0, d = info
        # selection is EXACTLY the full d x d square region (background included),
        # which is what Rotate / CopyI must act on -> bbox form is the intended mask
        rect = [r0, c0, d - 1, d - 1]
        src = I[r0:r0 + d, c0:c0 + d].copy()

        if bgc == 0:
            # keep the original square on the clipboard: it is re-laid over every turn
            ops.append(28); sels.append(rect)

        for _ in range(3):
            rot = grid.copy()
            rot[r0:r0 + d, c0:c0 + d] = np.rot90(grid[r0:r0 + d, c0:c0 + d], k=3)  # op25 = CW
            if np.array_equal(rot, grid):
                break
            ops.append(25); sels.append(rect)          # turn the square a quarter turn
            grid = rot

            if bgc == 0:
                # lay the untouched original square back on top (0 = transparent)
                new = grid.copy()
                sub = new[r0:r0 + d, c0:c0 + d]
                m = src != 0
                sub[m] = src[m]
                if not np.array_equal(new, grid):
                    ops.append(30); sels.append([r0, c0, 0, 0])
                    grid = new
            else:
                # non-zero background: Paste is not transparent, so redraw the
                # original square's own cells, one op per color
                for col in sorted({int(v) for v in src.flatten().tolist()}):
                    if col == bgc:
                        continue
                    cells = [(r0 + i, c0 + j) for i in range(d) for j in range(d)
                             if int(src[i, j]) == col and int(grid[r0 + i, c0 + j]) != col]
                    if cells:
                        ops.append(col); sels.append(sel_of(cells))
                        for (r, c) in cells:
                            grid[r, c] = col

            if np.array_equal(grid, O):
                break

    if not np.array_equal(grid, O):      # safety net, grouped per color
        rem = {}
        for r in range(h):
            for c in range(w):
                if grid[r, c] != O[r, c]:
                    rem.setdefault(int(O[r, c]), []).append((r, c))
        for col in sorted(rem):
            cells = rem[col]
            ops.append(col); sels.append(sel_of(cells))
            for (r, c) in cells:
                grid[r, c] = col

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
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
                                f"for task e40b9e2f"
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
                    f"Failed to build a complete episode for task e40b9e2f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e40b9e2f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
