"""
ARC Task: d8c310e9 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]  # number of CW quarter turns applied to the canonical layout


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, rot=None) -> dict:
    from dsl import (interval, remove, canvas, shift, paint,
                     identity, rot90, rot180, rot270)
    from random import sample, choice, randint
    from utils import unifint
    if bgc is None:
        bgc = random.choice(range(10))
    if rot is None:
        rot = random.choice(ROTS)
    cols = interval(0, 10, 1)
    # a quarter turn swaps the canvas dimensions
    hb, wb = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = unifint(diff_lb, diff_ub, (3, min(30, hb)))
    w = unifint(diff_lb, diff_ub, (10, min(30, wb)))
    p = unifint(diff_lb, diff_ub, (2, (w - 1) // 3))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    obj = set()
    for j in range(p):
        numcells = unifint(diff_lb, diff_ub, (1, h - 1))
        for ii in range(h - 1, h - numcells - 1, -1):
            obj.add((choice(ccols), (ii, j)))
    gi = canvas(bgc, (h, w))
    minobj = obj | shift(obj, (0, p))
    addonw = randint(0, p)
    addon = frozenset(c for c in obj if c[1][1] < addonw)
    fullobj = minobj | addon
    # the visible block must pin its period down: shifting too far off the edge
    # leaves fewer than (2p - 2) columns and admits a shorter false period
    leftshift = randint(0, min(addonw, 2))
    fullobj = shift(fullobj, (0, -leftshift))
    gi = paint(gi, fullobj)
    go = tuple(e for e in gi)
    for j in range(w // (2 * p) + 2):
        go = paint(go, shift(fullobj, (0, j * 2 * p)))
    fn = (identity, rot90, rot180, rot270)[rot]
    return {"input": fn(gi), "output": fn(go)}


def derive_operations(I, O=None, examples=None):
    """Reading: the repeating unit is ONE period strip.  The block hugs one edge; the
    one-period strip at that edge is copied from I and stamped again and again away
    from the edge until the grid ends.  Everything is measured from I; O is unused."""
    A = np.asarray(I, dtype=int)
    H, W = A.shape
    full = [0, 0, H - 1, W - 1]

    # frame: which quarter turn leaves top row + right column pure background
    axis = {0: (0, 1), 3: (-1, 0), 2: (0, -1), 1: (1, 0)}
    kk = None
    for k in (0, 3, 2, 1):
        C = np.rot90(A, k)
        if len(set(C[0].tolist())) == 1 and len(set(C[:, -1].tolist())) == 1:
            kk = k
            break
    if kk is None:
        return [34], [full]
    C = np.rot90(A, kk)
    bgc = int(C[0, -1])
    dr, dc = axis[kk]
    cells = np.argwhere(A != bgc)
    if len(cells) == 0:
        return [34], [full]
    r0, r1 = int(cells[:, 0].min()), int(cells[:, 0].max())
    c0, c1 = int(cells[:, 1].min()), int(cells[:, 1].max())

    horiz = dc != 0
    # lines across the march axis (columns if marching horizontally, rows otherwise)
    if horiz:
        lines = [A[r0:r1 + 1, c] for c in range(c0, c1 + 1)]
    else:
        lines = [A[r, c0:c1 + 1] for r in range(r0, r1 + 1)]
    n = len(lines)
    per = n
    for cand in range(1, n):
        if all(np.array_equal(lines[i], lines[i - cand]) for i in range(cand, n)):
            per = cand
            break
    step = dr + dc  # +1 marches toward larger index, -1 toward smaller
    lo, hi_ = (c0, c1) if horiz else (r0, r1)
    L = W if horiz else H
    # the one-period strip at the edge the block is anchored to
    t0 = lo if step > 0 else hi_ - per + 1

    grid = A.copy()
    ops, sels = [], []
    clip_rect = None
    k = 1
    while True:
        s = t0 + step * k * per
        e = s + per - 1
        if e < 0 or s >= L:
            break
        a, b = max(0, s), min(L - 1, e)       # part of this stamp inside the grid
        u0, u1 = a - s + t0, b - s + t0       # matching part of the strip
        if horiz:
            src = A[r0:r1 + 1, u0:u1 + 1]
            rect = [r0, u0, r1 - r0, u1 - u0]
            dst = (r0, a)
        else:
            src = A[u0:u1 + 1, c0:c1 + 1]
            rect = [u0, c0, u1 - u0, c1 - c0]
            dst = (a, c0)
        sh, sw = src.shape
        reg = grid[dst[0]:dst[0] + sh, dst[1]:dst[1] + sw]
        after = np.where(src != 0, src, reg)
        if not np.array_equal(after, reg):
            if clip_rect != rect:
                # full rectangle: the period strip itself (or the part that fits)
                ops.append(28); sels.append(list(rect))
                clip_rect = rect
            ops.append(30); sels.append([dst[0], dst[1], 0, 0])
            grid[dst[0]:dst[0] + sh, dst[1]:dst[1] + sw] = after
        # Paste never writes colour 0: black the strip's 0-cells at this stamp by hand
        black = [(dst[0] + i, dst[1] + j) for i in range(sh) for j in range(sw)
                 if src[i, j] == 0 and grid[dst[0] + i, dst[1] + j] != 0]
        if black:
            ops.append(0); sels.append(sel_of(black))
            for rr, cc in black:
                grid[rr, cc] = 0
        k += 1
    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
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
                                f"for task d8c310e9"
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
                    f"Failed to build a complete episode for task d8c310e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d8c310e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
