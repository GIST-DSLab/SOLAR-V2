"""
ARC Task: 3bd67248 (RE-ARC) — LLM-generated grid_maker
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
except Exception:  # pragma: no cover - fallback if helper unavailable
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


ROTATIONS = ["identity", "rot90", "rot180", "rot270"]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = min(max(a, int(a + diff_lb * (b - a))), b)
    hi = min(max(a, int(a + diff_ub * (b - a))), b)
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # generator: cols = interval(0,10,1) minus (2,4); bgc, linc = sample(cols, 2)
    cols = [c for c in range(10) if c not in (2, 4)]
    bgc, linc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTATIONS):
        examples = [{"rotation": r} for r in ROTATIONS]
        examples += [{"rotation": random.choice(ROTATIONS)}
                     for _ in range(n_ex - len(ROTATIONS))]
        random.shuffle(examples)
    else:
        examples = [{"rotation": r} for r in random.sample(ROTATIONS, n_ex)]
    plan = [dict(e) for e in examples] + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def _rot(g, rotation):
    if rotation == "identity":
        return [list(r) for r in g]
    if rotation == "rot90":   # CW
        return [list(r) for r in zip(*g[::-1])]
    if rotation == "rot180":
        return [list(r)[::-1] for r in g[::-1]]
    # rot270 == CCW
    return [list(r) for r in zip(*g)][::-1]


def _upscale(g, fac):
    out = []
    for row in g:
        newrow = []
        for v in row:
            newrow.extend([v] * fac)
        for _ in range(fac):
            out.append(list(newrow))
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, rotation=None, **kwargs) -> dict:
    if rotation is None:
        rotation = random.choice(ROTATIONS)

    keeps_shape = rotation in ("identity", "rot180")
    hmax = min(15, max_h if keeps_shape else max_w)
    wmax = min(15, max_w if keeps_shape else max_h)
    hmax = max(3, hmax)
    wmax = max(3, wmax)

    h = _unifint(diff_lb, diff_ub, (3, hmax))
    w = _unifint(diff_lb, diff_ub, (3, wmax))

    H0, W0 = (h, w) if keeps_shape else (w, h)
    facmax = min(30 // max(h, w), max(1, max_h // H0), max(1, max_w // W0))
    facmax = max(1, facmax)
    fac = _unifint(diff_lb, diff_ub, (1, facmax))

    # base input: background canvas with the line colour on the whole left column
    gi = [[bgc for _ in range(w)] for _ in range(h)]
    for r in range(h):
        gi[r][0] = linc

    # base output: colour 4 along the bottom row (excluding the line column),
    # and a colour-2 ray shooting up-right from just above/right of the corner
    go = [list(row) for row in gi]
    for c in range(1, w):
        go[h - 1][c] = 4
    r, c = h - 2, 1
    while 0 <= r < h and 0 <= c < w:
        go[r][c] = 2
        r -= 1
        c += 1

    gi = _upscale(_rot(gi, rotation), fac)
    go = _upscale(_rot(go, rotation), fac)
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ho, wo = O.shape

    # ---- read the structure out of I -------------------------------------
    cnt = Counter(I.flatten().tolist())
    # the line colour is always the minority colour (line = 1 base column of w>=3)
    linc = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0][0]

    edge = None
    if np.all(I[:, 0] == linc):
        edge = "left"
    elif np.all(I[:, W - 1] == linc):
        edge = "right"
    elif np.all(I[0, :] == linc):
        edge = "top"
    elif np.all(I[H - 1, :] == linc):
        edge = "bottom"
    if edge is None:
        ops = [34]
        sels = [sel_of([(r, c) for r in range(ho) for c in range(wo)])]
        return ops, sels

    # thickness of the line band == upscaling factor
    fac = 0
    if edge == "left":
        while fac < W and np.all(I[:, fac] == linc):
            fac += 1
    elif edge == "right":
        while fac < W and np.all(I[:, W - 1 - fac] == linc):
            fac += 1
    elif edge == "top":
        while fac < H and np.all(I[fac, :] == linc):
            fac += 1
    else:
        while fac < H and np.all(I[H - 1 - fac, :] == linc):
            fac += 1
    fac = max(1, fac)
    while fac > 1 and (H % fac or W % fac):
        fac -= 1

    RB, CB = H // fac, W // fac

    # (u, v) block coordinates: u = distance from the "4" edge,
    # v = distance from the line edge.  The "4" edge is the line edge turned
    # 90 deg counter-clockwise (left->bottom, top->left, right->top, bottom->right).
    if edge == "left":       # four edge = bottom
        U, V = RB, CB
        def blk(u, v): return (RB - 1 - u, v)
    elif edge == "top":      # four edge = left
        U, V = CB, RB
        def blk(u, v): return (v, u)
    elif edge == "right":    # four edge = top
        U, V = RB, CB
        def blk(u, v): return (u, CB - 1 - v)
    else:                    # edge == "bottom", four edge = right
        U, V = CB, RB
        def blk(u, v): return (RB - 1 - v, CB - 1 - u)

    def block_cells(rb, cb):
        return [(rb * fac + dr, cb * fac + dc)
                for dr in range(fac) for dc in range(fac)]

    ops, sels = [], []

    # 1) the colour-4 band: the whole edge strip next to the line, minus the
    #    block shared with the line band itself (v = 0).
    band = []
    for v in range(1, V):
        rb, cb = blk(0, v)
        band.extend(block_cells(rb, cb))
    if band:
        ops.append(4)
        sels.append(sel_of(band))

    # 2) the colour-2 ray: one block per step, starting at the block diagonally
    #    inward from the corner and marching away from both edges.
    K = min(U - 1, V - 1)
    for k in range(1, K + 1):
        rb, cb = blk(k, k)
        if 0 <= rb < RB and 0 <= cb < CB:
            ops.append(2)
            sels.append(sel_of(block_cells(rb, cb)))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(ho) for c in range(wo)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 3bd67248"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3bd67248"
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
                                f"for task 3bd67248"
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
                    f"Failed to build a complete episode for task 3bd67248 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3bd67248-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
