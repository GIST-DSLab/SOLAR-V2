"""
ARC Task: a79310a0 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    # generator: cols = interval(0,10) without 2; bgc and fgc both sampled from it
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        lo = max(a, min(lo, b))
        hi = max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    h = unifint(diff_lb, diff_ub, (2, max(2, max_h)))
    w = unifint(diff_lb, diff_ub, (2, max(2, max_w)))
    nc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 2 - 1)))

    bounds = {(i, j) for i in range(h) for j in range(w)}
    ch = random.choice(sorted(bounds))
    shp = {ch}
    bounds.discard(ch)
    for _ in range(nc - 1):
        cands = set()
        for (i, j) in shp:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        cands.add((i + di, j + dj))
        cands = sorted((cands & bounds) - shp)
        if not cands:
            break
        nxt = random.choice(cands)
        shp.add(nxt)
        bounds.discard(nxt)

    mi = min(i for i, j in shp)
    mj = min(j for i, j in shp)
    shp = {(i - mi, j - mj) for i, j in shp}
    oh = max(i for i, j in shp) + 1
    ow = max(j for i, j in shp) + 1

    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    plcd = {(i + loci, j + locj) for i, j in shp}

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    for (i, j) in plcd:
        gi[i][j] = fgc
    go = [[bgc for _ in range(w)] for _ in range(h)]
    for (i, j) in plcd:
        if 0 <= i + 1 < h:
            go[i + 1][j] = 2

    return {"input": gi, "output": go}


def derive_operations(I, O):
    """Rule (read off the examples): the one blob slides down a single row and is
    repainted 2; whatever is pushed past the bottom edge is lost.

    The blob is a single object, so there is no second region to pick first and no
    sweep direction to choose -- but there is a real choice of route, and it is
    settled here from the input alone:

      A. paint the blob 2 where it stands, then slide it down;
      B. slide the blob down first, then paint it 2 where it landed;
      C. slide it down, lay the background back over the whole footprint it
         started on, then draw the blob on top of that base in 2.

    A is forced when the blob's own colour is 0: ARCLE reads 0 as nothing there,
    so a Move cannot pick the blob up until it has been given a visible colour.
    """
    def sel_of(cells):
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background = the colour the generator paints the canvas with; the blob covers
    # strictly less than half the cells (nc <= h*w//2 - 1), so majority colour is safe.
    bgc = int(np.bincount(I.ravel(), minlength=10).argmax())

    # the single 8-connected blob = every non-background cell of I
    src = sorted((r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc)

    ops, sels = [], []
    if not src:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    fgc = int(I[src[0][0], src[0][1]])
    dst = sorted({(r + 1, c) for (r, c) in src if r + 1 < hi})   # where it lands
    hole = sorted(set(src) - set(dst))      # footprint the blob no longer covers
    overlap = sorted(set(src) & set(dst))   # footprint it still covers after sliding

    # Which route? A count and a corner of the blob decide it -- both read off the
    # input, so the same input always takes the same route and the answer grid is
    # never consulted.
    route = "ABC"[(len(src) + src[0][0] + src[0][1]) % 3]
    if fgc == 0 or not dst:
        # nothing for ARCLE to grab until it is coloured / nothing left to repaint
        route = "A"
    if route == "C" and bgc == 0 and not overlap:
        # the base would land only on cells the slide already left at 0: no visible
        # change, so take the plain route instead
        route = "B"

    if route == "A":
        # 1. paint the blob 2 where it stands
        ops.append(2)
        sels.append(sel_of(src))
        # 2. slide it down one row (one grab; ARCLE clips what goes off the bottom)
        ops.append(21)
        sels.append(sel_of(src))
        # 3. the grab zeroed the blob's old footprint -- give the part it no longer
        #    covers the background back
        if bgc != 0:
            ops.append(int(bgc))
            sels.append(sel_of(hole))
    elif route == "B":
        # 1. slide the blob down one row first, still in its own colour
        ops.append(21)
        sels.append(sel_of(src))
        # 2. background back over the part of the old footprint it left behind
        if bgc != 0:
            ops.append(int(bgc))
            sels.append(sel_of(hole))
        # 3. now paint it 2 where it has landed
        if dst:
            ops.append(2)
            sels.append(sel_of(dst))
    else:  # route C -- lay the background down, then draw the blob on top of it
        # 1. slide the blob down one row
        ops.append(21)
        sels.append(sel_of(src))
        # 2. lay the background back over the WHOLE footprint the blob started on,
        #    the part it left and the part it still stands on alike
        ops.append(int(bgc))
        sels.append(sel_of(src))
        # 3. draw the blob back in 2 over that base, at its new place
        ops.append(2)
        sels.append(sel_of(dst))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a79310a0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a79310a0"
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
                                f"for task a79310a0"
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
                    f"Failed to build a complete episode for task a79310a0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a79310a0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
