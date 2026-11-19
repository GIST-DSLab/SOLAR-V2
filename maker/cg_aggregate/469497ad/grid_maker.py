"""
ARC Task: 469497ad (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **kw) -> dict:
    from dsl import (remove, interval, canvas, backdrop, fill, rot180, shoot,
                     upscale, ofcolor, lrcorner, ulcorner, urcorner, llcorner,
                     identity, rot90, rot270)
    from random import sample, choice
    from utils import unifint
    cols = remove(2, interval(0, 10, 1))
    if bgc is None or sqc is None:
        bgc, sqc = random.sample(list(cols), 2)
    lim = max(3, min(6, max_h // 2, max_w // 2))
    h = unifint(diff_lb, diff_ub, (3, lim))
    w = unifint(diff_lb, diff_ub, (3, lim))
    gi = canvas(bgc, (h, w))
    sqh = random.randint(1, h - 2)
    sqw = random.randint(1, w - 2)
    sqloci = random.randint(0, h - sqh - 2)
    sqlocj = random.randint(0, w - sqw - 2)
    sq = backdrop(frozenset({(sqloci, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)}))
    gi = fill(gi, sqc, sq)
    numcub = min(min(h, w) + 1, min(max_h, max_w) // max(h, w), 7)
    numc = unifint(diff_lb, diff_ub, (2, max(2, numcub)))
    numaccc = numc - 1
    remcols = remove(bgc, remove(sqc, cols))
    ccols = sample(remcols, numaccc)
    gi = rot180(gi)
    locs = sample(interval(1, min(h, w), 1), numaccc - 1)
    locs = [0] + sorted(locs)
    for c, l in zip(ccols, locs):
        gi = fill(gi, c, shoot((0, l), (0, 1)))
        gi = fill(gi, c, shoot((l, 0), (1, 0)))
    gi = rot180(gi)
    go = upscale(gi, numc)
    rect = ofcolor(go, sqc)
    l1 = shoot(lrcorner(rect), (1, 1))
    l2 = shoot(ulcorner(rect), (-1, -1))
    l3 = shoot(urcorner(rect), (-1, 1))
    l4 = shoot(llcorner(rect), (1, -1))
    ll = l1 | l2 | l3 | l4
    go = fill(go, 2, ll & ofcolor(go, bgc))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    palette = sorted(set(I.flatten().tolist()))
    k = len(palette) - 1                       # upscale factor = numcolors(I) - 1 (from I, not O)
    ho, wo = hi * k, wi * k

    def ring_colors(r0, c0, r1, c1):
        out = set()
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1):
                    if 0 <= r < hi and 0 <= c < wi:
                        out.add(int(I[r, c]))
        return out

    def components(mask):
        seen = np.zeros((hi, wi), dtype=bool)
        comps = []
        for r in range(hi):
            for c in range(wi):
                if mask[r, c] and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    cells = []
                    while stack:
                        a, b = stack.pop()
                        cells.append((a, b))
                        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            na, nb = a + da, b + db
                            if 0 <= na < hi and 0 <= nb < wi and mask[na, nb] and not seen[na, nb]:
                                seen[na, nb] = True
                                stack.append((na, nb))
                    comps.append(cells)
        return comps

    # The square: the colour wrapped by one uniform ring (that ring colour is the
    # background); smallest bbox wins.  (Same reading as the maker.)
    best = None
    for c in palette:
        cells = np.argwhere(I == c)
        r0, c0 = [int(v) for v in cells.min(0)]
        r1, c1 = [int(v) for v in cells.max(0)]
        ring = ring_colors(r0, c0, r1, c1)
        if len(ring) == 1:
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if best is None or area < best[0]:
                best = (area, (r0, c0, r1, c1), next(iter(ring)), int(c))
    _, (sr0, sc0, sr1, sc1), bgc, sqc = best

    ops, sels = [], []
    # 1. grow the canvas to k-times size and lay the background base
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    if bgc != 0:
        ops.append(int(bgc)); sels.append([0, 0, ho - 1, wo - 1])   # whole canvas rectangle
    else:
        ops.append(0); sels.append([0, 0, hi - 1, wi - 1])          # stale input-footprint rectangle

    # 2. Golf reading: the 2-diagonals lie UNDER the picture (2 shows only where the
    #    picture is background).  Draw each diagonal ray from a corner of the blown-up
    #    square all the way to the canvas edge as ONE stroke.  (Rays, not full lines,
    #    because the "square" may be a non-square rectangle.)
    rr0, cc0 = sr0 * k, sc0 * k
    rr1, cc1 = (sr1 + 1) * k - 1, (sc1 + 1) * k - 1
    rays = [((rr0, cc0), (-1, -1)), ((rr0, cc1), (-1, 1)),
            ((rr1, cc1), (1, 1)), ((rr1, cc0), (1, -1))]
    for (r, c), (dr, dc) in rays:
        cells = []
        r += dr; c += dc
        while 0 <= r < ho and 0 <= c < wo:
            cells.append((r, c))
            r += dr; c += dc
        if cells:
            ops.append(2); sels.append(sel_of(cells))

    # 3. Paint every non-background object of I, blown up k times, on top: wherever
    #    the picture is not background it covers the rays.  One op per object.
    for col in [c for c in palette if c not in (bgc, sqc)] + [sqc]:
        for comp in components(I == col):
            big = [(r * k + a, c * k + b) for (r, c) in comp
                   for a in range(k) for b in range(k)]
            ops.append(int(col)); sels.append(sel_of(big))

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
                        f"num_examples+1 ({num_examples + 1}) for task 469497ad"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 469497ad"
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
                                f"for task 469497ad"
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
                    f"Failed to build a complete episode for task 469497ad "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"469497ad-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
