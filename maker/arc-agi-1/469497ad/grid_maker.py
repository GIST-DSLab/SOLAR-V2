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


def sample_colors(num_examples=None) -> dict:
    # generator samples: bgc, sqc, and (numc-1) band colors -- 2 is reserved for the diagonals
    cols = [c for c in range(10) if c != 2]
    picks = random.sample(cols, 8)
    return {"bgc": picks[0], "sqc": picks[1], "ccol_pool": tuple(picks[2:])}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, sqc=None, ccol_pool=None) -> dict:
    M = min(max_h, max_w)                       # rot90 can swap dims -> bound both by M
    hub = max(3, min(6, M // 2))
    h = unifint(diff_lb, diff_ub, (3, hub))
    w = unifint(diff_lb, diff_ub, (3, hub))
    if bgc is None or sqc is None or ccol_pool is None:
        cols = [c for c in range(10) if c != 2]
        picks = random.sample(cols, 8)
        bgc, sqc, ccol_pool = picks[0], picks[1], tuple(picks[2:])
    gi = canvas(bgc, (h, w))
    sqh = randint(1, h - 2)
    sqw = randint(1, w - 2)
    sqloci = randint(0, h - sqh - 2)
    sqlocj = randint(0, w - sqw - 2)
    sq = backdrop(frozenset({(sqloci, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)}))
    gi = fill(gi, sqc, sq)
    numcub = min(min(min(h, w) + 1, M // max(h, w)), 7)
    numcub = max(2, numcub)
    numc = unifint(diff_lb, diff_ub, (2, numcub))
    numaccc = numc - 1
    ccols = list(ccol_pool)[:numaccc]
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


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    k = ho // hi                                  # upscale factor = numcolors(I) - 1

    def ring_colors(r0, c0, r1, c1):
        out = set()
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if r == r0 - 1 or r == r1 + 1 or c == c0 - 1 or c == c1 + 1:
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

    def rects(cells):
        m = np.zeros((hi, wi), dtype=bool)
        for (r, c) in cells:
            m[r, c] = True
        out = []
        while m.any():
            r, c = [int(v) for v in np.argwhere(m)[0]]
            rw = 1
            while c + rw < wi and m[r, c + rw]:
                rw += 1
            rh = 1
            while r + rh < hi and m[r + rh, c:c + rw].all():
                rh += 1
            out.append((r, c, rh, rw))
            m[r:r + rh, c:c + rw] = False
        return out

    # --- rule step 1: locate the square = color fully wrapped by ONE uniform color
    #     (the ring colour is the background); ties broken by smallest bbox.
    best = None
    for c in sorted(set(I.flatten().tolist())):
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

    # --- rule step 2: upscale I by k -- grow the canvas, lay the background,
    #     then re-draw every non-background object as its k-times blown-up rectangles.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    if bgc != 0:
        ops.append(int(bgc)); sels.append([0, 0, ho - 1, wo - 1])
    else:
        ops.append(0); sels.append([0, 0, hi - 1, wi - 1])   # only the stale input footprint

    order = [c for c in sorted(set(I.flatten().tolist())) if c not in (bgc, sqc)] + [sqc]
    for c in order:
        for comp in components(I == c):
            for (r, cc, rh, rw) in rects(comp):
                ops.append(int(c))
                sels.append([r * k, cc * k, rh * k - 1, rw * k - 1])

    # --- rule step 3: shoot a diagonal outward from each corner of the upscaled square,
    #     marking 2 wherever the ray runs over background. One ray at a time, corner outward.
    rr0, cc0 = sr0 * k, sc0 * k
    rr1, cc1 = (sr1 + 1) * k - 1, (sc1 + 1) * k - 1
    rays = [((rr0, cc0), (-1, -1)), ((rr0, cc1), (-1, 1)),
            ((rr1, cc1), (1, 1)), ((rr1, cc0), (1, -1))]
    for (r, c), (dr, dc) in rays:
        r += dr; c += dc
        while 0 <= r < ho and 0 <= c < wo:
            if I[r // k, c // k] == bgc:
                ops.append(2); sels.append([int(r), int(c), 0, 0])
            r += dr; c += dc

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
