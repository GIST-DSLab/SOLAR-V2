"""
ARC Task: a48eeaf7 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    """bgc / sqc / dotc are the three colors the RE-ARC generator samples.

    dotc is kept non-zero: the dots are the objects that get MOVED, and ARCLE's
    object buffer only holds non-zero cells (a 0-colored object would be a NOOP).
    """
    cols = list(range(10))
    dotc = random.choice([c for c in cols if c != 0])
    rest = [c for c in cols if c != dotc]
    bgc, sqc = random.sample(rest, 2)
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc}


# ---------------------------------------------------------------- generator
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, sqc: int, dotc: int) -> dict:
    hlo, hhi = 8, max(8, max_h)
    wlo, whi = 8, max(8, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, hhi))
    w = unifint(diff_lb, diff_ub, (wlo, whi))
    ih = unifint(diff_lb, diff_ub, (2, h // 2))
    iw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = randint(2, h - ih - 2)
    locj = randint(2, w - iw - 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    sq = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    A = [(x, locj - 1) for x in interval(loci, loci + ih, 1)]
    Ap = [(x, randint(0, locj - 2)) for x in interval(loci, loci + ih, 1)]
    B = [(x, locj + iw) for x in interval(loci, loci + ih, 1)]
    Bp = [(x, randint(locj + iw + 1, w - 1)) for x in interval(loci, loci + ih, 1)]
    C = [(loci - 1, x) for x in interval(locj, locj + iw, 1)]
    Cp = [(randint(0, loci - 2), x) for x in interval(locj, locj + iw, 1)]
    D = [(loci + ih, x) for x in interval(locj, locj + iw, 1)]
    Dp = [(randint(loci + ih + 1, h - 1), x) for x in interval(locj, locj + iw, 1)]
    srarr = Ap + Bp + Cp + Dp
    dearr = A + B + C + D
    inds = interval(0, len(srarr), 1)
    num = unifint(diff_lb, diff_ub, (1, len(srarr)))
    locs = sample(inds, num)
    srarr = [e for j, e in enumerate(srarr) if j in locs]
    dearr = [e for j, e in enumerate(dearr) if j in locs]
    gi = fill(gi, sqc, sq)
    go = fill(go, sqc, sq)
    for s, d in zip(srarr, dearr):
        gi = fill(gi, dotc, {s})
        go = fill(go, dotc, {d})
    ncorn = unifint(diff_lb, diff_ub, (0, 4))
    fullinds = asindices(gi)
    if ncorn > 0:
        go = fill(go, dotc, {(loci - 1, locj - 1)})
        cands = shoot((loci - 2, locj - 2), (-1, -1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 1:
        go = fill(go, dotc, {(loci - 1, locj + iw)})
        cands = shoot((loci - 2, locj + iw + 1), (-1, 1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 2:
        go = fill(go, dotc, {(loci + ih, locj - 1)})
        cands = shoot((loci + ih + 1, locj - 2), (1, -1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 3:
        go = fill(go, dotc, {(loci + ih, locj + iw)})
        cands = shoot((loci + ih + 1, locj + iw + 1), (1, 1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- operations
def derive_operations(I, O):
    """Every dot slides to the nearest cell of the ring one step outside the
    solid rectangle.  That is a pure translation of a 1-cell object -> Move ops
    (grab once, then empty selections), then repair the vacated footprint when
    the background is non-zero."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # background = the canvas color the generator fills before placing anything
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    fg_colors = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    # the block: the non-bg color whose cells fill their whole bounding box,
    # largest such (mirrors fgpartition -> rectangles -> argmax by size)
    box_color, box_bounds, best = None, None, -1
    for c in fg_colors:
        rs, cs = np.where(I == c)
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        if len(rs) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(rs) > best:
            best = len(rs)
            box_color = c
            box_bounds = (r0, r1, c0, c1)
    if box_color is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    dot_colors = [c for c in fg_colors if c != box_color]
    if not dot_colors:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    dot_color = dot_colors[0]

    r0, r1, c0, c1 = box_bounds

    # ring one cell outside the block
    ring = []
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if 0 <= r < h and 0 <= c < w:
                ring.append((r, c))

    dots = [(int(r), int(c)) for r, c in zip(*np.where(I == dot_color))]

    def nearest(rd, cd):
        return min(ring, key=lambda p: (abs(p[0] - rd) + abs(p[1] - cd), p[0], p[1]))

    def side_key(rt, ct, rd, cd):
        if r0 <= rt <= r1 and ct == c0 - 1:
            grp = 0                      # left edge
        elif r0 <= rt <= r1 and ct == c1 + 1:
            grp = 1                      # right edge
        elif c0 <= ct <= c1 and rt == r0 - 1:
            grp = 2                      # top edge
        elif c0 <= ct <= c1 and rt == r1 + 1:
            grp = 3                      # bottom edge
        else:
            grp = 4                      # corners
        return (grp, rt, ct, rd, cd)

    plan = []
    for (rd, cd) in dots:
        rt, ct = nearest(rd, cd)
        if (rt, ct) == (rd, cd):
            continue
        plan.append((side_key(rt, ct, rd, cd), (rd, cd), (rt, ct)))
    plan.sort(key=lambda e: e[0])

    for _, (rd, cd), (rt, ct) in plan:
        dr = rt - rd
        dc = ct - cd
        grabbed = False
        vop = 20 if dr < 0 else 21
        for _ in range(abs(dr)):
            ops.append(vop)
            sels.append(sel_of([]) if grabbed else sel_of([(rd, cd)]))
            grabbed = True
        hop = 23 if dc < 0 else 22
        for _ in range(abs(dc)):
            ops.append(hop)
            sels.append(sel_of([]) if grabbed else sel_of([(rd, cd)]))
            grabbed = True
        # ARCLE leaves the grabbed cell's original footprint at 0
        if bgc != 0:
            ops.append(int(bgc))
            sels.append(sel_of([(rd, cd)]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])   # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task a48eeaf7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a48eeaf7"
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
                                f"for task a48eeaf7"
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
                    f"Failed to build a complete episode for task a48eeaf7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a48eeaf7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
