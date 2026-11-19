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


# ----------------------------------------------------------------------------
# Task a48eeaf7:
#   A solid rectangle (sqc) sits on a plain background (bgc), surrounded by
#   scattered single dots (dotc).  Every dot slides to the cell of the RING
#   one step outside the rectangle that is manhattan-closest to it.
#   Side dots slide straight in; diagonal (corner) dots slide to the ring's
#   corner.  Nothing else changes.
# ----------------------------------------------------------------------------

VARIANTS = [
    {"ncorn_mode": "none"},   # only straight side-dots
    {"ncorn_mode": "some"},   # at least one diagonal corner dot
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    # dotc must be non-zero: ARCLE's object ops treat 0 as "nothing", so a
    # 0-coloured dot could not be picked up and Moved.
    dotc = random.choice([c for c in cols if c != 0])
    rest = [c for c in cols if c != dotc]
    bgc = random.choice(rest)
    sqc = random.choice([c for c in rest if c != bgc])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, sqc=None, dotc=None, ncorn_mode=None) -> dict:
    if ncorn_mode is None:
        ncorn_mode = random.choice(VARIANTS)["ncorn_mode"]

    mh = max(8, min(int(max_h), 30))
    mw = max(8, min(int(max_w), 30))

    h = unifint(diff_lb, diff_ub, (8, mh))
    w = unifint(diff_lb, diff_ub, (8, mw))
    ih = unifint(diff_lb, diff_ub, (2, h // 2))
    iw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = random.randint(2, h - ih - 2)
    locj = random.randint(2, w - iw - 2)

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    sq = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))

    A = [(x, locj - 1) for x in interval(loci, loci + ih, 1)]
    Ap = [(x, random.randint(0, locj - 2)) for x in interval(loci, loci + ih, 1)]
    B = [(x, locj + iw) for x in interval(loci, loci + ih, 1)]
    Bp = [(x, random.randint(locj + iw + 1, w - 1)) for x in interval(loci, loci + ih, 1)]
    C = [(loci - 1, x) for x in interval(locj, locj + iw, 1)]
    Cp = [(random.randint(0, loci - 2), x) for x in interval(locj, locj + iw, 1)]
    D = [(loci + ih, x) for x in interval(locj, locj + iw, 1)]
    Dp = [(random.randint(loci + ih + 1, h - 1), x) for x in interval(locj, locj + iw, 1)]

    srarr = Ap + Bp + Cp + Dp
    dearr = A + B + C + D
    inds = interval(0, len(srarr), 1)
    num = unifint(diff_lb, diff_ub, (1, len(srarr)))
    locs = set(random.sample(list(inds), num))
    srarr = [e for j, e in enumerate(srarr) if j in locs]
    dearr = [e for j, e in enumerate(dearr) if j in locs]

    gi = fill(gi, sqc, sq)
    go = fill(go, sqc, sq)
    for s, d in zip(srarr, dearr):
        gi = fill(gi, dotc, {s})
        go = fill(go, dotc, {d})

    if ncorn_mode == "none":
        ncorn = 0
    else:
        ncorn = unifint(diff_lb, diff_ub, (1, 4))

    fullinds = asindices(gi)
    if ncorn > 0:
        go = fill(go, dotc, {(loci - 1, locj - 1)})
        cands = shoot((loci - 2, locj - 2), (-1, -1)) & fullinds
        locc = random.choice(sorted(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 1:
        go = fill(go, dotc, {(loci - 1, locj + iw)})
        cands = shoot((loci - 2, locj + iw + 1), (-1, 1)) & fullinds
        locc = random.choice(sorted(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 2:
        go = fill(go, dotc, {(loci + ih, locj - 1)})
        cands = shoot((loci + ih + 1, locj - 2), (1, -1)) & fullinds
        locc = random.choice(sorted(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 3:
        go = fill(go, dotc, {(loci + ih, locj + iw)})
        cands = shoot((loci + ih + 1, locj + iw + 1), (1, 1)) & fullinds
        locc = random.choice(sorted(cands))
        gi = fill(gi, dotc, {locc})

    rotf = random.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Everything below is measured from I only.

    1. background  = dominant colour of I (rect <= quarter of the area, dots sparse)
    2. block       = the fg colour whose cells exactly fill their bounding box
    3. ring        = the one-cell-thick frame immediately outside that block
    4. every remaining fg cell (a dot) is slid, one Move at a time, to the ring
       cell manhattan-closest to it; its vacated cell is repainted background.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    counts = Counter(I.flatten().tolist())
    bgc = counts.most_common(1)[0][0]

    cells_of = {}
    for col in counts:
        if col == bgc:
            continue
        cells_of[col] = [(r, c) for r in range(h) for c in range(w) if I[r, c] == col]

    # --- the solid block: fg colour whose cells exactly fill their bbox -------
    block_col, block_box, best = None, None, -1
    for col, pts in cells_of.items():
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        br, bc = min(rs), min(cs)
        bh, bw = max(rs) - br + 1, max(cs) - bc + 1
        if len(pts) == bh * bw and len(pts) > best:
            best, block_col, block_box = len(pts), col, (br, bc, bh, bw)

    ops, sels = [], []
    if block_col is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    br, bc, bh, bw = block_box

    # --- the ring one cell outside the block ---------------------------------
    ring = []
    for r in range(br - 1, br + bh + 1):
        for c in range(bc - 1, bc + bw + 1):
            if br <= r < br + bh and bc <= c < bc + bw:
                continue
            if 0 <= r < h and 0 <= c < w:
                ring.append((r, c))

    # --- the dots and their landing cells ------------------------------------
    dots = [p for col, pts in cells_of.items() if col != block_col for p in pts]

    plan = []
    for (sr, sc) in dots:
        tr, tc = min(ring, key=lambda t: (abs(t[0] - sr) + abs(t[1] - sc), t[0], t[1]))
        if (tr, tc) == (sr, sc):
            continue
        # walk the ring clockwise: top edge, right edge, bottom edge, left edge,
        # then the diagonal dots that land on the ring's corners
        if tr == br - 1 and bc <= tc < bc + bw:
            side, key = 0, tc
        elif tc == bc + bw and br <= tr < br + bh:
            side, key = 1, tr
        elif tr == br + bh and bc <= tc < bc + bw:
            side, key = 2, -tc
        elif tc == bc - 1 and br <= tr < br + bh:
            side, key = 3, -tr
        else:
            side, key = 4, tr * (w + 1) + tc
        plan.append((side, key, (sr, sc), (tr, tc)))
    plan.sort(key=lambda e: (e[0], e[1]))

    # --- slide each dot onto the ring ----------------------------------------
    for _, _, (sr, sc), (tr, tc) in plan:
        dr, dc = tr - sr, tc - sc
        grabbed = False
        vop = 21 if dr > 0 else 20
        for _ in range(abs(dr)):
            ops.append(vop)
            sels.append(sel_of([(sr, sc)]) if not grabbed else sel_of([]))
            grabbed = True
        hop = 22 if dc > 0 else 23
        for _ in range(abs(dc)):
            ops.append(hop)
            sels.append(sel_of([(sr, sc)]) if not grabbed else sel_of([]))
            grabbed = True
        # ARCLE leaves the dot's original cell at 0; restore the background there
        if bgc != 0:
            ops.append(int(bgc))
            sels.append(sel_of([(sr, sc)]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # full-grid bbox: submit
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
