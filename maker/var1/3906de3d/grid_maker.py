"""
ARC Task: 3906de3d (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, boxc = random.sample(cols, 2)
    # linc must be non-zero: ARCLE's Move grabs only NONZERO cells of the
    # selection, so a zero-coloured bar could not be slid at all.
    linc = random.choice([c for c in cols if c not in (bgc, boxc) and c != 0])

    # discrete structural variant = the final rotation applied to both grids,
    # i.e. which edge the comb hangs from and which way the bars travel.
    VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "boxc": boxc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, linc, rot=None) -> dict:
    import random as _r

    def _unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        lo = max(a, min(lo, b))
        hi = max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return _r.randint(lo, hi)

    def _rot90cw(g):
        return [list(r) for r in zip(*g[::-1])]

    if rot is None:
        rot = _r.choice([0, 1, 2, 3])

    hmax, wmax = max_h, max_w
    if rot % 2 == 1:                 # a 90/270 rotation swaps the final dims
        hmax, wmax = max_w, max_h
    hmax = max(5, min(int(hmax), 30))
    wmax = max(5, min(int(wmax), 30))

    h = _unifint(diff_lb, diff_ub, (5, hmax))
    w = _unifint(diff_lb, diff_ub, (5, wmax))
    oh = _unifint(diff_lb, diff_ub, (2, h // 2))
    ow = _unifint(diff_lb, diff_ub, (3, w - 2))
    locj = _r.randint(1, w - ow - 1)

    gi = [[bgc] * w for _ in range(h)]
    for i in range(oh):
        for j in range(locj, locj + ow):
            gi[i][j] = boxc

    rng = list(range(locj, locj + ow))
    cutoffs = [_r.randint(1, oh - 1) for _ in rng]
    for jj, co in zip(rng, cutoffs):          # carve the comb's teeth
        for i in range(co, oh):
            gi[i][jj] = bgc

    numlns = _unifint(diff_lb, diff_ub, (1, ow - 1))
    lnlocs = set(_r.sample(rng, numlns))

    go = [row[:] for row in gi]
    for jj, co in zip(rng, cutoffs):
        if jj in lnlocs:
            lineh = _r.randint(1, h - co - 1)
            for i in range(h - lineh, h):      # input: bar parked at far edge
                gi[i][jj] = linc
            for i in range(co, co + lineh):    # output: bar docked to the tooth
                go[i][jj] = linc

    for _ in range(rot % 4):
        gi = _rot90cw(gi)
        go = _rot90cw(go)

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    """Every 1-wide bar parked on one edge slides up its own lane until it
    docks against the end of the comb's tooth in that lane.

    Everything is read off I alone (O is never consulted), so a different
    answer grid yields the same operations.  Which bar is taken first, and
    whether the vacated footprints are tidied bar by bar or all in one go at
    the end, are decided from the input's own shape — one instance always
    gives one route, and different instances give different routes.
    """
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:                                   # pragma: no cover
        def sel_of(cells):
            uniq = sorted({(int(r), int(c)) for r, c in cells})
            return {"cells": [[r, c] for r, c in uniq]}

    I = np.asarray(I, dtype=int)
    H, W = I.shape              # the grid never changes size
    ops, sels = [], []

    # --- background: the comb never reaches an end of its edge and the bars
    #     sit strictly inside it, so every corner of the grid is background
    bgc = int(I[0, 0])

    # --- which edge carries the comb, which carries the bars ---------------
    # the comb's edge line is solid across its whole width (ow cells) while
    # the bars occupy at most ow-1 of those lanes, so the comb's edge always
    # holds strictly more non-background cells.  The other two edges are bare.
    top, bot = I[0, :], I[-1, :]
    lft, rgt = I[:, 0], I[:, -1]
    n_top, n_bot = int(np.sum(top != bgc)), int(np.sum(bot != bgc))
    n_lft, n_rgt = int(np.sum(lft != bgc)), int(np.sum(rgt != bgc))

    vertical = (n_top + n_bot) > 0
    if vertical:
        comb_first = n_top >= n_bot            # comb on the top row
        comb_edge, line_edge = (top, bot) if comb_first else (bot, top)
        mv = 20 if comb_first else 21          # bars travel toward the comb
        n_lanes = W
    else:
        comb_first = n_lft >= n_rgt            # comb on the left column
        comb_edge, line_edge = (lft, rgt) if comb_first else (rgt, lft)
        mv = 23 if comb_first else 22
        n_lanes = H

    bc = [int(v) for v in comb_edge.tolist() if int(v) != bgc]
    lc = [int(v) for v in line_edge.tolist() if int(v) != bgc]
    if not bc or not lc:
        ops.append(34); sels.append([0, 0, H - 1, W - 1])
        return ops, sels
    boxc = Counter(bc).most_common(1)[0][0]
    linc = Counter(lc).most_common(1)[0][0]

    def lane_cells(k):
        """this lane's cells, listed from the comb's edge inward"""
        if vertical:
            rows = range(H) if comb_first else range(H - 1, -1, -1)
            return [(r, k) for r in rows]
        cols = range(W) if comb_first else range(W - 1, -1, -1)
        return [(k, c) for c in cols]

    # --- read the comb's teeth and every bar, lane by lane -----------------
    teeth = {}          # comb lane -> depth of its tooth
    bars = []           # one entry per bar that has somewhere to go
    for k in range(n_lanes):
        cells = lane_cells(k)
        vals = [int(I[r, c]) for (r, c) in cells]
        depth = 0
        while depth < len(vals) and vals[depth] == boxc:
            depth += 1
        if depth == 0:
            continue                       # lane outside the comb: no bar here
        teeth[k] = depth
        idxs = [t for t, v in enumerate(vals) if v == linc]
        if not idxs:
            continue
        steps = min(idxs) - depth          # slide until it touches the tooth
        if steps <= 0:
            continue
        bars.append({"lane": k, "steps": steps,
                     "src": [cells[t] for t in idxs],
                     "dst": [cells[t - steps] for t in idxs]})

    # --- pick the route from the input's own shape -------------------------
    comb_lanes = sorted(teeth)
    n, ow = len(bars), len(comb_lanes)
    lo, hi = comb_lanes[0], comb_lanes[-1]
    # the deepest tooth is the comb's most prominent feature: start on its side
    deep_lane = max(comb_lanes, key=lambda k: (teeth[k], k))
    far_side = (deep_lane - lo) * 2 >= (hi - lo)
    crowded = 2 * n > ow               # more than half the teeth carry a bar
    if not crowded:                    # few bars: just sweep along the comb
        bars.sort(key=lambda b: -b["lane"] if far_side else b["lane"])
    elif far_side:                     # crowded: take the long slides first
        bars.sort(key=lambda b: (-b["steps"], b["lane"]))
    else:                              # crowded: clear the short ones first
        bars.sort(key=lambda b: (b["steps"], -b["lane"]))
    batched = n >= 3                   # several leftovers -> tidy them at once

    # --- slide each bar, then repair the footprint it left behind ----------
    holes = []
    for b in bars:
        src, dst, steps = b["src"], b["dst"], b["steps"]
        if linc != 0:
            ops.append(mv); sels.append(sel_of(src))      # grab this bar
            for _ in range(steps - 1):
                ops.append(mv); sels.append(sel_of([]))   # keep it gliding
        else:
            # colour 0 is "nothing" to ARCLE's object ops, so a 0-coloured bar
            # cannot be grabbed: draw it onto the lane it reaches instead
            ops.append(0); sels.append(sel_of(sorted(set(dst) - set(src))))
        hole = sorted(set(src) - set(dst))   # only the start is left behind
        if bgc != 0 and hole:
            if batched:
                holes.extend(hole)
            else:
                ops.append(bgc); sels.append(sel_of(hole))
    if bgc != 0 and holes:
        ops.append(bgc); sels.append(sel_of(holes))

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])   # bbox == the whole grid, exactly intended
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
                        f"num_examples+1 ({num_examples + 1}) for task 3906de3d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3906de3d"
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
                                f"for task 3906de3d"
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
                    f"Failed to build a complete episode for task 3906de3d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3906de3d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
