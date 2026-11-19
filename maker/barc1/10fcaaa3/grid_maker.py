"""
ARC Task: 10fcaaa3 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # 8 is hardcoded as the halo colour; the background is drawn from the others
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    # output is 2h x 2w, so each input dimension may be at most half the limit
    hmax = max(2, min(15, max_h // 2))
    wmax = max(2, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 6)))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = [c for c in cols if c != bgc]
    ccols = random.sample(remcols, min(ncols, len(remcols)))
    gi = np.full((h, w), bgc, dtype=int)
    inds = [(i, j) for i in range(h) for j in range(w)]
    locs = random.sample(inds, ncells)
    for (i, j) in locs:
        gi[i, j] = random.choice(ccols)
    go = np.vstack([np.hstack([gi, gi]), np.hstack([gi, gi])])
    fullocs = set()
    for (i, j) in locs:
        for (a, b) in [(0, 0), (0, w), (h, 0), (h, w)]:
            fullocs.add((i + a, j + b))
    ho, wo = go.shape
    topaint = set()
    for (i, j) in fullocs:
        for (di, dj) in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            pi, pj = i + di, j + dj
            if 0 <= pi < ho and 0 <= pj < wo and go[pi, pj] == bgc:
                topaint.add((pi, pj))
    for (pi, pj) in topaint:
        go[pi, pj] = 8
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    """
    Reference reading: blit the input tile 4 times (2x2) into the output, then
    scan the TILED grid column by column (outer loop over columns, inner over
    rows). Every coloured pixel (not background, not an already-painted 8) paints
    its diagonal neighbours 8, but only those that are still background at that
    moment. Each pixel's halo is one Color(8) op whose cells are that pixel's
    diagonal neighbours that are currently background.
    Everything is computed from I and the evolving grid; O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 2 * hi, 2 * wi

    # background = the canvas colour; the generator caps foreground at (h*w)//6
    # of the cells, so it is always the majority colour
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []

    # 1. expand canvas to the 2h x 2w size (whole rectangle intended)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    # simulate the working grid: the expanded canvas pads with 0
    grid = np.zeros((ho, wo), dtype=int)
    grid[:hi, :wi] = I

    # 2. blit the input tile into the remaining three quadrants
    #    (CopyI of the whole tile: a full rectangle is intended)
    ops.append(28); sels.append([0, 0, hi - 1, wi - 1])
    for (tr, tc) in [(0, wi), (hi, 0), (hi, wi)]:
        ops.append(30); sels.append([tr, tc, 0, 0])
        for r in range(hi):
            for c in range(wi):
                if I[r, c] != 0:
                    grid[tr + r, tc + c] = I[r, c]

    # 3. column-major scan of the tiled grid: each coloured pixel paints its
    #    diagonal neighbours that are still background
    diag = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for y in range(wo):
        for x in range(ho):
            v = grid[x, y]
            if v == bgc or v == 8:
                continue
            halo = []
            for (dx, dy) in diag:
                px, py = x + dx, y + dy
                if 0 <= px < ho and 0 <= py < wo and grid[px, py] == bgc:
                    halo.append((px, py))
            if halo:
                ops.append(8); sels.append(sel_of(halo))
                for (px, py) in halo:
                    grid[px, py] = 8

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
                        f"num_examples+1 ({num_examples + 1}) for task 10fcaaa3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 10fcaaa3"
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
                                f"for task 10fcaaa3"
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
                    f"Failed to build a complete episode for task 10fcaaa3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"10fcaaa3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
