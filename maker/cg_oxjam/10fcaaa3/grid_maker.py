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
    # 8 is hardcoded as the marker colour; the background is sampled from the rest.
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        return int(round(a + d * (b - a)))

    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    hmax = max(2, min(15, max_h // 2))
    wmax = max(2, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 6)))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = [c for c in cols if c != bgc]
    ccols = random.sample(remcols, min(ncols, len(remcols)))
    gi = [[bgc] * w for _ in range(h)]
    inds = [(r, c) for r in range(h) for c in range(w)]
    locs = random.sample(inds, ncells)
    for (r, c) in locs:
        gi[r][c] = random.choice(ccols)
    H, W = 2 * h, 2 * w
    go = [[gi[r % h][c % w] for c in range(W)] for r in range(H)]
    full = set()
    for (r, c) in locs:
        for (a, b) in [(0, 0), (0, w), (h, 0), (h, w)]:
            full.add((r + a, c + b))
    for (r, c) in full:
        for (dr, dc) in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and go[rr][cc] == bgc:
                go[rr][cc] = 8
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared with the golf entry): first the tile is doubled in both
    directions, and that whole 2h x 2w grid is treated as one unit. Then the
    rule is applied to each cell, not to each source pixel. The golf code
    sweeps the tiled grid one row at a time, looking at each row together
    with the rows above and below it. A background cell becomes 8 when any of
    its diagonal neighbours in the tiled grid is non-background. This includes
    neighbours on the other side of a tile seam.
    The tiling step is the same as the maker's. The marking step is grouped
    differently: there is one Color8 op per row of the tiled grid. It
    selects that row's background cells that have a non-background diagonal
    neighbour. Each selection is computed only from the tiled copy of I.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 2 * hi, 2 * wi

    # Find the background: foreground covers at most h*w/6 of the grid, so
    # the background is always the most common colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []

    # 1. Expand the canvas to 2h x 2w. The input tile stays in the top-left corner.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])  # full rectangle = new canvas

    # 2. Copy the whole tile into the other three quadrants.
    ops.append(28); sels.append([0, 0, hi - 1, wi - 1])  # the whole input tile (rectangle)
    for (tr, tc) in [(0, wi), (hi, 0), (hi, wi)]:
        ops.append(30); sels.append([tr, tc, 0, 0])

    # The tiled grid as it now stands, built only from I.
    T = np.tile(I, (2, 2))

    # 3. Go down the tiled grid one row at a time. For each row, look at the
    #    rows directly above and below it to find the background cells that
    #    have a non-background diagonal neighbour.
    for r in range(ho):
        row_cells = []
        for c in range(wo):
            if T[r, c] != bgc:
                continue
            hit = False
            for dr in (-1, 1):
                rr = r + dr
                if not (0 <= rr < ho):
                    continue
                for dc in (-1, 1):
                    cc = c + dc
                    if 0 <= cc < wo and T[rr, cc] != bgc:
                        hit = True
            if hit:
                row_cells.append((r, c))
        if row_cells:
            ops.append(8); sels.append(sel_of(row_cells))

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
