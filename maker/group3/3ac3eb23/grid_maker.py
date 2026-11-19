"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))

    def unif(lb, ub):
        ub = max(lb, ub)
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lb + d * (ub - lb)))

    k_rot = random.choice([0, 1, 2, 3])
    if k_rot % 2 == 1:
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    h = unif(3, min(30, hb))
    w = unif(3, min(30, wb))
    remcols = [c for c in range(10) if c != bgc]
    nlocs = unif(1, max(1, (w - 2) // 3))
    locopts = list(range(1, w - 1))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for _ in range(nlocs):
        if not locopts:
            break
        locj = random.choice(locopts)
        locopts = [x for x in locopts if not (locj - 2 <= x <= locj + 2)]
        col = random.choice(remcols)
        gi[0, locj] = col
        for p in range(0, h, 2):
            go[p, locj] = col
        for p in range(1, h, 2):
            go[p, locj - 1] = col
            go[p, locj + 1] = col
    gi = np.rot90(gi, k_rot)
    go = np.rot90(go, k_rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Same rule (each border seed grows a period-2 zigzag fan into the grid), but the
    unit of work is a full-width BAND of the grid, not a single seed:
      1. paint every seed's depth-1 wings at once, one Color op per seed colour
         -> the 2-deep band next to the seed border is now complete for ALL seeds;
      2. CopyO that whole band (full width/height of the border) and Paste it
         every 2 cells deeper, so all fans advance together band by band;
      3. colour-0 seeds are invisible to Copy/Paste, so after each pasted band
         their cells in that band are painted with Color0;
      4. if the strip depth is odd, the last lone row/column of seed lanes is
         painted per colour.
    O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    if all(r == 0 for r, _ in seeds):
        side, u, v, D = "top", (1, 0), (0, 1), hi
    elif all(r == hi - 1 for r, _ in seeds):
        side, u, v, D = "bottom", (-1, 0), (0, 1), hi
    elif all(c == 0 for _, c in seeds):
        side, u, v, D = "left", (0, 1), (1, 0), wi
    else:
        side, u, v, D = "right", (0, -1), (1, 0), wi

    def cell(s, d, t):
        return (s[0] + u[0] * d + v[0] * t, s[1] + u[1] * d + v[1] * t)

    def band_rect(d0):
        # full rectangle covering depths d0, d0+1 across the whole border
        if side == "top":
            return [d0, 0, 1, wi - 1]
        if side == "bottom":
            return [hi - 2 - d0, 0, 1, wi - 1]
        if side == "left":
            return [0, d0, hi - 1, 1]
        return [0, wi - 2 - d0, hi - 1, 1]

    colors = sorted(set(int(I[r, c]) for r, c in seeds))
    by_col = {col: [s for s in seeds if int(I[s[0], s[1]]) == col] for col in colors}
    zero_seeds = by_col.get(0, [])

    ops, sels = [], []

    # 1. depth-1 wings of all seeds, one op per colour
    for col in colors:
        cells = [cell(s, 1, t) for s in by_col[col] for t in (-1, 1)]
        ops.append(col)
        sels.append(sel_of(cells))

    P = D // 2
    has_visible = any(c != 0 for c in colors)
    if P > 1:
        if has_visible:
            # 2. copy the finished first band (whole rectangle) once
            ops.append(29)
            sels.append(band_rect(0))
        for k in range(1, P):
            d0 = 2 * k
            if has_visible:
                br = band_rect(d0)
                ops.append(30)
                sels.append([br[0], br[1], 0, 0])
            # 3. colour-0 seeds' cells inside this band
            if zero_seeds:
                cells = [cell(s, d0, 0) for s in zero_seeds]
                cells += [cell(s, d0 + 1, t) for s in zero_seeds for t in (-1, 1)]
                ops.append(0)
                sels.append(sel_of(cells))

    # 4. lone final row/column of seed lanes when depth is odd
    if D % 2 == 1:
        for col in colors:
            cells = [cell(s, D - 1, 0) for s in by_col[col]]
            ops.append(col)
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
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
                                f"for task 3ac3eb23"
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
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
