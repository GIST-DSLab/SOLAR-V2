"""
ARC Task: ea786f4a (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    """The generator samples exactly two colour things: `linc` (the canvas colour, which is
    also the unique centre marker and the colour the four diagonals are drawn in) and the
    pool of noise colours `ccols`.  Both are fixed once per episode so every instance in the
    episode shares the same palette; `linc` matters for the rule, so it must be stable."""
    cols = list(range(10))
    linc = random.choice(cols)
    pool = [c for c in cols if c != linc]
    random.shuffle(pool)
    return {"linc": linc, "ccols_pool": pool}


def generate(diff_lb, diff_ub, max_h, max_w, linc, ccols_pool) -> dict:
    try:
        ui = unifint  # RE-ARC helper when available
    except NameError:
        def ui(lb, ub, rng):
            a, b = rng
            return random.randint(a, b)

    hmax = max(1, min(14, (max_h - 1) // 2))
    wmax = max(1, min(14, (max_w - 1) // 2))
    hh = ui(diff_lb, diff_ub, (1, hmax))
    ww = ui(diff_lb, diff_ub, (1, wmax))
    h, w = 2 * hh + 1, 2 * ww + 1

    ncols = ui(diff_lb, diff_ub, (1, 9))
    ccols = list(ccols_pool)[:max(1, min(9, ncols))]

    gi = [[random.choice(ccols) for _ in range(w)] for _ in range(h)]
    gi[hh][ww] = linc  # the lone marker cell, in the canvas colour

    go = [row[:] for row in gi]
    for dr, dc in ((-1, -1), (1, 1), (-1, 1), (1, -1)):
        r, c = hh + dr, ww + dc
        while 0 <= r < h and 0 <= c < w:
            go[r][c] = linc
            r += dr
            c += dc

    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


def derive_operations(I, O):
    """Rule: the grid's exact centre cell holds a marker colour; four diagonal rays are
    shot from it to the grid edges and painted in that colour.

    The route (which ray/line/ring first, which sweep direction, whether a ray goes down as
    one stroke or cell by cell) is decided from the input grid alone — marker colour, centre
    position, ray length, number of distinct colours — so the same instance always replays
    identically while different instances are solved differently.  Only Color ops are used,
    so a marker colour of 0 travels exactly like any other colour."""
    import numpy as np
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cr, cc = hi // 2, wi // 2          # (2h+1)x(2w+1) grid -> exact centre
    marker = int(I[cr, cc])
    L = max(0, min(cr, cc))            # every diagonal ray has exactly this many cells

    dirs = {"UL": (-1, -1), "UR": (-1, 1), "DR": (1, 1), "DL": (1, -1)}
    ray = {}
    for k, (dr, dc) in dirs.items():
        cells = []
        for d in range(1, L + 1):
            r, c = cr + dr * d, cc + dc * d
            if 0 <= r < hi and 0 <= c < wi:
                cells.append((r, c))
        ray[k] = cells                 # ordered outward from the centre

    # --- route choice, read off the input -------------------------------------------
    ncol = len(set(I.flatten().tolist()))
    route = (marker + ncol + L + cc) % 6
    cyc = ["UL", "UR", "DR", "DL"]
    s = (marker + cr) % 4
    order = cyc[s:] + cyc[:s]          # which ray gets dealt with first

    ops, sels = [], []

    def stroke(cells):
        # drop cells that already hold the marker colour: an op there would be invisible
        cells = [(r, c) for (r, c) in cells if int(I[r, c]) != marker]
        if cells:
            ops.append(marker)
            sels.append(sel_of(cells))

    if route == 0:
        # two full diagonal lines, each drawn tip-to-tip in a single stroke
        main = ray["UL"][::-1] + ray["DR"]
        anti = ray["UR"][::-1] + ray["DL"]
        for line in ([main, anti] if (cr + cc) % 2 == 0 else [anti, main]):
            stroke(line)
    elif route == 1:
        # one stroke per ray
        for k in order:
            stroke(ray[k])
    elif route == 2 or route == 3:
        # concentric X: all four cells at the same distance from the centre together,
        # growing outward (2) or closing inward from the tips (3)
        steps = range(L) if route == 2 else range(L - 1, -1, -1)
        for d in steps:
            stroke([ray[k][d] for k in order if d < len(ray[k])])
    else:
        # cell by cell along each ray, outward from the centre (4) or inward from the
        # grid corner back to the centre (5)
        for k in order:
            cells = ray[k] if route == 4 else ray[k][::-1]
            for cell in cells:
                stroke([cell])

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
                        f"num_examples+1 ({num_examples + 1}) for task ea786f4a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ea786f4a"
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
                                f"for task ea786f4a"
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
                    f"Failed to build a complete episode for task ea786f4a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ea786f4a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
