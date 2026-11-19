"""
ARC Task: 1190e5a7 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule does not depend on any colour except the background.
    # Each frontier line's colour is drawn at random inside generate().
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(max(a + round(d * (b - a)), a), b)

    colopts = list(range(10))
    if bgc is None:
        bgc = random.choice(colopts)
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    grid = [[bgc] * w for _ in range(h)]
    nhf = unifint(diff_lb, diff_ub, (1, max(1, h // 3)))
    nvf = unifint(diff_lb, diff_ub, (1, max(1, w // 3)))

    hf_options = set(range(1, h - 1))
    hf_sel = []
    for _ in range(nhf):
        if not hf_options:
            break
        hf = random.choice(sorted(hf_options))
        hf_sel.append(hf)
        hf_options -= {hf - 1, hf, hf + 1}
    vf_options = set(range(1, w - 1))
    vf_sel = []
    for _ in range(nvf):
        if not vf_options:
            break
        vf = random.choice(sorted(vf_options))
        vf_sel.append(vf)
        vf_options -= {vf - 1, vf, vf + 1}

    remcols = [c for c in colopts if c != bgc]
    # horizontal frontiers first, then vertical (vertical painted on top, as in the original mapply/paint)
    for hf in hf_sel:
        col = random.choice(remcols)
        for j in range(w):
            grid[hf][j] = col
    for vf in vf_sel:
        col = random.choice(remcols)
        for i in range(h):
            grid[i][vf] = col

    nh, nv = len(hf_sel), len(vf_sel)
    out = [[bgc] * (nv + 1) for _ in range(nh + 1)]
    return {"input": grid, "output": out}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the corners are never on a frontier, so they always show the room colour.
    corner_vals = [int(I[0, 0]), int(I[0, wi - 1]), int(I[hi - 1, 0]), int(I[hi - 1, wi - 1])]
    bgc = Counter(corner_vals).most_common(1)[0][0]

    # Reading: the grid is a lattice of rooms separated by walls (full lines of non-bg).
    # Every run of non-wall rows is one room-row and every run of non-wall columns is
    # one room-column. Each room collapses to a single cell.
    wall_rows = [r for r in range(hi) if all(int(I[r, c]) != bgc for c in range(wi))]
    wall_cols = [c for c in range(wi) if all(int(I[r, c]) != bgc for r in range(hi))]
    wr, wc = set(wall_rows), set(wall_cols)

    def count_runs(n, walls):
        runs, prev_wall = 0, True
        for k in range(n):
            if k not in walls and prev_wall:
                runs += 1
            prev_wall = k in walls
        return runs

    ho = count_runs(hi, wr)   # number of room-rows
    wo = count_runs(wi, wc)   # number of room-columns

    ops, sels = [], []
    # 1. Crop the canvas to one cell per room. The window is taken from the top-left
    #    of the input (a full rectangle). Room cells and any wall pieces inside it are kept.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    G = I[:ho, :wo].copy()
    # 2. Dissolve every wall segment that sits inside the collapsed canvas:
    #    horizontal walls first (whole row of the window), then vertical walls
    #    (only the cells of that column that are not already room colour).
    for r in wall_rows:
        if r >= ho:
            continue
        cells = [(r, c) for c in range(wo) if G[r, c] != bgc]
        if cells:
            ops.append(int(bgc)); sels.append(sel_of(cells))
            for (a, b) in cells:
                G[a, b] = bgc
    for c in wall_cols:
        if c >= wo:
            continue
        cells = [(r, c) for r in range(ho) if G[r, c] != bgc]
        if cells:
            ops.append(int(bgc)); sels.append(sel_of(cells))
            for (a, b) in cells:
                G[a, b] = bgc

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
                        f"num_examples+1 ({num_examples + 1}) for task 1190e5a7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1190e5a7"
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
                                f"for task 1190e5a7"
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
                    f"Failed to build a complete episode for task 1190e5a7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1190e5a7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
