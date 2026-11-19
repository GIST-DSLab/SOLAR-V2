"""
ARC Task: aabf363d (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cola = random.choice(rem)
    rem = [c for c in rem if c != cola]
    colb = random.choice(rem)
    return {"bgc": bgc, "cola": cola, "colb": colb}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    return int(round(random.uniform(lo, hi)))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cola, colb, **kwargs) -> dict:
    # inner shape canvas is h x w; final grid is padded by 1 on all sides
    h = _unifint(diff_lb, diff_ub, (3, max(3, min(28, max_h - 2))))
    w = _unifint(diff_lb, diff_ub, (3, max(3, min(28, max_w - 2))))
    bounds = {(i, j) for i in range(h) for j in range(w)}

    def nbrs(cell):
        i, j = cell
        out = set()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                out.add((i + di, j + dj))
        return out

    sp = random.choice(sorted(bounds))
    ub = min(h * w - 1, max(1, (2 / 3) * h * w))
    ncells = _unifint(diff_lb, diff_ub, (1, int(ub)))
    shp = {sp}
    for _ in range(ncells):
        frontier = set()
        for cell in shp:
            frontier |= nbrs(cell)
        cands = sorted((frontier & bounds) - shp)
        if not cands:
            break
        shp.add(random.choice(cands))
    shp = {(i + 1, j + 1) for i, j in shp}

    gi = [[bgc] * (w + 2) for _ in range(h + 2)]
    go = [[bgc] * (w + 2) for _ in range(h + 2)]
    for i, j in shp:
        gi[i][j] = cola
        go[i][j] = colb
    free = [(i, j) for i in range(h + 2) for j in range(w + 2) if gi[i][j] == bgc]
    mi, mj = random.choice(free)
    gi[mi][mj] = colb
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # How this differs from the maker: the code-golf entry sees the output as a
    # two-colour picture. Every cell is either "shape" (painted in the marker's
    # colour) or "not shape" (background). The marker is just one more
    # non-shape cell. Here that means the marker is absorbed into the background
    # first, so the canvas is left as background plus one shape. Then the whole
    # shape is painted in the colour the marker had in I.
    # Both routes use the same selections (all cells of one colour, read from I).
    # Only the order is different, and with it the grouping of the marker
    # with the background.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The generator pads the grid with a 1-cell background frame, so the
    # border always shows the background colour.
    border = np.concatenate([I[0, :], I[-1, :], I[:, 0], I[:, -1]]).tolist()
    bgc = Counter(border).most_common(1)[0][0]

    counts = Counter(I.flatten().tolist())
    others = [c for c in counts if c != bgc]
    # The marker is a single cell. The shape always has at least 2 cells.
    marker = min(others, key=lambda c: counts[c])
    shape_cols = [c for c in others if c != marker]

    marker_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == marker]

    ops, sels = [], []

    # 1. The marker joins the background: paint the marker cell bgc.
    ops.append(int(bgc))
    sels.append(sel_of(marker_cells))

    # 2. Paint the shape (all cells of the shape colour) in the marker's colour.
    #    The shape is 8-connected, so use Color on its cells, not a 4-conn FloodFill.
    for sc in shape_cols:
        cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == sc]
        if cells:
            ops.append(int(marker))
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
                        f"num_examples+1 ({num_examples + 1}) for task aabf363d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task aabf363d"
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
                                f"for task aabf363d"
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
                    f"Failed to build a complete episode for task aabf363d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"aabf363d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
