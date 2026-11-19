"""
ARC Task: f76d97a5 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------- 1
def sample_colors(num_examples=None) -> dict:
    # The generator paints the canvas with 5 (hardcoded -> not sampled) and
    # sprinkles one randomly chosen colour `col` on it.  `col` is the colour the
    # rule fills the 5s with, so it must be fixed for the whole episode.
    cols = [c for c in range(10) if c not in (0, 5)]
    col = random.choice(cols)
    return {"col": col}


# ----------------------------------------------------------------------------- 2
def generate(diff_lb, diff_ub, max_h, max_w, col) -> dict:
    hmax = max(4, min(30, max_h))
    wmax = max(4, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (4, hmax))
    w = unifint(diff_lb, diff_ub, (4, wmax))
    gi = canvas(5, (h, w))
    go = canvas(col, (h, w))
    numdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    num = choice((numdev, h * w - numdev))
    num = min(max(1, num), h * w)
    inds = totuple(asindices(gi))
    locs = sample(inds, num)
    gi = fill(gi, col, locs)
    go = fill(go, 0, locs)
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------- 3
def derive_operations(I, O, examples=None):
    """Rule: the two colours trade places -- every `col` cell becomes 0, every 5
    becomes `col`.  The ROUTE taken to perform that swap (which colour is dealt
    with first, whether a 0 background is laid down and the shapes stamped on
    top, whether the grid is swept by rows / columns / quadrants, whether blobs
    are flood-filled one by one) is chosen deterministically from features of
    the INPUT alone, so the same puzzle always yields the same route while
    different puzzles are solved differently."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- which colour plays the "foreground" role (read from I, or, if this
    # --- instance happens to be all one colour, from the demonstrations) ------
    vals = sorted(set(int(v) for v in I.flatten().tolist()))
    non5 = [v for v in vals if v != 5]
    col = non5[0] if non5 else None
    if col is None and examples:
        for pair in examples:
            a = np.asarray(pair[0], dtype=int)
            cs = sorted(int(v) for v in set(a.flatten().tolist()) if v != 5)
            if cs:
                col = cs[0]
                break
    if col is None:
        col = 0

    col_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col]
    five_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 5]

    ops, sels = [], []

    def paint(colour, cells):
        """One Color op over exactly `cells` (skipped when there is nothing)."""
        if cells:
            ops.append(int(colour))
            sels.append(sel_of(cells))

    def components(cells):
        """Connected components of a cell set, raster order, as (seed, size)."""
        cs = set(cells)
        seen = set()
        out = []
        for r in range(hi):
            for c in range(wi):
                if (r, c) in cs and (r, c) not in seen:
                    stack = [(r, c)]
                    seen.add((r, c))
                    size = 0
                    while stack:
                        y, x = stack.pop()
                        size += 1
                        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            p = (y + dy, x + dx)
                            if p in cs and p not in seen:
                                seen.add(p)
                                stack.append(p)
                    out.append(((r, c), size))
        return out

    comps_col = components(col_cells)
    comps_five = components(five_cells)

    # --- pick the route from the input's own numbers --------------------------
    key = (col * 31 + hi * 7 + wi * 5 + len(col_cells) * 3 + len(five_cells)
           + len(comps_col))
    routes = ["base", "swap_fg", "swap_bg", "rows", "cols", "blocks"]
    if len(comps_col) + len(comps_five) <= 40:
        routes.append("flood")
    route = routes[key % len(routes)]
    sub = (key // len(routes)) % 2          # sweep direction / grouping toggle

    if route == "base":
        # Lay the whole canvas down as 0 (no cell is 0 yet, so every cell
        # changes), then stamp the shapes that were 5 back in `col` on top.
        # bbox == the whole grid rectangle exactly, so a bbox selection is fine.
        ops.append(0)
        sels.append([0, 0, hi - 1, wi - 1])
        if sub == 1 and 0 < len(comps_five) <= 40:
            # stamp blob by blob, raster order
            cs = set(five_cells)
            seen = set()
            for seed, _ in comps_five:
                stack = [seed]
                blob = []
                seen.add(seed)
                while stack:
                    y, x = stack.pop()
                    blob.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        p = (y + dy, x + dx)
                        if p in cs and p not in seen:
                            seen.add(p)
                            stack.append(p)
                paint(col, sorted(blob))
        else:
            paint(col, five_cells)

    elif route == "swap_fg":
        # clear the sprinkled colour first, then recolour the field
        paint(0, col_cells)
        paint(col, five_cells)

    elif route == "swap_bg":
        # recolour the field first, then clear the original sprinkles
        paint(col, five_cells)
        paint(0, col_cells)

    elif route == "rows":
        rows = range(hi) if sub == 0 else range(hi - 1, -1, -1)
        for r in rows:
            paint(0, [(r, c) for c in range(wi) if I[r, c] == col])
            paint(col, [(r, c) for c in range(wi) if I[r, c] == 5])

    elif route == "cols":
        colsrange = range(wi) if sub == 0 else range(wi - 1, -1, -1)
        for c in colsrange:
            paint(0, [(r, c) for r in range(hi) if I[r, c] == col])
            paint(col, [(r, c) for r in range(hi) if I[r, c] == 5])

    elif route == "blocks":
        mr, mc = hi // 2, wi // 2
        tl = (0, mr, 0, mc)
        tr = (0, mr, mc, wi)
        bl = (mr, hi, 0, mc)
        br = (mr, hi, mc, wi)
        quads = [tl, tr, br, bl] if sub == 0 else [tl, bl, tr, br]
        for r0, r1, c0, c1 in quads:
            blk = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
            paint(0, [p for p in blk if I[p[0], p[1]] == col])
            paint(col, [p for p in blk if I[p[0], p[1]] == 5])

    else:  # "flood" -- treat each blob as its own object
        order_col = list(comps_col)
        order_five = list(comps_five)
        if sub == 1:                      # biggest blobs first
            order_col.sort(key=lambda t: (-t[1], t[0]))
            order_five.sort(key=lambda t: (-t[1], t[0]))
        # the sprinkled colour must be cleared before the field is recoloured,
        # otherwise the field's fill would leak into it
        for (r, c), _ in order_col:
            ops.append(10)                # FloodFill0
            sels.append([r, c, 0, 0])
        for (r, c), _ in order_five:
            ops.append(10 + col)          # FloodFill<col>
            sels.append([r, c, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task f76d97a5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f76d97a5"
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
                                f"for task f76d97a5"
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
                    f"Failed to build a complete episode for task f76d97a5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f76d97a5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
