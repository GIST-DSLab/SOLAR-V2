"""
ARC Task: a68b268e (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, linc, c1, c2, c3, c4 = random.sample(range(10), 6)
    return {"bgc": bgc, "linc": linc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    v = random.randint(a, b)
    return max(a, min(b, v))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None,
             c1=None, c2=None, c3=None, c4=None) -> dict:
    if None in (bgc, linc, c1, c2, c3, c4):
        bgc, linc, c1, c2, c3, c4 = random.sample(range(10), 6)
    hmax = max(2, min(14, (max_h - 1) // 2))
    wmax = max(2, min(4, (max_w - 1) // 2))
    h = _unifint(diff_lb, diff_ub, (2, hmax))
    w = _unifint(diff_lb, diff_ub, (2, wmax))
    inds = [(r, c) for r in range(h) for c in range(w)]

    def ncount():
        d = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
        n = random.choice((d, h * w - d))
        return min(max(1, n), h * w - 1)

    sets = [random.sample(inds, ncount()) for _ in range(4)]
    cols = [c1, c2, c3, c4]
    go = [[bgc] * w for _ in range(h)]
    for col, cells in zip(cols, sets):
        for r, c in cells:
            go[r][c] = col
    quads = []
    for col, cells in zip(cols, sets):
        g = [[bgc] * w for _ in range(h)]
        for r, c in cells:
            g[r][c] = col
        quads.append(g)
    LR, LL, UR, UL = quads
    gi = [[linc] * (2 * w + 1) for _ in range(2 * h + 1)]
    for r in range(h):
        for c in range(w):
            gi[r + h + 1][c + w + 1] = LR[r][c]
            gi[r + h + 1][c] = LL[r][c]
            gi[r][c + w + 1] = UR[r][c]
            gi[r][c] = UL[r][c]
    return {"input": [tuple(x) for x in gi], "output": [tuple(x) for x in go]}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    h, w = hi // 2, wi // 2
    origins = {"UL": (0, 0), "UR": (0, wi - w), "LL": (hi - h, 0), "LR": (hi - h, wi - w)}
    quads = {k: I[r:r + h, c:c + w] for k, (r, c) in origins.items()}

    # background = the one colour every quadrant shares
    common = set.intersection(*[set(q.flatten().tolist()) for q in quads.values()])
    bgc = min(common) if common else Counter(I.flatten().tolist()).most_common(1)[0][0]

    def cells_of(grid, col, region=None):
        out = []
        for (r0, c0) in (region or [(0, 0)]):
            hh, ww = (h, w) if region else grid.shape
            for r in range(r0, r0 + hh):
                for c in range(c0, c0 + ww):
                    if grid[r, c] == col:
                        out.append((r, c))
        return out

    ops, sels = [], []
    g = I.copy()
    quad_regions = list(origins.values())

    # Step 1: make the background transparent (0) so quadrants can be stacked
    # with transparent Paste. If 0 is a real quadrant colour, park it on an
    # unused colour first.
    tmp = None
    if bgc != 0:
        zero_cells = cells_of(g, 0, quad_regions)
        if zero_cells:
            tmp = [c for c in range(1, 10) if c not in set(I.flatten().tolist())][0]
            ops.append(tmp); sels.append(sel_of(zero_cells))
            for r, c in zero_cells:
                g[r, c] = tmp
        bg_cells = cells_of(g, bgc, quad_regions)
        ops.append(0); sels.append(sel_of(bg_cells))
        for r, c in bg_cells:
            g[r, c] = 0

    # Step 2: stack the quadrants onto the LR quadrant, lowest priority first:
    # LR (already there) < LL < UR < UL.
    dr, dc = origins["LR"]
    for k in ("LL", "UR", "UL"):
        r0, c0 = origins[k]
        ops.append(29); sels.append([r0, c0, h - 1, w - 1])   # full quadrant rectangle
        ops.append(30); sels.append([dr, dc, 0, 0])            # paste onto LR corner
        blk = g[r0:r0 + h, c0:c0 + w].copy()
        tgt = g[dr:dr + h, dc:dc + w]
        tgt[blk != 0] = blk[blk != 0]

    # Step 3: crop to the stacked quadrant (full rectangle)
    ops.append(33); sels.append([dr, dc, h - 1, w - 1])
    g = g[dr:dr + h, dc:dc + w].copy()

    # Step 4: restore the background / parked colour
    if bgc != 0:
        holes = cells_of(g, 0)
        if holes:
            ops.append(bgc); sels.append(sel_of(holes))
        if tmp is not None:
            parked = cells_of(g, tmp)
            if parked:
                ops.append(0); sels.append(sel_of(parked))
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a68b268e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a68b268e"
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
                                f"for task a68b268e"
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
                    f"Failed to build a complete episode for task a68b268e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a68b268e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
