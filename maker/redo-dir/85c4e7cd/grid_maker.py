"""
ARC Task: 85c4e7cd (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. Episode-level colour sampling
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # The rule (reverse the colour order of the nested rings) is colour-agnostic,
    # but the palette is fixed for the whole episode so every instance shares it.
    # At least 2 distinct colours, otherwise input == output (degenerate).
    ncols = random.randint(2, 10)
    cols = random.sample(list(range(10)), ncols)
    return {"cols": cols}


# ----------------------------------------------------------------------------
# 2. Generator
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    try:
        return unifint(diff_lb, diff_ub, (a, b))  # noqa: F821  (re-arc util if present)
    except Exception:
        return random.randint(a, b)


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, cols=None) -> dict:
    if not cols or len(set(cols)) < 2:
        cols = random.sample(list(range(10)), random.randint(2, 10))

    hmax = max(2, max_h // 2)
    wmax = max(2, max_w // 2)
    h = _unifint(diff_lb, diff_ub, (2, hmax))
    w = _unifint(diff_lb, diff_ub, (2, wmax))
    n = min(h, w)

    # ring colours, outermost first; reject palindromes (they give input == output)
    for _ in range(200):
        colord = [random.choice(cols) for _ in range(n)]
        if colord != colord[::-1]:
            break
    else:
        colord = [random.choice(cols) for _ in range(n)]

    H, W = 2 * h, 2 * w
    gi = [[0] * W for _ in range(H)]
    go = [[0] * W for _ in range(H)]
    for idx in range(n):
        ci = colord[idx]
        co = colord[n - 1 - idx]
        r0, c0 = idx, idx
        r1, c1 = H - 1 - idx, W - 1 - idx
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r == r0 or r == r1 or c == c0 or c == c1:
                    gi[r][c] = ci
                    go[r][c] = co

    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


# ----------------------------------------------------------------------------
# 3. Trajectory
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    """
    The grid is an onion of concentric rectangular rings; the rule re-colours the
    rings in reversed order.  The onion is mirror-symmetric about both axes, so:
      (a) redraw the top-left quadrant as nested rectangles (outermost first),
      (b) replicate that quadrant into the top-right quadrant, mirrored L-R,
      (c) replicate the whole top half into the bottom half, mirrored U-D.
    Each destination is cleared to 0 first, because Paste never writes colour-0
    cells and colour 0 is an ordinary ring colour in this task.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    h, w = H // 2, W // 2
    n = min(h, w)

    ops, sels = [], []
    g = I.copy()

    def rect_cells(r, c, hh, ww):
        return [(rr, cc) for rr in range(r, r + hh) for cc in range(c, c + ww)]

    def color_rect(color, r, c, hh, ww):
        # selection is exactly this full rectangle
        if np.all(g[r:r + hh, c:c + ww] == color):
            return  # already that colour -> op would be invisible
        ops.append(int(color))
        sels.append(sel_of(rect_cells(r, c, hh, ww)))
        g[r:r + hh, c:c + ww] = color

    # (a) nested rectangles in the top-left quadrant, outermost ring first.
    #     Ring i's new colour is read off O's diagonal cell (i, i).
    for i in range(n):
        color_rect(int(O[i, i]), i, i, h - i, w - i)

    # (b) top-right quadrant = mirror of the quadrant we just drew
    quad = g[0:h, 0:w].copy()
    tr = np.fliplr(quad)
    if not np.array_equal(g[0:h, w:W], tr):
        color_rect(0, 0, w, h, w)                       # clear destination
        if tr.any():
            ops.append(29); sels.append(sel_of(rect_cells(0, 0, h, w)))   # CopyO quadrant
            ops.append(30); sels.append(sel_of([(0, w)]))                 # Paste at (0, w)
            g[0:h, w:W] = quad
            if not np.array_equal(quad, tr):
                ops.append(26)                                            # FlipH in place
                sels.append(sel_of(rect_cells(0, w, h, w)))
                g[0:h, w:W] = tr

    # (c) bottom half = mirror of the finished top half
    top = g[0:h, :].copy()
    bot = np.flipud(top)
    if not np.array_equal(g[h:H, :], bot):
        color_rect(0, h, 0, h, W)                       # clear destination
        if bot.any():
            ops.append(29); sels.append(sel_of(rect_cells(0, 0, h, W)))   # CopyO top half
            ops.append(30); sels.append(sel_of([(h, 0)]))                 # Paste at (h, 0)
            g[h:H, :] = top
            if not np.array_equal(top, bot):
                ops.append(27)                                            # FlipV in place
                sels.append(sel_of(rect_cells(h, 0, h, W)))
                g[h:H, :] = bot

    ops.append(34)
    sels.append(sel_of(rect_cells(0, 0, H, W)))
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
                        f"num_examples+1 ({num_examples + 1}) for task 85c4e7cd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 85c4e7cd"
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
                                f"for task 85c4e7cd"
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
                    f"Failed to build a complete episode for task 85c4e7cd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"85c4e7cd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
