"""
ARC Task: bc1d5164 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, objc = random.sample(range(10), 2)
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))
    if bgc is None or objc is None:
        bgc, objc = random.sample(range(10), 2)
    hub = max(3, min(15, (max_h + 1) // 2))
    wub = max(2, min(14, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (3, hub))
    w = unifint(diff_lb, diff_ub, (2, wub))
    fullh = 2 * h - 1
    fullw = 2 * w + 1
    inds = [(i, j) for i in range(h) for j in range(w)]

    def pick(f):
        pool = [ij for ij in inds if f(ij)]
        n = random.randint(1, (h - 1) * (w - 1) - 1)
        return random.sample(pool, n)
    A = pick(lambda ij: ij[0] < h - 1 and ij[1] < w - 1)
    B = pick(lambda ij: ij[0] < h - 1 and ij[1] > 0)
    C = pick(lambda ij: ij[0] > 0 and ij[1] < w - 1)
    D = pick(lambda ij: ij[0] > 0 and ij[1] > 0)
    gi = [[bgc] * fullw for _ in range(fullh)]
    go = [[bgc] * w for _ in range(h)]
    for S, (dr, dc) in [(A, (0, 0)), (B, (0, fullw - w)),
                        (C, (fullh - h, 0)), (D, (fullh - h, fullw - w))]:
        for i, j in S:
            gi[i + dr][j + dc] = objc
            go[i][j] = objc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading: the grid is folded twice. First the right half (full height)
    # is laid over the left half, then the bottom half of that strip is laid
    # over the top half; object cells win (union), background is transparent.
    I = np.asarray(I, dtype=int)
    fh, fw = I.shape
    h = (fh + 1) // 2
    w = (fw - 1) // 2
    bgc = int(I[h - 1, 0])                     # row h-1 is the empty seam row
    others = sorted(set(I.flatten().tolist()) - {bgc})
    objc = int(others[0]) if others else bgc

    ops, sels = [], []
    g = I.copy()

    def cells_of(grid, col, hh, ww):
        return [(r, c) for r in range(hh) for c in range(ww) if grid[r, c] == col]

    # Make background the clipboard-transparent colour (0) so folding = paste.
    tmp = None
    if objc == 0:
        tmp = next(c for c in range(1, 10) if c != bgc)
        cs = cells_of(g, 0, fh, fw)            # all object-coloured cells
        ops.append(tmp); sels.append(sel_of(cs))
        for r, c in cs:
            g[r, c] = tmp
    if bgc != 0:
        cs = cells_of(g, bgc, fh, fw)          # all background-coloured cells
        ops.append(0); sels.append(sel_of(cs))
        for r, c in cs:
            g[r, c] = 0
    recolored = (bgc != 0) or (objc == 0)

    # Fold 1: right half (full-height rectangle) over the left half.
    blk = g[0:fh, w + 1:2 * w + 1].copy()
    if np.any((blk != 0) & (g[0:fh, 0:w] != blk)):
        ops.append(29 if recolored else 28)
        sels.append([0, w + 1, fh - 1, w - 1])  # full rectangular right half
        ops.append(30); sels.append([0, 0, 0, 0])
        m = blk != 0
        g[0:fh, 0:w][m] = blk[m]

    # Fold 2: bottom half of the left strip over its top half.
    blk = g[h - 1:fh, 0:w].copy()
    if np.any((blk != 0) & (g[0:h, 0:w] != blk)):
        ops.append(29); sels.append([h - 1, 0, h - 1, w - 1])  # full rectangle
        ops.append(30); sels.append([0, 0, 0, 0])
        m = blk != 0
        g[0:h, 0:w][m] = blk[m]

    # Keep only the folded quadrant.
    ops.append(33); sels.append([0, 0, h - 1, w - 1])  # full rectangular region
    g = g[0:h, 0:w].copy()

    # Restore the real colours on the folded canvas.
    if bgc != 0:
        cs = cells_of(g, 0, h, w)              # all empty (background) cells
        if cs:
            ops.append(bgc); sels.append(sel_of(cs))
            for r, c in cs:
                g[r, c] = bgc
    if tmp is not None:
        cs = cells_of(g, tmp, h, w)            # all temp-coloured object cells
        if cs:
            ops.append(0); sels.append(sel_of(cs))
            for r, c in cs:
                g[r, c] = 0

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
                        f"num_examples+1 ({num_examples + 1}) for task bc1d5164"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bc1d5164"
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
                                f"for task bc1d5164"
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
                    f"Failed to build a complete episode for task bc1d5164 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bc1d5164-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
