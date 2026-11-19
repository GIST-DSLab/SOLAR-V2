"""
ARC Task: 77fdfe62 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    c1, c2, c3, c4, barc, bgc, inc = random.sample(cols, 7)
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4,
            "barc": barc, "bgc": bgc, "inc": inc}


def generate(diff_lb, diff_ub, max_h, max_w, c1, c2, c3, c4, barc, bgc, inc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + int(round((b - a) * lb))
        hi = a + int(round((b - a) * ub))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if lo > hi:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    hmax = max(1, min(13, (max_h - 4) // 2))
    wmax = max(1, min(13, (max_w - 4) // 2))
    h = unifint(diff_lb, diff_ub, (1, hmax))
    w = unifint(diff_lb, diff_ub, (1, wmax))
    inds = [(i, j) for i in range(h) for j in range(w)]
    fullh, fullw = 2 * h + 4, 2 * w + 4
    ns = [unifint(diff_lb, diff_ub, (1, h * w)) for _ in range(4)]
    picks = [random.sample(inds, n) for n in ns]

    gi = np.full((fullh, fullw), bgc, dtype=int)
    gi[1, :] = barc
    gi[fullh - 2, :] = barc
    gi[:, 1] = barc
    gi[:, fullw - 2] = barc
    gi[0, 0] = c1
    gi[0, fullw - 1] = c2
    gi[fullh - 1, 0] = c3
    gi[fullh - 1, fullw - 1] = c4
    offs_in = [(2, 2), (2, 2 + w), (2 + h, 2), (2 + h, 2 + w)]
    offs_out = [(0, 0), (0, w), (h, 0), (h, w)]
    cs = [c1, c2, c3, c4]
    go = np.full((2 * h, 2 * w), bgc, dtype=int)
    for p, (ri, ci), (ro, co), col in zip(picks, offs_in, offs_out, cs):
        for (i, j) in p:
            gi[ri + i, ci + j] = inc
            go[ro + i, co + j] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading shared by the maker and the golf entry: the interior is cut into four
    # quadrants, and in each quadrant the "ink" cells take the colour of the
    # matching outer corner (the golf entry does this as corner * cell). The two
    # readings agree; the only real choice is the order. Here each quadrant is
    # recoloured while its corner marker is still on the canvas, and the frame is
    # cropped away after that, once the corner markers have been used.
    # Every selection comes from I alone: quadrant rectangles and the ink-coloured
    # cells inside them.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    H, W = hi - 4, wi - 4          # interior size = output size
    h, w = H // 2, W // 2

    bgc = int(I[0, 2])             # the generator fills the frame's top row with bgc
    interior = I[2:2 + H, 2:2 + W]
    incs = [v for v in np.unique(interior).tolist() if v != bgc]
    inc = incs[0] if incs else bgc

    ops, sels = [], []
    corners = [((0, 0), (2, 2)),
               ((0, wi - 1), (2, 2 + w)),
               ((hi - 1, 0), (2 + h, 2)),
               ((hi - 1, wi - 1), (2 + h, 2 + w))]
    for (cr, cc), (r0, c0) in corners:
        col = int(I[cr, cc])
        cells = [(r0 + r, c0 + c) for r in range(h) for c in range(w)
                 if I[r0 + r, c0 + c] == inc]
        if cells and col != inc:
            ops.append(col); sels.append(sel_of(cells))

    # strip the two-cell frame (bars and corner markers): crop to the interior rectangle
    ops.append(33); sels.append([2, 2, H - 1, W - 1])

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 77fdfe62"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 77fdfe62"
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
                                f"for task 77fdfe62"
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
                    f"Failed to build a complete episode for task 77fdfe62 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"77fdfe62-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
