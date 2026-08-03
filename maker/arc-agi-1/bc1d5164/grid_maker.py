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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # bgc is the canvas / frontier colour, objc the painted cells.
    # bgc is pinned to 0 so the four quadrants can be overlaid with
    # CopyI/Paste (0 == "transparent" for the clipboard), which is what
    # the rule actually is: a union of the four h x w quadrants.
    bgc = 0
    objc = random.choice([c for c in range(10) if c != bgc])
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc) -> dict:
    # fullh = 2h-1, fullw = 2w+1  ->  bound h,w so the input fits max_h/max_w
    h_ub = min(15, (max_h + 1) // 2)
    w_ub = min(14, (max_w - 1) // 2)
    h_ub = max(3, h_ub)
    w_ub = max(2, w_ub)
    h = unifint(diff_lb, diff_ub, (3, h_ub))
    w = unifint(diff_lb, diff_ub, (2, w_ub))
    fullh = 2 * h - 1
    fullw = 2 * w + 1
    inds = asindices(canvas(-1, (h, w)))
    nA = randint(1, (h - 1) * (w - 1) - 1)
    nB = randint(1, (h - 1) * (w - 1) - 1)
    nC = randint(1, (h - 1) * (w - 1) - 1)
    nD = randint(1, (h - 1) * (w - 1) - 1)
    A = sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] < w - 1)), nA)
    B = sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] > 0)), nB)
    C = sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] < w - 1)), nC)
    D = sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] > 0)), nD)
    gi = canvas(bgc, (fullh, fullw))
    gi = fill(gi, objc, A)
    gi = fill(gi, objc, shift(B, (0, fullw - w)))
    gi = fill(gi, objc, shift(C, (fullh - h, 0)))
    gi = fill(gi, objc, shift(D, (fullh - h, fullw - w)))
    go = canvas(bgc, (h, w))
    go = fill(go, objc, set(A) | set(B) | set(C) | set(D))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    fh, fw = I.shape

    # --- geometry measured from I only ---------------------------------
    # input is 2h-1 tall and 2w+1 wide, so:
    h = (fh + 1) // 2
    w = (fw - 1) // 2

    # row h-1 is the all-background separator row (no quadrant reaches it),
    # column w is the all-background separator column.
    bgc = int(I[h - 1, 0])
    pal = sorted(set(I.flatten().tolist()))
    fg = [c for c in pal if c != bgc]
    if not fg:
        cnt = Counter(I.flatten().tolist())
        bgc = cnt.most_common(1)[0][0]
        fg = [c for c in pal if c != bgc]
    objc = int(fg[0]) if fg else int(bgc)

    # the four h x w quadrants, separated by the empty row/col
    quad_origins = [(0, 0), (0, w + 1), (h - 1, 0), (h - 1, w + 1)]

    ops, sels = [], []

    # accumulate the union onto the top-left quadrant (the working area)
    cur = I[0:h, 0:w].copy()

    for (r0, c0) in quad_origins[1:]:
        blk = I[r0:r0 + h, c0:c0 + w]
        new_cells = [(r, c) for r in range(h) for c in range(w)
                     if int(blk[r, c]) == objc and int(cur[r, c]) != objc]
        if not new_cells:
            continue  # this quadrant adds nothing -> emitting ops would be a no-op
        if bgc == 0:
            # background is transparent for the clipboard: CopyI + Paste is
            # exactly "overlay this quadrant's object cells onto the first one"
            ops.append(28); sels.append([r0, c0, h - 1, w - 1])   # full rectangular region
            ops.append(30); sels.append([0, 0, 0, 0])             # paste origin
        else:
            # non-zero background: paste would also carry background over, so
            # paint exactly this quadrant's contributed object cells
            ops.append(objc); sels.append(sel_of(new_cells))
        for (r, c) in new_cells:
            cur[r, c] = objc

    # keep only the overlaid quadrant as the answer canvas
    ops.append(33); sels.append([0, 0, h - 1, w - 1])   # full rectangular region
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
