"""
ARC Task: 7fe24cdd (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level colour role; the rule (rotations of
    # the whole tile) does not depend on which foreground colours appear.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, a + round(d * (b - a))))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    # output is 2h x 2h, so h is bounded by half the max size
    hmax = max(1, min(15, max_h // 2, max_w // 2))
    h = unifint(diff_lb, diff_ub, (1, hmax))
    w = h
    gi = np.full((h, w), bgc, dtype=int)
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for col in colsch:
        if not inds:
            break
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (i, j) in chos:
            gi[i, j] = col
        chs = set(chos)
        inds = [x for x in inds if x not in chs]
    r90 = np.rot90(gi, 3)    # CW
    r180 = np.rot90(gi, 2)
    r270 = np.rot90(gi, 1)   # CCW
    go1 = np.hstack([gi, r90])
    go2 = np.hstack([r270, r180])
    go = np.vstack([go1, go2])
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # The output is the input completed into a picture that is unchanged by a
    # quarter turn of the whole canvas. Build it one whole-canvas rotation at a time:
    #   1. Grow the canvas to 2h x 2h. The input sits in the top-left quarter.
    #   2. Copy the canvas, turn the WHOLE canvas a quarter turn clockwise
    #      (the input becomes rot90(I) in the top-right), then paste the copy back.
    #      The top half is now I | rot90(I).
    #   3. Copy the canvas, turn the WHOLE canvas a half turn (two clockwise quarter
    #      turns). The top half lands in the bottom half as rot270(I) | rot180(I).
    #      Paste the copy back to put the top half in place again.
    # All selections are the full square canvas (a full rectangle by intent), so Rotate is valid.
    I = np.asarray(I, dtype=int)
    h = I.shape[0]          # generator forces w == h
    n = 2 * h
    full = [0, 0, n - 1, n - 1]   # entire square canvas (intended full rectangle)

    ops, sels = [], []
    # 1. expand canvas to 2h x 2h
    ops.append(33); sels.append(full)

    if np.any(I != 0):
        # 2. quarter-turn of the whole canvas, keeping the original via clipboard
        ops.append(29); sels.append(full)          # CopyO whole canvas
        ops.append(25); sels.append(full)          # Rotate CW whole canvas
        ops.append(30); sels.append([0, 0, 0, 0])  # paste original back at origin
        # 3. half-turn of the whole canvas, keeping the top half via clipboard
        ops.append(29); sels.append(full)          # CopyO whole canvas
        ops.append(25); sels.append(full)          # Rotate CW
        ops.append(25); sels.append(full)          # Rotate CW again (half turn total)
        ops.append(30); sels.append([0, 0, 0, 0])  # paste top half back

    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 7fe24cdd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7fe24cdd"
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
                                f"for task 7fe24cdd"
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
                    f"Failed to build a complete episode for task 7fe24cdd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7fe24cdd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
