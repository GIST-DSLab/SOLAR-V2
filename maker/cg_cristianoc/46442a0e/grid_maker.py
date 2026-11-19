"""
ARC Task: 46442a0e (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of  # noqa: F401  (all selections here are full square quadrants)


def sample_colors(num_examples=None) -> dict:
    # Only the background colour is sampled as a role. The other colours are
    # scattered noise, and the rule (a rotation pinwheel) does not depend on them.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    if bgc is None:
        bgc = random.choice(range(10))
    cap = max(1, min(15, max_h // 2, max_w // 2))
    h = unifint(diff_lb, diff_ub, (1, cap))
    w = h
    gi = np.full((h, w), bgc, dtype=int)
    remcols = [c for c in range(10) if c != bgc]
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
    r90 = np.rot90(gi, 3)
    r180 = np.rot90(gi, 2)
    r270 = np.rot90(gi, 1)
    go = np.vstack([np.hstack([gi, r90]), np.hstack([r270, r180])])
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading: the output is a pinwheel. Going clockwise around the quadrants
    # TL -> TR -> BR -> BL, each quadrant is the previous one turned 90 deg clockwise.
    # So: copy the quadrant just finished (from the working grid), paste it into the
    # next quadrant clockwise, and turn that quadrant CW. Every selection is a full
    # n x n quadrant, which can be named from I's size alone (a whole-rectangle bbox).
    I = np.asarray(I, dtype=int)
    n = I.shape[0]  # the generator makes square inputs (w = h)
    N = 2 * n

    ops, sels = [], []
    G = np.zeros((N, N), dtype=int)
    G[:n, :n] = I

    # 1. Expand the canvas to 2n x 2n. I stays at the top-left and the rest is zero.
    ops.append(33); sels.append([0, 0, N - 1, N - 1])

    quads = [(0, 0), (0, n), (n, n), (n, 0)]  # TL, TR, BR, BL (clockwise)
    for k in range(3):
        sr, sc = quads[k]
        dr, dc = quads[k + 1]
        block = G[sr:sr + n, sc:sc + n].copy()
        if not block.any():
            # An all-zero block: copying and pasting it would change nothing.
            continue
        # Copy the quadrant just completed (whole n x n rectangle) from the working grid.
        ops.append(29); sels.append([sr, sc, n - 1, n - 1])
        # Paste it into the next quadrant clockwise (it is all zeros, so the paste is exact).
        ops.append(30); sels.append([dr, dc, 0, 0])
        G[dr:dr + n, dc:dc + n] = block
        # Turn that quadrant 90 deg clockwise (op25). The selection is square, so this
        # is safe. Skip it when the turn would leave the block unchanged.
        rotated = np.rot90(block, 3)
        if not np.array_equal(rotated, block):
            ops.append(25); sels.append([dr, dc, n - 1, n - 1])
            G[dr:dr + n, dc:dc + n] = rotated

    ops.append(34); sels.append([0, 0, N - 1, N - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 46442a0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 46442a0e"
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
                                f"for task 46442a0e"
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
                    f"Failed to build a complete episode for task 46442a0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"46442a0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
