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


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    lo_i = max(a, min(b, lo_i))
    hi_i = max(lo_i, min(b, hi_i))
    return random.randint(lo_i, hi_i)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    hmax = max(1, min(15, max_h // 2, max_w // 2))
    h = _unifint(diff_lb, diff_ub, (1, hmax))
    w = h
    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in cols if c != bgc]
    numc = _unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(r, c) for r in range(h) for c in range(w)]
    for col in colsch:
        if not inds:
            break
        num = _unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (r, c) in chos:
            gi[r][c] = col
        chs = set(chos)
        inds = [x for x in inds if x not in chs]
    g = np.array(gi, dtype=int)
    top = np.hstack([g, np.rot90(g, 3)])
    bot = np.hstack([np.rot90(g, 1), np.rot90(g, 2)])
    go = np.vstack([top, bot])
    return {"input": g.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading: the output is the top band [I | rot90cw(I)] followed by that whole
    # band turned 180 degrees underneath it (O is 4-fold rotation symmetric).
    # Build the top band, then copy the band as ONE unit, paste it below and
    # rotate it 180 with FlipH + FlipV.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    H, W = 2 * h, 2 * w
    ops, sels = [], []

    # simulated working grid, to skip ops with no visible effect
    G = np.zeros((H, W), dtype=int)
    G[:h, :w] = I

    # 1. Expand canvas to 2h x 2w (I stays at top-left)
    ops.append(33); sels.append([0, 0, H - 1, W - 1])

    # 2. Copy the input, paste into top-right quadrant (an empty, zero region)
    if np.any(I != 0):
        ops.append(28); sels.append([0, 0, h - 1, w - 1])
        ops.append(30); sels.append([0, w, 0, 0])
        G[:h, w:] = I
        # 3. Rotate that pasted square CW (full square region, zeros included)
        R = np.rot90(I, 3)
        if not np.array_equal(R, G[:h, w:]):
            ops.append(25); sels.append([0, w, h - 1, w - 1])
            G[:h, w:] = R

    top = G[:h, :].copy()
    if np.any(top != 0):
        # 4. Copy the whole top band from the working grid, paste it as the bottom band
        ops.append(29); sels.append([0, 0, h - 1, W - 1])
        ops.append(30); sels.append([h, 0, 0, 0])
        G[h:, :] = top
        # 5. Turn the bottom band (full h x 2w rectangle) 180 degrees: FlipH then FlipV
        f1 = np.fliplr(G[h:, :])
        if not np.array_equal(f1, G[h:, :]):
            ops.append(26); sels.append([h, 0, h - 1, W - 1])
            G[h:, :] = f1
        f2 = np.flipud(G[h:, :])
        if not np.array_equal(f2, G[h:, :]):
            ops.append(27); sels.append([h, 0, h - 1, W - 1])
            G[h:, :] = f2

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
