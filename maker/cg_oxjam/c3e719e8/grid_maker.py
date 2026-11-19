"""
ARC Task: c3e719e8 (RE-ARC) — LLM-generated grid_maker
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
import math
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # Output background is always 0 (hardcoded). The main colour mc is the only
    # colour sampled for its role. The rule uses "the most common colour",
    # never a fixed hue, so mc is fixed once per episode only to keep the
    # episode consistent.
    mc = random.choice(list(range(1, 10)))
    return {"mc": mc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, hi_i))


def generate(diff_lb, diff_ub, max_h, max_w, mc=None) -> dict:
    cols = list(range(1, 10))
    if mc is None:
        mc = random.choice(cols)
    hmax = max(2, min(5, int(math.isqrt(max_h))))
    wmax = max(2, min(5, int(math.isqrt(max_w))))
    h = _unifint(diff_lb, diff_ub, (2, hmax))
    w = _unifint(diff_lb, diff_ub, (2, wmax))
    ncols = _unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    nmc = random.randint(max(1, (h * w) // (ncols + 1) + 1), h * w)
    inds = [(r, c) for r in range(h) for c in range(w)]
    gi = [[-1] * w for _ in range(h)]
    mcc = random.sample(inds, nmc)
    for (r, c) in mcc:
        gi[r][c] = mc
    mset = set(mcc)
    inds = [x for x in inds if x not in mset]
    remcols = [c for c in cols if c != mc]
    ocols = random.sample(remcols, ncols)
    k = len(inds) // ncols + 1
    for ocol in ocols:
        if len(inds) == 0:
            break
        ub = min(nmc - 1, len(inds))
        ub = min(ub, k)
        ub = max(ub, 1)
        locs = random.sample(inds, _unifint(diff_lb, diff_ub, (1, ub)))
        ls = set(locs)
        inds = [x for x in inds if x not in ls]
        for (r, c) in locs:
            gi[r][c] = ocol
    gi = [[mc if v == -1 else v for v in row] for row in gi]
    go = [[0] * (w * w) for _ in range(h * h)]
    for br in range(h):
        for bc in range(w):
            if gi[br][bc] == mc:
                for r in range(h):
                    for c in range(w):
                        go[br * h + r][bc * w + c] = gi[r][c]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # The code-golf entry reads the task the same way the maker does:
    # out[block(br,bc)][r][c] = I[r][c] * (I[br][bc] == mostcolor(I)).
    # That is, the mask "cells of I holding the most common colour" is
    # Kronecker-multiplied by I. Each mc cell of I picks one block of the
    # (h*h, w*w) canvas, and that block gets a full copy of I. The unit
    # (one whole copy of I per block), the condition (most common colour)
    # and the order (copy, resize, fill blocks) all match, so nothing changes.
    # Every selection comes from I alone: the whole input, the output canvas
    # (shape h*h x w*w), and the top-left corners of the blocks picked by mc.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi * hi, wi * wi

    # mc is the strict majority colour by generator construction
    mc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []

    # 1. Copy the whole input to the clipboard (full rectangle, so a bbox is
    #    fine). Input colours are never 0, so every cell is copied.
    ops.append(28)
    sels.append([0, 0, hi - 1, wi - 1])

    # 2. Grow the canvas to (hi*hi, wi*wi). This is a transparent resize:
    #    I stays in block (0,0) and every other cell is 0 (the output bg).
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])

    targets = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == mc]

    # 3. Block (0,0) still holds I from the resize. If I[0,0] is not mc, this
    #    block must be empty, so clear the whole block to 0.
    if (0, 0) not in targets:
        ops.append(0)
        sels.append([0, 0, hi - 1, wi - 1])

    # 4. Paste a copy of I into each block picked by an mc cell, in the
    #    row-major order of the mask cells in I.
    for (r, c) in targets:
        if (r, c) == (0, 0):
            continue
        ops.append(30)
        sels.append([r * hi, c * wi, 0, 0])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c3e719e8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c3e719e8"
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
                                f"for task c3e719e8"
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
                    f"Failed to build a complete episode for task c3e719e8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c3e719e8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
