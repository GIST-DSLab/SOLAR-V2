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
from collections import Counter
import numpy as np


def unifint(diff_lb, diff_ub, bounds):
    lo, hi = bounds
    a = lo + int(round((hi - lo) * diff_lb))
    b = lo + int(round((hi - lo) * diff_ub))
    a = max(lo, min(a, hi))
    b = max(lo, min(b, hi))
    if b < a:
        a, b = b, a
    return random.randint(a, b)


def sample_colors(num_examples=None) -> dict:
    # Rule is pattern-based (placement follows the MOST-COMMON color of I).
    # Only the most-common color role matters; fix it for episode consistency.
    cols = list(range(1, 10))
    mc = random.choice(cols)
    return {"mc": mc}


def generate(diff_lb, diff_ub, max_h, max_w, mc, **color_kwargs) -> dict:
    cols = list(range(1, 10))
    # output is (h*h, w*w); keep within max_h/max_w and original (2,5) bounds
    hmax = max(2, min(5, int(math.isqrt(int(max_h)))))
    wmax = max(2, min(5, int(math.isqrt(int(max_w)))))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))

    hw = h * w
    ncols = unifint(diff_lb, diff_ub, (1, min(hw - 1, 8)))
    nmc = random.randint(max(1, hw // (ncols + 1) + 1), hw)

    allinds = [(r, c) for r in range(h) for c in range(w)]
    remcols = [c for c in cols if c != mc]

    mcc = random.sample(allinds, nmc)
    mcc_set = set(mcc)
    inds_rem = [i for i in allinds if i not in mcc_set]

    gridmap = {i: mc for i in mcc}
    ocols = random.sample(remcols, ncols)
    k = len(inds_rem) // ncols + 1
    for ocol in ocols:
        if len(inds_rem) == 0:
            break
        ub = min(nmc - 1, len(inds_rem))
        ub = min(ub, k)
        ub = max(ub, 1)
        locs = random.sample(inds_rem, unifint(diff_lb, diff_ub, (1, ub)))
        locs_set = set(locs)
        inds_rem = [i for i in inds_rem if i not in locs_set]
        for l in locs:
            gridmap[l] = ocol
    # leftover cells (the -1 -> mc replacement in original)
    for i in allinds:
        if i not in gridmap:
            gridmap[i] = mc

    gi = [[gridmap[(r, c)] for c in range(w)] for r in range(h)]

    ho, wo = h * h, w * w
    go = [[0] * wo for _ in range(ho)]
    for r in range(h):
        for c in range(w):
            if gi[r][c] == mc:
                for rr in range(h):
                    for cc in range(w):
                        go[r * h + rr][c * w + cc] = gi[rr][cc]

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background of the OUTPUT canvas is 0; mc = most-common color of I,
    # which is exactly the color whose cells select which blocks get a copy of I.
    mc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops = []
    sels = []

    # 1. Copy the whole input to clipboard (full rectangle -> bbox ok).
    ops.append(28)
    sels.append([0, 0, hi - 1, wi - 1])

    # 2. Expand canvas to the (hi*hi, wi*wi) output size. Transparent resize
    #    leaves I in the top-left block, zeros elsewhere.
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])

    # blocks to fill: every (r,c) where I[r,c] == mc
    targets = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == mc]

    # 3. Repair the top-left block (0,0) left over from resize:
    #    if it is NOT a target, clear it to 0 (full-rectangle Color0).
    if (0, 0) not in targets:
        ops.append(0)
        sels.append([0, 0, hi - 1, wi - 1])

    # 4. Paste a copy of I into every target block (skip (0,0) which resize
    #    already placed correctly when it is a target).
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
