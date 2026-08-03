"""
ARC Task: 662c240a (RE-ARC) — LLM-generated grid_maker
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
    # Rule depends only on structural symmetry of sub-blocks, not on colors,
    # and there is no background color to fix. No color roles needed.
    return {}


def _make_symmetric_block(d, cols):
    nc = random.randint(2, min(9, d * d))
    tcolset = random.sample(cols, nc)
    g = [[0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            col = random.choice(tcolset)
            g[i][j] = col
            g[j][i] = col
    return g, tcolset


def _make_broken_block(d, cols):
    # symmetric block, then break symmetry on off-diagonal cells
    g, tcolset = _make_symmetric_block(d, cols)
    offdiag = [(i, j) for i in range(d) for j in range(d) if i != j]
    tot = d * (d - 1) // 2
    ndistinv = random.randint(0, tot - 1)
    ndist = tot - ndistinv
    distinds = random.sample(offdiag, min(ndist, len(offdiag)))
    for (i, j) in distinds:
        if g[i][j] == g[j][i]:
            choices = [c for c in tcolset if c != g[i][j]]
            g[i][j] = random.choice(choices)
        else:
            g[i][j] = g[j][i]
    return g


def generate(diff_lb, diff_ub, max_h, max_w, **color_kwargs) -> dict:
    cols = list(range(10))

    d = random.randint(2, max(2, min(7, max_h, max_w)))

    # pick orientation that fits at least 2 blocks along the long axis
    options = []
    if max_w // d >= 2:
        options.append('h')
    if max_h // d >= 2:
        options.append('v')
    if not options:
        d = max(2, min(7, min(max_h, max_w) // 2))
        if max_w // d >= 2:
            options.append('h')
        if max_h // d >= 2:
            options.append('v')
    concatf = random.choice(options)

    if concatf == 'h':
        cap = min(max_w // d, 30 // d)
    else:
        cap = min(max_h // d, 30 // d)
    cap = max(2, cap)
    ng = random.randint(2, cap)

    # the ONE non-symmetric block = the output
    for _ in range(200):
        first_block = _make_broken_block(d, cols)
        fb = np.array(first_block, dtype=int)
        if not np.array_equal(fb, fb.T):
            break
    else:
        # fallback: force asymmetry
        first_block[0][1] = (first_block[1][0] + 1) % 10
        fb = np.array(first_block, dtype=int)

    blocks = [first_block]
    for _ in range(ng - 1):
        b, _ = _make_symmetric_block(d, cols)
        if random.choice([True, False]):
            blocks.append(b)
        else:
            blocks.insert(0, b)

    arrs = [np.array(b, dtype=int) for b in blocks]
    if concatf == 'h':
        grid = np.hstack(arrs)
    else:
        grid = np.vstack(arrs)

    return {"input": grid.tolist(), "output": fb.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    d = min(hi, wi)

    # blocks lie along the long axis; short axis length = d
    if hi >= wi:
        n = hi // d
        origins = [(k * d, 0) for k in range(n)]
    else:
        n = wi // d
        origins = [(0, k * d) for k in range(n)]

    # measure the rule from I: the target block is the one NOT equal to its
    # main-diagonal mirror (transpose). All other blocks are symmetric.
    r0, c0 = origins[0]
    for (br, bc) in origins:
        block = I[br:br + d, bc:bc + d]
        if not np.array_equal(block, block.T):
            r0, c0 = br, bc
            break

    # crop the working canvas down to that block
    ops.append(33)
    sels.append([r0, c0, d - 1, d - 1])
    ops.append(34)
    sels.append([0, 0, d - 1, d - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 662c240a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 662c240a"
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
                                f"for task 662c240a"
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
                    f"Failed to build a complete episode for task 662c240a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"662c240a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
