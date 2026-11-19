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
    # Every color in this task is drawn freshly for each block from all 10 colors.
    # There is no background color and no color role: the rule depends only on
    # whether a block is symmetric. So no color needs to be fixed per episode.
    return {}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, max(lo_i, hi_i)))


def _sym_block(d, cols, nc):
    tcolset = random.sample(cols, nc)
    g = [[0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            v = random.choice(tcolset)
            g[i][j] = v
            g[j][i] = v
    return g, tcolset


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kwargs) -> dict:
    cols = list(range(10))
    lim = min(max_h, max_w)
    d = _unifint(diff_lb, diff_ub, (2, min(7, lim)))
    concat_h = random.choice((True, False))
    long_lim = max_w if concat_h else max_h
    ng = _unifint(diff_lb, diff_ub, (2, max(2, long_lim // d)))
    nc = _unifint(diff_lb, diff_ub, (2, min(9, d ** 2)))

    gik, tcolset = _sym_block(d, cols, nc)
    offd = [(i, j) for i in range(d) for j in range(d) if i != j]
    ndistinv = _unifint(diff_lb, diff_ub, (0, d * (d - 1) // 2 - 1))
    ndist = d * (d - 1) // 2 - ndistinv
    for (i, j) in random.sample(offd, ndist):
        if gik[i][j] == gik[j][i]:
            gik[i][j] = random.choice([c for c in tcolset if c != gik[i][j]])
        else:
            gik[i][j] = gik[j][i]
    target = np.array(gik, dtype=int)
    if np.array_equal(target, target.T):
        # guarantee asymmetry
        i, j = offd[0]
        target[i, j] = random.choice([c for c in tcolset if c != target[j, i]])

    blocks = [target]
    for _ in range(ng - 1):
        b, _ = _sym_block(d, cols, nc)
        b = np.array(b, dtype=int)
        if random.choice((True, False)):
            blocks.append(b)
        else:
            blocks.insert(0, b)
    gi = np.hstack(blocks) if concat_h else np.vstack(blocks)
    return {"input": gi.tolist(), "output": target.tolist()}


def derive_operations(I, O, examples=None):
    # The golf entry reads the task the same way the maker does: cut the grid
    # into d x d blocks along the long side, look at them in order, and keep the
    # first block that differs from its own transpose. That is one crop of the
    # canvas to that block. The two readings agree, so nothing changes here.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    d = min(hi, wi)
    if hi >= wi:
        origins = [(k * d, 0) for k in range(hi // d)]
    else:
        origins = [(0, k * d) for k in range(wi // d)]

    r0, c0 = origins[0]
    for (br, bc) in origins:
        block = I[br:br + d, bc:bc + d]
        if not np.array_equal(block, block.T):
            r0, c0 = br, bc
            break

    # Full d x d rectangle of the asymmetric block -> crop canvas to it
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
