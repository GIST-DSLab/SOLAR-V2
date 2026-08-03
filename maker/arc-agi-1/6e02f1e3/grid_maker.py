"""
ARC Task: 6e02f1e3 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


VARIANTS = [{"numcols": 1}, {"numcols": 2}, {"numcols": 3}]


def sample_colors(num_examples=None) -> dict:
    c1, c2, c3 = random.sample(list(range(10)), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"c1": c1, "c2": c2, "c3": c3, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, c1, c2, c3, numcols=None) -> dict:
    if numcols is None:
        numcols = random.choice((1, 2, 3))
    ub = max(3, min(30, max_h, max_w))
    d = _unifint(diff_lb, diff_ub, (3, ub))
    kk = d * d

    inds = [(r, c) for r in range(d) for c in range(d)]
    random.shuffle(inds)

    gi = np.zeros((d, d), dtype=int)
    go = np.zeros((d, d), dtype=int)

    if numcols == 1:
        gi[:, :] = c1
        go[0, :] = 5
    elif numcols == 2:
        mp = kk // 2
        nc1 = _unifint(diff_lb, diff_ub, (1, mp))
        for (r, c) in inds[:nc1]:
            gi[r, c] = c1
        for (r, c) in inds[nc1:]:
            gi[r, c] = c2
        for k in range(d):
            go[k, k] = 5
    else:
        a = int(1 / 3 * kk)
        b = int(2 / 3 * kk)
        adev = _unifint(diff_lb, diff_ub, (0, a - 1))
        bdev = _unifint(diff_lb, diff_ub, (0, kk - b - 1))
        a -= adev
        b -= bdev
        for (r, c) in inds[:a]:
            gi[r, c] = c1
        for (r, c) in inds[a:b]:
            gi[r, c] = c2
        for (r, c) in inds[b:]:
            gi[r, c] = c3
        for k in range(d):
            go[d - 1 - k, k] = 5

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    d = hi

    # Rule read off the INPUT: how many distinct colors does it use?
    n = len(set(I.flatten().tolist()))

    ops, sels = [], []

    # Input content is fully consumed by the count above; blank the canvas.
    ops.append(0)
    sels.append([0, 0, hi - 1, wi - 1])

    if n == 1:
        # single color -> top row line (rectangular: one op)
        ops.append(5)
        sels.append([0, 0, 0, wi - 1])
    else:
        if n == 2:
            cells = [(k, k) for k in range(d)]              # main diagonal
        else:
            cells = [(d - 1 - k, k) for k in range(d)]      # anti-diagonal, bottom-left up
        for (r, c) in cells:
            ops.append(5)
            sels.append([r, c, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 6e02f1e3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6e02f1e3"
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
                                f"for task 6e02f1e3"
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
                    f"Failed to build a complete episode for task 6e02f1e3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6e02f1e3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
