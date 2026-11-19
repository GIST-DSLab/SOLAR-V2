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
from maker.sel_helpers import sel_of

VARIANTS = [{"numcols": 1}, {"numcols": 2}, {"numcols": 3}]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    lo_i = max(a, lo_i)
    hi_i = min(b, max(lo_i, hi_i))
    return random.randint(lo_i, hi_i)


def sample_colors(num_examples=None) -> dict:
    # The rule depends only on HOW MANY colours the input uses, not on which ones,
    # so no colour role is fixed. The case count (1/2/3) is planned so every case
    # appears among the examples whenever there are enough example slots.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, numcols=None, **kwargs) -> dict:
    if numcols is None:
        numcols = random.choice((1, 2, 3))
    dmax = max(3, min(30, max_h, max_w))
    d = _unifint(diff_lb, diff_ub, (3, dmax))
    inds = [(i, j) for i in range(d) for j in range(d)]
    random.shuffle(inds)
    chcols = random.sample(list(range(10)), numcols)
    gi = [[0] * d for _ in range(d)]
    go = [[0] * d for _ in range(d)]
    if numcols == 1:
        gi = [[chcols[0]] * d for _ in range(d)]
        for j in range(d):
            go[0][j] = 5
    elif numcols == 2:
        c1, c2 = chcols
        mp = (d ** 2) // 2
        nc1 = _unifint(diff_lb, diff_ub, (1, mp))
        for k, (i, j) in enumerate(inds):
            gi[i][j] = c1 if k < nc1 else c2
        for k in range(d):
            go[k][k] = 5
    else:
        c1, c2, c3 = chcols
        kk = d ** 2
        a = int(1 / 3 * kk)
        b = int(2 / 3 * kk)
        adev = _unifint(diff_lb, diff_ub, (0, a - 1))
        bdev = _unifint(diff_lb, diff_ub, (0, kk - b - 1))
        a -= adev
        b -= bdev
        for k, (i, j) in enumerate(inds):
            gi[i][j] = c1 if k < a else (c2 if k < b else c3)
        for k in range(d):
            go[d - 1 - k][k] = 5
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # Rule read off the INPUT only: how many distinct colours does it use?
    n = len(set(I.flatten().tolist()))

    ops, sels = [], []

    # The input's content is used up by that count, so clear the canvas.
    # (Skip this when the grid is already all 0, because it would change nothing.)
    if np.any(I != 0):
        ops.append(0)
        sels.append([0, 0, hi - 1, wi - 1])  # whole grid rectangle

    # Column-wise reading (from the golf entry): every column gets exactly one 5,
    # and its row is a function of the column index that depends on the colour count:
    #   1 colour  -> row 0 in every column       (top edge)
    #   2 colours -> row c                       (main diagonal)
    #   3 colours -> row d-1-c                   (anti-diagonal)
    # Paint the columns left to right, one marked cell per column.
    def row_for(c):
        if n == 1:
            return 0
        if n == 2:
            return c
        return hi - 1 - c

    for c in range(wi):
        ops.append(5)
        sels.append(sel_of([(row_for(c), c)]))

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
