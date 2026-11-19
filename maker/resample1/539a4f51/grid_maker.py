"""
ARC Task: 539a4f51 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # identity / rot90 / rot180 / rot270 applied to both grids


def sample_colors(num_examples=None) -> dict:
    # Background is hardcoded 0 and the rule reads its colours from I itself,
    # so the only per-instance choice is which corner the pattern grows from.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append({"rot": random.choice(ex)})
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, rot=None, **kwargs) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    cols = list(range(1, 10))
    dmax = max(2, min(15, max_h // 2, max_w // 2))
    d = unifint(diff_lb, diff_ub, (2, dmax))
    gi = canvas(0, (d, d))
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(cols, numc)
    numocc = unifint(diff_lb, diff_ub, (1, d))
    arr = [choice(ccols) for k in range(numocc)]
    while len(set(arr)) == 1:
        arr = [choice(ccols) for k in range(d)]
    for j, col in enumerate(arr):
        gi = fill(gi, col, connect((j, 0), (j, j)) | connect((0, j), (j, j)))
    go = canvas(0, (2 * d, 2 * d))
    for j in range(2 * d):
        col = arr[j % len(arr)]
        go = fill(go, col, connect((j, 0), (j, j)) | connect((0, j), (j, j)))
    rotf = (identity, rot90, rot180, rot270)[rot]
    return {"input": rotf(gi), "output": rotf(go)}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    d = I.shape[0]
    n = 2 * d

    # Find the corner the nested L-shapes grow from: nonzero cells are exactly
    # those within Chebyshev distance < k of it, each band one colour.
    corner, k, arr = None, None, None
    rr, cc_ = np.indices(I.shape)
    nz = I != 0
    for cr, cc in [(0, 0), (0, d - 1), (d - 1, 0), (d - 1, d - 1)]:
        if not nz[cr, cc]:
            continue
        dist = np.maximum(np.abs(rr - cr), np.abs(cc_ - cc))
        kk = int(dist[nz].max()) + 1
        if not np.array_equal(nz, dist < kk):
            continue
        a, ok = [], True
        for j in range(kk):
            vals = set(I[dist == j].tolist())
            if len(vals) != 1:
                ok = False
                break
            a.append(vals.pop())
        if ok:
            corner, k, arr = (cr, cc), kk, a
            break
    cr, cc = corner

    ops, sels = [], []
    # 1. Expand canvas to 2d x 2d (input lands at top-left, padding is bg 0).
    ops.append(33); sels.append([0, 0, n - 1, n - 1])

    # 2. The pattern's corner must sit at the matching corner of the bigger
    #    canvas: slide the whole pattern there (grab once, then empty moves).
    dr = d if cr == d - 1 else 0
    dc = d if cc == d - 1 else 0
    cells = [(r, c) for r in range(d) for c in range(d) if I[r, c] != 0]
    first = True
    for op, steps in ((21, dr), (22, dc)):
        for _ in range(steps):
            ops.append(op)
            sels.append(sel_of(cells) if first else sel_of([]))
            first = False
    # bg is 0, so the vacated footprint needs no repair; step 3 paints it.

    # 3. Continue the nested L-shapes outward, cycling the colour sequence.
    ocr = 0 if cr == 0 else n - 1
    occ = 0 if cc == 0 else n - 1
    for j in range(k, n):
        Lcells = [(r, c) for r in range(n) for c in range(n)
                  if max(abs(r - ocr), abs(c - occ)) == j]
        ops.append(int(arr[j % k])); sels.append(sel_of(Lcells))

    ops.append(34); sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 539a4f51"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 539a4f51"
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
                                f"for task 539a4f51"
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
                    f"Failed to build a complete episode for task 539a4f51 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"539a4f51-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
