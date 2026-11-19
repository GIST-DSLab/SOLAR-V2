"""
ARC Task: 3428a4f5 (RE-ARC) — LLM-generated grid_maker
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

ORIENTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (0, 3)]
    barcol = random.choice(cols)
    rem = [c for c in cols if c != barcol]
    cola = random.choice(rem)
    colb = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        examples = [dict(v) for v in ORIENTS]
        examples += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"barcol": barcol, "cola": cola, "colb": colb, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    d = random.uniform(diff_lb, diff_ub)
    return max(lo, min(hi, lo + round(d * (hi - lo))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, barcol=8, cola=1, colb=2,
             transpose=None, **kwargs) -> dict:
    if transpose is None:
        transpose = random.choice([False, True])
    # non-transposed input: (h, 2w+1); transposed: (2w+1, h)
    lim_h = max_w if transpose else max_h
    lim_wfull = max_h if transpose else max_w
    h = _unifint(diff_lb, diff_ub, 1, min(30, lim_h))
    w = _unifint(diff_lb, diff_ub, 1, min(14, (lim_wfull - 1) // 2))
    bgc = 0
    inds = [(i, j) for i in range(h) for j in range(w)]
    mp = (h * w) // 2

    def count():
        dev = _unifint(diff_lb, diff_ub, 0, mp) * random.choice((1, -1))
        return max(min(h * w - 1, mp + dev), 1) if h * w > 1 else 1

    numa, numb = count(), count()
    a = random.sample(inds, min(numa, len(inds)))
    b = random.sample(inds, min(numb, len(inds)))
    gia = np.full((h, w), bgc, dtype=int)
    gib = np.full((h, w), bgc, dtype=int)
    for (i, j) in a:
        gia[i, j] = cola
    for (i, j) in b:
        gib[i, j] = colb
    gbar = np.full((h, 1), barcol, dtype=int)
    gi = np.hstack([gia, gbar, gib])
    go = np.zeros((h, w), dtype=int)
    for (i, j) in (set(a) | set(b)) - (set(a) & set(b)):
        go[i, j] = 3
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Locate the separator bar from I: the middle column is a single-colour line
    # (left/right halves); otherwise the middle row is (top/bottom halves).
    mid_c = wi // 2
    col_vals = set(I[:, mid_c].tolist())
    if len(col_vals) == 1 and 0 not in col_vals and wi % 2 == 1:
        ho, wo = hi, mid_c
        A = I[:, :wo]
        B = I[:, mid_c + 1:]
    else:
        mid_r = hi // 2
        ho, wo = mid_r, wi
        A = I[:ho, :]
        B = I[mid_r + 1:, :]

    # 1. Crop the canvas down to the first pattern (bgc == 0, transparent copy is fine)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    # Overlay the second pattern on the first: cells where the two patterns differ
    # turn green; cells where both are filled are equal and go black.
    diff_cells, both_cells = [], []
    for r in range(ho):
        for c in range(wo):
            a = A[r, c] != 0
            b = B[r, c] != 0
            if a != b:
                diff_cells.append((r, c))
            elif a and b:
                both_cells.append((r, c))

    if diff_cells:
        ops.append(3); sels.append(sel_of(diff_cells))
    if both_cells:
        ops.append(0); sels.append(sel_of(both_cells))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3428a4f5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3428a4f5"
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
                                f"for task 3428a4f5"
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
                    f"Failed to build a complete episode for task 3428a4f5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3428a4f5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
