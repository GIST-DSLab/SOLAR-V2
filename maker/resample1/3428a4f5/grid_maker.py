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

def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)

VARIANTS = [{"transpose": False}, {"transpose": True}]

def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(1, 10) if c != 3]
    barcol = random.choice(cols)
    rem = [c for c in cols if c != barcol]
    cola = random.choice(rem)
    colb = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"barcol": barcol, "cola": cola, "colb": colb, "instance_plan": plan}

def generate(diff_lb, diff_ub, max_h=30, max_w=30, barcol=5, cola=2, colb=8,
             transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice([False, True])
    if transpose:
        hmax, wmax = max_w, (max_h - 1) // 2
    else:
        hmax, wmax = max_h, (max_w - 1) // 2
    wmax = max(1, min(14, wmax))
    hmax = max(1, hmax)
    h = _unifint(diff_lb, diff_ub, (1, hmax))
    w = _unifint(diff_lb, diff_ub, (1, wmax))
    inds = [(i, j) for i in range(h) for j in range(w)]
    mp = (h * w) // 2
    deva = _unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
    devb = _unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
    numa = max(min(h * w - 1, mp + deva), 1)
    numb = max(min(h * w - 1, mp + devb), 1)
    numa = min(numa, len(inds)); numb = min(numb, len(inds))
    a = set(random.sample(inds, numa))
    b = set(random.sample(inds, numb))
    ga = np.zeros((h, w), dtype=int); gb = np.zeros((h, w), dtype=int)
    go = np.zeros((h, w), dtype=int)
    for (i, j) in a: ga[i, j] = cola
    for (i, j) in b: gb[i, j] = colb
    for (i, j) in (a ^ b): go[i, j] = 3
    gi = np.hstack([ga, np.full((h, 1), barcol), gb])
    if transpose:
        gi = gi.T; go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    vertical = False
    if wi % 2 == 1:
        col = I[:, wi // 2]
        if col[0] != 0 and np.all(col == col[0]):
            others = np.delete(I, wi // 2, axis=1)
            if not np.any(others == col[0]):
                vertical = True
    if vertical:
        hw = wi // 2
        A = I[:, :hw]; B = I[:, hw + 1:]
        h, w = hi, hw
    else:
        hh = hi // 2
        A = I[:hh, :]; B = I[hh + 1:, :]
        h, w = hh, wi
    both = [(r, c) for r in range(h) for c in range(w) if A[r, c] != 0 and B[r, c] != 0]
    aonly = [(r, c) for r in range(h) for c in range(w) if A[r, c] != 0 and B[r, c] == 0]
    bonly = [(r, c) for r in range(h) for c in range(w) if A[r, c] == 0 and B[r, c] != 0]
    ops, sels = [], []
    # cells marked in both halves cancel out -> clear them in the first half
    if both:
        ops.append(0); sels.append(sel_of(both))
    # cells marked only in the first half -> recolour to 3
    if aonly:
        ops.append(3); sels.append(sel_of(aonly))
    # cells marked only in the second half -> add 3 at the matching first-half position
    if bonly:
        ops.append(3); sels.append(sel_of(bonly))
    # crop to the first half (the full h x w rectangle is exactly the intended region)
    ops.append(33); sels.append([0, 0, h - 1, w - 1])
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
