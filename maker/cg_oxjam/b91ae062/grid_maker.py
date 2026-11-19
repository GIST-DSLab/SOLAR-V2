"""
ARC Task: b91ae062 (RE-ARC) — LLM-generated grid_maker
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
    # Rule is colour-agnostic (k = number of distinct colours - 1); colours stay per-instance.
    return {}

def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, round(random.uniform(lo, hi))))
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (2, min(5, max_h)))
    w = unifint(diff_lb, diff_ub, (2, min(5, max_w)))
    ub = min(h * w, 10, min(max_h, max_w) // max(h, w) + 1, 30 // max(h, w) + 1)
    numc = unifint(diff_lb, diff_ub, (3, max(3, ub)))
    ccols = random.sample(cols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    fix = random.sample(inds, numc)
    g = [[None] * w for _ in range(h)]
    for c, (i, j) in zip(ccols, fix):
        g[i][j] = c
    for (i, j) in inds:
        if g[i][j] is None:
            g[i][j] = random.choice(ccols)
    k = numc - 1
    out = [[g[i // k][j // k] for j in range(w * k)] for i in range(h * k)]
    return {"input": g, "output": out}

def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Rule (measured from I only): k = numcolors(I) - 1.
    # Reading: first stretch every row sideways (each cell -> a run of k cells),
    # then repeat every stretched row k times downward.
    k = len(np.unique(I)) - 1
    ho, wo = hi * k, wi * k
    ops, sels = [], []

    # Expand canvas to output size (whole rectangle; I transparently kept at top-left).
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    cur = np.zeros((ho, wo), dtype=int)
    cur[:hi, :wi] = I

    # Phase 1: stretch each source row horizontally, in place.
    # Right-to-left so every source cell is read before a wider run covers it.
    for i in range(hi):
        for j in range(wi - 1, -1, -1):
            col = int(I[i, j])
            cells = [(i, j * k + t) for t in range(k)]   # the run this cell becomes
            if all(cur[r, c] == col for r, c in cells):
                continue
            ops.append(col); sels.append(sel_of(cells))
            for r, c in cells:
                cur[r, c] = col

    # Phase 2: repeat each stretched row k times downward.
    # Bottom-up so every stretched source row is copied before it is overwritten.
    for i in range(hi - 1, -1, -1):
        row = cur[i].copy()
        has_zero = bool((I[i] == 0).any())   # transparent paste can't write colour 0
        copied = False
        for d in range(i * k, i * k + k):
            if d == i:
                continue
            if has_zero and d < hi and cur[d].any():
                # stale, already-replicated source row: clear the whole row first
                ops.append(0); sels.append([d, 0, 0, wo - 1])   # full row rectangle
                cur[d] = 0
            new = np.where(row != 0, row, cur[d])
            if np.array_equal(new, cur[d]):
                continue
            if not copied:
                ops.append(29); sels.append([i, 0, 0, wo - 1])  # copy the whole stretched row
                copied = True
            ops.append(30); sels.append([d, 0, 0, 0])            # paste at row d
            cur[d] = new

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
                        f"num_examples+1 ({num_examples + 1}) for task b91ae062"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b91ae062"
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
                                f"for task b91ae062"
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
                    f"Failed to build a complete episode for task b91ae062 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b91ae062-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
