"""
ARC Task: eb5a1d5d (RE-ARC) — LLM-generated grid_maker
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
    if b <= a:
        return a
    lb = diff_lb * (b - a) + a
    ub = diff_ub * (b - a) + a
    lo, hi = int(round(lb)), int(round(ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # The rule (squeeze duplicated rows/cols) is colour-agnostic; ring colours
    # are sampled freshly per instance, exactly like the original generator.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **color_kwargs) -> dict:
    lim = min(max_h, max_w)
    dmax = max(2, min(10, (lim + 1) // 2))
    d = _unifint(diff_lb, diff_ub, (2, dmax))
    n = 2 * d - 1
    colss = random.sample(range(10), d)
    go = [[0] * n for _ in range(n)]
    for j, cc in enumerate(colss):
        for r in range(j, n - j):
            for c in range(j, n - j):
                go[r][c] = cc
    nvenl = _unifint(diff_lb, diff_ub, (0, max(0, max_h - n)))
    nhenl = _unifint(diff_lb, diff_ub, (0, max(0, max_w - n)))
    gi = [row[:] for row in go]
    enl = [nvenl, nhenl]
    while enl[0] > 0 or enl[1] > 0:
        opts = []
        if enl[0] > 0 and len(gi) < max_h:
            opts.append(0)
        if enl[1] > 0 and len(gi[0]) < max_w:
            opts.append(1)
        if not opts:
            break
        ch = random.choice(opts)
        if ch == 0:
            idx = random.randint(0, len(gi) - 1)
            gi = gi[:idx + 1] + [gi[idx][:]] + gi[idx + 1:]
        else:
            idx = random.randint(0, len(gi[0]) - 1)
            gi = [row[:idx + 1] + [row[idx]] + row[idx + 1:] for row in gi]
        enl[ch] -= 1
    return {"input": [list(r) for r in gi], "output": go}


def derive_operations(I, O=None, examples=None):
    # Rule: collapse every run of identical rows and identical columns -> the
    # compact (2d-1)x(2d-1) nested-ring figure, each ring 1 thick.
    # Realization in large strokes: resize the canvas to the compact size once,
    # then paint the rings as nested full rectangles, outermost first (each
    # rectangle = "ring j and everything inside it"; inner rings layer on top).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def dedupe_rows(A):
        keep = [0] + [i for i in range(1, A.shape[0])
                      if not np.array_equal(A[i], A[i - 1])]
        return A[keep]

    C = dedupe_rows(dedupe_rows(I).T).T   # compact figure, computed from I only
    n = C.shape[0]
    d = (n + 1) // 2
    ring_cols = [int(C[j, j]) for j in range(d)]

    ops, sels = [], []
    G = I.copy()
    if (n, n) != (hi, wi):
        # resize canvas to n x n (transparent copy of I's top-left corner)
        ops.append(33); sels.append([0, 0, n - 1, n - 1])
        G = I[:n, :n].copy()

    for j, col in enumerate(ring_cols):
        # full rectangle rows/cols j..n-1-j
        region = G[j:n - j, j:n - j]
        if np.all(region == col):
            continue          # op would change nothing on the current grid
        ops.append(col)
        sels.append([j, j, n - 1 - 2 * j, n - 1 - 2 * j])
        G[j:n - j, j:n - j] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
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
                                f"for task eb5a1d5d"
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
                    f"Failed to build a complete episode for task eb5a1d5d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"eb5a1d5d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
