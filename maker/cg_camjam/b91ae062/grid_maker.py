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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule (upscale by numcolors-1) depends only on how many distinct colours
    # the input has, not on which ones. There is no background, so no colour role
    # needs to be fixed across the episode.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **color_kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))

    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    cap = min(h * w, 10, min(max_h // h, max_w // w) + 1)
    cap = max(3, cap)
    numc = unifint(diff_lb, diff_ub, (3, cap))
    ccols = random.sample(cols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    fixinds = random.sample(inds, numc)
    gi = [[-1] * w for _ in range(h)]
    for cc, (i, j) in zip(ccols, fixinds):
        gi[i][j] = cc
    for (i, j) in inds:
        if gi[i][j] == -1:
            gi[i][j] = random.choice(ccols)
    k = numc - 1
    go = [[gi[r // k][c // k] for c in range(w * k)] for r in range(h * k)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (as in the golf entry): upscaling by k = numcolors(I) - 1 splits into two
    # separate steps instead of k x k cell blocks:
    #   1) stretch every input row sideways: each cell becomes a run of k cells
    #   2) repeat every stretched row k times downward (CopyO the row, Paste it).
    # Everything here is measured from I alone. O is never read.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    k = len(np.unique(I)) - 1
    ho, wo = hi * k, wi * k

    ops, sels = [], []

    # Make the canvas output-sized. I is copied into the top-left and the rest is 0.
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])
    cur = np.zeros((ho, wo), dtype=int)
    cur[:hi, :wi] = I

    # Step 1: stretch rows 0..hi-1 in place. Go right to left inside each row.
    # Colours come from I. Skip a run that already shows its colour.
    for i in range(hi):
        for j in range(wi - 1, -1, -1):
            col = int(I[i, j])
            cells = [(i, j * k + d) for d in range(k)]
            if all(cur[r, c] == col for r, c in cells):
                continue
            ops.append(col)
            sels.append(sel_of(cells))
            for r, c in cells:
                cur[r, c] = col

    # Step 2: repeat rows, bottom-up. This way stretched row i is still in place
    # when it gets copied into rows i*k .. i*k+k-1.
    for i in range(hi - 1, -1, -1):
        src = cur[i].copy()
        copied = False
        for d in range(k):
            r = i * k + d
            if r == i:
                continue  # the row is its own copy
            # Paste does not write 0. Where the stretched row is 0 but the target row
            # still holds old non-zero content, clear the row's 0-runs first.
            zero_cols = [c for c in range(wo) if src[c] == 0]
            if any(cur[r, c] != 0 for c in zero_cols):
                ops.append(0)
                sels.append(sel_of([(r, c) for c in zero_cols]))
                cur[r, zero_cols] = 0
            if np.array_equal(cur[r], src):
                continue
            if not copied:
                ops.append(29)  # CopyO: the stretched row i (full row rectangle)
                sels.append([i, 0, 0, wo - 1])
                copied = True
            ops.append(30)
            sels.append([r, 0, 0, 0])
            nz = src != 0
            cur[r, nz] = src[nz]

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
