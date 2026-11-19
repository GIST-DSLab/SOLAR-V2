"""
ARC Task: 90c28cc7 (RE-ARC) — LLM-generated grid_maker
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
    # The frame colour 0 is fixed in the generator. The picture's colours are drawn at random
    # for each instance, and the rule does not depend on them, so no colour role is fixed here.
    return {}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, int(round(a + d * (b - a)))))


def _dedupe_rows(g):
    out = []
    for r in g:
        if r not in out:
            out.append(r)
    return out


def _T(g):
    return [list(x) for x in zip(*g)]


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kw) -> dict:
    cols = list(range(1, 10))
    h = _unifint(diff_lb, diff_ub, (2, 10))
    w = _unifint(diff_lb, diff_ub, (2, 10))
    nc = _unifint(diff_lb, diff_ub, (2, 9))
    colss = random.sample(cols, nc)
    gi = [[random.choice(colss) for _ in range(w)] for _ in range(h)]
    gi = _T(_dedupe_rows(_T(_dedupe_rows(gi))))
    go = [list(r) for r in gi]
    h, w = len(gi), len(gi[0])
    if h > max_h or w > max_w:
        raise ValueError("too big")
    fullh = _unifint(diff_lb, diff_ub, (h, max_h))
    fullw = _unifint(diff_lb, diff_ub, (w, max_w))
    inh = _unifint(diff_lb, diff_ub, (h, fullh))
    inw = _unifint(diff_lb, diff_ub, (w, fullw))
    while h < inh or w < inw:
        if h < inh:
            idx = random.randint(0, h - 1)
            gi = gi[:idx + 1] + [list(gi[idx])] + gi[idx + 1:]
        else:
            t = _T(gi)
            idx = random.randint(0, w - 1)
            t = t[:idx + 1] + [list(t[idx])] + t[idx + 1:]
            gi = _T(t)
        h, w = len(gi), len(gi[0])
    while h < fullh or w < fullw:
        if h < fullh:
            z = [0] * w
            gi = [z] + gi if random.random() < 0.5 else gi + [z]
        else:
            t = _T(gi)
            z = [0] * h
            t = [z] + t if random.random() < 0.5 else t + [z]
            gi = _T(t)
        h, w = len(gi), len(gi[0])
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Reading (golf solution): a line is redundant if it is empty (all 0) or repeats a line
    already seen. Columns are pruned first, then rows, and the frame is handled the same way
    as duplicates. There is no separate "find the frame" step.
    How the ops do it:
      1. Columns: each distinct non-empty column keeps its first occurrence. Every contiguous
         block of these survivors slides to the left edge of the canvas
         (CopyI of the block, then Paste at its packed column).
      2. Rows: the same on the column-packed strip (CopyO of each block of surviving rows,
         then Paste at its packed row).
      3. CropGrid the canvas down to the packed block at (0, 0).
    Every operation is computed from I alone.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    nz = np.argwhere(I != 0)
    r0, r1 = int(nz[:, 0].min()), int(nz[:, 0].max())

    def first_reps(lines):
        seen, reps = [], []
        for i, ln in enumerate(lines):
            t = tuple(ln)
            if not any(t):
                continue
            if t in seen:
                continue
            seen.append(t)
            reps.append(i)
        return reps

    def runs(reps):
        out, s = [], 0
        for k in range(1, len(reps) + 1):
            if k == len(reps) or reps[k] - k != reps[s] - s:
                out.append((s, k - 1))
                s = k
        return out

    # --- 1. columns: pack the surviving columns against the left edge ---
    col_reps = first_reps(I.T)
    wo = len(col_reps)
    for a, b in runs(col_reps):
        src = col_reps[a]
        if src == a:
            continue
        # full rectangle: the content rows across this block of surviving columns
        ops.append(28); sels.append([r0, src, r1 - r0, b - a])   # CopyI
        ops.append(30); sels.append([r0, a, 0, 0])               # Paste at packed column

    # --- 2. rows: pack the surviving rows against the top edge (working-grid source) ---
    packed = I[:, col_reps]
    row_reps = first_reps(packed)
    ho = len(row_reps)
    for a, b in runs(row_reps):
        src = row_reps[a]
        if src == a:
            continue
        # full rectangle: this block of surviving rows across the packed columns
        ops.append(29); sels.append([src, 0, b - a, wo - 1])     # CopyO
        ops.append(30); sels.append([a, 0, 0, 0])                # Paste at packed row

    # --- 3. drop everything outside the packed block ---
    if (ho, wo) != (hi, wi):
        ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
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
                                f"for task 90c28cc7"
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
                    f"Failed to build a complete episode for task 90c28cc7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"90c28cc7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
