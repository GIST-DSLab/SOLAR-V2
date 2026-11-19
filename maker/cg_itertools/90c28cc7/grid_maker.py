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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Rule is colour-agnostic (drop empty / repeated lines); the frame colour 0 is hardcoded.
    return {}


def _dedupe(rows):
    out = []
    for r in rows:
        if r not in out:
            out.append(r)
    return out


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + round(lb * (b - a))
        hi = a + round(ub * (b - a))
        return random.randint(min(lo, hi), max(lo, hi))

    T = lambda g: [list(x) for x in zip(*g)]
    while True:
        h = unifint(diff_lb, diff_ub, (2, min(10, max_h)))
        w = unifint(diff_lb, diff_ub, (2, min(10, max_w)))
        nc = unifint(diff_lb, diff_ub, (2, 9))
        colss = random.sample(range(1, 10), nc)
        gi = [[random.choice(colss) for _ in range(w)] for _ in range(h)]
        gi = T(_dedupe(T(_dedupe(gi))))
        # keep only pictures whose rows and cols are all distinct (rule well-defined)
        if len(_dedupe(gi)) != len(gi) or len(_dedupe(T(gi))) != len(gi[0]):
            continue
        break
    go = [list(r) for r in gi]
    h, w = len(gi), len(gi[0])
    fullh = unifint(diff_lb, diff_ub, (h, max_h))
    fullw = unifint(diff_lb, diff_ub, (w, max_w))
    inh = unifint(diff_lb, diff_ub, (h, fullh))
    inw = unifint(diff_lb, diff_ub, (w, fullw))
    while h < inh or w < inw:
        if h < inh:
            idx = random.randint(0, h - 1)
            gi = gi[:idx + 1] + [list(gi[idx])] + gi[idx + 1:]
        else:
            g = T(gi)
            idx = random.randint(0, w - 1)
            g = g[:idx + 1] + [list(g[idx])] + g[idx + 1:]
            gi = T(g)
        h, w = len(gi), len(gi[0])
    while h < fullh or w < fullw:
        if h < fullh:
            z = [0] * w
            gi = [z] + gi if random.random() < 0.5 else gi + [z]
        else:
            g = T(gi)
            z = [0] * h
            g = [z] + g if random.random() < 0.5 else g + [z]
            gi = T(g)
        h, w = len(gi), len(gi[0])
    return {"input": [list(r) for r in gi], "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (one axis at a time, columns first): a column that is all frame (0), or that
    repeats a column already seen, carries nothing - drop it; the surviving columns close
    up against the left edge. Then the same for rows of the column-collapsed picture,
    closing up against the top edge. Finally cut the canvas down to the closed-up block.
    Frame lines and duplicate lines are treated as the same kind of thing (lines to drop).
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def first_reps(lines):
        # indices of lines that are non-empty and not seen before
        reps, seen = [], []
        for i, ln in enumerate(lines):
            t = tuple(int(x) for x in ln)
            if any(t) and t not in seen:
                seen.append(t)
                reps.append(i)
        return reps

    def groups(reps):
        # consecutive kept lines with a constant shift move together as one block
        out, s = [], 0
        for k in range(1, len(reps) + 1):
            if k == len(reps) or reps[k] - k != reps[s] - s:
                out.append((s, k - 1))
                s = k
        return out

    # pass 1: columns (read from the input) close up to column 0
    col_reps = first_reps(I.T)
    for a, b in groups(col_reps):
        src = col_reps[a]
        if src == a:
            continue  # block already in place
        ops.append(28); sels.append([0, src, hi - 1, b - a])   # CopyI full-height block of kept columns
        ops.append(30); sels.append([0, a, 0, 0])              # Paste at its closed-up column
    wo = len(col_reps)
    C = I[:, col_reps]

    # pass 2: rows (read from the working grid) close up to row 0
    row_reps = first_reps(C)
    for a, b in groups(row_reps):
        src = row_reps[a]
        if src == a:
            continue
        ops.append(29); sels.append([src, 0, b - a, wo - 1])   # CopyO block of kept rows (collapsed width)
        ops.append(30); sels.append([a, 0, 0, 0])              # Paste at its closed-up row
    ho = len(row_reps)

    # single crop: the closed-up block is exactly the full rectangle [0,0,ho,wo]
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
