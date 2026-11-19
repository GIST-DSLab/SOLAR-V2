"""
ARC Task: cdecee7f (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, random.randint(1, 9))
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    h = max(3, min(h, max_h))
    w = max(3, min(w, max_w))
    numc = unifint(diff_lb, diff_ub, (1, min(9, w)))
    numc = max(1, min(numc, 9, w))
    locs = sorted(random.sample(range(w), numc))
    gi = [[bgc] * w for _ in range(h)]
    go = []
    for j in locs:
        iloc = random.randint(0, h - 1)
        col = random.choice(ccols)
        gi[iloc][j] = col
        go.append(col)
    go = go + [bgc] * (9 - len(go))
    go_grid = [go[:3], go[3:6][::-1], go[6:]]
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go_grid]}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: marks are sparse (<= w cells on an h*w canvas, h >= 3)
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # every column holds at most one mark; read them left to right
    marks = sorted([(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) != bgc],
                   key=lambda rc: rc[1])
    colors = [int(I[r, c]) for r, c in marks]

    ops, sels = [], []
    DIRS = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}

    def relocate(src, dst, col):
        """Slide a single-cell mark from src to dst: vertical leg first, then horizontal."""
        if src == dst:
            return
        if col == 0:
            # a colour-0 mark is invisible to ARCLE's object grab -> repaint it instead
            ops.append(bgc); sels.append(sel_of([src]))
            ops.append(0); sels.append(sel_of([dst]))
            return
        steps = []
        dr = dst[0] - src[0]
        dc = dst[1] - src[1]
        steps += [(1 if dr > 0 else -1, 0)] * abs(dr)
        steps += [(0, 1 if dc > 0 else -1)] * abs(dc)
        first = True
        for st in steps:
            ops.append(DIRS[st])
            sels.append(sel_of([src]) if first else sel_of([]))
            first = False
        # only the original footprint is left at 0 by the grab snapshot
        if bgc != 0:
            ops.append(bgc); sels.append(sel_of([src]))

    pos = list(marks)

    # stage 1: collapse every column's mark up onto row 0 (one mark per column -> no clashes)
    for k in range(len(pos)):
        r, c = pos[k]
        relocate((r, c), (0, c), colors[k])
        pos[k] = (0, c)

    # stage 2: pack the row leftwards so mark k sits in column k (reading order kept)
    for k in range(len(pos)):
        r, c = pos[k]
        relocate((r, c), (0, k), colors[k])
        pos[k] = (0, k)

    # stage 3: fold the row into a 3x3 snake: row 1 runs right->left, row 2 left->right
    slots = [(0, 0), (0, 1), (0, 2),
             (1, 2), (1, 1), (1, 0),
             (2, 0), (2, 1), (2, 2)]
    for k in range(3, len(pos)):
        relocate(pos[k], slots[k], colors[k])
        pos[k] = slots[k]

    # keep only the 3x3 answer patch (full rectangle)
    if not (hi == 3 and wi == 3):
        ops.append(33); sels.append([0, 0, 2, 2])

    ops.append(34); sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
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
                                f"for task cdecee7f"
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
                    f"Failed to build a complete episode for task cdecee7f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"cdecee7f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
