"""
ARC Task: d631b094 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # bgc is hardcoded 0 in the generator; the mark colour is sampled.
    fgc = random.choice(range(1, 10))
    return {"fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, fgc=None, **kw) -> dict:
    from random import sample

    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    if fgc is None:
        fgc = random.choice(range(1, 10))
    bgc = 0
    h = unifint(diff_lb, diff_ub, (2, max(2, max_h)))
    w = unifint(diff_lb, diff_ub, (2, max(2, max_w)))
    nc = unifint(diff_lb, diff_ub, (1, max(1, min(30, (h * w) // 2 - 1))))
    gi = [[bgc] * w for _ in range(h)]
    cels = sample([(i, j) for i in range(h) for j in range(w)], nc)
    for i, j in cels:
        gi[i][j] = fgc
    go = [[fgc] * nc]
    return {"input": gi, "output": go}


MOVE = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}


def derive_operations(I, O=None, examples=None):
    # Reading: the marks, taken in reading order, are gathered into one row.
    # Each mark is an object that slides into the next free slot of the top row;
    # the gathered top row is then cropped out as the answer.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = 0  # the generator paints the canvas 0
    marks = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]  # reading order
    fgc = Counter(int(I[r, c]) for r, c in marks).most_common(1)[0][0]
    nc = len(marks)
    W = max(w, nc)
    ops, sels = [], []
    grid = np.zeros((h, W), dtype=int)
    grid[:, :w] = I
    seen = {I.tobytes()} if W == w else {I.tobytes(), grid.tobytes()}

    # Widen the canvas when the top row is too short to hold every mark.
    if W > w:
        ops.append(33); sels.append([0, 0, h - 1, W - 1])  # whole widened canvas (rectangle)

    # Gather each mark, in reading order, into slot k of the top row.
    for k, (sr, sc) in enumerate(marks):
        tgt = (0, k)
        if (sr, sc) == tgt:
            continue
        snap = grid.copy(); snap[sr, sc] = 0
        occ = snap != 0
        # Shortest path to the slot that glides around the other marks
        # (landing on one would hide the moving mark -> invisible step).
        prev = {(sr, sc): None}
        dq = deque([(sr, sc)])
        while dq:
            p = dq.popleft()
            if p == tgt:
                break
            for d in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                q = (p[0] + d[0], p[1] + d[1])
                if not (0 <= q[0] < h and 0 <= q[1] < W) or q in prev:
                    continue
                if occ[q]:
                    continue
                g2 = snap.copy(); g2[q] = fgc
                if g2.tobytes() in seen:
                    continue
                prev[q] = p; dq.append(q)
        if tgt in prev:
            path = [tgt]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            path.reverse()
            for i in range(1, len(path)):
                g2 = snap.copy(); g2[path[i]] = fgc; seen.add(g2.tobytes())
                d = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
                ops.append(MOVE[d])
                # first step grabs the mark; the rest continue the same grab
                sels.append(sel_of([(sr, sc)]) if i == 1 else sel_of([]))
            # bgc == 0: the vacated footprint already reads background, no repair needed
        else:
            # no free path (practically never): place the mark in its slot, clear its origin
            ops.append(fgc); sels.append(sel_of([tgt]))
            ops.append(bgc); sels.append(sel_of([(sr, sc)]))
            g2 = grid.copy(); g2[tgt] = fgc; seen.add(g2.tobytes())
            seen.add(snap.tobytes()); g2[sr, sc] = 0; seen.add(g2.tobytes())
        grid = snap
        grid[tgt] = fgc

    # The gathered top row of nc marks is the answer: crop to it (rectangle).
    ops.append(33); sels.append([0, 0, 0, nc - 1])
    ops.append(34); sels.append([0, 0, 0, nc - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d631b094"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d631b094"
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
                                f"for task d631b094"
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
                    f"Failed to build a complete episode for task d631b094 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d631b094-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
