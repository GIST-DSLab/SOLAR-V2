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
    # bgc is hardcoded 0 in the generator; only the mark colour is sampled.
    fgc = random.choice(range(1, 10))
    return {"fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, fgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return max(a, min(b, round(random.uniform(lo, hi_))))
    bgc = 0
    if fgc is None:
        fgc = random.choice(range(1, 10))
    h = unifint(diff_lb, diff_ub, (2, max(2, max_h)))
    w = unifint(diff_lb, diff_ub, (2, max(2, max_w)))
    nc = unifint(diff_lb, diff_ub, (1, max(1, min(30, max_w, (h * w) // 2 - 1))))
    gi = [[bgc] * w for _ in range(h)]
    cands = [(i, j) for i in range(h) for j in range(w)]
    for (i, j) in random.sample(cands, nc):
        gi[i][j] = fgc
    go = [[fgc] * nc]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (as in the golf entry): flatten the grid in reading order and keep only
    # the marks -> every mark is GATHERED, in reading order, into one row at (0,0..nc-1),
    # then the canvas is cropped to that row. O is never read.
    I = np.asarray(I, dtype=int)
    for extra in range(0, 4):
        try:
            return _gather(I, extra)
        except ValueError:
            continue
    raise ValueError("no clean gathering found")


def _gather(I, extra):
    bgc = 0  # generator paints the canvas with 0
    hi, wi = I.shape
    marks = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    nc = len(marks)
    ops, sels = [], []
    g = I.copy()
    # Make room: the row of nc marks needs nc columns; when the grid is too cramped
    # to slide marks past each other, add empty working rows below (cropped away later).
    nw = max(wi, nc)
    if nw > wi or extra:
        ops.append(33); sels.append([0, 0, hi + extra - 1, nw - 1])  # full rectangle canvas
        ng = np.zeros((hi + extra, nw), dtype=int); ng[:hi, :wi] = g; g = ng
    H, W = g.shape
    STEP = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}

    def seg(a, b):
        (r0, c0), (r1, c1) = a, b
        out = []
        while (r0, c0) != (r1, c1):
            if r0 != r1:
                r0 += 1 if r1 > r0 else -1
            else:
                c0 += 1 if c1 > c0 else -1
            out.append((r0, c0))
        return out

    def simulate(src, path):
        # one grab: snapshot with the grabbed cell zeroed, object re-pasted each step
        snap = g.copy(); col = snap[src]; snap[src] = 0
        prev = g.copy()
        for q in path:
            ng = snap.copy(); ng[q] = col
            if np.array_equal(ng, prev):
                return None  # a step with no visible effect -> reject this path
            prev = ng
        return prev

    def emit(src, path):
        prev = src
        for i, q in enumerate(path):
            ops.append(STEP[(q[0] - prev[0], q[1] - prev[1])])
            sels.append(sel_of([src]) if i == 0 else sel_of([]))  # grab once, then empties
            prev = q

    for k, (r, c) in enumerate(marks):
        tgt = (0, k)
        if (r, c) == tgt:
            continue
        cands = [[(r, k)], [(0, c)]]
        cands += [[(rr, c), (rr, k)] for rr in range(H) if rr not in (r, 0)]
        cands += [[(r, cc), (0, cc)] for cc in range(W) if cc not in (c, k)]
        done = False
        for wps in cands:
            path, p = [], (r, c)
            for wp in wps + [tgt]:
                path += seg(p, wp); p = wp
            if len(set(path)) != len(path) or (r, c) in path:
                continue
            res = simulate((r, c), path)
            if res is None:
                continue
            emit((r, c), path)
            g = res  # bgc == 0: vacated footprint already reads background
            done = True
            break
        if not done:
            # fallback: shortest path on which every unit step visibly changes the grid
            snap = g.copy(); snap[r, c] = 0
            par = {(r, c): None}; dq = deque([(r, c)])
            while dq:
                p = dq.popleft()
                if p == tgt:
                    break
                for d in STEP:
                    q = (p[0] + d[0], p[1] + d[1])
                    if not (0 <= q[0] < H and 0 <= q[1] < W) or q in par:
                        continue
                    if p != (r, c) and snap[p] != bgc and snap[q] != bgc:
                        continue
                    par[q] = p; dq.append(q)
            if tgt not in par:
                raise ValueError("no clean path")
            path, p = [], tgt
            while p != (r, c):
                path.append(p); p = par[p]
            path.reverse()
            res = simulate((r, c), path)
            if res is None:
                raise ValueError("no clean path")
            emit((r, c), path)
            g = res
    row = [0, 0, 0, nc - 1]
    ops.append(33); sels.append(row)  # crop to the gathered row (full rectangle)
    ops.append(34); sels.append(row)
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
