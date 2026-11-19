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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    fgc = random.choice(range(1, 10))
    return {"fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, fgc=None, **kwargs) -> dict:
    from random import sample as rsample

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if fgc is None:
        fgc = random.choice(range(1, 10))
    bgc = 0
    h = unifint(diff_lb, diff_ub, (2, max(2, max_h)))
    w = unifint(diff_lb, diff_ub, (2, max(2, max_w)))
    nc = unifint(diff_lb, diff_ub, (1, max(1, min(30, max_w, (h * w) // 2 - 1))))
    cands = [(i, j) for i in range(h) for j in range(w)]
    cels = rsample(cands, nc)
    gi = [[bgc] * w for _ in range(h)]
    for i, j in cels:
        gi[i][j] = fgc
    go = [[fgc] * nc]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = 0  # generator paints the canvas 0

    # Reading: the marks are GATHERED, in reading order, into one row.
    # The k-th mark (row-major) slides to cell (0, k); then the canvas is
    # cut down to that row.
    marks = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    nc = len(marks)
    fgc = Counter(int(I[r, c]) for r, c in marks).most_common(1)[0][0]

    def fallback():
        row = [0, 0, 0, nc - 1]
        return [33, fgc, 34], [row, row, row]

    W = max(wi, nc)
    ops, sels = [], []
    grid = np.zeros((hi, W), dtype=int)
    grid[:, :wi] = I
    seen = {grid.tobytes()}
    if nc > wi:
        # widen the canvas so the gathered row has room for every mark
        ops.append(33); sels.append([0, 0, hi - 1, W - 1])

    def legs_to_steps(legs):
        steps = []
        for kind, d in legs:
            if kind == "v":
                steps += [(20 if d < 0 else 21, (-1 if d < 0 else 1, 0))] * abs(d)
            else:
                steps += [(23 if d < 0 else 22, (0, -1 if d < 0 else 1))] * abs(d)
        return steps

    def candidate_routes(src, dst):
        # Shortest L-shaped paths first; then three-leg detours through
        # another row / column so the glide never passes mark-onto-mark
        # (which would leave the visible grid unchanged).
        (r, c), (tr, tc) = src, dst
        yield [("v", tr - r), ("h", tc - c)]
        yield [("h", tc - c), ("v", tr - r)]
        for y in sorted(range(hi), key=lambda y: abs(y - r) + abs(y - tr)):
            yield [("v", y - r), ("h", tc - c), ("v", tr - y)]
        for x in sorted(range(W), key=lambda x: abs(x - c) + abs(x - tc)):
            yield [("h", x - c), ("v", tr - r), ("h", tc - x)]

    def simulate(g, src, steps):
        snap = g.copy(); snap[src] = 0
        col = g[src]
        pos = src
        cur = g.copy()
        local = set()
        for _, (dr, dc) in steps:
            pos = (pos[0] + dr, pos[1] + dc)
            new = snap.copy(); new[pos] = col
            key = new.tobytes()
            if np.array_equal(new, cur) or key in seen or key in local:
                return None  # no visible change, or a state already held
            local.add(key)
            cur = new
        seen.update(local)
        return cur

    for k, src in enumerate(marks):
        dst = (0, k)
        if src == dst:
            continue
        done = False
        for legs in candidate_routes(src, dst):
            steps = legs_to_steps([lg for lg in legs if lg[1] != 0])
            res = simulate(grid, src, steps)
            if res is None:
                continue
            for i, (op, _) in enumerate(steps):
                ops.append(op)
                # first step grabs the mark; later steps continue the same grab
                sels.append(sel_of([src]) if i == 0 else sel_of([]))
            grid = res
            done = True
            break
        if not done:
            return fallback()

    row = [0, 0, 0, nc - 1]
    ops.append(33); sels.append(row)  # keep only the gathered row
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
