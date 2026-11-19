"""
ARC Task: 2204b7a8 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ccol = random.choice(rem)
    rem2 = [c for c in rem if c != ccol]
    c1 = random.choice(rem2)
    c2 = random.choice([c for c in rem2 if c != c1])
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(examples)
    else:
        examples = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccol": ccol, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccol, c1, c2, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    for _ in range(1000):
        if transpose:
            h = _unifint(diff_lb, diff_ub, (4, max(4, max_w)))
            w = _unifint(diff_lb, diff_ub, (4, max(4, max_h)))
        else:
            h = _unifint(diff_lb, diff_ub, (4, max(4, max_h)))
            w = _unifint(diff_lb, diff_ub, (4, max(4, max_w)))
        inds = [(i, j + 1) for i in range(h) for j in range(w - 2)]
        hi_b = max(1, (h * (w - 2)) // 2 - 1)
        nc = _unifint(diff_lb, diff_ub, (1, hi_b))
        locs = random.sample(inds, min(nc, len(inds)))
        if w % 2 == 1:
            locs = [(i, j) for (i, j) in locs if j != w // 2]
        if not locs:
            continue
        gi = np.full((h, w), bgc, dtype=int)
        gi[:, 0] = c1
        gi[:, w - 1] = c2
        for (i, j) in locs:
            gi[i, j] = ccol
        go = gi.copy()
        for (i, j) in locs:
            go[i, j] = c1 if j < w // 2 else c2
        if len(set(gi.flatten().tolist())) != 4:
            continue
        if transpose:
            gi = gi.T.copy()
            go = go.T.copy()
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # Lines are the first/last COLUMN (vertical) or first/last ROW (horizontal).
    vertical = (len(set(I[:, 0].tolist())) == 1) and (len(set(I[:, -1].tolist())) == 1)

    # Work in "line-oriented" coordinates: lines at positions 0 and L-1 along axis `a`.
    if vertical:
        L = w
        col_a, col_b = int(I[0, 0]), int(I[0, -1])
        interior = I[:, 1:w - 1]
        cell = lambda pos, k: (k, pos)      # pos = column, k = row
        span = h
    else:
        L = h
        col_a, col_b = int(I[0, 0]), int(I[-1, 0])
        interior = I[1:h - 1, :]
        cell = lambda pos, k: (pos, k)      # pos = row, k = column
        span = w

    cnt = Counter(interior.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    markers = [c for c in cnt if c != bgc]
    if markers:
        marker = markers[0]

        def marker_cells_at(pos):
            out = []
            for k in range(span):
                r, c = cell(pos, k)
                if int(I[r, c]) == marker:
                    out.append((r, c))
            return out

        # Proximity reading: each marker takes the colour of the nearest line.
        # Walk outward from line A, strip by strip (distance 1, 2, ...), then from line B.
        for d in range(1, L - 1):
            pos = d
            if d < L - 1 - pos:            # strictly closer to line A
                cells = marker_cells_at(pos)
                if cells:
                    ops.append(col_a); sels.append(sel_of(sorted(cells)))
        for d in range(1, L - 1):
            pos = L - 1 - d
            if d < pos:                    # strictly closer to line B
                cells = marker_cells_at(pos)
                if cells:
                    ops.append(col_b); sels.append(sel_of(sorted(cells)))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2204b7a8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2204b7a8"
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
                                f"for task 2204b7a8"
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
                    f"Failed to build a complete episode for task 2204b7a8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2204b7a8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
