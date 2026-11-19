"""
ARC Task: a2fd1cf0 (RE-ARC) — LLM-generated grid_maker
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
    # 2, 3 and 8 are fixed by the task; only the background is random.
    cols = [c for c in range(10) if c not in (2, 3, 8)]
    return {"bgc": random.choice(cols)}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _rot90(g):
    return [list(r) for r in zip(*g[::-1])]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice([c for c in range(10) if c not in (2, 3, 8)])
    h = _unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = _unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    gi_ = _unifint(diff_lb, diff_ub, (1, h - 1))
    gj_ = _unifint(diff_lb, diff_ub, (1, w - 1))
    g = [[bgc] * w for _ in range(h)]
    # rot180 of a 3 at (gi_, gj_)
    ii, jj = h - 1 - gi_, w - 1 - gj_
    g[ii][jj] = 3
    ri = _unifint(diff_lb, diff_ub, (ii + 1, h - 1))
    rj = _unifint(diff_lb, diff_ub, (jj + 1, w - 1))
    g[ri][rj] = 2
    k = random.choice([0, 1, 2, 3])
    for _ in range(k):
        g = _rot90(g)
    gi = [row[:] for row in g]
    H, W = len(gi), len(gi[0])
    a = b = c = d = None
    for r in range(H):
        for cc in range(W):
            if gi[r][cc] == 2:
                a, b = r, cc
            elif gi[r][cc] == 3:
                c, d = r, cc
    go = [row[:] for row in gi]
    for cc in range(min(b, d), max(b, d) + 1):
        go[a][cc] = 8
    for r in range(min(a, c), max(a, c) + 1):
        go[r][d] = 8
    go[a][b] = 2
    go[c][d] = 3
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Rule: an 8-coloured L-shaped path runs from the 2 along the 2's row to
    # the 3's column, then turns along that column up to the 3. Background
    # cells only; 2 and 3 stay as they are.
    # The golf entry treats the task the same way: one row segment and one
    # column segment, both bounded by the two markers. So the decomposition
    # matches the maker's. Each arm is one straight line segment named from
    # the marker positions in I. The corner belongs to the horizontal arm.
    # The vertical arm then covers only the cells strictly between the corner
    # and the 3, so no cell is painted twice.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    rs2, cs2 = np.where(I == 2)
    rs3, cs3 = np.where(I == 3)
    a, b = int(rs2[0]), int(cs2[0])
    c, d = int(rs3[0]), int(cs3[0])

    ops, sels = [], []

    # Horizontal arm: on the 2's row, from beside the 2 up to the corner (a, d).
    step = 1 if d > b else -1
    h_cells = [(a, cc) for cc in range(b + step, d + step, step)]
    ops.append(8); sels.append(sel_of(h_cells))

    # Vertical arm: on the 3's column, from just past the corner up to just before the 3.
    if abs(c - a) > 1:
        step = 1 if c > a else -1
        v_cells = [(r, d) for r in range(a + step, c, step)]
        ops.append(8); sels.append(sel_of(v_cells))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a2fd1cf0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a2fd1cf0"
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
                                f"for task a2fd1cf0"
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
                    f"Failed to build a complete episode for task a2fd1cf0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a2fd1cf0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
