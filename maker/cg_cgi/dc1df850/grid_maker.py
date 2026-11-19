"""
ARC Task: dc1df850 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # 1 (halo) and 2 (seed) are hardcoded; only the background is sampled.
    # Other object colours are scattered at random and play no role in the rule.
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    v = random.uniform(lo, hi)
    return max(a, min(b, int(round(v))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kwargs) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    remcols = [c for c in cols if c != bgc]
    h = _unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = _unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    grid = [[bgc] * w for _ in range(h)]
    nc = _unifint(diff_lb, diff_ub, (0, (h * w) // 2 - 1))
    nreddev = _unifint(diff_lb, diff_ub, (0, nc // 2))
    nred = random.choice((nreddev, nc - nreddev))
    nred = min(max(0, nred), nc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    occ = random.sample(inds, nc)
    reds = random.sample(occ, nred)
    redset = set(reds)
    for (i, j) in occ:
        if (i, j) in redset:
            grid[i][j] = 2
        else:
            grid[i][j] = random.choice(remcols)
    gi = [row[:] for row in grid]
    go = [row[:] for row in grid]
    for (i, j) in reds:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                a, b = i + di, j + dj
                if 0 <= a < h and 0 <= b < w and gi[a][b] == bgc:
                    go[a][b] = 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Background: the generator covers fewer than half the cells, so bgc is
    # strictly the most common colour of I (and is never 1 or 2).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 2]
    cur = I.copy()

    def sweep(offsets):
        # Paint 1 onto every cell at one of these offsets from a seed,
        # but only where the cell is still background.
        cells = set()
        for (r, c) in seeds:
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < hi and 0 <= cc < wi and cur[rr, cc] == bgc:
                    cells.add((rr, cc))
        if cells:
            cells = sorted(cells)
            ops.append(1)
            sels.append(sel_of(cells))
            for (rr, cc) in cells:
                cur[rr, cc] = 1

    # The golf solution's reading: the halo is built by directional one-step
    # sweeps. First a horizontal smear (right, then left) gives each seed a
    # 1x3 strip. Then a vertical smear of that strip (down, then up) fills
    # the 3x3 box. The strip offsets come from each seed itself, so a blocked
    # arm cell never stops its diagonal from being filled.
    sweep([(0, 1)])                        # right arms
    sweep([(0, -1)])                       # left arms
    sweep([(1, -1), (1, 0), (1, 1)])       # strip pushed down
    sweep([(-1, -1), (-1, 0), (-1, 1)])    # strip pushed up

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task dc1df850"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task dc1df850"
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
                                f"for task dc1df850"
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
                    f"Failed to build a complete episode for task dc1df850 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"dc1df850-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
