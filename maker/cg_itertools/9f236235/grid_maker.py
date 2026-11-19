"""
ARC Task: 9f236235 (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(range(10))
    frontcol = random.choice([c for c in range(10) if c != bgc])
    return {"bgc": bgc, "frontcol": frontcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, frontcol=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))
    if bgc is None:
        bgc = random.choice(range(10))
    if frontcol is None:
        frontcol = random.choice([c for c in range(10) if c != bgc])
    remcols = [c for c in range(10) if c != frontcol]
    numh = unifint(diff_lb, diff_ub, (2, min(14, (max_h + 1) // 2)))
    numw = unifint(diff_lb, diff_ub, (2, min(14, (max_w + 1) // 2)))
    h = unifint(diff_lb, diff_ub, (1, (max_h + 1) // numh - 1))
    w = unifint(diff_lb, diff_ub, (1, (max_w + 1) // numw - 1))
    numcells = unifint(diff_lb, diff_ub, (1, numh * numw))
    cands = [(a, b) for a in range(numh) for b in range(numw)]
    locs = set(random.sample(cands, numcells))
    gi = [[frontcol] * (w * numw + numw - 1) for _ in range(h * numh + numh - 1)]
    go = [[bgc] * numw for _ in range(numh)]
    for (a, b) in cands:
        col = random.choice(remcols) if (a, b) in locs else bgc
        for i in range(h):
            for j in range(w):
                gi[a * (h + 1) + i][b * (w + 1) + j] = col
        go[a][b] = col
    go = [row[::-1] for row in go]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Cell size: the first fully uniform row/column is the first frontier line
    # (cell colours are never the frontier colour, so only frontier lines are uniform).
    h = next(r for r in range(hi) if len(set(I[r].tolist())) == 1)
    w = next(c for c in range(wi) if len(set(I[:, c].tolist())) == 1)
    numh = (hi + 1) // (h + 1)
    numw = (wi + 1) // (w + 1)
    ops, sels = [], []

    # 1. Mirror the whole mosaic left<->right FIRST (whole grid = exact rectangle).
    #    Frontier lines are symmetric, so cell block b lands exactly on block numw-1-b.
    F = I[:, ::-1].copy()
    if not np.array_equal(F, I):
        ops.append(26); sels.append([0, 0, hi - 1, wi - 1])

    # 2. Shrink the canvas to one pixel per cell (keeps the top-left numh x numw corner).
    ops.append(33); sels.append([0, 0, numh - 1, numw - 1])
    cur = np.zeros((numh, numw), dtype=int)
    sub = F[:numh, :numw]
    cur[sub != 0] = sub[sub != 0]

    # 3. Collapse each (mirrored) cell block to its pixel: one Color op per colour
    #    class of the blocks, most common block colour first.
    blk = F[::h + 1, ::w + 1]
    order = [c for c, _ in Counter(blk.flatten().tolist()).most_common()]
    for col in order:
        cells = [(a, b) for a in range(numh) for b in range(numw)
                 if blk[a, b] == col and cur[a, b] != col]
        if cells:
            ops.append(int(col)); sels.append(sel_of(cells))
            for a, b in cells:
                cur[a, b] = col

    ops.append(34); sels.append([0, 0, numh - 1, numw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9f236235"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9f236235"
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
                                f"for task 9f236235"
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
                    f"Failed to build a complete episode for task 9f236235 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9f236235-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
