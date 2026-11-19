"""
ARC Task: 1e0a9b12 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background is a fixed role; the rule (column gravity) ignores fg colours.
    return {"bgc": random.choice(list(range(10)))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None:
        bgc = random.choice(list(range(10)))
    remcols = [c for c in range(10) if c != bgc]
    while True:
        h = max(3, min(max_h, unifint(diff_lb, diff_ub, (3, max_h))))
        w = max(3, min(max_w, unifint(diff_lb, diff_ub, (3, max_w))))
        nc = max(1, min(w, unifint(diff_lb, diff_ub, (1, w))))
        gi = [[bgc] * w for _ in range(h)]
        scols = [random.choice(remcols) for _ in range(nc)]
        slocs = random.sample(range(w), nc)
        for col, l in zip(scols, slocs):
            nc2 = random.randint(1, h - 1)
            for r in random.sample(range(h), nc2):
                gi[r][l] = col
        go = [[bgc] * w for _ in range(h)]
        for c in range(w):
            fg = [gi[r][c] for r in range(h) if gi[r][c] != bgc]
            for i, v in enumerate(fg):
                go[h - len(fg) + i][c] = v
        cnt = Counter(v for row in gi for v in row)
        others = [cnt[k] for k in cnt if k != bgc]
        if not others or cnt[bgc] > max(others):
            break
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (from the column-sort): a column is one unit; its non-background cells
    # keep their order and the background rises. Done as gap closing from the floor:
    # find the lowest background gap that has colour above it, and let EVERYTHING
    # above that gap fall as one block by the gap's height. Repeat upward.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # the generator guarantees the background is the strict majority colour
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    G = I.copy()                 # the working grid as ARCLE holds it
    ops, sels = [], []

    for c in range(w):
        L = [int(I[r, c]) for r in range(h)]   # logical column contents
        while True:
            fg = [r for r in range(h) if L[r] != bgc]
            if not fg:
                break
            # lowest background cell that still has colour above it
            gap_bot = None
            for r in range(h - 1, -1, -1):
                if L[r] == bgc and any(L[q] != bgc for q in range(r)):
                    gap_bot = r
                    break
            if gap_bot is None:
                break
            gap_top = gap_bot
            while gap_top - 1 >= 0 and L[gap_top - 1] == bgc:
                gap_top -= 1
            g = gap_bot - gap_top + 1
            block = [(r, c) for r in range(gap_top) if L[r] != bgc]  # everything above the gap
            carried = [(r, cc) for (r, cc) in block if L[r] != 0]
            if carried:
                ops.append(21); sels.append(sel_of(block))   # grab the block once
                for _ in range(g - 1):
                    ops.append(21); sels.append(sel_of([]))  # keep it falling
                snap = G.copy()
                for (r, cc) in carried:
                    snap[r, cc] = 0
                for (r, cc) in carried:
                    snap[r + g, cc] = L[r]
                G = snap
            newL = list(L)
            for (r, cc) in block:
                newL[r] = bgc
            for (r, cc) in block:
                newL[r + g] = L[r]
            # colour-0 cells are not carried by Move: paint them where they landed
            zero_dst = [(r + g, cc) for (r, cc) in block if L[r] == 0 and G[r + g, cc] != 0]
            if zero_dst:
                ops.append(0); sels.append(sel_of(zero_dst))
                for p in zero_dst:
                    G[p] = 0
            # cells the block left behind are now background: repair those not holding it
            left = [(r, c) for r in range(gap_bot + 1) if newL[r] == bgc and G[r, c] != bgc]
            if left:
                ops.append(bgc); sels.append(sel_of(left))
                for p in left:
                    G[p] = bgc
            L = newL

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # Submit: whole grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 1e0a9b12"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1e0a9b12"
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
                                f"for task 1e0a9b12"
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
                    f"Failed to build a complete episode for task 1e0a9b12 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1e0a9b12-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
