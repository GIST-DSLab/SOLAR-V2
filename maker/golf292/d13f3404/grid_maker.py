"""
ARC Task: d13f3404 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; dot colours don't matter to the rule.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    if bgc is None:
        bgc = random.choice(range(10))
    Hmax = max(3, min(15, max_h // 2))
    Wmax = max(3, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (3, Hmax))
    w = unifint(diff_lb, diff_ub, (3, Wmax))
    opts = [(i, 0) for i in range(h)] + [(0, j) for j in range(1, w)]
    num = unifint(diff_lb, diff_ub, (1, len(opts)))
    locs = random.sample(opts, num)
    remcols = [c for c in range(10) if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * (2 * w) for _ in range(2 * h)]
    for (a, b) in locs:
        ln = [(a + k, b + k) for k in range(min(h - a, w - b))]
        r, c = random.choice(ln)
        col = random.choice(remcols)
        gi[r][c] = col
        k = 0
        while r + k < 2 * h and c + k < 2 * w:
            go[r + k][c + k] = col
            k += 1
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Rule (read from I only): canvas doubles to (2h, 2w); every non-background dot
    shoots a ray of its own colour down-right to the edge.  Each diagonal holds at
    most one dot, so the output is the union of copies of I shifted by (k, k).
    Strokes: turn the background to 0 (so copies are transparent), double the canvas,
    then copy the grid and paste it shifted by 1, 2, 4, ... (each paste doubles every
    ray), then turn all remaining 0 cells back to the background, and finally draw any
    colour-0 dot's ray (0 can't travel through a transparent paste).
    """
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [(int(r), int(c)) for r, c in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    H, W = 2 * h, 2 * w

    # background colour: majority vote over demonstration inputs + I
    grids = []
    for pair in (examples or []):
        try:
            g = np.asarray(pair[0], dtype=int)
            if g.ndim == 2 and g.size:
                grids.append(g)
        except Exception:
            pass
    grids.append(I)
    votes, totals = Counter(), Counter()
    for g in grids:
        cnt = Counter(g.flatten().tolist())
        votes[cnt.most_common(1)[0][0]] += 1
        totals.update(cnt)
    top = max(votes.values())
    bgc = max([c for c, v in votes.items() if v == top], key=lambda c: totals[c])

    dots = [(r, c, int(I[r, c])) for r in range(h) for c in range(w) if I[r, c] != bgc]
    zero_dots = [(r, c) for r, c, col in dots if col == 0]
    ops, sels = [], []

    # 1) all background cells -> 0, so copy/paste is transparent around the dots
    if bgc != 0:
        ops.append(0)
        sels.append(sel_of([(r, c) for r in range(h) for c in range(w) if I[r, c] == bgc]))

    # 2) double the canvas (whole rectangle intended)
    ops.append(33)
    sels.append([0, 0, H - 1, W - 1])

    # 3) grow every ray by doubling: copy the top-left rectangle, paste it at (s, s)
    reach = max([min(H - 1 - r, W - 1 - c) for r, c, col in dots if col != 0] or [0])
    s = 1
    while s <= reach:
        ops.append(29); sels.append([0, 0, H - s - 1, W - s - 1])  # full rectangle
        ops.append(30); sels.append([s, s, 0, 0])
        s *= 2

    if bgc != 0:
        # 4) every cell that is 0 right now goes back to the background colour
        g = np.zeros((H, W), dtype=int)
        for r, c, col in dots:
            if col == 0:
                continue
            k = 0
            while r + k < H and c + k < W:
                g[r + k, c + k] = col
                k += 1
        empties = [(r, c) for r in range(H) for c in range(W) if g[r, c] == 0]
        if empties:
            ops.append(int(bgc)); sels.append(sel_of(empties))
        # 5) a colour-0 dot draws its own ray (0 is invisible to Copy/Paste)
        for r, c in zero_dots:
            ray = []
            k = 0
            while r + k < H and c + k < W:
                ray.append((r + k, c + k))
                k += 1
            ops.append(0); sels.append(sel_of(ray))

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d13f3404"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d13f3404"
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
                                f"for task d13f3404"
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
                    f"Failed to build a complete episode for task d13f3404 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d13f3404-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
