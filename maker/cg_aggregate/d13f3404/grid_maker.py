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
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if bgc is None:
        bgc = random.choice(list(range(10)))
    hmax = max(3, min(15, max_h // 2))
    wmax = max(3, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (3, hmax))
    w = unifint(diff_lb, diff_ub, (3, wmax))
    vopts = {(ii, 0) for ii in range(h)}
    hopts = {(0, jj) for jj in range(1, w)}
    opts = tuple(vopts | hopts)
    num = unifint(diff_lb, diff_ub, (1, len(opts)))
    locs = random.sample(opts, num)
    remcols = [c for c in range(10) if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * (2 * w) for _ in range(2 * h)]
    for (r0, c0) in locs:
        ln = []
        r, c = r0, c0
        while r < h and c < w:
            ln.append((r, c))
            r += 1
            c += 1
        lr, lc = random.choice(ln)
        col = random.choice(remcols)
        gi[lr][lc] = col
        r, c = lr, lc
        while r < 2 * h and c < 2 * w:
            go[r][c] = col
            r += 1
            c += 1
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Reading (row-propagation / stamped pattern): the output is the input's dot
    pattern stamped repeatedly, each stamp displaced one more step down-right
    (d, d), d = 1, 2, ..., on a doubled background canvas.  Every stamp is one
    unit: for each step d, the dots of each colour shifted by (d, d) are painted
    with a single Color op.  O is never inspected.
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
    if examples:
        for pair in examples:
            try:
                gi = np.asarray(pair[0], dtype=int)
            except Exception:
                continue
            if gi.ndim == 2 and gi.size:
                grids.append(gi)
    grids.append(I)
    votes, totals = Counter(), Counter()
    for g in grids:
        c = Counter(g.flatten().tolist())
        votes[c.most_common(1)[0][0]] += 1
        totals.update(c)
    top = max(votes.values())
    tied = [col for col, v in votes.items() if v == top]
    bgc = max(tied, key=lambda col: totals[col])

    ops, sels = [], []

    # 1) grow canvas to 2h x 2w (whole rectangle intended)
    ops.append(33)
    sels.append([0, 0, H - 1, W - 1])

    # 2) lay background over the newly added (zero) area; skip when bgc == 0
    if bgc != 0:
        ops.append(int(bgc)); sels.append([h, 0, h - 1, W - 1])   # bottom band, exact rect
        ops.append(int(bgc)); sels.append([0, w, h - 1, w - 1])   # top-right block, exact rect

    # 3) stamp the whole dot pattern at (d, d), one Color op per colour per stamp
    dots = [(r, c, int(I[r, c])) for r in range(h) for c in range(w)
            if int(I[r, c]) != bgc]
    colours = sorted({col for _, _, col in dots})
    d = 1
    while d < max(H, W):
        any_in = False
        for col in colours:
            cells = [(r + d, c + d) for (r, c, cc) in dots
                     if cc == col and r + d < H and c + d < W]
            if cells:
                any_in = True
                ops.append(col)
                sels.append(sel_of(cells))
        if not any_in:
            break
        d += 1

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
