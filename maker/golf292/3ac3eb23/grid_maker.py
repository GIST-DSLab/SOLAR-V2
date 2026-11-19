"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= 4:
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - 4)]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))
    if rot is None:
        rot = random.choice(ROTS)
    if rot in (1, 3):
        h = unifint(diff_lb, diff_ub, (3, max_w))
        w = unifint(diff_lb, diff_ub, (3, max_h))
    else:
        h = unifint(diff_lb, diff_ub, (3, max_h))
        w = unifint(diff_lb, diff_ub, (3, max_w))
    remcols = [c for c in range(10) if c != bgc]
    nlocs = unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = list(range(1, w - 1))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for _ in range(nlocs):
        if not locopts:
            break
        locj = random.choice(locopts)
        locopts = [x for x in locopts if not (locj - 2 <= x <= locj + 2)]
        col = random.choice(remcols)
        gi[0, locj] = col
        go[0::2, locj] = col
        go[1::2, locj - 1] = col
        go[1::2, locj + 1] = col
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    # border carrying the seeds -> u inward, v along border, D depth
    if all(r == 0 for r, _ in seeds):
        u, v, D = (1, 0), (0, 1), hi
    elif all(r == hi - 1 for r, _ in seeds):
        u, v, D = (-1, 0), (0, 1), hi
    elif all(c == 0 for _, c in seeds):
        u, v, D = (0, 1), (1, 0), wi
    else:
        u, v, D = (0, -1), (1, 0), wi

    def cell(s, d, t):
        return (s[0] + u[0] * d + v[0] * t, s[1] + u[1] * d + v[1] * t)

    def band(d0, n):
        # full-width rectangle covering depths d0 .. d0+n-1 as [r, c, h-1, w-1]
        if u[0] == 1:
            return [d0, 0, n - 1, wi - 1]
        if u[0] == -1:
            return [hi - d0 - n, 0, n - 1, wi - 1]
        if u[1] == 1:
            return [0, d0, hi - 1, n - 1]
        return [0, wi - d0 - n, hi - 1, n - 1]

    ops, sels = [], []
    # 1. each seed grows its two wings one step in (one Color op per seed colour)
    cols = []
    for s in seeds:
        col = int(I[s])
        if col not in cols:
            cols.append(col)
    for col in cols:
        wings = [cell(s, 1, t) for s in seeds if int(I[s]) == col for t in (-1, 1)]
        ops.append(col); sels.append(sel_of(wings))

    # 2. the border band of depth 0..1 is one period of the pattern:
    #    copy the finished band and paste it further in, doubling each time
    if any(int(I[s]) != 0 for s in seeds):
        held = [None]

        def copy(n):                                          # copy depths 0..n-1 once
            if held[0] != n:
                ops.append(29); sels.append(band(0, n))
                held[0] = n

        b = 2                                                 # depths 0..b-1 are done
        while b < D:
            r = D - b
            if r >= b:
                copy(b)                                       # double the finished band
                ops.append(30); sels.append(band(b, b)[:2] + [0, 0])
                b *= 2
            else:
                n = held[0]
                fwd = u[0] == 1 or u[1] == 1
                if n is not None and n >= r and (D - n) % 2 == 0:
                    # held band laid flush against the far border (overlap is identical)
                    ops.append(30); sels.append(band(D - n, n)[:2] + [0, 0])
                elif n is not None and n >= r and fwd:
                    # paste inward; the part past the far border is clipped off
                    ops.append(30); sels.append(band(b, n)[:2] + [0, 0])
                else:
                    copy(r)                                   # just the missing depths
                    ops.append(30); sels.append(band(b, r)[:2] + [0, 0])
                b = D

    # 3. colour 0 is invisible to Copy/Paste: extend 0-coloured fans directly
    zs = [s for s in seeds if int(I[s]) == 0]
    if zs and D > 2:
        rest = [cell(s, d, t) for s in zs for d in range(2, D)
                for t in ((0,) if d % 2 == 0 else (-1, 1))]
        ops.append(0); sels.append(sel_of(rest))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
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
                                f"for task 3ac3eb23"
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
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
