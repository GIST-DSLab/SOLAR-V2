"""
ARC Task: 05269061 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [
    {"mode": "diag", "mirror": False},
    {"mode": "diag", "mirror": True},
    {"mode": "horiz", "mirror": False},
]


def sample_colors(num_examples=None) -> dict:
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, mode=None, mirror=None) -> dict:
    if mode is None:
        mode = random.choice(["diag", "horiz"])
    if mirror is None:
        mirror = random.choice([True, False])
    dim_bounds = (2, min(max_h, max_w))
    colopts = interval(1, 10, 1)
    d = unifint(diff_lb, diff_ub, dim_bounds)
    go = canvas(0, (d, d))
    gi = canvas(0, (d, d))
    if mode == "diag":
        period_bounds = (2, min(2 * d - 2, 9))
        num = unifint(diff_lb, diff_ub, period_bounds)
        cols = tuple(choice(colopts) for k in range(num))
        keeps = [choice(interval(j, 2 * d - 1, num)) for j in range(num)]
        for k, col in enumerate((cols * 30)[:2 * d - 1]):
            lin = shoot(toivec(k), UP_RIGHT)
            go = fill(go, col, lin)
            if keeps[k % num] == k:
                gi = fill(gi, col, lin)
    else:
        period_bounds = (2, min(d, 9))
        num = unifint(diff_lb, diff_ub, period_bounds)
        cols = tuple(choice(colopts) for k in range(num))
        keeps = [choice(interval(j, d, num)) for j in range(num)]
        for k, col in enumerate((cols * 30)[:d]):
            lin = hfrontier(toivec(k))
            go = fill(go, col, lin)
            if keeps[k % num] == k:
                gi = fill(gi, col, lin)
    if mirror:
        gi = vmirror(gi)
        go = vmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O):
    """
    Rule (read from I alone): the non-zero cells of I are p monochrome seed lines of
    one family -- full rows, anti-diagonals (r+c) or main diagonals (r-c).  There is
    exactly one seed per residue class mod p, so p = number of seed lines, and every
    line of the family takes the colour of the seed in its class.
    Route: paint the p classes inside the top-left period tile (one Color per class:
    the family's lines crossing the tile), CopyO the tile and Paste it along the top
    strip, then CopyO the strip and Paste it down the grid.
    """
    I = np.asarray(I, dtype=int)
    d, w = I.shape
    nz = [(r, c) for r in range(d) for c in range(w) if I[r, c] != 0]

    def fam_ok(key):
        groups = {}
        for r, c in nz:
            groups.setdefault(key(r, c), set()).add((r, c))
        for k, g in groups.items():
            full = {(r, c) for r in range(d) for c in range(w) if key(r, c) == k}
            if g != full or len({int(I[r, c]) for r, c in g}) != 1:
                return None
        return groups

    fams = [lambda r, c: r, lambda r, c: r + c, lambda r, c: r - c]
    key, groups = None, None
    for f in fams:
        g = fam_ok(f)
        if g is not None:
            key, groups = f, g
            break
    horiz = key is fams[0]
    p = len(groups)
    colour = {}
    for k, g in groups.items():
        r, c = next(iter(g))
        colour[k % p] = int(I[r, c])

    th = min(p, d)
    tw = w if horiz else min(p, w)
    G = I.copy()
    ops, sels = [], []

    # 1. paint each class inside the period tile
    for j in sorted(colour):
        cells = [(r, c) for r in range(th) for c in range(tw) if key(r, c) % p == j]
        if not cells or all(G[r, c] == colour[j] for r, c in cells):
            continue
        for r, c in cells:
            G[r, c] = colour[j]
        ops.append(colour[j]); sels.append(sel_of(cells))

    def copy_paste(h, wd, origins):
        clip = G[:h, :wd].copy()
        pend = []
        for (r0, c0) in origins:
            changed = False
            for r in range(h):
                for c in range(wd):
                    rr, cc = r0 + r, c0 + c
                    if rr < d and cc < w and clip[r, c] != 0 and G[rr, cc] != clip[r, c]:
                        G[rr, cc] = clip[r, c]
                        changed = True
            if changed:
                pend.append((r0, c0))
        if pend:
            ops.append(29); sels.append([0, 0, h - 1, wd - 1])   # whole tile/strip rectangle
            for r0, c0 in pend:
                ops.append(30); sels.append([r0, c0, 0, 0])

    # 2. tile along the top strip (diagonal families only)
    if tw < w:
        copy_paste(th, tw, [(0, c0) for c0 in range(tw, w, tw)])
    # 3. strip down the grid
    if th < d:
        copy_paste(th, w, [(r0, 0) for r0 in range(th, d, th)])

    ops.append(34); sels.append([0, 0, d - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 05269061"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 05269061"
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
                                f"for task 05269061"
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
                    f"Failed to build a complete episode for task 05269061 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"05269061-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
