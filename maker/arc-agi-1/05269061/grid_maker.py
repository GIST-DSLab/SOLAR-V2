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
    {"mode": "diag",  "mirror": False},
    {"mode": "diag",  "mirror": True},
    {"mode": "horiz", "mirror": False},
    {"mode": "horiz", "mirror": True},
]


def sample_colors(num_examples=None) -> dict:
    # background is hardcoded 0 in the generator; line colours are irrelevant to the
    # rule (it depends only on line positions / period), so only the discrete
    # structural variants (diagonal vs horizontal family, mirrored or not) are planned.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, mode=None, mirror=None, **kw) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        val = int(round(a + (b - a) * (lb + (ub - lb) * random.random())))
        return max(a, min(b, val))

    if mode is None or mirror is None:
        v = random.choice(VARIANTS)
        mode = v["mode"]
        mirror = v["mirror"]

    dmax = min(30, int(max_h), int(max_w))
    if dmax < 2:
        dmax = 2
    d = unifint(diff_lb, diff_ub, (2, dmax))

    gi = np.zeros((d, d), dtype=int)
    go = np.zeros((d, d), dtype=int)
    colopts = list(range(1, 10))

    if mode == "diag":
        K = 2 * d - 1                       # anti-diagonals r+c = 0 .. 2d-2
        num = unifint(diff_lb, diff_ub, (2, max(2, min(2 * d - 2, 9))))
        num = max(2, min(num, K - 1))
        cols = [random.choice(colopts) for _ in range(num)]
        keeps = [random.choice(list(range(j, K, num))) for j in range(num)]
        for k in range(K):
            col = cols[k % num]
            cells = [(r, k - r) for r in range(d) if 0 <= k - r < d]
            for (r, c) in cells:
                go[r, c] = col
            if keeps[k % num] == k:
                for (r, c) in cells:
                    gi[r, c] = col
    else:
        K = d                                # full rows 0 .. d-1
        num = unifint(diff_lb, diff_ub, (2, max(2, min(d, 9))))
        num = max(2, min(num, d))
        cols = [random.choice(colopts) for _ in range(num)]
        keeps = [random.choice(list(range(j, K, num))) for j in range(num)]
        for k in range(K):
            col = cols[k % num]
            go[k, :] = col
            if keeps[k % num] == k:
                gi[k, :] = col

    if mirror:
        gi = np.fliplr(gi)
        go = np.fliplr(go)

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    """
    Rule measured from I alone:
      * I contains a few monochrome seed LINES, either full rows (hfrontier family)
        or full diagonals (shoot UP_RIGHT family, possibly mirrored -> main diagonals).
      * The number of seed lines IS the period p of the line family.
      * Each seed line with family-index k0 is stamped onto every line whose index
        k satisfies k % p == k0 % p, tiling the whole grid.
    Ops: one Color op per stamped line (the line's exact cells), grouped seed by seed.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    d, w = I.shape
    ops, sels = [], []

    nz = [(r, c) for r in range(d) for c in range(w) if I[r, c] != 0]

    # ---- horizontal (full-row) family? --------------------------------------
    rows = sorted({r for r, _ in nz})
    horiz = bool(rows) and len(nz) == len(rows) * w and all(
        all(I[r, c] == I[r, 0] and I[r, c] != 0 for c in range(w)) for r in rows
    )

    def row_cells(k):
        return [(k, c) for c in range(w)]

    def anti_cells(k):
        return [(r, k - r) for r in range(d) if 0 <= k - r < w]

    def main_cells(k):
        # key = r - c + (d-1)
        return [(r, r - k + d - 1) for r in range(d) if 0 <= r - k + d - 1 < w]

    if horiz:
        cells_of = row_cells
        K = d
        seeds = [(r, int(I[r, 0])) for r in rows]
    else:
        # ---- diagonal family: figure out orientation from I ------------------
        def try_orient(cells_fn, key_fn, K):
            groups = {}
            for (r, c) in nz:
                groups.setdefault(key_fn(r, c), []).append((r, c))
            for k, g in groups.items():
                if set(g) != set(cells_fn(k)):
                    return None
                if len({int(I[r, c]) for r, c in g}) != 1:
                    return None
            return groups

        K = d + w - 1
        got = try_orient(anti_cells, lambda r, c: r + c, K)
        if got is not None:
            cells_of = anti_cells
        else:
            got = try_orient(main_cells, lambda r, c: r - c + d - 1, K)
            cells_of = main_cells
        seeds = sorted((k, int(I[g[0][0], g[0][1]])) for k, g in got.items())

    p = len(seeds)  # period == number of seed lines

    # ---- stamp each seed line at every multiple of the period ---------------
    for k0, col in seeds:
        for k in range(K):
            if k == k0 or k % p != k0 % p:
                continue
            cells = cells_of(k)
            if not cells:
                continue
            ops.append(int(col))
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
