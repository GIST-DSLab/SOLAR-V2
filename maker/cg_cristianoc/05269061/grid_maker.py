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

VARIANTS = [
    {"family": "horiz", "mirror": False},
    {"family": "diag", "mirror": False},
    {"family": "diag", "mirror": True},
]


def sample_colors(num_examples=None) -> dict:
    # Background is the hard-coded 0 and the rule never depends on which line
    # colours appear, so no colour role needs fixing. The structural variants
    # (rows / anti-diagonals / main diagonals) are planned up front instead.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    return {"instance_plan": ex + [dict(random.choice(ex))]}


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    x = random.uniform(lb, ub)
    return max(a, min(b, int(round(a + (b - a) * x))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, family=None, mirror=None, **kw):
    if family is None or mirror is None:
        v = random.choice(VARIANTS)
        family, mirror = v["family"], v["mirror"]
    d = _unifint(diff_lb, diff_ub, (2, min(30, max_h, max_w)))
    go = [[0] * d for _ in range(d)]
    gi = [[0] * d for _ in range(d)]
    colopts = list(range(1, 10))
    if family == "diag":
        nlines = 2 * d - 1
        num = _unifint(diff_lb, diff_ub, (2, min(2 * d - 2, 9)))
        cells = lambda k: [(r, k - r) for r in range(d) if 0 <= k - r < d]
    else:
        nlines = d
        num = _unifint(diff_lb, diff_ub, (2, min(d, 9)))
        cells = lambda k: [(k, c) for c in range(d)]
    cols = [random.choice(colopts) for _ in range(num)]
    keeps = [random.choice(list(range(j, nlines, num))) for j in range(num)]
    for k in range(nlines):
        col = cols[k % num]
        for r, c in cells(k):
            go[r][c] = col
            if keeps[k % num] == k:
                gi[r][c] = col
    if mirror:
        gi = [row[::-1] for row in gi]
        go = [row[::-1] for row in go]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Reading (taken from the golf entry, which differs from the maker's
    line-by-line stamping): each cell's colour depends only on its residue
    class along the line family -- r % p (rows), (r+c) % p (anti-diagonals)
    or (r-c) % p (main diagonals, mirrored case) -- where p = number of seed
    lines in I. Each seed line is the one visible member of its class.
    So the unit here is the WHOLE RESIDUE CLASS, not each line: one Color op
    per seed paints its whole class except the seed line, which already
    holds that colour. Every selection is computed from I alone.
    """
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    nz = [(r, c) for r in range(h) for c in range(w) if I[r, c] != 0]

    def lines_ok(key):
        groups = {}
        for r, c in nz:
            groups.setdefault(key(r, c), []).append((r, c))
        for k, g in groups.items():
            full = [(r, c) for r in range(h) for c in range(w) if key(r, c) == k]
            if set(g) != set(full) or len({int(I[r, c]) for r, c in g}) != 1:
                return None
        return groups

    # Find which line family the seed lines belong to: full rows,
    # anti-diagonals (r+c), or main diagonals (r-c).
    keys = [lambda r, c: r,
            lambda r, c: r + c,
            lambda r, c: r - c + w - 1]
    key, groups = keys[0], None
    for kf in keys:
        g = lines_ok(kf)
        if g is not None:
            key, groups = kf, g
            break
    if groups is None:
        groups = {}
    seeds = sorted(groups.items())
    p = len(seeds)  # period = number of seed lines (one per residue class)

    ops, sels = [], []
    for k0, g in seeds:
        col = int(I[g[0]])
        cls = [(r, c) for r in range(h) for c in range(w)
               if key(r, c) % p == k0 % p and key(r, c) != k0]
        if cls:
            ops.append(col)
            sels.append(sel_of(cls))
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
