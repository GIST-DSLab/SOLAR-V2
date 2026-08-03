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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. sample_colors
#    Rule = gravity of every column's non-background cells to the bottom.
#    It depends only on the PRESENCE pattern of non-bg cells, never on which
#    foreground colors were drawn -> only the background must be fixed.
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


# ---- helpers shared by generate() -------------------------------------------
def _gravity(g, bgc):
    h = len(g)
    w = len(g[0])
    out = [[bgc] * w for _ in range(h)]
    for c in range(w):
        vals = [g[r][c] for r in range(h) if g[r][c] != bgc]
        for i, v in enumerate(vals):
            out[h - len(vals) + i][c] = v
    return out


def _runs_of_column(g, bgc, c, h):
    """maximal contiguous same-color non-bg runs of column c, top-down."""
    rows = [r for r in range(h) if g[r][c] != bgc]
    if not rows:
        return []
    runs = []
    s = p = rows[0]
    for r in rows[1:]:
        if r == p + 1 and g[r][c] == g[p][c]:
            p = r
        else:
            runs.append((s, p))
            s = p = r
    runs.append((s, p))
    return runs


def _move_cost(g, bgc):
    """total number of 1-cell Move steps derive_operations will emit."""
    h = len(g)
    w = len(g[0])
    total = 0
    for c in range(w):
        runs = _runs_of_column(g, bgc, c, h)
        floor = h
        for a, b in reversed(runs):
            total += (floor - 1) - b
            floor -= b - a + 1
    return total


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lo, hi):
        return random.randint(lo + int((hi - lo) * diff_lb),
                              lo + int((hi - lo) * diff_ub))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]

    hmax = max(3, min(30, int(max_h)))
    wmax = max(3, min(30, int(max_w)))

    cap = 300          # bound on trajectory length (total Move steps)
    attempts = 0
    while True:
        attempts += 1
        h = unifint(3, hmax)
        w = unifint(3, wmax)
        nc = unifint(1, w)

        gi = [[bgc] * w for _ in range(h)]
        slocs = random.sample(range(w), nc)      # each column used at most once
        for l in slocs:
            col = random.choice(remcols)
            k = random.randint(1, h - 1)
            for r in random.sample(range(h), k):
                gi[r][l] = col

        cnt = Counter(v for row in gi for v in row)
        others = [n for cc, n in cnt.items() if cc != bgc]
        if not others:
            continue
        if cnt[bgc] <= max(others):              # bg must be strict majority
            continue
        if _move_cost(gi, bgc) > cap and attempts < 200:
            continue

        go = _gravity(gi, bgc)
        if go == gi:                             # nothing to do -> useless pair
            continue

        return {"input": tuple(tuple(r) for r in gi),
                "output": tuple(tuple(r) for r in go)}


# ----------------------------------------------------------------------------
# 3. derive_operations
#    Every column's non-bg cells fall to the bottom keeping their order.
#    Each maximal contiguous run is an object that TRANSLATES down -> MoveD.
#    Runs are settled bottom-most first, so a sliding run never lands on
#    anything but background.
#    Special case: a run whose color is 0 cannot be grabbed by ARCLE
#    (object buffer keeps nonzero cells only) -> paint it with Color0 instead.
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background = the color the generator fills the canvas with; guaranteed
    # to be the strict majority color of I.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []

    for c in range(w):
        rows = [r for r in range(h) if I[r, c] != bgc]
        if not rows:
            continue

        # maximal contiguous same-color runs of this column, top-down
        runs = []
        s = p = rows[0]
        for r in rows[1:]:
            if r == p + 1 and I[r, c] == I[p, c]:
                p = r
            else:
                runs.append((s, p))
                s = p = r
        runs.append((s, p))

        floor = h                       # first still-free row from the bottom
        for a, b in reversed(runs):     # settle the lowest run first
            m = b - a + 1
            d = (floor - 1) - b         # how far this run must fall
            floor -= m
            if d <= 0:
                continue                # already resting -> a Move would be a no-op

            color = int(I[a, c])
            src = [(r, c) for r in range(a, b + 1)]
            dst = [(r + d, c) for r in range(a, b + 1)]

            if color != 0:
                # grab the run once, then keep sliding with empty selections
                ops.append(21)
                sels.append(sel_of(src))
                for _ in range(d - 1):
                    ops.append(21)
                    sels.append(sel_of([]))
            else:
                # 0 is transparent to ARCLE's object ops: paint the run instead
                ops.append(0)
                sels.append(sel_of(dst))

            # the cells the run left behind (source minus destination)
            hole = sorted(set(src) - set(dst))
            if hole and bgc != 0:
                ops.append(bgc)
                sels.append(sel_of(hole))

    # Submit: selection is the whole grid rectangle (background included)
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
