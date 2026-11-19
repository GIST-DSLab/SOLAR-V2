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
    # Only the background colour is sampled once. The column colours can be
    # anything and the rule does not depend on them.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi = a + int(round((b - a) * ub))
        lo = max(a, min(lo, b))
        hi = max(lo, min(hi, b))
        return random.randint(lo, hi)

    while True:
        h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
        w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
        nc = unifint(diff_lb, diff_ub, (1, w))
        gi = [[bgc] * w for _ in range(h)]
        remcols = [c for c in cols if c != bgc]
        scols = [random.choice(remcols) for _ in range(nc)]
        slocs = random.sample(range(w), nc)
        for col, l in zip(scols, slocs):
            nc2 = random.randint(1, h - 1)
            rows = random.sample(range(h), nc2)
            for r in rows:
                gi[r][l] = col
        # gravity: every column's non-bg cells fall to the bottom, keeping their order
        go = [[bgc] * w for _ in range(h)]
        for c in range(w):
            vals = [gi[r][c] for r in range(h) if gi[r][c] != bgc]
            for k, v in enumerate(vals):
                go[h - len(vals) + k][c] = v
        cnt = Counter(v for row in gi for v in row)
        others = [cnt[v] for v in cnt if v != bgc]
        if not others or cnt[bgc] > max(others):
            break
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background = the colour the generator fills the canvas with; the generator
    # guarantees it is the strict majority colour of I.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    G = I.copy()  # tracks the working grid as ARCLE will hold it
    ops, sels = [], []

    for c in range(w):
        rows = [r for r in range(h) if I[r, c] != bgc]
        if not rows:
            continue

        # maximal contiguous same-colour runs of this column, top-down
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
                continue                # already resting, so a Move would be a no-op

            color = int(I[a, c])
            src = [(r, c) for r in range(a, b + 1)]
            dst = [(r + d, c) for r in range(a, b + 1)]

            if color != 0:
                # grab the run once, then keep sliding it with empty selections
                ops.append(21)
                sels.append(sel_of(src))
                for _ in range(d - 1):
                    ops.append(21)
                    sels.append(sel_of([]))
                # ARCLE: the grabbed footprint becomes 0, the path is restored,
                # and the run is pasted at its destination
                for (r, cc) in src:
                    G[r, cc] = 0
                for (r, cc) in dst:
                    G[r, cc] = color
            else:
                # ARCLE treats 0 as empty, so it would not carry a 0 run: paint it at its destination
                ops.append(0)
                sels.append(sel_of(dst))
                for (r, cc) in dst:
                    G[r, cc] = 0

        # The column has settled. Rows above the stack are the column's empty
        # top. Any cell there that does not hold the background now (a vacated
        # footprint left at 0) is repaired with one Color(bgc).
        if bgc != 0:
            top = [(r, c) for r in range(floor) if G[r, c] != bgc]
            if top:
                ops.append(bgc)
                sels.append(sel_of(top))
                for (r, cc) in top:
                    G[r, cc] = bgc

    # Submit: the selection is the whole grid rectangle, background included
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
