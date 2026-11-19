"""
ARC Task: 1bfc4729 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)
    return {"bgc": bgc, "acol": acol, "bcol": bcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, acol=None, bcol=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return random.randint(int(round(lo)), int(round(max(lo, hi))))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if acol is None:
        acol = random.choice([c for c in cols if c != bgc])
    if bcol is None:
        bcol = random.choice([c for c in cols if c not in (bgc, acol)])

    H = max(4, min(30, max_h))
    W = max(4, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (4, H))
    w = unifint(diff_lb, diff_ub, (4, W))
    if h % 2 == 1:
        opts = [x for x in (max(4, h - 1), min(H, h + 1)) if x % 2 == 0 and 4 <= x <= H]
        h = random.choice(opts) if opts else 4
    alocj = unifint(diff_lb, diff_ub, (w // 2, w - 1))
    if random.choice((True, False)):
        alocj = max(min(w // 2, alocj - w // 2), 1)
    aloci = random.randint(1, h // 2 - 1)
    blocj = unifint(diff_lb, diff_ub, (w // 2, w - 1))
    if random.choice((True, False)):
        blocj = max(min(w // 2, blocj - w // 2), 1)
    bloci = random.randint(h // 2, h - 2)

    gi = np.full((h, w), bgc, dtype=int)
    gi[aloci, alocj] = acol
    gi[bloci, blocj] = bcol
    go = gi.copy()
    half = h // 2
    go[aloci, :] = acol
    go[bloci, :] = bcol
    go[0, :] = acol
    go[h - 1, :] = bcol
    go[:half, 0] = acol
    go[:half, w - 1] = acol
    go[half:, 0] = bcol
    go[half:, w - 1] = bcol
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading: the grid splits into two halves; each half is owned by the single
    # marker inside it. Each marker draws, in its own colour, its own half's
    # frame: the line through itself, the outer border row of its half, and the
    # two side columns of its half. Work half by half (marker by marker).
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    half = h // 2

    # background: the canvas colour, read from the demonstrations (and I)
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_in, _ex_out in examples:
            cnt.update(np.asarray(ex_in, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    if not np.any(I == bgc):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def marker_in(rows):
        for r in rows:
            for c in range(w):
                if I[r, c] != bgc:
                    return r, int(I[r, c])
        return None

    top = marker_in(range(0, half))
    bot = marker_in(range(half, h))

    grid = I.copy()
    ops, sels = [], []

    def paint(color, cells):
        if any(grid[r, c] != color for r, c in cells):
            for r, c in cells:
                grid[r, c] = color
            ops.append(color)
            sels.append(sel_of(cells))

    halves = []
    if top is not None:
        halves.append((top, 0, range(0, half)))
    if bot is not None:
        halves.append((bot, h - 1, range(half, h)))

    for (mr, col), border_row, rows in halves:
        # line through the marker
        paint(col, [(mr, c) for c in range(w)])
        # outer border row of this half
        paint(col, [(border_row, c) for c in range(w)])
        # both side columns of this half
        paint(col, [(r, 0) for r in rows] + [(r, w - 1) for r in rows])

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
                        f"num_examples+1 ({num_examples + 1}) for task 1bfc4729"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1bfc4729"
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
                                f"for task 1bfc4729"
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
                    f"Failed to build a complete episode for task 1bfc4729 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1bfc4729-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
