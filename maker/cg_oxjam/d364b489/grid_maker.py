"""
ARC Task: d364b489 (RE-ARC) — LLM-generated grid_maker
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
    # The halo colours 2, 6, 7 and 8 are hardcoded. Only the background colour
    # and the mark colour are sampled.
    cols = [c for c in range(10) if c not in (2, 6, 7, 8)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 6, 7, 8)]
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    h = unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    inds = set((r, c) for r in range(h) for c in range(w))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 5)))
    res = set()
    for _ in range(num):
        if not inds:
            break
        r = random.choice(sorted(inds))
        inds.discard(r)
        # Clear all 8 neighbours of the mark.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                inds.discard((r[0] + dr, r[1] + dc))
        # Also clear the 4 cells at distance 2 in a straight line.
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            inds.discard((r[0] + dr, r[1] + dc))
        res.add(r)
    for (r, c) in res:
        gi[r][c] = fgc
    go = [row[:] for row in gi]
    for dr, dc, col in ((0, -1, 7), (0, 1, 6), (1, 0, 8), (-1, 0, 2)):
        for (r, c) in res:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                go[rr][cc] = col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """The rule, read from I only: every mark gets a 4-cell halo. The cell
    above it becomes 2, below 8, left 7 and right 6.

    How the golf entry reads it: one halo side is one unit. It turns the grid
    a quarter turn at a time and paints the cell next to every mark on one
    side, for all marks at once. The maker's four directional passes split the
    task the same way, so nothing structural changes here. The one thing the
    golf adds is the order of the sides: it goes around the mark in rotation
    order (2, 7, 8, 6 = up, left, down, right). The generator keeps halo cells
    from ever colliding with each other or with a mark, so the order does not
    change the result. Each selection is "every mark shifted one step in one
    direction", which comes from I alone. Halo cells that fall off the grid
    are dropped."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Background = the canvas colour. Marks are sparse, so it is the majority
    # colour. The example inputs share the same background, so pool them in
    # when they are given.
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_in, _ in examples:
            cnt.update(np.asarray(ex_in, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    if bgc not in set(I.flatten().tolist()):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    marks = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    # Go around the mark in rotation order: up, left, down, right.
    for dr, dc, color in ((-1, 0, 2), (0, -1, 7), (1, 0, 8), (0, 1, 6)):
        cells = []
        for (r, c) in marks:
            rr, cc = r + dr, c + dc
            if 0 <= rr < hi and 0 <= cc < wi:
                cells.append((rr, cc))
        if cells:
            ops.append(int(color))
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d364b489"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d364b489"
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
                                f"for task d364b489"
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
                    f"Failed to build a complete episode for task d364b489 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d364b489-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
