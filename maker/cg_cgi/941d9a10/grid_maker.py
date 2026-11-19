"""
ARC Task: 941d9a10 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # 1, 2, 3 are fixed output marker colours; bgc and the grid-line colour are random.
    cols = [c for c in range(10) if c not in (1, 2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * max(0.0, diff_lb)))
    hi_ = a + int(round((b - a) * min(1.0, max(d, diff_lb))))
    lo = max(a, min(lo, b))
    hi_ = max(lo, min(hi_, b))
    return random.randint(lo, hi_)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    if bgc is None or fgc is None:
        cols = [c for c in range(10) if c not in (1, 2, 3)]
        bgc, fgc = random.sample(cols, 2)
    h = _unifint(diff_lb, diff_ub, (5, max(5, min(30, max_h))))
    w = _unifint(diff_lb, diff_ub, (5, max(5, min(30, max_w))))

    def line_positions(n):
        opts = list(range(2, (n - 1) // 2 + 1, 2))
        k = opts[_unifint(diff_lb, diff_ub, (0, len(opts) - 1))]
        # 2k+1 slots, lines at odd slots, then insert (n - (2k+1)) blank slots
        slots = [(j % 2 == 1) for j in range(2 * k + 1)]
        for _ in range(n - (2 * k + 1)):
            loc = random.randint(0, len(slots) - 1)
            slots.insert(loc, False)
        return [i for i, s in enumerate(slots) if s]

    hl = line_positions(h)
    wl = line_positions(w)
    gi = np.full((h, w), bgc, dtype=int)
    for r in hl:
        gi[r, :] = fgc
    for c in wl:
        gi[:, c] = fgc

    def bands(n, lines):
        ls = set(lines)
        out, cur = [], []
        for i in range(n):
            if i in ls:
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(i)
        if cur:
            out.append(cur)
        return out

    rb = bands(h, hl)
    cb = bands(w, wl)
    go = gi.copy()
    for color, ri, ci in ((1, 0, 0), (3, len(rb) - 1, len(cb) - 1),
                          (2, len(rb) // 2, len(cb) // 2)):
        for r in rb[ri]:
            for c in cb[ci]:
                go[r, c] = color
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background = colour at the grid corners (corners are always cell interiors).
    corners = [I[0, 0], I[0, wi - 1], I[hi - 1, 0], I[hi - 1, wi - 1]]
    bgc = Counter(int(x) for x in corners).most_common(1)[0][0]

    # Reading: full-length lines cut the grid into row bands and column bands.
    line_rows = {r for r in range(hi) if all(I[r, c] != bgc for c in range(wi))}
    line_cols = {c for c in range(wi) if all(I[r, c] != bgc for r in range(hi))}

    def bands(n, lines):
        out, cur = [], []
        for i in range(n):
            if i in lines:
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(i)
        if cur:
            out.append(cur)
        return out

    rb = bands(hi, line_rows)
    cb = bands(wi, line_cols)

    ops, sels = [], []
    # Each cell = (row band) x (column band); one connected bgc region -> one FloodFill.
    # first x first -> 1, middle x middle -> 2, last x last -> 3
    for color, ri, ci in ((1, 0, 0),
                          (2, len(rb) // 2, len(cb) // 2),
                          (3, len(rb) - 1, len(cb) - 1)):
        r0, c0 = rb[ri][0], cb[ci][0]
        ops.append(10 + color)
        sels.append([r0, c0, 0, 0])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 941d9a10"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 941d9a10"
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
                                f"for task 941d9a10"
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
                    f"Failed to build a complete episode for task 941d9a10 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"941d9a10-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
