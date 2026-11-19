"""
ARC Task: d4f3cd78 (RE-ARC) — LLM-generated grid_maker
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

try:
    from dsl import *  # noqa
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)

    def unif(lo, hi):
        if hi <= lo:
            return lo
        lb = lo + (hi - lo) * diff_lb
        ub = lo + (hi - lo) * diff_ub
        a = int(round(lb)); b = int(round(ub))
        a = max(lo, min(hi, a)); b = max(a, min(hi, b))
        return random.randint(a, b)

    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unif(10, H)
    w = unif(10, W)
    ih = unif(3, h // 3 * 2)
    iw = unif(3, w // 3 * 2)
    loci = random.randint(1, h - ih - 1)
    locj = random.randint(1, w - iw - 1)
    r0, c0, r1, c1 = loci, locj, loci + ih - 1, locj + iw - 1

    box = set()
    for c in range(c0, c1 + 1):
        box.add((r0, c)); box.add((r1, c))
    for r in range(r0, r1 + 1):
        box.add((r, c0)); box.add((r, c1))
    fullcrns = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    opts = sorted(box - fullcrns)
    nholes = unif(1, len(opts))
    holes = random.sample(opts, nholes)

    gi = [[bgc] * w for _ in range(h)]
    for (r, c) in box - set(holes):
        gi[r][c] = fgc
    go = [row[:] for row in gi]
    A, B, C, D = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    for r in range(A, C + 1):
        for c in range(B, D + 1):
            go[r][c] = 8
    for (r, c) in holes:
        dr = 1 if r > C else (-1 if r < A else 0)
        dc = 1 if c > D else (-1 if c < B else 0)
        while 0 <= r < h and 0 <= c < w:
            go[r][c] = 8
            r += dr; c += dc
    return {"input": gi, "output": go}


def _runs(vals):
    vals = sorted(vals)
    runs = []
    for v in vals:
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return runs


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background: the canvas colour (the outline is a thin perimeter, never the majority)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    r0 = min(r for r, _ in fg); r1 = max(r for r, _ in fg)
    c0 = min(c for _, c in fg); c1 = max(c for _, c in fg)

    ops, sels = [], []

    # 1) fill the box interior with 8 (full rectangle)
    ops.append(8)
    sels.append([r0 + 1, c0 + 1, (r1 - 1) - (r0 + 1), (c1 - 1) - (c0 + 1)])

    # 2) gaps in the outline leak rays of 8 out to the grid edge
    top = {c for c in range(c0 + 1, c1) if I[r0, c] == bgc}
    bot = {c for c in range(c0 + 1, c1) if I[r1, c] == bgc}
    left = {r for r in range(r0 + 1, r1) if I[r, c0] == bgc}
    right = {r for r in range(r0 + 1, r1) if I[r, c1] == bgc}

    # columns open at both top and bottom: the whole column (full-height rectangle)
    for a, b in _runs(top & bot):
        ops.append(8); sels.append([0, a, h - 1, b - a])
    # runs of adjacent gaps on the top side: rectangle from row 0 down to the outline row
    for a, b in _runs(top - bot):
        ops.append(8); sels.append([0, a, r0, b - a])
    # runs on the bottom side: from the outline row down to the last row
    for a, b in _runs(bot - top):
        ops.append(8); sels.append([r1, a, h - 1 - r1, b - a])
    # rows open on both left and right: the whole row
    for a, b in _runs(left & right):
        ops.append(8); sels.append([a, 0, b - a, w - 1])
    for a, b in _runs(left - right):
        ops.append(8); sels.append([a, 0, b - a, c0])
    for a, b in _runs(right - left):
        ops.append(8); sels.append([a, c1, b - a, w - 1 - c1])

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
                        f"num_examples+1 ({num_examples + 1}) for task d4f3cd78"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d4f3cd78"
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
                                f"for task d4f3cd78"
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
                    f"Failed to build a complete episode for task d4f3cd78 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d4f3cd78-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
