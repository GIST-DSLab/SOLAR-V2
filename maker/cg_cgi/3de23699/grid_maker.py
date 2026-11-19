"""
ARC Task: 3de23699 (RE-ARC) — LLM-generated grid_maker
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
    ccol = random.choice(rem)
    rem = [c for c in rem if c != ccol]
    ncol = random.choice(rem)
    return {"bgc": bgc, "ccol": ccol, "ncol": ncol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccol=None, ncol=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccol is None:
        ccol = random.choice([c for c in cols if c != bgc])
    if ncol is None:
        ncol = random.choice([c for c in cols if c not in (bgc, ccol)])

    H = max(5, min(30, max_h))
    W = max(5, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (5, H))
    w = unifint(diff_lb, diff_ub, (5, W))
    gi = [[bgc] * w for _ in range(h)]
    hi = unifint(diff_lb, diff_ub, (4, h))
    wi = unifint(diff_lb, diff_ub, (4, w))
    loci = random.randint(0, h - hi)
    locj = random.randint(0, w - wi)
    r1, c1 = loci + hi - 1, locj + wi - 1
    cnds = [(r, c) for r in range(loci + 1, r1) for c in range(locj + 1, c1)]
    mp = len(cnds) // 2
    dev = unifint(diff_lb, diff_ub, (0, mp))
    ncnds = random.choice((dev, len(cnds) - dev))
    ncnds = min(max(0, ncnds), len(cnds))
    ss = random.sample(cnds, ncnds)
    for (r, c) in [(loci, locj), (loci, c1), (r1, locj), (r1, c1)]:
        gi[r][c] = ccol
    for (r, c) in ss:
        gi[r][c] = ncol
    go = [[ccol if gi[r][c] == ncol else gi[r][c] for c in range(locj + 1, c1)]
          for r in range(loci + 1, r1)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Reading (the golf entry's carving): the four corner markers frame a window.
      1. Trim the background margin away until the markers sit on the canvas edge
         (CropGrid to the marker rectangle - a full rectangle, background included).
      2. Peel off one more layer - the frame row/columns holding the markers
         (CropGrid to the inner rectangle).
      3. Everything left in the window that is not background is noise: paint it
         all to the marker colour (exact cells, read off the current grid).
    Nothing here inspects O.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    border = []
    for cc in range(w):
        border += [int(I[0, cc]), int(I[h - 1, cc])]
    for rr in range(1, h - 1):
        border += [int(I[rr, 0]), int(I[rr, w - 1])]
    bgc = Counter(border).most_common(1)[0][0]

    cells = [(r, c) for r in range(h) for c in range(w) if int(I[r, c]) != bgc]
    if not cells:
        return [34], [[0, 0, h - 1, w - 1]]
    r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
    ccol = int(I[r0, c0])
    if r1 - r0 < 2 or c1 - c0 < 2:
        return [34], [[0, 0, h - 1, w - 1]]

    ops, sels = [], []
    G = I.copy()

    # 1. trim the background margin: crop to the marker rectangle (whole rectangle)
    if not (r0 == 0 and c0 == 0 and r1 == h - 1 and c1 == w - 1):
        ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])
        G = G[r0:r1 + 1, c0:c1 + 1].copy()

    # 2. peel off the frame layer carrying the markers (whole inner rectangle)
    gh, gw = G.shape
    ops.append(33); sels.append([1, 1, gh - 3, gw - 3])
    G = G[1:gh - 1, 1:gw - 1].copy()
    ho, wo = G.shape

    # 3. every non-background cell left in the window is noise -> marker colour
    noise = [(r, c) for r in range(ho) for c in range(wo)
             if int(G[r, c]) != bgc and int(G[r, c]) != ccol]
    if noise:
        ops.append(ccol); sels.append(sel_of(sorted(noise)))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3de23699"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3de23699"
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
                                f"for task 3de23699"
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
                    f"Failed to build a complete episode for task 3de23699 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3de23699-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
