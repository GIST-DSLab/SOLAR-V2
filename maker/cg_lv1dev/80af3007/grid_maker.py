"""
ARC Task: 80af3007 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None:
        bgc = random.randint(0, 9)
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c != bgc])

    hmax = 5
    while hmax > 2 and hmax ** 2 + 2 > max_h:
        hmax -= 1
    wmax = 5
    while wmax > 2 and wmax ** 2 + 2 > max_w:
        wmax -= 1
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))

    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = random.choice((numcd, h * w - numcd))
    numc = min(max(0, numc), h * w)
    inds = [(i, j) for i in range(h) for j in range(w)]
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    locs = set(random.sample(inds, numc)) | set(random.sample(corners, 3))

    gi = [[bgc] * w for _ in range(h)]
    for (i, j) in locs:
        gi[i][j] = fgc

    go = [[bgc] * (w * w) for _ in range(h * h)]
    for (li, lj) in locs:
        for (i, j) in locs:
            go[li * h + i][lj * w + j] = fgc

    fullh = unifint(diff_lb, diff_ub, (h ** 2 + 2, max_h))
    fullw = unifint(diff_lb, diff_ub, (w ** 2 + 2, max_w))
    fullh = max(fullh, h ** 2 + 2)
    fullw = max(fullw, w ** 2 + 2)
    loci = random.randint(1, fullh - h ** 2 - 1)
    locj = random.randint(1, fullw - w ** 2 - 1)
    grid = [[bgc] * fullw for _ in range(fullh)]
    for r in range(h * h):
        for c in range(w * w):
            grid[loci + r][locj + c] = gi[r // h][c // w]
    return {"input": grid, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)

    # Generator floods the canvas with bgc and leaves a >=1 margin -> corner is bgc.
    bgc = int(I[0, 0])
    fg = np.argwhere(I != bgc)
    r0, c0 = int(fg[:, 0].min()), int(fg[:, 1].min())
    r1, c1 = int(fg[:, 0].max()), int(fg[:, 1].max())
    fgc = int(I[fg[0][0], fg[0][1]])

    # Object bbox is exactly h^2 x w^2 (3 of 4 pattern corners are always set).
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    h = int(round(oh ** 0.5))
    w = int(round(ow ** 0.5))

    # Reading: downscale the upscaled object back to its h x w pattern,
    # then output = pattern (x) pattern: each "on" cell of the pattern
    # becomes a copy of the pattern, each "off" cell an empty block.
    pat = [[I[r0 + i * h, c0 + j * w] == fgc for j in range(w)] for i in range(h)]
    pat_cells = [(i, j) for i in range(h) for j in range(w) if pat[i][j]]

    ops, sels = [], []

    # 1. Canvas becomes the h^2 x w^2 output frame (the object's bounding box).
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])  # full rectangle

    # 2. Fresh background canvas: the whole frame is laid with bgc.
    ops.append(bgc); sels.append([0, 0, oh - 1, ow - 1])    # full rectangle

    # 3. Draw the downscaled pattern once, in the first "on" block.
    si, sj = pat_cells[0]
    ops.append(fgc)
    sels.append(sel_of([(si * h + i, sj * w + j) for (i, j) in pat_cells]))

    others = pat_cells[1:]
    if fgc != 0:
        # 4. Copy that pattern block and stamp it at every other "on" block.
        if others:
            ops.append(29); sels.append([si * h, sj * w, h - 1, w - 1])  # full block
            for (bi, bj) in others:
                ops.append(30); sels.append([bi * h, bj * w, 0, 0])
    else:
        # fgc == 0 cannot travel through the clipboard; draw each copy directly.
        for (bi, bj) in others:
            ops.append(fgc)
            sels.append(sel_of([(bi * h + i, bj * w + j) for (i, j) in pat_cells]))

    ops.append(34); sels.append([0, 0, oh - 1, ow - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 80af3007"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 80af3007"
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
                                f"for task 80af3007"
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
                    f"Failed to build a complete episode for task 80af3007 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"80af3007-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
