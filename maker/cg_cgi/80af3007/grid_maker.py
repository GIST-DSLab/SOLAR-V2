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

try:
    from dsl import *
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, fgc=None, **kwargs) -> dict:
    fullcols = list(range(10))
    if bgc is None:
        bgc = random.choice(fullcols)
    if fgc is None:
        fgc = random.choice([c for c in fullcols if c != bgc])
    hmax = 2
    while (hmax + 1) ** 2 + 2 <= max_h and hmax < 5:
        hmax += 1
    wmax = 2
    while (wmax + 1) ** 2 + 2 <= max_w and wmax < 5:
        wmax += 1
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    c = canvas(bgc, (h, w))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcd, h * w - numcd))
    numc = min(max(0, numc), h * w)
    inds = totuple(asindices(c))
    locs = tuple(set(sample(inds, numc)) | set(sample(totuple(corners(inds)), 3)))
    gi = fill(c, fgc, locs)
    go = canvas(bgc, (h ** 2, w ** 2))
    for loc in locs:
        go = fill(go, fgc, shift(locs, multiply(loc, (h, w))))
    fullh = unifint(diff_lb, diff_ub, (h ** 2 + 2, max_h))
    fullw = unifint(diff_lb, diff_ub, (w ** 2 + 2, max_w))
    fullg = canvas(bgc, (fullh, fullw))
    loci = randint(1, fullh - h ** 2 - 1)
    locj = randint(1, fullw - w ** 2 - 1)
    giups = hupscale(vupscale(gi, h), w)
    gi = paint(fullg, shift(asobject(giups), (loci, locj)))
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (golf view): shrink the upscaled object back to its small h x w pattern P,
    # then the answer is a blank h^2 x w^2 canvas with a copy of P stamped at every
    # block position where P itself is foreground.
    I = np.asarray(I, dtype=int)
    bgc = int(I[0, 0])  # generator always leaves a >=1 background margin
    fg = np.argwhere(I != bgc)
    r0, c0 = int(fg[:, 0].min()), int(fg[:, 1].min())
    r1, c1 = int(fg[:, 0].max()), int(fg[:, 1].max())
    fgc = int(I[fg[0][0], fg[0][1]])
    ho, wo = r1 - r0 + 1, c1 - c0 + 1
    h = int(round(ho ** 0.5))
    w = int(round(wo ** 0.5))
    # small pattern, read once per block of the upscaled input object
    pat = [[I[r0 + i * h, c0 + j * w] == fgc for j in range(w)] for i in range(h)]

    ops, sels = [], []
    # 1. Crop to the object's bbox: canvas is now h^2 x w^2 (the output size).
    ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])  # full rectangle

    full = all(all(row) for row in pat)
    if not full:
        # 2. Blank canvas: lay the background over the whole output area.
        ops.append(bgc); sels.append([0, 0, ho - 1, wo - 1])  # full rectangle

        # 3. Draw the small pattern once, at a block that P marks as foreground
        #    (a corner block; 3 of 4 corners are always foreground).
        src = None
        for p in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
            if pat[p[0]][p[1]]:
                src = p
                break
        i0, j0 = src
        for a in range(h):
            row_cells = [(i0 * h + a, j0 * w + b) for b in range(w) if pat[a][b]]
            if row_cells:
                ops.append(fgc); sels.append(sel_of(row_cells))

        # 4. Stamp the pattern at every other foreground block position of P.
        targets = [(i, j) for i in range(h) for j in range(w)
                   if pat[i][j] and (i, j) != (i0, j0)]
        if targets:
            ops.append(29); sels.append([i0 * h, j0 * w, h - 1, w - 1])  # the drawn block
            for (i, j) in targets:
                if fgc == 0:
                    # clipboard only carries the (nonzero) bgc cells: lay a fgc base first
                    ops.append(0); sels.append([i * h, j * w, h - 1, w - 1])
                ops.append(30); sels.append([i * h, j * w, 0, 0])
    # (a fully-foreground pattern yields an all-fgc output: the crop already is it)

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
