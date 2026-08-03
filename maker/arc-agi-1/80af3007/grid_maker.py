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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    def _cap(m):
        k = 2
        while k < 5 and (k + 1) ** 2 + 2 <= m:
            k += 1
        return k

    h_hi = _cap(max_h)
    w_hi = _cap(max_w)

    h = unifint(diff_lb, diff_ub, (2, h_hi))
    w = unifint(diff_lb, diff_ub, (2, w_hi))
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
    loc = (loci, locj)
    giups = hupscale(vupscale(gi, h), w)
    gi = paint(fullg, shift(asobject(giups), loc))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)

    # Background: the generator floods the full canvas with bgc and always leaves a
    # >=1 cell margin around the object, so the grid corner is bgc.
    bgc = int(I[0, 0])

    fg = np.argwhere(I != bgc)
    r0, c0 = int(fg[:, 0].min()), int(fg[:, 1].min())
    r1, c1 = int(fg[:, 0].max()), int(fg[:, 1].max())
    fgc = int(I[fg[0][0], fg[0][1]])

    # The object is a block-upscaled h x w pattern; its bbox is exactly h^2 x w^2.
    ho, wo = r1 - r0 + 1, c1 - c0 + 1
    h = int(round(ho ** 0.5))
    w = int(round(wo ** 0.5))

    # Read the small pattern back off the block grid (each block is uniform).
    pat = [[I[r0 + i * h, c0 + j * w] == fgc for j in range(w)] for i in range(h)]

    ops, sels = [], []

    # 1. Keep only the upscaled object: the canvas becomes the h^2 x w^2 block grid.
    ops.append(33)
    sels.append([r0, c0, ho - 1, wo - 1])

    # 2. Pick one solid block (a corner block, always part of the pattern) and carve
    #    the pattern's holes into it -> that block now holds the pattern at 1:1 scale.
    cand = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    src = None
    for p in cand:
        if pat[p[0]][p[1]]:
            src = p
            break
    if src is None:
        for i in range(h):
            for j in range(w):
                if pat[i][j]:
                    src = (i, j)
                    break
            if src is not None:
                break

    i0, j0 = src
    carved = False
    for a in range(h):
        b = 0
        while b < w:
            if not pat[a][b]:
                b2 = b
                while b2 < w and not pat[a][b2]:
                    b2 += 1
                ops.append(bgc)
                sels.append([i0 * h + a, j0 * w + b, 0, b2 - b - 1])
                carved = True
                b = b2
            else:
                b += 1

    # 3. Stamp that carved block onto every other solid block (blocks that are already
    #    background stay background). If the pattern has no holes, every block is
    #    already correct and nothing needs stamping.
    if carved:
        targets = [(i, j) for i in range(h) for j in range(w)
                   if pat[i][j] and (i, j) != (i0, j0)]
        if targets:
            ops.append(29)
            sels.append([i0 * h, j0 * w, h - 1, w - 1])
            for (i, j) in targets:
                if bgc == 0:
                    # clipboard carries only the fgc cells; clear the solid block first
                    ops.append(0)
                    sels.append([i * h, j * w, h - 1, w - 1])
                ops.append(30)
                sels.append([i * h, j * w, 0, 0])

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
