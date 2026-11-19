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
    if bgc is None:
        bgc = random.randrange(10)
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c != bgc])
    hmax = max(2, min(5, int((max_h - 2) ** 0.5)))
    wmax = max(2, min(5, int((max_w - 2) ** 0.5)))
    h = random.randint(2, hmax)
    w = random.randint(2, wmax)
    numcd = random.randint(0, (h * w) // 2)
    numc = random.choice((numcd, h * w - numcd))
    numc = min(max(0, numc), h * w)
    inds = [(i, j) for i in range(h) for j in range(w)]
    corn = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    locs = set(random.sample(inds, numc)) | set(random.sample(corn, 3))
    gi = [[fgc if (i, j) in locs else bgc for j in range(w)] for i in range(h)]
    go = [[bgc] * (w * w) for _ in range(h * h)]
    for (a, b) in locs:
        for (i, j) in locs:
            go[a * h + i][b * w + j] = fgc
    fullh = random.randint(h * h + 2, max(h * h + 2, max_h))
    fullw = random.randint(w * w + 2, max(w * w + 2, max_w))
    g = [[bgc] * fullw for _ in range(fullh)]
    li = random.randint(1, fullh - h * h - 1)
    lj = random.randint(1, fullw - w * w - 1)
    for i in range(h * h):
        for j in range(w * w):
            g[li + i][lj + j] = gi[i // h][j // w]
    return {"input": g, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)

    # Background: the generator paints the canvas with bgc and keeps a >=1 cell
    # margin around the object, so the corner is always bgc.
    bgc = int(I[0, 0])
    fg = np.argwhere(I != bgc)
    r0, c0 = int(fg[:, 0].min()), int(fg[:, 1].min())
    r1, c1 = int(fg[:, 0].max()), int(fg[:, 1].max())
    fgc = int(I[fg[0][0], fg[0][1]])

    # The object is an h x w pattern upscaled by (h, w); at least 3 pattern corners
    # are set, so its bbox is exactly h^2 x w^2.
    H, W = r1 - r0 + 1, c1 - c0 + 1
    h = int(round(H ** 0.5))
    w = int(round(W ** 0.5))
    pat = [[I[r0 + i * h, c0 + j * w] == fgc for j in range(w)] for i in range(h)]

    ops, sels = [], []

    # 1. Crop to the upscaled object (full rectangle, background blocks included).
    #    The canvas is now the coarse layer: the pattern blown up to h^2 x w^2.
    ops.append(33)
    sels.append([r0, c0, H - 1, W - 1])
    cur = I[r0:r1 + 1, c0:c1 + 1].copy()

    # 2. Overlay the second layer: the pattern tiled at 1:1 scale. Output keeps fgc
    #    only where both layers agree, so every hole (a, b) of the small pattern
    #    knocks out the lattice {(a + k*h, b + l*w)} across the whole grid.
    #    One Color(bgc) per hole, on the lattice cells that currently hold fgc.
    for a in range(h):
        for b in range(w):
            if pat[a][b]:
                continue
            cells = [(r, c) for r in range(a, H, h) for c in range(b, W, w)
                     if cur[r, c] != bgc]
            if not cells:
                continue
            ops.append(bgc)
            sels.append(sel_of(cells))
            for r, c in cells:
                cur[r, c] = bgc

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
