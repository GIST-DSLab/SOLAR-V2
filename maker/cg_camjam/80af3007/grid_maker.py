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
    bgc = random.choice(range(10))
    fgc = random.choice([c for c in range(10) if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fgc=1, **kw) -> dict:
    from random import randint, sample, choice

    def unif(lo, hi):
        if hi < lo:
            hi = lo
        d = random.uniform(diff_lb, diff_ub)
        return lo + int(round(d * (hi - lo)))

    hmax = 2
    while (hmax + 1) ** 2 + 2 <= max_h and hmax < 5:
        hmax += 1
    wmax = 2
    while (wmax + 1) ** 2 + 2 <= max_w and wmax < 5:
        wmax += 1
    h = unif(2, hmax)
    w = unif(2, wmax)
    inds = [(i, j) for i in range(h) for j in range(w)]
    numcd = unif(0, (h * w) // 2)
    numc = choice((numcd, h * w - numcd))
    numc = min(max(0, numc), h * w)
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    locs = set(sample(inds, numc)) | set(sample(corners, 3))
    pat = np.full((h, w), bgc, dtype=int)
    for (i, j) in locs:
        pat[i, j] = fgc
    go = np.full((h * h, w * w), bgc, dtype=int)
    for (i, j) in locs:
        for (a, b) in locs:
            go[i * h + a, j * w + b] = fgc
    fullh = unif(h * h + 2, max_h)
    fullw = unif(w * w + 2, max_w)
    gi = np.full((fullh, fullw), bgc, dtype=int)
    li = randint(1, fullh - h * h - 1)
    lj = randint(1, fullw - w * w - 1)
    up = np.kron(pat, np.ones((h, w), dtype=int))
    gi[li:li + h * h, lj:lj + w * w] = up
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (shared with the golf entry): the object is an h x w pattern upscaled by
    # (h, w). Reduce it to that pattern first (one sample per block), then rebuild the
    # output as pattern-of-patterns: the pattern stamped into every foreground block.
    I = np.asarray(I, dtype=int)
    # Generator floods the canvas with bgc and leaves a >=1 cell margin -> corner is bgc.
    bgc = int(I[0, 0])
    fg = np.argwhere(I != bgc)
    r0, c0 = int(fg[:, 0].min()), int(fg[:, 1].min())
    r1, c1 = int(fg[:, 0].max()), int(fg[:, 1].max())
    fgc = int(I[fg[0][0], fg[0][1]])
    H, W = r1 - r0 + 1, c1 - c0 + 1
    h = int(round(H ** 0.5))
    w = int(round(W ** 0.5))
    # Small pattern: every h-th row / every w-th column of the object (block samples).
    pat = [[I[r0 + i * h, c0 + j * w] == fgc for j in range(w)] for i in range(h)]
    ops, sels = [], []

    # 1. Downscale in place: the object's top-left h x w corner takes, at cell (i, j),
    #    the colour of block (i, j). That corner is uniform (block (0,0)), so only the
    #    cells whose block has the other colour get painted.
    corner_fg = pat[0][0]
    diff = [(r0 + i, c0 + j) for i in range(h) for j in range(w)
            if pat[i][j] != corner_fg]
    if diff:
        ops.append(bgc if corner_fg else fgc)
        sels.append(sel_of(diff))

    # 2. Keep only the reduced h x w pattern (full rectangle crop).
    ops.append(33)
    sels.append([r0, c0, h - 1, w - 1])

    # 3. Grow the canvas to h^2 x w^2 (full rectangle); fill the added cells with bgc.
    ops.append(33)
    sels.append([0, 0, H - 1, W - 1])
    added = [(r, c) for r in range(H) for c in range(W) if r >= h or c >= w]
    if bgc != 0:
        ops.append(bgc)
        sels.append(sel_of(added))

    # 4. Stamp the pattern into each block whose pattern cell is foreground.
    targets = [(i, j) for i in range(h) for j in range(w)
               if pat[i][j] and (i, j) != (0, 0)]
    has_bg_cells = any(not pat[i][j] for i in range(h) for j in range(w))
    clip_nonempty = (fgc != 0) or has_bg_cells
    if targets and clip_nonempty:
        ops.append(29)
        sels.append([0, 0, h - 1, w - 1])  # whole pattern rectangle
    if not pat[0][0]:
        # block (0,0) is a background block: clear it once the pattern is copied
        ops.append(bgc)
        sels.append([0, 0, h - 1, w - 1])  # full block rectangle
    for (i, j) in targets:
        if fgc == 0:
            # clipboard cannot carry 0: lay the block solid fgc, then stamp bgc holes
            ops.append(0)
            sels.append([i * h, j * w, h - 1, w - 1])  # full block rectangle
            if has_bg_cells:
                ops.append(30)
                sels.append([i * h, j * w, 0, 0])
        else:
            ops.append(30)
            sels.append([i * h, j * w, 0, 0])

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
