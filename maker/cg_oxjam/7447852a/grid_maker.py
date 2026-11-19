"""
ARC Task: 7447852a (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(lo_i, hi_i)


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, linc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, linc)]
    numcols = random.randint(1, 7)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "linc": linc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, linc=1, ccols=None, **kwargs) -> dict:
    if ccols is None or len(ccols) == 0:
        ccols = [c for c in range(10) if c not in (4, bgc, linc)][:1]
    # pre-rotation grid is h x w; after rot90 the input is w rows x h cols
    wmax = max(2, min(8, max_h))
    w = _unifint(diff_lb, diff_ub, (2, wmax))
    hmax = max(w + 1, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (w + 1, hmax))
    g = [[bgc] * w for _ in range(h)]
    sp = (h - 1, 0)
    g[sp[0]][sp[1]] = linc
    direc = 1
    while True:
        sp = (sp[0] - 1, sp[1] + direc)
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        if sp[0] < 0 or sp[1] < 0 or sp[1] >= w or g[sp[0]][sp[1]] == linc:
            break
        g[sp[0]][sp[1]] = linc
    gi = np.rot90(np.array(g, dtype=int), k=3)  # DSL rot90 = clockwise
    H, W = gi.shape
    # bgc regions (4-connected), ordered by leftmost column
    seen = np.zeros((H, W), dtype=bool)
    regions = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or gi[r, c] != bgc:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and gi[ny, nx] == bgc:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            regions.append(cells)
    regions.sort(key=lambda cs: min(cc for _, cc in cs))
    inds = [(r, c) for r in range(H) for c in range(W) if gi[r, c] == bgc]
    ncells = _unifint(diff_lb, diff_ub, (0, len(inds)))
    for (r, c) in random.sample(inds, ncells):
        gi[r, c] = random.choice(ccols)
    go = gi.copy()
    for cells in regions[0::3]:
        for (r, c) in cells:
            go[r, c] = 4
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # The zigzag line starts at the top-left corner.
    linc = int(I[0, 0])

    # Label the strips between the line's crossings (4-connected non-line cells),
    # ordered left to right as the zigzag sweeps across the grid.
    label = -np.ones((h, w), dtype=int)
    regions = []
    for r in range(h):
        for c in range(w):
            if label[r, c] >= 0 or I[r, c] == linc:
                continue
            idx = len(regions)
            q = deque([(r, c)])
            label[r, c] = idx
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and label[ny, nx] < 0 and I[ny, nx] != linc:
                        label[ny, nx] = idx
                        q.append((ny, nx))
            regions.append(cells)
    order = sorted(range(len(regions)),
                   key=lambda i: (min(cc for _, cc in regions[i]),
                                  min(rr for rr, _ in regions[i])))
    rank = {reg: k for k, reg in enumerate(order)}
    chosen = {reg for reg in order if rank[reg] % 3 == 0}

    # Row-scan reading: walk along each row, splitting it at the line's crossings.
    # Each run between crossings is part of one strip. Paint the run if its strip
    # is one of every third strip.
    ops, sels = [], []
    for r in range(h):
        c = 0
        while c < w:
            if I[r, c] == linc:
                c += 1
                continue
            start = c
            while c < w and I[r, c] != linc:
                c += 1
            if label[r, start] in chosen:
                ops.append(4)
                sels.append(sel_of([(r, x) for x in range(start, c)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 7447852a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7447852a"
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
                                f"for task 7447852a"
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
                    f"Failed to build a complete episode for task 7447852a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7447852a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
