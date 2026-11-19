"""
ARC Task: 31aa019c (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, max(lo_i, hi_i)))


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    while True:
        h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
        w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
        h = min(h, max_h)
        w = min(w, max_w)
        canv = [[bgc] * w for _ in range(h)]
        inds = [(i, j) for i in range(h) for j in range(w)]
        mp = (h * w) // 2 - 1
        ncols = _unifint(diff_lb, diff_ub, (2, max(2, min(9, mp // 2 - 1))))
        chcols = random.sample(cols, ncols)
        trgcol = chcols[0]
        chcols = chcols[1:]
        dic = {c: set() for c in chcols}
        nnoise = _unifint(diff_lb, diff_ub, (2 * (ncols - 1), max(2 * (ncols - 1), mp)))
        locc = random.choice(inds)
        inds.remove(locc)
        if nnoise > len(inds):
            continue
        noise = random.sample(inds, nnoise)
        for c in chcols:
            ij = random.choice(inds)
            dic[c].add(ij)
            inds.remove(ij)
        for c in chcols:
            ij = random.choice(inds)
            dic[c].add(ij)
            inds.remove(ij)
        for ij in noise:
            c = random.choice(chcols)
            dic[c].add(ij)
            if ij in inds:
                inds.remove(ij)
        gi = [row[:] for row in canv]
        for c, ss in dic.items():
            for (i, j) in ss:
                gi[i][j] = c
        gi[locc[0]][locc[1]] = trgcol
        cnt = Counter(v for row in gi for v in row)
        if sum(1 for c in cnt if cnt[c] == 1) == 1:
            break
    cnt = Counter(v for row in gi for v in row)
    lc = min(cnt.keys(), key=lambda k: (cnt[k], k))
    mc = max(cnt.keys(), key=lambda k: cnt[k])
    loc = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == lc][0]
    go = [[mc] * w for _ in range(h)]
    go[loc[0]][loc[1]] = lc
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            a, b = loc[0] + di, loc[1] + dj
            if 0 <= a < h and 0 <= b < w:
                go[a][b] = 2
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading: the single unique pixel is KEPT in place untouched; everything
    # else is wiped to background; then a ring of 2 is drawn around the kept pixel.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cnt = Counter(I.flatten().tolist())
    trg = min(cnt.keys(), key=lambda k: (cnt[k], k))      # leastcolor (unique pixel)
    bgc = max(cnt.keys(), key=lambda k: cnt[k])           # mostcolor
    pos = np.argwhere(I == trg)
    r, c = int(pos[0][0]), int(pos[0][1])

    ops, sels = [], []
    grid = I.copy()

    # 1. Erase everything except the unique pixel (one selection: all cells but it)
    others = [(i, j) for i in range(hi) for j in range(wi) if (i, j) != (r, c)]
    if any(grid[i, j] != bgc for i, j in others):
        ops.append(int(bgc)); sels.append(sel_of(others))
        for i, j in others:
            grid[i, j] = bgc

    # 2. Draw the ring of 8 neighbours (clipped to grid) in color 2
    ring = [(r + di, c + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
            if (di or dj) and 0 <= r + di < hi and 0 <= c + dj < wi]
    if any(grid[i, j] != 2 for i, j in ring):
        ops.append(2); sels.append(sel_of(ring))
        for i, j in ring:
            grid[i, j] = 2

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 31aa019c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 31aa019c"
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
                                f"for task 31aa019c"
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
                    f"Failed to build a complete episode for task 31aa019c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"31aa019c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
