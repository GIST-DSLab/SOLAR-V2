"""
ARC Task: 6e19193c (RE-ARC) — LLM-generated grid_maker
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
    # Only the background is a real colour role; each object's colour is free
    # (the ray takes on whatever colour its object has).
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    v = int(round(random.uniform(min(lo, hi), max(lo, hi))))
    _ = d
    return max(a, min(b, v))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]
    dirs = (((0, 0), (-1, -1)), ((0, 1), (-1, 1)), ((1, 0), (1, -1)), ((1, 1), (1, 1)))
    base = {(0, 0), (1, 0), (0, 1), (1, 1)}

    def shoot(start, d):
        return {(start[0] + k * d[0], start[1] + k * d[1]) for k in range(43)}

    cands = []
    for corner, d in dirs:
        ci = base - {corner}
        co = (base | shoot(corner, d)) - {corner}
        cands.append((ci, co))

    h = _unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = _unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    num = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 8)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    fullinds = {(r, c) for r in range(h) for c in range(w)}
    inds = set(fullinds)
    kk, tr = 0, 0
    maxtrials = num * 4
    while kk < num and tr < maxtrials:
        if not inds:
            break
        loc = random.choice(sorted(inds))
        obji, objo = random.choice(cands)
        obji = {(r + loc[0], c + loc[1]) for r, c in obji}
        objo = {(r + loc[0], c + loc[1]) for r, c in objo} & fullinds
        if objo.issubset(inds) and obji.issubset(objo):
            col = random.choice(remcols)
            for r, c in obji:
                gi[r][c] = col
            for r, c in objo:
                go[r][c] = col
            nb = set()
            for r, c in obji:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb.add((r + dr, c + dc))
            inds = (inds - objo) - nb
            kk += 1
        tr += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # Background is the canvas colour; objects (3 cells each, <= hw/8 of them)
    # never reach the majority.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Read (shared with the golf entry, which rotates the grid four times and
    # extends one diagonal direction per pass): each L-tromino (a 2x2 block
    # missing one corner) fires a diagonal LINE of its colour out of its missing
    # corner to the grid edge. Rays never cross, and they only cover background.
    # Where this differs from the maker: the unit here is the WHOLE RAY (one
    # line, one Color op), and the work is grouped BY DIRECTION, one pass per
    # diagonal, as in the golf reading, instead of object-by-object and
    # cell-by-cell.

    seen = np.zeros((hi, wi), dtype=bool)
    objs = []  # (missing corner, (dy, dx), colour)
    for sr in range(hi):
        for sc in range(wi):
            if seen[sr, sc] or I[sr, sc] == bgc:
                continue
            col = int(I[sr, sc])
            stack = [(sr, sc)]
            seen[sr, sc] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) != 3:
                continue
            r0 = min(p[0] for p in comp)
            c0 = min(p[1] for p in comp)
            if max(p[0] for p in comp) - r0 != 1 or max(p[1] for p in comp) - c0 != 1:
                continue
            cells = set(comp)
            missing = [(y, x) for y in (r0, r0 + 1) for x in (c0, c0 + 1) if (y, x) not in cells]
            if len(missing) != 1:
                continue
            my, mx = missing[0]
            d = (-1 if my == r0 else 1, -1 if mx == c0 else 1)
            objs.append(((my, mx), d, col))

    ops, sels = [], []
    # One pass per diagonal direction (UL, UR, DR, DL: one quarter-turn each).
    for d in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        for (my, mx), od, col in objs:
            if od != d:
                continue
            ray = []
            y, x = my + d[0], mx + d[1]
            while 0 <= y < hi and 0 <= x < wi:
                ray.append((y, x))
                y += d[0]
                x += d[1]
            if not ray:
                continue  # missing corner sits on the border: nothing to shoot
            ops.append(col)
            sels.append(sel_of(ray))

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
                        f"num_examples+1 ({num_examples + 1}) for task 6e19193c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6e19193c"
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
                                f"for task 6e19193c"
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
                    f"Failed to build a complete episode for task 6e19193c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6e19193c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
