"""
ARC Task: caa06a1f (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    tric = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"tric": tric, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, tric, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    vp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    hp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    remcols = [c for c in cols if c != tric]
    bgc = choice(remcols)
    numc = unifint(diff_lb, diff_ub, (2, min(8, max(2, hp * vp))))
    ccols = sample(remcols, numc)
    obj = {(choice(ccols), ij) for ij in asindices(canvas(-1, (vp, hp)))}
    go = canvas(bgc, (h, w))
    gi = canvas(bgc, (h, w))
    for a in range(-vp, h + 1, vp):
        for b in range(-hp, w + 1, hp):
            go = paint(go, shift(obj, (a, b + 1)))
    for a in range(-vp, h + 1, vp):
        for b in range(-hp, w + 1, hp):
            gi = paint(gi, shift(obj, (a, b)))
    ioffs = unifint(diff_lb, diff_ub, (1, h - 2 * vp))
    joffs = unifint(diff_lb, diff_ub, (1, w - 2 * hp))
    for a in range(ioffs):
        gi = fill(gi, tric, connect((a, 0), (a, w - 1)))
    for b in range(joffs):
        gi = fill(gi, tric, connect((0, b), (h - 1, b)))
    rotf = (identity, rot90, rot180, rot270)[rot]
    return {"input": rotf(gi), "output": rotf(go)}


def derive_operations(I, O):
    """Reading: the output is the visible wallpaper read one cell later, repeated
    out to the edges.  So: slide the visible wallpaper block one cell toward the
    uncovered corner FIRST (clipping its far line, vacating the line beside the
    band), then wipe the opaque band (only if the wallpaper needs 0s there), then
    tile the shifted block outward over the vacated line + band, one period strip
    at a time, first along rows, then along columns.  Strips are aligned to the
    grid edge so every paste lands fully inside the grid; one copy per axis."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    corner_pos = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
    ccols = [int(I[r, c]) for r, c in corner_pos]
    tric = Counter(ccols).most_common(1)[0][0]
    fi = [i for i, v in enumerate(ccols) if v != tric][0]
    sr, sc = [(0, -1), (-1, 0), (0, 1), (1, 0)][fi]
    band_top = fi in (2, 3)
    band_left = fi in (1, 2)

    full_r = [all(I[r, :] == tric) for r in range(h)]
    full_c = [all(I[:, c] == tric) for c in range(w)]
    nbr, nbc = sum(full_r), sum(full_c)
    vr0, vr1 = (nbr, h - 1) if band_top else (0, h - 1 - nbr)
    vc0, vc1 = (nbc, w - 1) if band_left else (0, w - 1 - nbc)
    V = I[vr0:vr1 + 1, vc0:vc1 + 1]
    Hv, Wv = V.shape
    pv = next(p for p in range(1, Hv + 1)
              if np.array_equal(V[p:, :], V[:Hv - p, :]))
    ph = next(p for p in range(1, Wv + 1)
              if np.array_equal(V[:, p:], V[:, :Wv - p]))

    def ow(r, c):  # wallpaper value after the one-cell slide
        return int(V[(r - sr - vr0) % pv, (c - sc - vc0) % ph])

    # 1. slide the whole visible wallpaper block one cell (full rectangle selection)
    ops.append({(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(sr, sc)])
    sels.append([vr0, vc0, vr1 - vr0, vc1 - vc0])
    ar0, ar1 = max(vr0, vr0 + sr), min(vr1, vr1 + sr)
    ac0, ac1 = max(vc0, vc0 + sc), min(vc1, vc1 + sc)

    # 2. wipe the band (all tric cells) when the wallpaper must show 0 there
    band = [(r, c) for r in range(h) for c in range(w) if I[r, c] == tric]
    if tric != 0 and any(ow(r, c) == 0 for r, c in band):
        ops.append(0)
        sels.append(sel_of(band))

    # 3. tile outward: rows first (within valid cols), then columns (full height)
    def tile(lo, hi, n, p, axis, cross0, cross1):
        if lo > 0:
            poss = list(range(0, lo, p))[::-1]
            s = lo + ((-lo) % p)
        elif hi < n - 1:
            poss = [x for x in range(n - p, -p, -p) if x + p - 1 > hi][::-1]
            s = lo + ((n - lo) % p)
        else:
            return
        todo = []
        for x in poss:
            fill_idx = [k for k in range(x, x + p) if k < lo or k > hi]
            if axis == 0:
                vals = [ow(k, c) for k in fill_idx for c in range(cross0, cross1 + 1)]
            else:
                vals = [ow(r, k) for k in fill_idx for r in range(cross0, cross1 + 1)]
            if any(vals):
                todo.append(x)
        if not todo:
            return
        if axis == 0:
            ops.append(29); sels.append([s, cross0, p - 1, cross1 - cross0])
            for x in todo:
                ops.append(30); sels.append([x, cross0, 0, 0])
        else:
            ops.append(29); sels.append([cross0, s, cross1 - cross0, p - 1])
            for x in todo:
                ops.append(30); sels.append([cross0, x, 0, 0])

    tile(ar0, ar1, h, pv, 0, ac0, ac1)
    tile(ac0, ac1, w, ph, 1, 0, h - 1)

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
                        f"num_examples+1 ({num_examples + 1}) for task caa06a1f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task caa06a1f"
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
                                f"for task caa06a1f"
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
                    f"Failed to build a complete episode for task caa06a1f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"caa06a1f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
