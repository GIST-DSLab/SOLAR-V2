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
from collections import Counter
import numpy as np


def sample_colors(num_examples=None) -> dict:
    """Fix the whole palette for the episode.

    Colors are drawn from 1..9 only: every clipboard/object op in ARCLE treats 0 as
    "nothing there", and this task copies and slides the wallpaper as a whole, so no
    cell of the wallpaper may be 0.

    The 4 rotations of the scene are the discrete structural cases (they decide which
    corner stays uncovered and therefore which way the wallpaper slides), so they are
    planned per instance and all four are shown when there are enough example slots.
    """
    pool = list(range(1, 10))
    random.shuffle(pool)
    bgc = pool[0]            # canvas color (entirely covered by the wallpaper)
    tric = pool[1]           # color of the opaque L-shaped band
    ccols_pool = pool[2:]    # palette the wallpaper tile is drawn from

    rots = [0, 1, 2, 3]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(rots):
        examples = [{"rot": r} for r in rots]
        examples += [{"rot": random.choice(rots)} for _ in range(n_ex - len(rots))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(rots, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test case is one of the shown cases
    return {"bgc": bgc, "tric": tric, "ccols_pool": ccols_pool, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, tric, ccols_pool, rot=None) -> dict:
    if rot is None:
        rot = random.randrange(4)

    def uni(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    def cyc_period(seq):
        n = len(seq)
        for p in range(1, n + 1):
            if n % p:
                continue
            if all(seq[i] == seq[(i + p) % n] for i in range(n)):
                return p
        return n

    def rot_cw(g, k):
        for _ in range(k % 4):
            g = [list(r) for r in zip(*g[::-1])]
        return g

    hmax = min(30, max(10, max_h))
    wmax = min(30, max(10, max_w))
    h = uni(diff_lb, diff_ub, (10, hmax))
    w = uni(diff_lb, diff_ub, (10, wmax))
    vp = uni(diff_lb, diff_ub, (2, h // 2 - 1))
    hp = uni(diff_lb, diff_ub, (2, w // 2 - 1))
    numc = uni(diff_lb, diff_ub, (2, min(len(ccols_pool), max(2, hp * vp))))
    ccols = list(ccols_pool[:numc])

    # base tile: require its true minimal periods to be exactly (vp, hp), so the
    # one-cell slide of the wallpaper is always a visible change.
    tile = None
    for _ in range(500):
        cand = [[random.choice(ccols) for _ in range(hp)] for _ in range(vp)]
        if len({v for row in cand for v in row}) < 2:
            continue
        if cyc_period([tuple(r) for r in cand]) != vp:
            continue
        if cyc_period([tuple(cand[i][j] for i in range(vp)) for j in range(hp)]) != hp:
            continue
        tile = cand
        break
    if tile is None:
        tile = [[ccols[0]] * hp for _ in range(vp)]
        tile[0][0] = ccols[1]

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            gi[r][c] = tile[r % vp][c % hp]          # wallpaper
            go[r][c] = tile[r % vp][(c - 1) % hp]    # same wallpaper, slid one cell

    ioffs = uni(diff_lb, diff_ub, (1, h - 2 * vp))
    joffs = uni(diff_lb, diff_ub, (1, w - 2 * hp))
    for r in range(ioffs):
        for c in range(w):
            gi[r][c] = tric
    for c in range(joffs):
        for r in range(h):
            gi[r][c] = tric

    gi = rot_cw(gi, rot)
    go = rot_cw(go, rot)
    return {"input": tuple(tuple(r) for r in gi), "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    """A periodic wallpaper is covered along two adjacent edges by an opaque L-shaped
    band.  Rule: extend the wallpaper over the band, then slide the whole wallpaper one
    cell toward the corner that stayed uncovered.

    Trajectory: period-sized copy/paste strips grow the wallpaper across the band, one
    Move slides the finished wallpaper, and the line the slide exposes is refilled from
    the line one period away.  Every selection below is exactly the full rectangle it
    names (strips, whole grid, single line, paste origin), so the bbox form is used.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # --- read the scene: 3 of the 4 corners are band color, one corner is free
    corner_pos = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
    ccols = [int(I[r, c]) for r, c in corner_pos]
    tric = Counter(ccols).most_common(1)[0][0]
    free = [i for i, v in enumerate(ccols) if v != tric]
    fi = free[0] if free else 0
    sr, sc = [(0, -1), (-1, 0), (0, 1), (1, 0)][fi]   # slide toward the free corner

    band_rows = [r for r in range(h) if all(int(I[r, c]) == tric for c in range(w))]
    band_cols = [c for c in range(w) if all(int(I[r, c]) == tric for r in range(h))]
    top = 0 in band_rows
    left = 0 in band_cols
    vr0, vr1 = (len(band_rows), h - 1) if top else (0, h - 1 - len(band_rows))
    vc0, vc1 = (len(band_cols), w - 1) if left else (0, w - 1 - len(band_cols))

    V = I[vr0:vr1 + 1, vc0:vc1 + 1]                   # the still-visible wallpaper
    Hv, Wv = V.shape
    pv = Hv
    for p in range(1, Hv + 1):
        if all(V[r, c] == V[r + p, c] for r in range(Hv - p) for c in range(Wv)):
            pv = p
            break
    ph = Wv
    for p in range(1, Wv + 1):
        if all(V[r, c] == V[r, c + p] for r in range(Hv) for c in range(Wv - p)):
            ph = p
            break
    # clipboard ops treat 0 as "nothing": if the wallpaper itself uses color 0, clear
    # each destination strip first so its 0 cells are already correct when pasting.
    has_zero = bool((V == 0).any())

    # --- 1. grow the wallpaper across the horizontal band, one period-tall strip at a
    #        time, over the columns that are still visible.
    if top:
        d = vr0 - pv
        while d > -pv:
            lo = max(d, 0)
            ln = d + pv - lo
            src = lo + pv
            ops.append(28 if src >= vr0 else 29)      # CopyI while source is untouched
            sels.append([src, vc0, ln - 1, vc1 - vc0])   # full rectangle: the strip
            if has_zero:
                ops.append(0)
                sels.append([lo, vc0, ln - 1, vc1 - vc0])  # full rectangle: destination
            ops.append(30)
            sels.append([lo, vc0, 0, 0])              # paste origin
            d -= pv
    else:
        d = vr1 + 1
        while d <= h - 1:
            ln = min(pv, h - d)
            src = d - pv
            ops.append(28 if src + ln - 1 <= vr1 else 29)
            sels.append([src, vc0, ln - 1, vc1 - vc0])
            if has_zero:
                ops.append(0)
                sels.append([d, vc0, ln - 1, vc1 - vc0])
            ops.append(30)
            sels.append([d, vc0, 0, 0])
            d += pv

    # --- 2. grow it sideways across the vertical band, one period-wide strip at a time,
    #        now over the full height (the rows were completed in step 1).
    if left:
        d = vc0 - ph
        while d > -ph:
            lo = max(d, 0)
            ln = d + ph - lo
            src = lo + ph
            ops.append(29)                            # source includes cells just made
            sels.append([0, src, h - 1, ln - 1])      # full rectangle: the strip
            if has_zero:
                ops.append(0)
                sels.append([0, lo, h - 1, ln - 1])
            ops.append(30)
            sels.append([0, lo, 0, 0])
            d -= ph
    else:
        d = vc1 + 1
        while d <= w - 1:
            ln = min(ph, w - d)
            src = d - ph
            ops.append(29)
            sels.append([0, src, h - 1, ln - 1])
            if has_zero:
                ops.append(0)
                sels.append([0, d, h - 1, ln - 1])
            ops.append(30)
            sels.append([0, d, 0, 0])
            d += ph

    # --- 3. slide the whole, now complete, wallpaper one cell toward the free corner.
    #        Selection is the entire grid rectangle - every cell belongs to the object.
    ops.append({(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(sr, sc)])
    sels.append([0, 0, h - 1, w - 1])

    # --- 4. the line the slide exposed repeats the line one period away
    if sc != 0:
        e = w - 1 if sc < 0 else 0
        s_col = e - ph if e > 0 else e + ph
        ops.append(29)
        sels.append([0, s_col, h - 1, 0])             # full rectangle: one column
        ops.append(30)
        sels.append([0, e, 0, 0])
    else:
        e = h - 1 if sr < 0 else 0
        s_row = e - pv if e > 0 else e + pv
        ops.append(29)
        sels.append([s_row, 0, 0, w - 1])             # full rectangle: one row
        ops.append(30)
        sels.append([e, 0, 0, 0])

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
