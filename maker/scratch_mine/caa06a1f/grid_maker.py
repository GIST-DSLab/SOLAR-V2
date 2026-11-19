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
    pool = list(range(10))
    random.shuffle(pool)
    bgc = pool[0]            # canvas colour (fully covered by the wallpaper)
    tric = pool[1]           # colour of the opaque L-shaped band
    ccols_pool = pool[2:]    # palette for the wallpaper tile
    rots = [0, 1, 2, 3]      # which corner stays uncovered -> shift direction
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(rots):
        examples = [{"rot": r} for r in rots]
        examples += [{"rot": random.choice(rots)} for _ in range(n_ex - len(rots))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(rots, n_ex)]
    plan = examples + [dict(random.choice(examples))]
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

    # The solution extends the wallpaper over a canvas one line larger than the
    # grid, so each side is kept <= 29 (ARCLE's canvas tops out at 30).
    hmax = min(29, max(10, max_h))
    wmax = min(29, max(10, max_w))
    h = uni(diff_lb, diff_ub, (10, hmax))
    w = uni(diff_lb, diff_ub, (10, wmax))
    vp = uni(diff_lb, diff_ub, (2, h // 2 - 1))
    hp = uni(diff_lb, diff_ub, (2, w // 2 - 1))
    numc = uni(diff_lb, diff_ub, (2, min(len(ccols_pool), max(2, hp * vp))))
    ccols = list(ccols_pool[:numc])

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
            gi[r][c] = tile[r % vp][c % hp]
            go[r][c] = tile[r % vp][(c - 1) % hp]

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


def derive_operations(I, O, examples=None):
    """Reading: the visible wallpaper is a translationally symmetric pattern.  Extend
    it (its orbit under the period translations) over a canvas ONE LINE LARGER than
    the grid along the shift axis, then cut that extra line away so the pattern
    lands shifted one cell toward the corner the band left uncovered.

      shift left/up   : grow canvas by one line at the far end, extend, crop off line 0
      shift right/down: grow canvas by one line, push the input one cell into the new
                        line (making room at the front), extend, crop off the far line
    Extension = period-sized strips copied from the strip one period closer to the
    visible pattern and pasted next to it.  Every selection is a full rectangle
    (canvas, strip, paste origin) so the bbox form is used.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # band colour = the colour of three corners; the fourth corner is uncovered
    corner_pos = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
    ccv = [int(I[r, c]) for r, c in corner_pos]
    tric = Counter(ccv).most_common(1)[0][0]
    free = [i for i, v in enumerate(ccv) if v != tric]
    fi = free[0] if free else 0
    sr, sc = [(0, -1), (-1, 0), (0, 1), (1, 0)][fi]

    band_rows = [r for r in range(h) if all(int(I[r, c]) == tric for c in range(w))]
    band_cols = [c for c in range(w) if all(int(I[r, c]) == tric for r in range(h))]
    top = 0 in band_rows
    left = 0 in band_cols
    vr0, vr1 = (len(band_rows), h - 1) if top else (0, h - 1 - len(band_rows))
    vc0, vc1 = (len(band_cols), w - 1) if left else (0, w - 1 - len(band_cols))

    V = I[vr0:vr1 + 1, vc0:vc1 + 1]
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

    # --- 1. enlarge the canvas by one line along the shift axis
    H, W = (h + 1, w) if sr != 0 else (h, w + 1)
    ops.append(33); sels.append([0, 0, H - 1, W - 1])        # whole new canvas
    cur = np.zeros((H, W), dtype=int)
    cur[:h, :w] = I

    # --- 2. positive shift: push the input one cell into the new line (room at front)
    dr = dc = 0
    if sr > 0 or sc > 0:
        dr, dc = max(sr, 0), max(sc, 0)
        ops.append(21 if dr else 22); sels.append([0, 0, h - 1, w - 1])  # whole input
        nxt = np.zeros_like(cur)
        nxt[dr:dr + h, dc:dc + w] = cur[:h, :w]
        cur = nxt
    vr0, vr1, vc0, vc1 = vr0 + dr, vr1 + dr, vc0 + dc, vc1 + dc

    def wall(r, c):
        return int(V[(r - vr0) % pv, (c - vc0) % ph])

    clip = {"key": None}

    def strip(r0, nr, c0, nc, sr0, sc0):
        """Fill dest rect (r0,c0,nr,nc) with a copy of the same-size rect at (sr0,sc0)."""
        tgt = np.array([[wall(r, c) for c in range(c0, c0 + nc)]
                        for r in range(r0, r0 + nr)])
        dst = cur[r0:r0 + nr, c0:c0 + nc]
        if np.array_equal(dst, tgt):
            return
        if np.any((tgt == 0) & (dst != 0)):          # Paste can't write 0: clear first
            ops.append(0); sels.append([r0, c0, nr - 1, nc - 1])
            dst[:] = 0
        if np.any(tgt != 0):
            key = (sr0, sc0, nr, nc)
            if clip["key"] != key:
                ops.append(29); sels.append([sr0, sc0, nr - 1, nc - 1])
            ops.append(30); sels.append([r0, c0, 0, 0])
            dst[:] = np.where(tgt != 0, tgt, dst)
            clip["key"] = (r0, c0, nr, nc)          # pasted strip == clipboard content

    nc = vc1 - vc0 + 1
    # --- 3a. extend vertically (over the visible columns)
    d = vr0 - pv
    while d + pv > 0:
        lo = max(d, 0); ln = d + pv - lo
        strip(lo, ln, vc0, nc, lo + pv, vc0)
        d -= pv
    d = vr1 + 1
    while d <= H - 1:
        ln = min(pv, H - d)
        strip(d, ln, vc0, nc, d - pv, vc0)
        d += pv
    # --- 3b. extend horizontally (full height)
    d = vc0 - ph
    while d + ph > 0:
        lo = max(d, 0); ln = d + ph - lo
        strip(0, H, lo, ln, 0, lo + ph)
        d -= ph
    d = vc1 + 1
    while d <= W - 1:
        ln = min(ph, W - d)
        strip(0, H, d, ln, 0, d - ph)
        d += ph

    # --- 4. cut the extra line away: the pattern now sits shifted by one
    if sr < 0:
        ops.append(33); sels.append([1, 0, h - 1, w - 1])
    elif sc < 0:
        ops.append(33); sels.append([0, 1, h - 1, w - 1])
    else:
        ops.append(33); sels.append([0, 0, h - 1, w - 1])

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
