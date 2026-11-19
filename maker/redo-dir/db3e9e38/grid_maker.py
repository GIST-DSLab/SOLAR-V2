"""
ARC Task: db3e9e38 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:                       # fallback: mask given as explicit cell list
    def sel_of(cells):
        cells = list(cells)
        if not cells:
            return {"cells": []}
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# The only discrete structural variant: which edge the bar is glued to
# (identity / rot90 / rot180 / rot270 of the canonical "bar hanging from the top" grid).
ROT_VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]      # 8 is reserved for the alternating bands
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROT_VARIANTS):
        examples = [dict(v) for v in ROT_VARIANTS]
        examples += [dict(random.choice(ROT_VARIANTS)) for _ in range(n_ex - len(ROT_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ROT_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test orientation was shown
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    # an odd rot swaps the axes -> sample the canonical grid so the rotated one still fits
    lim_h, lim_w = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = _unifint(diff_lb, diff_ub, (3, max(3, lim_h)))
    w = _unifint(diff_lb, diff_ub, (3, max(3, lim_w)))
    barth = _unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    barth = max(1, min(barth, w - 2))
    loci = _unifint(diff_lb, diff_ub, (1, h - 2))
    locj = random.randint(1, w - barth - 1)

    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    gi[0:loci + 1, locj:locj + barth] = fgc

    for k in range(0, 32):
        plen = loci + 1 - k
        if plen <= 0:
            break
        col = fgc if k % 2 == 0 else 8
        for sgn in (1, -1):
            j0 = locj + sgn * k * barth
            jj0, jj1 = max(0, j0), min(w, j0 + barth)
            if jj0 < jj1:
                go[0:plen, jj0:jj1] = col
            if k == 0:
                break

    gi = np.rot90(gi, k=rot)
    go = np.rot90(go, k=rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # --- read the scene out of I -------------------------------------------------
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    pts = np.argwhere(I != bgc)
    r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    r1, c1 = int(pts[:, 0].max()), int(pts[:, 1].max())
    fgc = int(I[r0, c0])
    MARK = 8

    # the bar is glued to exactly one edge; bands step sideways along that edge
    if r0 == 0:
        axis, anchor = 'col', 'top'
    elif r1 == h - 1:
        axis, anchor = 'col', 'bottom'
    elif c0 == 0:
        axis, anchor = 'row', 'left'
    else:
        axis, anchor = 'row', 'right'

    if axis == 'col':
        p0, p1, pmax = c0, c1, w - 1        # lateral = columns
        plen = r1 - r0 + 1                  # length perpendicular to the glued edge
    else:
        p0, p1, pmax = r0, r1, h - 1        # lateral = rows
        plen = c1 - c0 + 1
    b = p1 - p0 + 1                         # bar thickness = lateral step
    A = p0 + p1                             # mirror axis: lateral q  <->  A - q

    def perp_range(k):
        if anchor == 'top':
            return 0, r1 - k
        if anchor == 'bottom':
            return r0 + k, h - 1
        if anchor == 'left':
            return 0, c1 - k
        return c0 + k, w - 1

    def rect_cells(lat_lo, lat_hi, pa, pb):
        if axis == 'col':
            return [(r, c) for r in range(pa, pb + 1) for c in range(lat_lo, lat_hi + 1)]
        return [(r, c) for r in range(lat_lo, lat_hi + 1) for c in range(pa, pb + 1)]

    G = I.copy()

    # draw the branch that has more room; its mirror then covers the other branch whole
    s = 1 if (pmax - p1) >= p0 else -1

    # --- 1. paint the staircase bands outward from the bar, one band = one rectangle ---
    k = 1
    drawn = 0
    while plen - k >= 1:
        lat_lo, lat_hi = p0 + s * k * b, p1 + s * k * b
        if lat_lo > pmax or lat_hi < 0:
            break
        lat_lo, lat_hi = max(0, lat_lo), min(pmax, lat_hi)
        pa, pb = perp_range(k)
        cells = rect_cells(lat_lo, lat_hi, pa, pb)          # exactly a full rectangle
        col = fgc if k % 2 == 0 else MARK
        ops.append(col)
        sels.append(sel_of(cells))
        for (rr, cc) in cells:
            G[rr, cc] = col
        drawn += 1
        k += 1

    if drawn:
        # --- 2. reflect that branch onto the other side of the bar -------------------
        if s > 0:
            src_lo, src_hi = p0, min(A, pmax)
        else:
            src_lo, src_hi = max(0, A - pmax), p1
        dst_lo, dst_hi = A - src_hi, A - src_lo
        pa, pb = perp_range(0)                               # the bar's full depth

        src_cells = rect_cells(src_lo, src_hi, pa, pb)       # full rectangle
        dst_cells = rect_cells(dst_lo, dst_hi, pa, pb)       # full rectangle
        dst_r, dst_c = (pa, dst_lo) if axis == 'col' else (dst_lo, pa)

        ops.append(29)                                       # CopyO: the branch as drawn
        sels.append(sel_of(src_cells))

        if any(G[rr, cc] == 0 for (rr, cc) in src_cells):
            # Paste never writes colour-0 cells, so clear the target first
            ops.append(0)
            sels.append(sel_of(dst_cells))
            for (rr, cc) in dst_cells:
                G[rr, cc] = 0

        ops.append(30)                                       # Paste at the target corner
        sels.append(sel_of([(dst_r, dst_c)]))
        patch = np.array([[G[rr, cc] for cc in range(src_lo if axis != 'col' else 0, 0)]]) \
            if False else None
        src_arr = (G[pa:pb + 1, src_lo:src_hi + 1] if axis == 'col'
                   else G[src_lo:src_hi + 1, pa:pb + 1]).copy()
        if axis == 'col':
            G[pa:pb + 1, dst_lo:dst_hi + 1] = src_arr
        else:
            G[dst_lo:dst_hi + 1, pa:pb + 1] = src_arr

        ops.append(26 if axis == 'col' else 27)              # mirror it in place
        sels.append(sel_of(dst_cells))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
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
                                f"for task db3e9e38"
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
                    f"Failed to build a complete episode for task db3e9e38 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"db3e9e38-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
