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
import numpy as np
import random
from collections import Counter

from maker.sel_helpers import sel_of

VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    tric = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "tric": tric, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, tric, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    def _uni(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if rot % 2 == 1:
        cap_h, cap_w = max_w, max_h
    else:
        cap_h, cap_w = max_h, max_w
    h = max(_uni(diff_lb, diff_ub, (min(10, cap_h), cap_h)), 6)
    w = max(_uni(diff_lb, diff_ub, (min(10, cap_w), cap_w)), 6)
    vp = _uni(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
    hp = _uni(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
    vp = max(2, min(vp, (h - 1) // 2))
    hp = max(2, min(hp, (w - 1) // 2))

    remcols = [c for c in range(10) if c != bgc and c != tric]
    numc = _uni(diff_lb, diff_ub, (2, min(8, max(2, hp * vp))))
    numc = max(2, min(numc, len(remcols)))
    ccols = random.sample(remcols, numc)
    tile = [[random.choice(ccols) for _ in range(hp)] for _ in range(vp)]

    gi = [[tile[r % vp][c % hp] for c in range(w)] for r in range(h)]
    go = [[tile[r % vp][(c - 1) % hp] for c in range(w)] for r in range(h)]

    ioffs = _uni(diff_lb, diff_ub, (1, max(1, h - 2 * vp)))
    joffs = _uni(diff_lb, diff_ub, (1, max(1, w - 2 * hp)))
    ioffs = min(ioffs, h - 2 * vp) if h - 2 * vp >= 1 else 1
    joffs = min(joffs, w - 2 * hp) if w - 2 * hp >= 1 else 1
    for a in range(ioffs):
        for c in range(w):
            gi[a][c] = tric
    for b in range(joffs):
        for r in range(h):
            gi[r][b] = tric

    ai = np.array(gi, dtype=int)
    ao = np.array(go, dtype=int)
    if rot:
        ai = np.rot90(ai, k=-rot)
        ao = np.rot90(ao, k=-rot)
    return {"input": ai.tolist(), "output": ao.tolist()}


def derive_operations(I, O):
    """Reading: the answer is ONE period tile of the wallpaper laid edge to edge
    over the whole grid, with its phase pushed one cell toward the uncovered
    corner.  So the unit is the tile, not the band: find the period, pick the
    tile in the visible wallpaper whose phase is already the shifted one
    (it sits one cell against the shift direction from a period boundary),
    CopyI it once, and paste it at every tile origin (i*pv, j*ph) of the grid.
    Everything is read from I; O is only used for the final Submit size."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # occluding band colour = colour of the three covered corners
    corner_pos = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
    ccols = [int(I[r, c]) for r, c in corner_pos]
    tric = Counter(ccols).most_common(1)[0][0]
    free = [i for i, v in enumerate(ccols) if v != tric]
    fi = free[0] if free else 0
    # origin free -> LEFT, ur free -> UP, lr free -> RIGHT, ll free -> DOWN
    dr, dc = [(0, -1), (-1, 0), (0, 1), (1, 0)][fi]

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
        if np.array_equal(V[p:, :], V[:Hv - p, :]):
            pv = p
            break
    ph = Wv
    for p in range(1, Wv + 1):
        if np.array_equal(V[:, p:], V[:, :Wv - p]):
            ph = p
            break

    # source tile: top-left (a, b) inside the visible wallpaper with
    # a = -dr (mod pv), b = -dc (mod ph): its content is the shifted tile
    a = vr0 + ((-dr - vr0) % pv)
    b = vc0 + ((-dc - vc0) % ph)
    if a + pv - 1 > vr1:
        a -= pv
    if b + ph - 1 > vc1:
        b -= ph
    tile = I[a:a + pv, b:b + ph]

    ops.append(28)                       # CopyI the full period tile (rectangle)
    sels.append([a, b, pv - 1, ph - 1])

    G = I.copy()                         # the working grid as it stands
    for r0 in range(0, h, pv):
        for c0 in range(0, w, ph):
            rh = min(pv, h - r0)
            cw = min(ph, w - c0)
            t = tile[:rh, :cw]
            cur = G[r0:r0 + rh, c0:c0 + cw]
            if np.array_equal(cur, t):
                continue                 # this tile slot already shows the tile
            # Paste never writes 0: if the tile carries 0 where the grid does
            # not, lay a 0 base over the whole slot (full rectangle) first.
            if np.any((t == 0) & (cur != 0)):
                ops.append(0)
                sels.append([r0, c0, rh - 1, cw - 1])
                cur[:, :] = 0
            ops.append(30)               # Paste at the slot origin
            sels.append([r0, c0, 0, 0])
            cur[t != 0] = t[t != 0]

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
