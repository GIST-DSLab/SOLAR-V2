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

ROTATIONS = [0, 1, 2, 3]   # number of CW quarter turns applied to the generated pair


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTATIONS):
        ex = [{"rot": r} for r in ROTATIONS]
        ex += [{"rot": random.choice(ROTATIONS)} for _ in range(n_ex - len(ROTATIONS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTATIONS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, rot=None, **kw) -> dict:
    if rot is None:
        rot = random.choice(ROTATIONS)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    if rot in (1, 3):
        h, w = w, h            # rotated output must still fit max_h x max_w
    vp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    hp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    numc = unifint(diff_lb, diff_ub, (2, min(8, max(2, hp * vp))))
    remcols = remove(bgc, cols)
    ccols = sample(remcols, numc)
    remcols = difference(remcols, ccols)
    tric = choice(remcols)
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
    rotf = [identity, rot90, rot180, rot270][rot]
    return {"input": rotf(gi), "output": rotf(go)}


def derive_operations(I, O, examples=None):
    """Reading: the output is nothing but the wallpaper TILE laid edge-to-edge over the
    whole canvas, at a phase one cell off from where I shows it (toward the corner the
    L-band left uncovered).  So the unit is the period tile: copy one tile -- taken from
    the visible wallpaper at the shifted phase -- and paste it at every tile origin
    (i*pv, j*ph).  The band and the old phase are both simply painted over; pastes that
    overhang the right/bottom edge are clipped by the canvas.  If the tile contains 0
    (transparent to Paste) the canvas is first cleared to 0 as a base."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    corner_pos = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
    ccols = [int(I[r, c]) for r, c in corner_pos]
    tric = Counter(ccols).most_common(1)[0][0]
    free = [i for i, v in enumerate(ccols) if v != tric]
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
    pv = next(p for p in range(1, Hv + 1)
              if all(V[r, c] == V[r + p, c] for r in range(Hv - p) for c in range(Wv)))
    ph = next(p for p in range(1, Wv + 1)
              if all(V[r, c] == V[r, c + p] for r in range(Hv) for c in range(Wv - p)))

    # output cell (r, c) carries wallpaper cell (r - sr, c - sc); the tile for origin
    # (0, 0) therefore starts in I at the visible cell with that phase.
    r0 = vr0 + ((-sr - vr0) % pv)
    c0 = vc0 + ((-sc - vc0) % ph)
    tile = I[r0:r0 + pv, c0:c0 + ph]

    G = I.copy()                       # the grid as my own ops leave it
    ops.append(28); sels.append([r0, c0, pv - 1, ph - 1])   # CopyI: one full tile (rect)

    if (tile == 0).any():
        # Paste cannot write 0: lay a 0 base over the whole canvas first, if that changes
        # any cell the tile wants to be 0.
        need = any(tile[(r % pv), (c % ph)] == 0 and G[r, c] != 0
                   for r in range(h) for c in range(w))
        if need:
            ops.append(0); sels.append([0, 0, h - 1, w - 1])  # whole canvas (rect)
            G[:, :] = 0

    for ti in range(0, h, pv):
        for tj in range(0, w, ph):
            eh, ew = min(pv, h - ti), min(ph, w - tj)
            patch = tile[:eh, :ew]
            region = G[ti:ti + eh, tj:tj + ew]
            new = np.where(patch != 0, patch, region)
            if np.array_equal(new, region):
                continue               # this paste would change nothing visible
            ops.append(30); sels.append([ti, tj, 0, 0])
            G[ti:ti + eh, tj:tj + ew] = new

    if not any(o == 30 for o in ops):
        # the 0 base alone already laid the (all-0) wallpaper: no tile was ever used
        del ops[0], sels[0]
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
