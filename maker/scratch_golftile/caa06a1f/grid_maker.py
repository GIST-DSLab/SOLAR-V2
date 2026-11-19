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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    pool = list(range(10))
    random.shuffle(pool)
    bgc = pool[0]
    tric = pool[1]
    ccols_pool = pool[2:]
    rots = [0, 1, 2, 3]
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
    hmax = min(30, max(10, max_h))
    wmax = min(30, max(10, max_w))
    if rot % 2 == 1:
        hmax, wmax = wmax, hmax
    h = unifint(diff_lb, diff_ub, (10, hmax))
    w = unifint(diff_lb, diff_ub, (10, wmax))
    vp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    hp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    numc = unifint(diff_lb, diff_ub, (2, min(8, len(ccols_pool), max(2, hp * vp))))
    ccols = list(ccols_pool[:numc])
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
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading (shared with the golf entry): the output is ONE period tile repeated on
    the lattice anchored at the grid's top-left corner -- no band, no slide.  That tile
    is the wallpaper one cell displaced toward the uncovered corner, and it can be read
    whole from the visible part of I.  So: copy that phase-shifted period tile out of
    I and stamp it at every lattice origin (r*pv, c*ph) over the whole canvas; lattice
    cells clipped by the right/bottom border get the matching top-left sub-tile.
    Since Paste never writes 0, the canvas is first cleared to 0 when the tile has 0s.
    Every selection is a full rectangle (a tile block in I, the whole grid) or a paste
    origin, so the bbox form is used.  Nothing here reads O.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # band colour = colour shared by the three covered corners; free corner -> shift
    corner_pos = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
    ccols = [int(I[r, c]) for r, c in corner_pos]
    tric = Counter(ccols).most_common(1)[0][0]
    free = [i for i, v in enumerate(ccols) if v != tric]
    fi = free[0] if free else 0
    sr, sc = [(0, -1), (-1, 0), (0, 1), (1, 0)][fi]   # content displacement

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

    # Output cell (r, c) shows wallpaper cell (r - sr, c - sc).  The tile for lattice
    # origin (0, 0) is therefore the visible pv x ph block whose top-left is congruent
    # to (-sr, -sc) modulo the periods.
    R = vr0 + ((-sr - vr0) % pv)
    C = vc0 + ((-sc - vc0) % ph)
    tile = I[R:R + pv, C:C + ph]

    g = I.copy()                            # simulated working grid
    if tric != 0 and (tile == 0).any():     # Paste can't write 0: lay a 0 base
        ops.append(0); sels.append([0, 0, h - 1, w - 1])   # whole grid
        g[:, :] = 0

    # lattice origins grouped by the (clipped) size of tile they receive
    groups = {}
    for r0 in range(0, h, pv):
        for c0 in range(0, w, ph):
            groups.setdefault((min(pv, h - r0), min(ph, w - c0)), []).append((r0, c0))
    for (th, tw) in sorted(groups, key=lambda k: (-k[0] * k[1], -k[0])):
        sub = tile[:th, :tw]
        if not (sub != 0).any():
            continue
        copied = False
        for (r0, c0) in groups[(th, tw)]:
            new = g.copy()
            blk = new[r0:r0 + th, c0:c0 + tw]
            blk[sub != 0] = sub[sub != 0]
            if np.array_equal(new, g):
                continue                    # this stamp would change nothing
            if not copied:
                ops.append(28); sels.append([R, C, th - 1, tw - 1])   # tile block in I
                copied = True
            ops.append(30); sels.append([r0, c0, 0, 0])
            g = new

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
