"""
ARC Task: 3906de3d (RE-ARC) — LLM-generated grid_maker
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
import math
import random

import numpy as np

from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- helpers ---

def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = int(math.ceil(a + (b - a) * diff_lb))
    hi = int(math.ceil(a + (b - a) * diff_ub))
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _apply_rot(g, rot):
    a = np.array(g, dtype=int)
    if rot == "cw":
        a = np.rot90(a, k=3)
    elif rot == "ccw":
        a = np.rot90(a, k=1)
    elif rot == "rot180":
        a = np.rot90(a, k=2)
    return [[int(v) for v in row] for row in a]


# the only discrete structural variant: which edge the bar-chart block sits on
VARIANTS = [
    {"rot": "identity"},
    {"rot": "cw"},
    {"rot": "rot180"},
    {"rot": "ccw"},
]


# ----------------------------------------------------------- sample_colors ---

def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    boxc = random.choice([c for c in cols if c != bgc])
    # linc must be non-zero: ARCLE's object grab (ops 20-23) only captures
    # non-zero cells, and the sliding bars are exactly what has to be moved.
    linc = random.choice([c for c in cols if c not in (bgc, boxc) and c != 0])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "boxc": boxc, "linc": linc, "instance_plan": plan}


# ----------------------------------------------------------------- generate ---

def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, linc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(VARIANTS)["rot"]

    # after a 90/270 rotation the dimensions swap, so bound them accordingly
    if rot in ("cw", "ccw"):
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    hb = max(5, min(30, int(hb)))
    wb = max(5, min(30, int(wb)))

    h = _unifint(diff_lb, diff_ub, (5, hb))
    w = _unifint(diff_lb, diff_ub, (5, wb))
    oh = _unifint(diff_lb, diff_ub, (2, h // 2))
    ow = _unifint(diff_lb, diff_ub, (3, w - 2))
    locj = random.randint(1, w - ow - 1)

    gi = [[bgc] * w for _ in range(h)]
    for r in range(oh):
        for c in range(locj, locj + ow):
            gi[r][c] = boxc

    rng = list(range(locj, locj + ow))
    cutoffs = [random.randint(1, oh - 1) for _ in rng]
    for jj, co in zip(rng, cutoffs):
        for r in range(co, oh):
            gi[r][jj] = bgc

    numlns = _unifint(diff_lb, diff_ub, (1, ow - 1))
    lnlocs = set(random.sample(rng, numlns))

    go = [row[:] for row in gi]
    for jj, co in zip(rng, cutoffs):
        if jj in lnlocs:
            lineh = random.randint(1, h - co - 1)
            for r in range(h - lineh, h):
                gi[r][jj] = linc          # bar parked against the far edge
            for r in range(co, co + lineh):
                go[r][jj] = linc          # bar docked under its column of the block

    return {"input": _apply_rot(gi, rot), "output": _apply_rot(go, rot)}


# -------------------------------------------------------- derive_operations ---

def derive_operations(I, O):
    """Each free-floating bar slides across the gap until it docks against the
    bottom of its own column of the block.  Everything (which bars, which axis,
    how far) is measured from I."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    ops, sels = [], []

    # corners are always background (the block never reaches a side edge)
    bgc = int(I[0, 0])
    fg = sorted({int(v) for v in I.flatten().tolist()} - {bgc})

    # The block occupies a solid run along one whole edge (one cell per column
    # of its span); the bars touch the opposite edge on a strict subset of those
    # columns.  So the longest edge-run identifies the block colour and its edge.
    best = None
    for col in fg:
        for e, arr in (("top", I[0, :]), ("bottom", I[h - 1, :]),
                       ("left", I[:, 0]), ("right", I[:, w - 1])):
            cnt = int((arr == col).sum())
            if cnt and (best is None or cnt > best[0]):
                best = (cnt, col, e)
    _, boxc, box_edge = best
    linc = [c for c in fg if c != boxc][0]

    # canonical view: block hanging from the top, bars parked at the bottom
    if box_edge == "top":
        V = I
        mv = 20                                     # bars travel up
        real = lambda r, c: (r, c)
    elif box_edge == "bottom":
        V = I[::-1, :]
        mv = 21                                     # bars travel down
        real = lambda r, c: (h - 1 - r, c)
    elif box_edge == "left":
        V = I.T
        mv = 23                                     # bars travel left
        real = lambda r, c: (c, r)
    else:  # "right"
        V = I[:, ::-1].T
        mv = 22                                     # bars travel right
        real = lambda r, c: (c, w - 1 - r)

    H, W = V.shape

    # walk the block's columns in order and dock each bar it owns
    for c in range(W):
        colv = V[:, c]
        depth = int((colv == boxc).sum())            # block reaches rows 0..depth-1
        rows = [int(r) for r in np.flatnonzero(colv == linc)]
        if not rows or depth == 0:
            continue
        lineh = len(rows)
        top = rows[0]
        steps = top - depth                          # size of the empty gap
        if steps <= 0:
            continue

        src = [real(r, c) for r in rows]
        dst = [real(r, c) for r in range(depth, depth + lineh)]

        # one grab, then empty selections keep the same bar sliding
        ops.append(mv)
        sels.append(sel_of(src))
        for _ in range(steps - 1):
            ops.append(mv)
            sels.append(sel_of([]))

        # ARCLE zeroed the grabbed footprint; repair only what the bar vacated
        hole = sorted(set(src) - set(dst))
        if bgc != 0 and hole:
            ops.append(bgc)
            sels.append(sel_of(hole))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3906de3d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3906de3d"
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
                                f"for task 3906de3d"
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
                    f"Failed to build a complete episode for task 3906de3d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3906de3d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
