"""
ARC Task: 5521c0d9 (RE-ARC) — LLM-generated grid_maker
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

# Bars can grow from any of the four edges (the generator rotates the whole grid).
# Planning one rotation per example makes every direction appear before the test.
ROTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(2, 9)
    ccols = random.sample(remcols, ncols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, rot=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        remcols = remove(bgc, cols)
        ncols = unifint(diff_lb, diff_ub, (2, 9))
        ccols = sample(remcols, ncols)
    ccols = list(ccols)
    if rot is None:
        rot = choice(ROTS)
    # after rot90/rot270 the canvas is transposed
    if rot in ("rot90", "rot270"):
        hmax, wmax = max_w, max_h
    else:
        hmax, wmax = max_h, max_w
    h = unifint(diff_lb, diff_ub, (4, min(30, hmax)))
    w = unifint(diff_lb, diff_ub, (6, min(30, wmax)))
    inds = interval(0, w, 1)
    nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
    speps = sample(inds, nobjs * 2)
    while 0 in speps or w - 1 in speps:
        nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
        speps = sample(inds, nobjs * 2)
    speps = sorted(speps)
    starts = speps[::2]
    ends = speps[1::2]
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    forb = -1
    for sp, ep in zip(starts, ends):
        col = choice(remove(forb, ccols))
        forb = col
        hdev = unifint(diff_lb, diff_ub, (0, h // 2))
        hei = choice((hdev, h - hdev))
        hei = min(max(1, hei), h - 1)
        ulc = (h - hei, sp)
        lrc = (h - 1, ep)
        obj = backdrop(frozenset({ulc, lrc}))
        gi = fill(gi, col, obj)
        go = fill(go, col, shift(obj, (-hei, 0)))
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Every bar stands on one grid edge and slides away from it by its own
    thickness. The code-golf reading shifts each column by the number of
    non-background cells in it. All columns of one bar have the same count, so
    that reading and the bar-as-object reading are the same: the unit and the
    distance agree and nothing changes. The bar is kept as the moved object.
    Pure translation -> one grab, then empty-selection Moves."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: three of the four border lines are always all background
    border = ([(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)]
              + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)])
    bgc = Counter(int(I[r, c]) for r, c in border).most_common(1)[0][0]

    # the edge the bars stand on: the only border line with non-background cells
    lines = [
        ("bottom", [(h - 1, c) for c in range(w)]),
        ("top", [(0, c) for c in range(w)]),
        ("left", [(r, 0) for r in range(h)]),
        ("right", [(r, w - 1) for r in range(h)]),
    ]
    side = None
    for name, cells in lines:
        if any(I[r, c] != bgc for r, c in cells):
            side = name
            break
    if side is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # canonical view V: bars stand on V's last row and travel "up" in V
    if side == "bottom":
        V = I; back = lambda i, j: (i, j); mop = 20
    elif side == "top":
        V = I[::-1, :]; back = lambda i, j: (h - 1 - i, j); mop = 21
    elif side == "left":
        V = np.rot90(I, 1); back = lambda i, j: (j, w - 1 - i); mop = 22
    else:
        V = np.rot90(I, 3); back = lambda i, j: (h - 1 - j, i); mop = 23
    H, W = V.shape

    # bars: maximal runs of adjacent columns sharing (colour, top row)
    bars = []
    for c in range(W):
        if V[H - 1, c] == bgc:
            continue
        col = int(V[H - 1, c]); r = H - 1
        while r - 1 >= 0 and V[r - 1, c] == col:
            r -= 1
        if bars and bars[-1][0] == col and bars[-1][1] == r and bars[-1][3] == c - 1:
            bars[-1][3] = c
        else:
            bars.append([col, r, c, c])

    for col, r, c0, c1 in bars:
        ht = H - r                      # thickness = shift distance
        src = [back(i, j) for i in range(r, H) for j in range(c0, c1 + 1)]

        if col == 0:
            # colour 0 is transparent to ARCLE object ops, so Move cannot carry
            # this bar: erase it, then draw it at its shifted (clipped) place.
            dst = [back(i, j) for i in range(r - ht, H - ht) if i >= 0
                   for j in range(c0, c1 + 1)]
            ops.append(bgc); sels.append(sel_of(src))
            ops.append(0); sels.append(sel_of(dst))
            continue

        # On a 0 background every one of the ht steps is visible: the part
        # pushed past the far edge is clipped, and the uncovered footprint row
        # already reads 0 = background. On a non-zero background the bar slides
        # until it reaches the far wall (min(ht, r) steps). Then the original
        # footprint is painted to background in one op. That fills the vacated
        # hole and trims the part that would have gone past the wall.
        steps = ht if bgc == 0 else min(ht, r)
        ops.append(mop); sels.append(sel_of(src))           # grab the bar
        for _ in range(steps - 1):
            ops.append(mop); sels.append(sel_of([]))        # keep it gliding

        if bgc != 0:
            ops.append(bgc); sels.append(sel_of(src))       # original footprint

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
                        f"num_examples+1 ({num_examples + 1}) for task 5521c0d9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5521c0d9"
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
                                f"for task 5521c0d9"
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
                    f"Failed to build a complete episode for task 5521c0d9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5521c0d9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
