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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """Episode-level colours + the per-instance rotation plan.

    The generator samples: bgc, and ncols/ccols (the block palette).
    Object colours exclude 0 here: ARCLE's object ops treat 0 as transparent,
    so a 0-coloured block cannot be carried by a Move (derive_operations still
    handles that case by painting, for pairs coming from the raw RE-ARC gen).
    `rot` is the discrete structural variant (which edge the blocks sit on).
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    avail = [c for c in cols if c != bgc and c != 0]
    ncols = random.randint(2, len(avail))
    ccols = random.sample(avail, ncols)

    ROTS = ["identity", "rot90", "rot180", "rot270"]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols, rot=None) -> dict:
    """Bars grow from one edge; each bar jumps away from that edge by its own
    thickness (clipped at the far edge). `rot` picks which edge."""

    def _unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if rot is None:
        rot = random.choice(["identity", "rot90", "rot180", "rot270"])

    # a 90/270 rotation swaps the axes -> bound h,w so the rotated grid fits
    if rot in ("rot90", "rot270"):
        h_ub, w_ub = max_w, max_h
    else:
        h_ub, w_ub = max_h, max_w
    h_ub = max(4, min(30, int(h_ub)))
    w_ub = max(6, min(30, int(w_ub)))

    h = _unifint(diff_lb, diff_ub, (4, h_ub))
    w = _unifint(diff_lb, diff_ub, (6, w_ub))

    inds = list(range(w))
    while True:
        nobjs = _unifint(diff_lb, diff_ub, (1, w // 3))
        speps = random.sample(inds, nobjs * 2)
        if 0 not in speps and (w - 1) not in speps:
            break
    speps = sorted(speps)
    starts, ends = speps[::2], speps[1::2]

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    forb = -1
    for sp, ep in zip(starts, ends):
        pool = [c for c in ccols if c != forb] or list(ccols)
        col = random.choice(pool)
        forb = col
        hdev = _unifint(diff_lb, diff_ub, (0, h // 2))
        hei = random.choice((hdev, h - hdev))
        hei = min(max(1, hei), h - 1)
        for r in range(h - hei, h):
            for c in range(sp, ep + 1):
                gi[r][c] = col
        for r in range(h - 2 * hei, h - hei):          # shifted up by its own height
            if 0 <= r < h:                              # clipped at the far edge
                for c in range(sp, ep + 1):
                    go[r][c] = col

    def _rot(g):
        if rot == "identity":
            return [list(r) for r in g]
        if rot == "rot90":                              # clockwise
            return [list(r) for r in zip(*g[::-1])]
        if rot == "rot180":
            return [list(r)[::-1] for r in g[::-1]]
        return [list(r) for r in zip(*g)][::-1]         # counter-clockwise

    return {"input": _rot(gi), "output": _rot(go)}


def derive_operations(I, O):
    """Every bar touches ONE grid edge; it slides away from that edge by its own
    thickness (stopping when it hits the opposite wall). Pure translation ->
    Move ops: one grab, then empty selections to keep the same object gliding."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: 3 of the 4 border lines are always fully background here
    border = ([(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)]
              + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)])
    bgc = Counter(int(I[r, c]) for r, c in border).most_common(1)[0][0]

    # which edge the bars grow from (only one border line carries non-bgc cells)
    lines = [
        ("bottom", [(h - 1, c) for c in range(w)]),
        ("top",    [(0, c) for c in range(w)]),
        ("left",   [(r, 0) for r in range(h)]),
        ("right",  [(r, w - 1) for r in range(h)]),
    ]
    side = None
    for name, cells in lines:
        if any(I[r, c] != bgc for r, c in cells):
            side = name
            break
    if side is None:                                    # no bars at all
        ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
        return ops, sels

    # canonical view V: bars sit on V's LAST row and travel upward in V
    if side == "bottom":
        V = I
        back = lambda i, j: (i, j)
        mop = 20                                        # MoveU
    elif side == "top":
        V = I[::-1, :]
        back = lambda i, j: (h - 1 - i, j)
        mop = 21                                        # MoveD
    elif side == "left":
        V = np.rot90(I, 1)
        back = lambda i, j: (j, w - 1 - i)
        mop = 22                                        # MoveR
    else:
        V = np.rot90(I, 3)
        back = lambda i, j: (h - 1 - j, i)
        mop = 23                                        # MoveL
    H, W = V.shape

    # bars = maximal runs of columns sharing (colour, top row) — measured from I
    objs = []
    for c in range(W):
        if V[H - 1, c] == bgc:
            continue
        col = int(V[H - 1, c]); r = H - 1
        while r - 1 >= 0 and V[r - 1, c] == col:
            r -= 1
        if objs and objs[-1][0] == col and objs[-1][1] == r and objs[-1][3] == c - 1:
            objs[-1][3] = c
        else:
            objs.append([col, r, c, c])

    for col, r, c0, c1 in objs:
        ht = H - r                                      # bar thickness
        src = [back(i, j) for i in range(r, H) for j in range(c0, c1 + 1)]
        dst_rows = [i for i in range(r - ht, r) if 0 <= i < H]

        if col == 0:
            # 0 is transparent to ARCLE object ops: this bar cannot be carried
            # by a Move, so erase it and draw it at its destination.
            ops.append(bgc); sels.append(sel_of(src))
            dst = [back(i, j) for i in dst_rows for j in range(c0, c1 + 1)]
            if dst:
                ops.append(0); sels.append(sel_of(dst))
            continue

        # slide by its own thickness; when the bar is thicker than the gap it
        # simply stops against the far wall (further steps only push cells off).
        steps = ht if bgc == 0 else min(ht, r)
        if steps > 0:
            ops.append(mop); sels.append(sel_of(src))            # grab the bar
            for _ in range(steps - 1):
                ops.append(mop); sels.append(sel_of([]))         # keep it gliding

        # the grab zeroed the bar's original footprint; restore it to background
        # (and clear any part of the bar still standing there after the slide)
        need = []
        for i in range(r, H):
            still_bar = (max(0, r - steps) <= i <= H - 1 - steps)
            if bgc != 0 or still_bar:
                need.extend(back(i, j) for j in range(c0, c1 + 1))
        if need:
            ops.append(bgc); sels.append(sel_of(need))

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
