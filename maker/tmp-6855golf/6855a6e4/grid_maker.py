"""
ARC Task: 6855a6e4 (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # the pattern colour is kept non-zero: ARCLE Flip/Move treat 0 as "nothing"
    objc = random.choice([c for c in cols if c not in (bgc, 0)])
    boxc = random.choice([c for c in cols if c not in (bgc, objc)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, boxc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))
    lim_h, lim_w = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (10, lim_h))
    w = unifint(diff_lb, diff_ub, (4, lim_w))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bcanv = canvas(bgc, (h, w))
    loci = randint(0, h - fullh)
    locj = randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        # the pattern reaches the outer edge of its band, so that the output
        # (pattern mirrored, resting against the stubs) matches the verifier
        while True:
            cells = frozenset(sample(totuple(cands), ncells))
            if any(i == 0 for i, _ in cells):
                break
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    # background: the corner is never part of the frame or a pattern
    bgc = int(I[0, 0])
    others = sorted(int(c) for c in set(I.flatten().tolist()) if c != bgc)

    # the frame: two full parallel rails, each with two inward stubs at its ends
    def frame_of(c):
        pts = np.argwhere(I == c)
        r0, r1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        c0, c1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh >= 4 and ww >= 3 and len(pts) == 2 * ww + 4 \
                and all(I[r0, j] == c and I[r1, j] == c for j in range(c0, c1 + 1)) \
                and all(I[a, b] == c for a, b in ((r0 + 1, c0), (r0 + 1, c1),
                                                   (r1 - 1, c0), (r1 - 1, c1))):
            return True, (r0, r1, c0, c1)
        if ww >= 4 and hh >= 3 and len(pts) == 2 * hh + 4 \
                and all(I[i, c0] == c and I[i, c1] == c for i in range(r0, r1 + 1)) \
                and all(I[a, b] == c for a, b in ((r0, c0 + 1), (r1, c0 + 1),
                                                   (r0, c1 - 1), (r1, c1 - 1))):
            return False, (r0, r1, c0, c1)
        return None

    boxc, info = None, None
    for c in others:
        info = frame_of(c)
        if info is not None:
            boxc = c
            break
    horizontal, (fr0, fr1, fc0, fc1) = info
    objc = [c for c in others if c != boxc][0]

    # Work in "rail coordinates": t runs across the rails, s along them.
    # horizontal rails -> t = row, s = col; vertical rails -> t = col, s = row.
    def cell(t, s):
        return (t, s) if horizontal else (s, t)

    if horizontal:
        rails = (fr0, fr1)
        s_lo, s_hi = fc0 + 1, fc1 - 1          # interior span between the stubs
        flip_op = 27                           # up<->down
        toward = {0: 20, 1: 21}                # rail 0 is above: move up; rail 1: down
    else:
        rails = (fc0, fc1)
        s_lo, s_hi = fr0 + 1, fr1 - 1
        flip_op = 26                           # left<->right
        toward = {0: 23, 1: 22}

    G = I.copy()
    obj = [(int(r), int(c)) for r, c in np.argwhere(I == objc)]
    tcoord = (lambda p: p[0]) if horizontal else (lambda p: p[1])

    for side in (0, 1):
        rail = rails[side]
        cells = [p for p in obj if (tcoord(p) < rail if side == 0 else tcoord(p) > rail)]
        if not cells:
            continue
        ts = [tcoord(p) for p in cells]
        near = max(ts) if side == 0 else min(ts)    # pattern edge nearest the rail
        far = min(ts) if side == 0 else max(ts)     # pattern edge farthest out
        k = abs(rail - far)

        # 1. reflect the band across its rail: the window spans k lines on each
        #    side of the rail, across the interior between the stubs (whole
        #    rectangle is the intent: the rail segment maps onto itself, the
        #    pattern lands mirrored inside, the empty inside lands outside).
        t_lo, t_hi = rail - k, rail + k
        if horizontal:
            win = [t_lo, s_lo, t_hi - t_lo, s_hi - s_lo]
            G[t_lo:t_hi + 1, s_lo:s_hi + 1] = np.flipud(G[t_lo:t_hi + 1, s_lo:s_hi + 1])
        else:
            win = [s_lo, t_lo, s_hi - s_lo, t_hi - t_lo]
            G[s_lo:s_hi + 1, t_lo:t_hi + 1] = np.fliplr(G[s_lo:s_hi + 1, t_lo:t_hi + 1])
        ops.append(flip_op); sels.append(win)

        # 2. the reflected pattern now sits as far inside as it was gapped
        #    outside; slide it in to rest against the stubs (rail +/- 2).
        refl = [(2 * rail - tcoord(p), (p[1] if horizontal else p[0])) for p in cells]
        cur = [cell(t, s) for t, s in refl]
        near_in = 2 * rail - near
        target = rail + 2 if side == 0 else rail - 2
        steps = abs(near_in - target)
        if steps:
            dt = -1 if side == 0 else 1
            src = list(cur)
            mv = toward[side]
            ops.append(mv); sels.append(sel_of(cur))
            for i in range(steps):
                if i:
                    ops.append(mv); sels.append(sel_of([]))
                cur = [((r + dt, c) if horizontal else (r, c + dt)) for r, c in cur]
            hole = sorted(set(src) - set(cur))
            if bgc != 0 and hole:
                ops.append(bgc); sels.append(sel_of(hole))

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
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
                                f"for task 6855a6e4"
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
                    f"Failed to build a complete episode for task 6855a6e4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6855a6e4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
