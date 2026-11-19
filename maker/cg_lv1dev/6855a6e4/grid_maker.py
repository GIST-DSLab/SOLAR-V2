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
from random import randint, choice, sample
from collections import Counter
import numpy as np
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    # objc is kept non-zero: the pattern travels as an ARCLE object, and ARCLE
    # treats 0 as "nothing", so a 0-coloured pattern could not be grabbed.
    cols = list(range(10))
    bgc = random.choice(cols)
    objc = random.choice([c for c in cols if c not in (bgc, 0)])
    boxc = random.choice([c for c in cols if c not in (bgc, objc)])
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, objc=None,
             boxc=None, transpose=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or objc is None or boxc is None:
        bgc, objc, boxc = sample(cols, 3)
    if transpose is None:
        transpose = choice((True, False))
    # the grid is transposed for the vertical variant, so bound accordingly
    HH, WW = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (10, HH))
    w = unifint(diff_lb, diff_ub, (4, WW))
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
        cells = frozenset(sample(totuple(cands), ncells))
        canvi = fill(canvi, objc, cells)
        # each cell is mirrored across the rail row (objh + 1)
        canvo = fill(canvo, objc, frozenset((2 * (objh + 1) - i, j) for i, j in cells))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in sorted(set(cells))]}

    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])
    others = sorted(int(c) for c in cnt if c != bgc)

    # --- the frame: two full parallel rails, each with two inward stubs ---
    boxc, horizontal = None, True
    for c in others:
        pts = np.argwhere(I == c)
        r0, r1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        c0, c1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if not all((int(r) in (r0, r1)) or (int(q) in (c0, c1)) for r, q in pts):
            continue
        rows_full = bool(np.all(I[r0, c0:c1 + 1] == c) and np.all(I[r1, c0:c1 + 1] == c))
        cols_full = bool(np.all(I[r0:r1 + 1, c0] == c) and np.all(I[r0:r1 + 1, c1] == c))
        stubs_h = hh >= 4 and all(I[a, b] == c for a, b in
                                  ((r0 + 1, c0), (r0 + 1, c1), (r1 - 1, c0), (r1 - 1, c1)))
        stubs_v = ww >= 4 and all(I[a, b] == c for a, b in
                                  ((r0, c0 + 1), (r1, c0 + 1), (r0, c1 - 1), (r1, c1 - 1)))
        if rows_full and stubs_h and len(pts) == 2 * ww + 4:
            boxc, horizontal = c, True
            break
        if cols_full and stubs_v and len(pts) == 2 * hh + 4:
            boxc, horizontal = c, False
            break
    if boxc is None:
        boxc = others[0]
        pts = np.argwhere(I == boxc)
        r0, c0, c1 = int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 1].max())
        horizontal = bool(np.all(I[r0, c0:c1 + 1] == boxc))
    objc = [c for c in others if c != boxc][0]

    bx = np.argwhere(I == boxc)
    br0, br1 = int(bx[:, 0].min()), int(bx[:, 0].max())
    bc0, bc1 = int(bx[:, 1].min()), int(bx[:, 1].max())

    obj = [(int(r), int(c)) for r, c in np.argwhere(I == objc)]
    if horizontal:
        groups = [[p for p in obj if p[0] < br0], [p for p in obj if p[0] > br1]]
    else:
        groups = [[p for p in obj if p[1] < bc0], [p for p in obj if p[1] > bc1]]

    # Each pattern outside a rail is ONE object mirrored across that rail's
    # axis.  Reflection about the rail = flip within its own extent, then a
    # slide through the rail by 2*rail - (near + far edge).
    for gi, cells in enumerate(groups):
        if not cells:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if horizontal:
            flip_op = 27
            flipped = [(r0 + r1 - r, c) for r, c in cells]
            R = br0 if gi == 0 else br1          # the rail this pattern faces
            shift_ = 2 * R - r0 - r1             # reflection = flip + this shift
            step_op, d = (21, (1, 0)) if shift_ > 0 else (20, (-1, 0))
        else:
            flip_op = 26
            flipped = [(r, c0 + c1 - c) for r, c in cells]
            R = bc0 if gi == 0 else bc1
            shift_ = 2 * R - c0 - c1
            step_op, d = (22, (0, 1)) if shift_ > 0 else (23, (0, -1))
        n = abs(shift_)

        grabbed = False
        if set(flipped) != set(cells):
            ops.append(flip_op); sels.append(sel_of(cells))   # grab + mirror
            grabbed = True
        cur = flipped
        for k in range(n):
            if not grabbed:
                ops.append(step_op); sels.append(sel_of(cur))  # grab + first step
                grabbed = True
            else:
                ops.append(step_op); sels.append(sel_of([]))   # same object
            cur = [(r + d[0], c + d[1]) for r, c in cur]
        # the grab left the original footprint at 0
        hole = sorted(set(cells) - set(cur))
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
