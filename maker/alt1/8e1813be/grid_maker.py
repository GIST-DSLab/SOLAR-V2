"""
ARC Task: 8e1813be (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    # strip colours: everything except background / square colour.
    # colour 0 is excluded on purpose: ARCLE's object ops (Move) and Copy/Paste
    # treat 0 as "nothing", so a 0-coloured strip could not be carried around.
    pool = [c for c in cols if c not in (bgc, sqc) and c != 0]

    # discrete structural variant: strips run as rows (mirror=False) or as
    # columns (mirror=True).  Both must be demonstrated.
    VARIANTS = [{"mirror": False}, {"mirror": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "ccols_pool": pool, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, ccols_pool, mirror=None, **kwargs) -> dict:
    if mirror is None:
        mirror = choice((True, False))
    # gi is built with h2 rows / w cols and is transposed at the end when mirror
    lim_h = min(30, max_w if mirror else max_h)
    lim_w = min(30, max_h if mirror else max_w)
    nmax = min(8, len(ccols_pool), lim_h // 3, lim_w - 3)
    if nmax < 3:
        raise ValueError("grid limits too small for task 8e1813be")
    nbars = unifint(diff_lb, diff_ub, (3, nmax))
    ccols = list(sample(ccols_pool, nbars))
    w = unifint(diff_lb, diff_ub, (nbars + 3, lim_w))
    hmarg = unifint(diff_lb, diff_ub, (2 * nbars, lim_h - nbars))
    go = tuple(repeat(col, nbars) for col in ccols)
    gi = tuple(repeat(col, w) for col in ccols)
    r = repeat(bgc, w)
    for k in range(hmarg):
        idx = randint(0, nbars - 1)
        gi = gi[:idx] + (r,) + gi[idx:]
    h2 = nbars + hmarg
    oh, ow = nbars, nbars
    loci = randint(1, h2 - oh - 2)
    locj = randint(1, w - ow - 2)
    sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, sqc, sq)
    gi = fill(gi, bgc, outbox(sq))
    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule: the one-cell-thick strips carry an order of colours read across the grid.
    Trajectory: slide the strips together so they form that sequence, read the
    sequence off one clean line across them, then lay that one sequence down as
    many times as there are strips.
    """
    import numpy as np
    from collections import defaultdict
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in cells]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    # ---- find the strips: colours whose bounding box is one cell thick -------
    cellmap = defaultdict(list)
    for r in range(H):
        for c in range(W):
            cellmap[int(I[r, c])].append((r, c))
    strips = []
    for col, pts in cellmap.items():
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        bh = max(rs) - min(rs) + 1
        bw = max(cs) - min(cs) + 1
        if bh == 1 or bw == 1:
            strips.append({"color": col, "bh": bh, "bw": bw,
                           "pos": min(rs) if bh == 1 else min(cs),
                           "cells": pts})
    row_strips = [s for s in strips if s["bh"] == 1]
    col_strips = [s for s in strips if s["bw"] == 1]
    horizontal = len(row_strips) > len(col_strips)     # strips are rows
    S = sorted(row_strips if horizontal else col_strips, key=lambda s: s["pos"])
    n = len(S)
    pos = [s["pos"] for s in S]
    dim = H if horizontal else W

    # ---- where to gather them: n consecutive lines, cheapest common offset ---
    offs = sorted(p - k for k, p in enumerate(pos))
    s0 = offs[(n - 1) // 2]
    s0 = max(0, min(s0, dim - n))
    dest = [s0 + k for k in range(n)]

    ops, sels = [], []
    moved_from = set()
    # strips that must slide towards smaller indices first (ascending), then the
    # ones sliding towards larger indices (descending) -> no strip is overrun.
    order = [k for k in range(n) if pos[k] > dest[k]] + \
            [k for k in range(n) if pos[k] < dest[k]][::-1]
    for k in order:
        if S[k]["color"] == 0:
            continue                       # colour 0 cannot be grabbed by Move
        steps = abs(pos[k] - dest[k])
        if horizontal:
            op = 20 if pos[k] > dest[k] else 21          # MoveU / MoveD
        else:
            op = 23 if pos[k] > dest[k] else 22          # MoveL / MoveR
        ops.append(op)
        sels.append(sel_of(S[k]["cells"]))               # grab this strip
        for _ in range(steps - 1):
            ops.append(op)
            sels.append(sel_of([]))                      # keep the same grab
        moved_from.add(pos[k])

    # ---- the line across the strips along which the order is read -----------
    line = 0
    span = W if horizontal else H
    for t in range(span):
        ok = True
        for k in range(n):
            v = I[pos[k], t] if horizontal else I[t, pos[k]]
            if int(v) != S[k]["color"]:
                ok = False
                break
        if ok:
            line = t
            break

    # a strip whose colour is 0 could not be slid; its slot has to be restored
    zero_idx = None
    for k in range(n):
        if S[k]["color"] == 0:
            zero_idx = k
    fix_zero = False
    if zero_idx is not None:
        d = dest[zero_idx]
        if d in moved_from:
            v = 0
        else:
            v = int(I[d, line]) if horizontal else int(I[line, d])
        fix_zero = (v != 0)

    # ---- read the sequence, then repeat it once per strip -------------------
    # (every bbox below is exactly the full rectangle it names)
    if horizontal:
        ops.append(33); sels.append([s0, line, n - 1, 0])   # crop the reading line
        ops.append(33); sels.append([0, 0, n - 1, n - 1])   # canvas -> n x n
        ops.append(29); sels.append([0, 0, n - 1, 0])       # the sequence itself
        for j in range(1, n):
            ops.append(30); sels.append([0, j, 0, 0])       # lay it down again
        if fix_zero:
            ops.append(0); sels.append(sel_of([(zero_idx, j) for j in range(n)]))
    else:
        ops.append(33); sels.append([line, s0, 0, n - 1])   # crop the reading line
        ops.append(33); sels.append([0, 0, n - 1, n - 1])   # canvas -> n x n
        ops.append(29); sels.append([0, 0, 0, n - 1])       # the sequence itself
        for i in range(1, n):
            ops.append(30); sels.append([i, 0, 0, 0])       # lay it down again
        if fix_zero:
            ops.append(0); sels.append(sel_of([(i, zero_idx) for i in range(n)]))

    ops.append(34); sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
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
                                f"for task 8e1813be"
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
                    f"Failed to build a complete episode for task 8e1813be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e1813be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
