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
import random
import numpy as np
from collections import defaultdict, Counter

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback if helper unavailable
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """bgc / boxc (the bars hanging off one edge) / linc (the segments that slide).

    linc must be non-zero: ARCLE's object mode only grabs NONZERO cells, so a
    zero-coloured line could not be Moved at all.

    The generator also picks one of 4 rotations per instance -> a discrete
    structural variant (which edge carries the bars, which way the lines slide).
    Plan it up front so every orientation the test can show also appears in the
    examples.
    """
    cols = list(range(10))
    bgc, boxc = random.sample(cols, 2)
    linc = random.choice([c for c in cols if c not in (bgc, boxc) and c != 0])

    variants = [{"rotk": k} for k in range(4)]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "boxc": boxc, "linc": linc, "instance_plan": plan}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, linc, rotk=None) -> dict:
    try:
        _uf = unifint  # noqa: F821  (RE-ARC helper when available)
    except NameError:
        def _uf(lb, ub, bounds):
            a, b = bounds
            return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if rotk is None:
        rotk = random.randint(0, 3)
    rotk = int(rotk) % 4

    mh = max(5, min(int(max_h), 30))
    mw = max(5, min(int(max_w), 30))
    # a 90/270 rotation swaps the dimensions -> sample so the rotated grid fits
    if rotk % 2 == 1:
        hub, wub = mw, mh
    else:
        hub, wub = mh, mw

    h = _uf(diff_lb, diff_ub, (5, hub))
    w = _uf(diff_lb, diff_ub, (5, wub))
    oh = _uf(diff_lb, diff_ub, (2, h // 2))
    ow = _uf(diff_lb, diff_ub, (3, w - 2))
    locj = random.randint(1, w - ow - 1)

    gi = [[bgc] * w for _ in range(h)]

    rng = list(range(locj, locj + ow))
    cutoffs = [random.randint(1, oh - 1) for _ in rng]
    # bars hanging from the top edge, one per column of the box
    for jj, co in zip(rng, cutoffs):
        for r in range(co):
            gi[r][jj] = boxc

    go = [row[:] for row in gi]

    numlns = _uf(diff_lb, diff_ub, (1, ow - 1))
    lnlocs = set(random.sample(rng, numlns))
    for jj, co in zip(rng, cutoffs):
        if jj in lnlocs:
            lineh = random.randint(1, h - co - 1)
            for r in range(h - lineh, h):      # input: segment glued to bottom edge
                gi[r][jj] = linc
            for r in range(co, co + lineh):    # output: segment glued under its bar
                go[r][jj] = linc

    def rot_cw(g):
        return [list(row) for row in zip(*g[::-1])]

    for _ in range(rotk):
        gi = rot_cw(gi)
        go = rot_cw(go)

    return {
        "input": tuple(tuple(int(v) for v in row) for row in gi),
        "output": tuple(tuple(int(v) for v in row) for row in go),
    }


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    """Rule (measured from I only):

    One colour forms BARS growing out of one edge of the grid, one bar per
    line (column or row, depending on the rotation).  Another colour forms
    SEGMENTS stuck to the OPPOSITE edge, each inside a line that also holds a
    bar.  Each segment slides along its own line, keeping its length, until it
    touches the end of the bar in that line.

    Everything below (background, axis, which colour is bar vs segment, which
    edge, each segment's length and travel distance) is read out of I.
    O is never inspected.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def mono(vec):
        return len(set(vec.tolist())) == 1

    # --- axis + background -------------------------------------------------
    # The box never touches the first/last line-index, so the two extreme
    # lines perpendicular to the segments are pure background.
    axis, bgc = None, None
    if mono(I[:, 0]) and mono(I[:, -1]) and int(I[0, 0]) == int(I[0, -1]):
        axis, bgc = "v", int(I[0, 0])          # segments run along columns
    elif mono(I[0, :]) and mono(I[-1, :]) and int(I[0, 0]) == int(I[-1, 0]):
        axis, bgc = "h", int(I[0, 0])          # segments run along rows
    else:                                       # defensive fallback
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
        axis = "v" if mono(I[:, 0]) else "h"

    n = h if axis == "v" else w                 # cells along one line
    m = w if axis == "v" else h                 # number of lines

    def cell(k, p):
        return (p, k) if axis == "v" else (k, p)

    occ = defaultdict(lambda: defaultdict(list))   # colour -> line -> positions
    for k in range(m):
        for p in range(n):
            r, c = cell(k, p)
            v = int(I[r, c])
            if v != bgc:
                occ[v][k].append(p)

    colours = sorted(occ.keys(), key=lambda c: -len(occ[c]))
    if len(colours) < 2:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    # every segment lives in a line that also holds a bar, and there are
    # strictly fewer segments than bars -> the bar colour spans more lines.
    boxc, linc = colours[0], colours[1]

    # which edge do the bars grow from?
    box_at_start = all(min(ps) == 0 for ps in occ[boxc].values())

    for k in sorted(occ[linc].keys()):
        seg = sorted(occ[linc][k])
        L = len(seg)
        co = len(occ[boxc].get(k, []))          # bar length in this line
        if co == 0:
            continue

        if box_at_start:
            src = list(range(n - L, n))         # segment sits at the far edge
            dst_start = co                      # land right after the bar
            step = -1
            mv = 20 if axis == "v" else 23      # MoveU / MoveL
        else:
            src = list(range(0, L))
            dst_start = n - co - L
            step = +1
            mv = 21 if axis == "v" else 22      # MoveD / MoveR

        steps = abs(dst_start - src[0])
        if steps <= 0:
            continue

        src_cells = [cell(k, p) for p in src]
        dst_cells = [cell(k, p) for p in range(dst_start, dst_start + L)]

        # grab the segment once, then keep sliding with empty selections so
        # ARCLE re-pastes it over its own background snapshot each step.
        ops.append(mv); sels.append(sel_of(src_cells))
        for _ in range(steps - 1):
            ops.append(mv); sels.append(sel_of([]))

        # only the vacated part of the ORIGINAL footprint is left at 0
        hole = sorted(set(src_cells) - set(dst_cells))
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])   # whole-grid rectangle
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
