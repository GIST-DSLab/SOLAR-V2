"""
ARC Task: d687bc17 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # bgc may be any colour; the four border colours must be NON-ZERO because the
    # interior dots are moved with ARCLE Move ops (object buffer keeps nonzero cells only).
    bgc = random.choice(list(range(10)))
    pool = [c for c in range(1, 10) if c != bgc]
    c1, c2, c3, c4 = random.sample(pool, 4)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4) -> dict:
    max_h = max(5, int(max_h))
    max_w = max(5, int(max_w))
    h = _unifint(diff_lb, diff_ub, (5, max_h))
    w = _unifint(diff_lb, diff_ub, (5, max_w))

    gi = [[bgc] * w for _ in range(h)]
    for j in range(w):
        gi[0][j] = c1
    for i in range(h):
        gi[i][0] = c2
    for i in range(h):
        gi[i][w - 1] = c3
    for j in range(w):
        gi[h - 1][j] = c4
    for (i, j) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        gi[i][j] = bgc

    go = [row[:] for row in gi]

    # candidate dot cells: rows 2..h-3, cols 2..w-3
    cands = [(i, j) for i in range(2, h - 2) for j in range(2, w - 2)]
    ndots = _unifint(diff_lb, diff_ub, (1, min(len(cands), h + h + w + w)))
    ndots = max(1, min(ndots, len(cands)))
    picked = random.sample(cands, ndots)
    colored = [(random.choice((c1, c2, c3, c4)), ij) for ij in picked]

    by_color = {c1: [], c2: [], c3: [], c4: []}
    for (col, ij) in colored:
        by_color[col].append(ij)

    # c1 -> projects onto row 1 (one dot per column)
    cov = sorted({j for (i, j) in by_color[c1]})
    if len(cov) == w - 4 and w > 5:
        cov.remove(random.choice(cov))
    for jj in cov:
        loci = random.choice([i for (i, j) in by_color[c1] if j == jj])
        gi[loci][jj] = c1
        go[1][jj] = c1

    # c2 -> projects onto column 1 (one dot per row)
    cov = sorted({i for (i, j) in by_color[c2]})
    if len(cov) == h - 4 and h > 5:
        cov.remove(random.choice(cov))
    for ii in cov:
        locj = random.choice([j for (i, j) in by_color[c2] if i == ii])
        gi[ii][locj] = c2
        go[ii][1] = c2

    # c4 -> projects onto row h-2 (one dot per column)
    cov = sorted({j for (i, j) in by_color[c4]})
    if len(cov) == w - 4 and w > 5:
        cov.remove(random.choice(cov))
    for jj in cov:
        loci = random.choice([i for (i, j) in by_color[c4] if j == jj])
        gi[loci][jj] = c4
        go[h - 2][jj] = c4

    # c3 -> projects onto column w-2 (one dot per row)
    cov = sorted({i for (i, j) in by_color[c3]})
    if len(cov) == h - 4 and h > 5:
        cov.remove(random.choice(cov))
    for ii in cov:
        locj = random.choice([j for (i, j) in by_color[c3] if i == ii])
        gi[ii][locj] = c3
        go[ii][w - 2] = c3

    noisecands = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc]
    noisecols = [c for c in range(10) if c not in (bgc, c1, c2, c3, c4)]
    ub = ((h * w) - 2 * h - 2 * (w - 2)) // 2 - ndots - 1
    nnoise = _unifint(diff_lb, diff_ub, (0, max(0, ub)))
    nnoise = max(0, min(nnoise, len(noisecands)))
    for (i, j) in random.sample(noisecands, nnoise):
        gi[i][j] = random.choice(noisecols)

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # border colours (border cells are never noise: noise only lands on bgc cells)
    c1 = int(I[0, 1])        # top row
    c2 = int(I[1, 0])        # left column
    c3 = int(I[1, w - 1])    # right column
    c4 = int(I[h - 1, 1])    # bottom row

    interior_vals = [int(I[r, c]) for r in range(1, h - 1) for c in range(1, w - 1)]
    bgc = Counter(interior_vals).most_common(1)[0][0]
    border_cols = {c1, c2, c3, c4}

    ops, sels = [], []

    # ---- 1. wipe everything that is not a dot: interior noise + non-bgc corners ----
    junk = []
    for (r, c) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if int(I[r, c]) != bgc:
            junk.append((r, c))

    tops, lefts, bottoms, rights = [], [], [], []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = int(I[r, c])
            if v == bgc:
                continue
            if v == c1:
                tops.append((r, c))
            elif v == c2:
                lefts.append((r, c))
            elif v == c4:
                bottoms.append((r, c))
            elif v == c3:
                rights.append((r, c))
            else:
                junk.append((r, c))

    if junk:
        ops.append(bgc)
        sels.append(sel_of(junk))

    # ---- 2. slide every dot to its own border side ----
    def slide(r, c, dr, dc, steps, color):
        if steps <= 0:
            return
        if dr < 0:
            op = 20
        elif dr > 0:
            op = 21
        elif dc > 0:
            op = 22
        else:
            op = 23
        if color == 0:
            # ARCLE's object buffer drops zero cells, so a 0-coloured dot cannot be
            # moved; paint it at its destination instead (bgc != 0 in that case).
            ops.append(0)
            sels.append(sel_of([(r + dr * steps, c + dc * steps)]))
        else:
            ops.append(op)
            sels.append(sel_of([(r, c)]))          # first Move grabs the dot
            for _ in range(steps - 1):
                ops.append(op)
                sels.append(sel_of([]))            # empty -> keep sliding same object
        # the grabbed cell's original footprint was zeroed by the grab
        if bgc != 0:
            ops.append(bgc)
            sels.append(sel_of([(r, c)]))

    for (r, c) in sorted(tops, key=lambda p: (p[1], p[0])):        # up to row 1
        slide(r, c, -1, 0, r - 1, c1)
    for (r, c) in sorted(lefts, key=lambda p: (p[0], p[1])):       # left to col 1
        slide(r, c, 0, -1, c - 1, c2)
    for (r, c) in sorted(bottoms, key=lambda p: (p[1], p[0])):     # down to row h-2
        slide(r, c, 1, 0, (h - 2) - r, c4)
    for (r, c) in sorted(rights, key=lambda p: (p[0], p[1])):      # right to col w-2
        slide(r, c, 0, 1, (w - 2) - c, c3)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox = exactly the whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task d687bc17"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d687bc17"
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
                                f"for task d687bc17"
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
                    f"Failed to build a complete episode for task d687bc17 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d687bc17-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
