"""
ARC Task: e48d4e1a (RE-ARC) — LLM-generated grid_maker
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

# Discrete structural variant: which way the generator rotated the grid. This sets
# which quadrant holds the dots, and so which way the cross shifts.
ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc, fgc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]          # test direction was shown in examples
    return {"bgc": bgc, "fgc": fgc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, dotc, rot=None) -> dict:
    from utils import unifint
    from dsl import (asindices, canvas, totuple, fill, hfrontier,
                     vfrontier, identity, rot90, rot180, rot270)
    if rot is None:
        rot = random.choice(ROTS)
    # odd rotations swap h/w, so bound the pre-rotation dims accordingly
    hb, wb = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = unifint(diff_lb, diff_ub, (3, max(3, min(30, hb))))
    w = unifint(diff_lb, diff_ub, (3, max(3, min(30, wb))))
    loci = random.randint(1, h - 2)
    locj = random.randint(1, w - 2)
    inds = asindices(canvas(-1, (loci, locj)))
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = unifint(diff_lb, diff_ub, (1, maxn))
    ss = random.sample(totuple(inds), nn)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = fill(gi, fgc, hfrontier((loci, 0)) | vfrontier((0, locj)))
    gi = fill(gi, dotc, ss)
    go = fill(go, fgc, hfrontier((loci + nn, 0)) | vfrontier((0, locj + nn)))
    rotf = (identity, rot90, rot180, rot270)[rot]
    return {'input': rotf(gi), 'output': rotf(go)}


def derive_operations(I, O, examples=None):
    # Reading (taken from the golf entry, which shifts the whole picture by the dot
    # count d): the cross is NOT erased and redrawn. It is the same object, translated.
    # The number of dots sets how far it goes; the dots' side sets the direction (away
    # from them). The maker erases and repaints. This version MOVES the cross instead,
    # as its two lines:
    #   - the vertical line (its cells off the horizontal line) slides sideways nn cells,
    #   - then the horizontal line (its cells off the now-shifted vertical line) slides
    #     up/down nn cells.
    # A full-width row slid vertically stays a full row, and a full-height column slid
    # sideways stays a full column. Nothing clips, and the crossing cell is carried by
    # whichever line is still standing. Each chain grabs once, then uses empty selections.
    # The maker and this version agree on how the dots are read (count = distance, side =
    # direction) and that they are consumed, so that part is kept.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)

    # the cross: the vertical line crosses every row, so the only single-colour row is
    # the cross's horizontal line; its colour also fills exactly one full column.
    hrow = next(r for r in range(hi) if np.all(I[r, :] == I[r, 0]))
    fgc = int(I[hrow, 0])
    vcol = next(c for c in range(wi) if np.all(I[:, c] == fgc))
    # off-cross colours: background strictly outnumbers the dots (generator bound)
    rest = Counter(int(v) for v in I.flatten() if v != fgc)
    bgc = rest.most_common(1)[0][0]
    dotc = [c for c in rest if c != bgc][0]
    dots = [(int(r), int(c)) for r, c in np.argwhere(I == dotc)]
    nn = len(dots)
    di = sign(hrow - sum(r for r, _ in dots) / nn)
    dj = sign(vcol - sum(c for _, c in dots) / nn)
    new_hrow, new_vcol = hrow + di * nn, vcol + dj * nn

    ops, sels = [], []
    # 1. the dots are a counter, consumed: all cells of the dot colour -> background
    ops.append(bgc); sels.append(sel_of(dots))

    if fgc == 0:
        # Move cannot carry colour-0 content (0 is "empty" to the object buffer),
        # so here the lines have to be redrawn instead.
        ops.append(bgc)
        sels.append(sel_of([(hrow, c) for c in range(wi)] + [(r, vcol) for r in range(hi)]))
        ops.append(fgc)
        sels.append(sel_of([(new_hrow, c) for c in range(wi)] + [(r, new_vcol) for r in range(hi)]))
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    def slide(cells, dr, dc, steps):
        op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        start = list(cells)
        cur = list(cells)
        for k in range(steps):
            # first step grabs the line, later steps continue the same grab
            ops.append(op); sels.append(sel_of(cur) if k == 0 else sel_of([]))
            cur = [(r + dr, c + dc) for r, c in cur]
        # only the line's original footprint is left at 0; restore it to background
        hole = sorted(set(start) - set(cur))
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))

    # 2. vertical line slides sideways, away from the dots
    if dj != 0:
        slide([(r, vcol) for r in range(hi) if r != hrow], 0, dj, nn)
    # 3. horizontal line slides up/down, away from the dots
    if di != 0:
        slide([(hrow, c) for c in range(wi) if c != new_vcol], di, 0, nn)

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
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
                                f"for task e48d4e1a"
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
                    f"Failed to build a complete episode for task e48d4e1a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e48d4e1a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
