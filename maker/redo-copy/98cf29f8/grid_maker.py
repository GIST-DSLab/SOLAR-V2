"""
ARC Task: 98cf29f8 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for r, c in cells]}


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    objc = random.choice([c for c in cols if c != bgc])
    # the moving block must be non-zero: ARCLE's Move only carries non-zero cells
    otherc = random.choice([c for c in cols if c not in (bgc, objc) and c != 0])
    return {"bgc": bgc, "objc": objc, "otherc": otherc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, objc=None, otherc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or objc is None or otherc is None:
        bgc, objc, otherc = sample(cols, 3)
        if otherc == 0:
            otherc, objc = objc, otherc
    hlo = max(7, min(10, max_h))
    wlo = max(7, min(10, max_w))
    h = unifint(diff_lb, diff_ub, (hlo, max(hlo, max_h)))
    w = unifint(diff_lb, diff_ub, (wlo, max(wlo, max_w)))
    objh = unifint(diff_lb, diff_ub, (2, h - 5))
    objw = unifint(diff_lb, diff_ub, (2, w - 5))
    loci = randint(0, h - objh)
    locj = randint(0, w - objw)
    obj = backdrop(frozenset({(loci, locj), (loci + objh - 1, locj + objw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, obj)
    bmarg = h - (loci + objh)
    rmarg = w - (locj + objw)
    tmarg = loci
    lmarg = locj
    margs = (bmarg, rmarg, tmarg, lmarg)
    options = [idx for idx, marg in enumerate(margs) if marg > 2]
    pos = choice(options)
    for k in range(pos):
        gi = rot90(gi)
    h, w = shape(gi)
    ofc = ofcolor(gi, objc)
    locis = randint(lowermost(ofc) + 2, h - 2)
    locie = randint(locis + 1, h - 1)
    locjs = randint(0, min(w - 2, rightmost(ofc)))
    locje = randint(max(locjs + 1, leftmost(ofc)), w - 1)
    otherobj = backdrop(frozenset({(locis, locjs), (locie, locje)}))
    ub = min(rightmost(ofc), rightmost(otherobj))
    lb = max(leftmost(ofc), leftmost(otherobj))
    jloc = randint(lb, ub)
    ln = connect((lowermost(ofc) + 1, jloc), (uppermost(otherobj) - 1, jloc))
    gib = tuple(e for e in gi)
    gi = fill(gi, otherc, otherobj)
    gi = fill(gi, otherc, ln)
    go = fill(gib, otherc, shift(otherobj, (-len(ln), 0)))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read from I only):
      - one solid rectangle (the anchor) and one block of another colour that is
        attached to the anchor by a 1-cell-wide stem.
      - erase the stem, then slide the block along the stem until it touches the anchor.
    O is never inspected.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- background: most common colour on the border ring (generator's definition) ---
    ring = []
    for c in range(wi):
        ring.append(int(I[0, c]))
        ring.append(int(I[hi - 1, c]))
    for r in range(hi):
        ring.append(int(I[r, 0]))
        ring.append(int(I[r, wi - 1]))
    bgc = Counter(ring).most_common(1)[0][0]

    def cells_of(col):
        return [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) == col]

    def bbox(cs):
        rs = [r for r, _ in cs]
        cs_ = [c for _, c in cs]
        return min(rs), min(cs_), max(rs), max(cs_)

    def is_solid(cs):
        r0, c0, r1, c1 = bbox(cs)
        return len(cs) == (r1 - r0 + 1) * (c1 - c0 + 1)

    def is_bg(r, c):
        if r < 0 or r >= hi or c < 0 or c >= wi:
            return True            # outside the grid counts as background
        return int(I[r, c]) == bgc

    def stem_of(cs):
        out = set()
        for (r, c) in cs:
            if (is_bg(r, c - 1) and is_bg(r, c + 1)) or (is_bg(r - 1, c) and is_bg(r + 1, c)):
                out.add((r, c))
        return out

    fg = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]
    solid = [c for c in fg if is_solid(cells_of(c))]
    nonsolid = [c for c in fg if not is_solid(cells_of(c))]
    if len(solid) == 1 and len(nonsolid) == 1:
        anchor_c, mover_c = solid[0], nonsolid[0]
    else:
        # fall back on "which colour owns a 1-wide stem"
        mover_c = max(fg, key=lambda c: len(stem_of(cells_of(c))))
        anchor_c = [c for c in fg if c != mover_c][0]

    anchor_cells = cells_of(anchor_c)
    mover_cells = cells_of(mover_c)
    stem = stem_of(mover_cells)
    body = [p for p in mover_cells if p not in stem]

    ar0, ac0, ar1, ac1 = bbox(anchor_cells)
    br0, bc0, br1, bc1 = bbox(body)

    rows_disjoint = (ar1 < br0) or (br1 < ar0)
    cols_disjoint = (ac1 < bc0) or (bc1 < ac0)

    if rows_disjoint and not cols_disjoint:
        vertical = True
    elif cols_disjoint and not rows_disjoint:
        vertical = False
    else:
        srows = {r for r, _ in stem}
        scols = {c for _, c in stem}
        vertical = len(scols) <= len(srows)

    if vertical:
        if ar1 < br0:                      # anchor above -> slide up
            d = br0 - ar1 - 1
            move_op, dr, dc = 20, -1, 0
        else:                              # anchor below -> slide down
            d = ar0 - br1 - 1
            move_op, dr, dc = 21, 1, 0
    else:
        if ac1 < bc0:                      # anchor to the left -> slide left
            d = bc0 - ac1 - 1
            move_op, dr, dc = 23, 0, -1
        else:                              # anchor to the right -> slide right
            d = ac0 - bc1 - 1
            move_op, dr, dc = 22, 0, 1

    d = max(0, int(d))

    ops, sels = [], []

    # 1. erase the connecting stem
    if stem:
        ops.append(int(bgc))
        sels.append(sel_of(sorted(stem)))

    # 2. slide the block until it touches the anchor (grab once, then empty selections)
    cur = sorted(body)
    if d > 0:
        ops.append(move_op)
        sels.append(sel_of(cur))
        cur = [(r + dr, c + dc) for r, c in cur]
        for _ in range(d - 1):
            ops.append(move_op)
            sels.append(sel_of([]))
            cur = [(r + dr, c + dc) for r, c in cur]

        # 3. repair only the footprint the block no longer covers
        hole = sorted(set(body) - set(cur))
        if bgc != 0 and hole:
            ops.append(int(bgc))
            sels.append(sel_of(hole))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 98cf29f8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 98cf29f8"
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
                                f"for task 98cf29f8"
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
                    f"Failed to build a complete episode for task 98cf29f8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"98cf29f8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
