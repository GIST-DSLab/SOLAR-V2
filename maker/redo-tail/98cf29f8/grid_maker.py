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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # generator does: bgc, objc, otherc = sample(cols, 3)  -> three distinct colors
    cols = list(range(10))
    bgc, objc, otherc = random.sample(cols, 3)
    return {"bgc": bgc, "objc": objc, "otherc": otherc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, objc: int, otherc: int) -> dict:
    hlo = max(8, min(10, max_h))
    wlo = max(8, min(10, max_w))
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
    Rule: two non-background objects.  One is a solid rectangle (the anchor).
    The other is a solid rectangle with a 1-cell-wide stem pointing at the anchor.
    The stemmed rectangle SLIDES along the stem until it touches the anchor; the
    stem disappears.  -> Move chain + cleanup of the vacated footprint + erase of
    whatever part of the stem is still visible afterwards.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    full_sel = [0, 0, h - 1, w - 1]   # bbox = the whole grid (exactly the rectangle meant)

    # --- background: most common colour along the grid border (task's own definition)
    border = []
    for c in range(w):
        border.append(int(I[0, c]))
        border.append(int(I[h - 1, c]))
    for r in range(h):
        border.append(int(I[r, 0]))
        border.append(int(I[r, w - 1]))
    bgc = Counter(border).most_common(1)[0][0]

    def cells_of(col):
        rs, cs = np.where(I == col)
        return [(int(r), int(c)) for r, c in zip(rs, cs)]

    def bbox(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), min(cs), max(rs), max(cs)

    def is_full_rect(cells):
        if not cells:
            return False
        r0, c0, r1, c1 = bbox(cells)
        return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)

    colors = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    if len(colors) != 2:
        # degenerate safety net: repaint only the cells that actually differ
        diff = {}
        for r in range(min(h, O.shape[0])):
            for c in range(min(w, O.shape[1])):
                if I[r, c] != O[r, c]:
                    diff.setdefault(int(O[r, c]), []).append((r, c))
        for col, cs in diff.items():
            ops.append(col); sels.append(sel_of(cs))
        ops.append(34); sels.append(full_sel)
        return ops, sels

    a, b = colors
    ca, cb = cells_of(a), cells_of(b)
    if is_full_rect(ca) and not is_full_rect(cb):
        static_col, mover_col = a, b
    elif is_full_rect(cb) and not is_full_rect(ca):
        static_col, mover_col = b, a
    else:
        na = int(np.sum((I == a) != (O == a)))
        nb = int(np.sum((I == b) != (O == b)))
        mover_col, static_col = (a, b) if na >= nb else (b, a)

    mover_cells = cells_of(mover_col)
    static_cells = cells_of(static_col)

    # --- split the mover into its solid block and its 1-cell-wide stem
    P = np.full((h + 2, w + 2), bgc, dtype=int)
    P[1:h + 1, 1:w + 1] = I
    stem, block = [], []
    for (r, c) in mover_cells:
        thin_h = (P[r + 1, c] == bgc and P[r + 1, c + 2] == bgc)
        thin_v = (P[r, c + 1] == bgc and P[r + 2, c + 1] == bgc)
        (stem if (thin_h or thin_v) else block).append((r, c))
    if not block:
        block, stem = list(mover_cells), []

    br0, bc0, br1, bc1 = bbox(block)
    sr0, sc0, sr1, sc1 = bbox(static_cells)
    cols_overlap = not (bc1 < sc0 or sc1 < bc0)
    rows_overlap = not (br1 < sr0 or sr1 < br0)

    if cols_overlap and not rows_overlap:
        if br0 > sr1:
            dr, dc, dist = -1, 0, br0 - sr1 - 1
        else:
            dr, dc, dist = 1, 0, sr0 - br1 - 1
    elif rows_overlap and not cols_overlap:
        if bc0 > sc1:
            dr, dc, dist = 0, -1, bc0 - sc1 - 1
        else:
            dr, dc, dist = 0, 1, sc0 - bc1 - 1
    else:
        # fall back on the stem's own geometry
        bcr, bcc = (br0 + br1) / 2.0, (bc0 + bc1) / 2.0
        scr, scc = (sr0 + sr1) / 2.0, (sc0 + sc1) / 2.0
        if abs(scr - bcr) >= abs(scc - bcc):
            dr, dc = (1, 0) if scr > bcr else (-1, 0)
            dist = abs(sr0 - br1 - 1) if scr > bcr else abs(br0 - sr1 - 1)
        else:
            dr, dc = (0, 1) if scc > bcc else (0, -1)
            dist = abs(sc0 - bc1 - 1) if scc > bcc else abs(bc0 - sc1 - 1)
        if stem:
            dist = len(stem)
    dist = max(0, int(dist))

    move_op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]

    # ARCLE object ops treat colour 0 as transparent, so a 0-coloured block cannot
    # be carried by Move.  Give it a temporary visible colour, slide it, restore.
    use_temp = (mover_col == 0)
    temp = None
    if use_temp:
        temp = next(c for c in range(1, 10) if c != bgc and c != static_col)
        ops.append(temp); sels.append(sel_of(block))

    cur = list(block)
    if dist > 0:
        ops.append(move_op); sels.append(sel_of(cur))       # first step GRABS the block
        cur = [(r + dr, c + dc) for r, c in cur]
        for _ in range(dist - 1):
            ops.append(move_op); sels.append(sel_of([]))    # empty -> keep same object
            cur = [(r + dr, c + dc) for r, c in cur]

    # ARCLE zeroed the grabbed footprint; repair only what the block no longer covers
    hole = sorted(set(block) - set(cur))
    if bgc != 0 and hole:
        ops.append(bgc); sels.append(sel_of(hole))

    # the stem is still in the grid wherever the block did not come to rest on it
    leftover = sorted(set(stem) - set(cur))
    if leftover:
        ops.append(bgc); sels.append(sel_of(leftover))

    if use_temp:
        ops.append(0); sels.append(sel_of(cur))

    ops.append(34); sels.append(full_sel)
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
