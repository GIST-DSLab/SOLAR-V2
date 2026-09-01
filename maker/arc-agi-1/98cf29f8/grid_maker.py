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
import numpy as np
from collections import Counter
from itertools import permutations

from maker.sel_helpers import sel_of

DIRECTIONS = ["up", "down", "left", "right"]


def sample_colors(num_examples=None) -> dict:
    # otherc is the colour of the block that gets MOVED -> it must be non-zero,
    # otherwise ARCLE's object buffer (non-zero cells only) cannot carry it.
    otherc = random.choice([c for c in range(1, 10)])
    rest = [c for c in range(10) if c != otherc]
    bgc, objc = random.sample(rest, 2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(DIRECTIONS):
        ex = [d for d in DIRECTIONS]
        ex += [random.choice(DIRECTIONS) for _ in range(n_ex - len(DIRECTIONS))]
        random.shuffle(ex)
    else:
        ex = random.sample(DIRECTIONS, n_ex)
    plan = [{"direction": d} for d in ex]
    plan.append({"direction": random.choice(ex)})
    return {"bgc": bgc, "objc": objc, "otherc": otherc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, objc: int, otherc: int, direction=None) -> dict:
    if direction is None:
        direction = choice(DIRECTIONS)
    lim = min(max_h, max_w)
    if lim < 10:
        lim = 10
    h = unifint(diff_lb, diff_ub, (10, lim))
    w = unifint(diff_lb, diff_ub, (10, lim))
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
    # canonical movement direction is "up"; a planned dihedral transform sets the
    # actual direction so every direction shows up across the episode
    dirmap = {
        "up": (identity, vmirror),
        "down": (hmirror, rot180),
        "left": (dmirror, rot270),
        "right": (rot90, cmirror),
    }
    fn = choice(dirmap[direction])
    gi = fn(gi)
    go = fn(go)
    return {'input': gi, 'output': go}


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _analyze(I, ca, cb, strict=True):
    """Try to read the task's structure out of I only.

    ca = candidate colour of the big anchor rectangle,
    cb = candidate colour of the block + its 1-cell-wide connector stem.
    """
    hi, wi = I.shape
    A = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == ca]
    B = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == cb]
    if not A or not B:
        return None
    ar0, ar1, ac0, ac1 = _bbox(A)
    if strict:
        if len(A) != (ar1 - ar0 + 1) * (ac1 - ac0 + 1):
            return None
        if ar1 - ar0 + 1 < 2 or ac1 - ac0 + 1 < 2:
            return None

    rowcnt = Counter(r for r, _ in B)
    colcnt = Counter(c for _, c in B)
    thin_rows = set(r for r, n in rowcnt.items() if n == 1)
    thin_cols = set(c for c, n in colcnt.items() if n == 1)
    if bool(thin_rows) == bool(thin_cols):
        return None                      # need exactly one thin orientation
    vertical = bool(thin_rows)
    if vertical:
        stem = [(r, c) for (r, c) in B if r in thin_rows]
    else:
        stem = [(r, c) for (r, c) in B if c in thin_cols]
    stem_set = set(stem)
    rect = [p for p in B if p not in stem_set]
    if not rect:
        return None
    rr0, rr1, rc0, rc1 = _bbox(rect)
    if len(rect) != (rr1 - rr0 + 1) * (rc1 - rc0 + 1):
        return None
    if rr1 - rr0 + 1 < 2 or rc1 - rc0 + 1 < 2:
        return None

    if vertical:
        cols = set(c for _, c in stem)
        if len(cols) != 1:
            return None
        jc = next(iter(cols))
        rows = sorted(r for r, _ in stem)
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        if not (rc0 <= jc <= rc1):
            return None
        if strict and not (ac0 <= jc <= ac1):
            return None
        if rows[-1] == rr0 - 1:
            step = (-1, 0)
            if strict and rows[0] - 1 != ar1:
                return None
        elif rows[0] == rr1 + 1:
            step = (1, 0)
            if strict and rows[-1] + 1 != ar0:
                return None
        else:
            return None
        nsteps = len(rows)
    else:
        rows = set(r for r, _ in stem)
        if len(rows) != 1:
            return None
        ir = next(iter(rows))
        cols = sorted(c for _, c in stem)
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        if not (rr0 <= ir <= rr1):
            return None
        if strict and not (ar0 <= ir <= ar1):
            return None
        if cols[-1] == rc0 - 1:
            step = (0, -1)
            if strict and cols[0] - 1 != ac1:
                return None
        elif cols[0] == rc1 + 1:
            step = (0, 1)
            if strict and cols[-1] + 1 != ac0:
                return None
        else:
            return None
        nsteps = len(cols)

    return {"anchor": ca, "block": cb, "rect": rect, "stem": stem,
            "step": step, "nsteps": nsteps}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    submit_sel = [0, 0, hi - 1, wi - 1]

    colors = sorted(set(I.flatten().tolist()))
    counts = Counter(I.flatten().tolist())

    found = None
    for strict in (True, False):
        cands = []
        for ca, cb in permutations(colors, 2):
            res = _analyze(I, ca, cb, strict=strict)
            if res is None:
                continue
            others = [c for c in colors if c != ca and c != cb]
            if not others:
                continue
            bg = max(others, key=lambda c: counts[c])
            res["bg"] = bg
            cands.append(res)
        if cands:
            # background is the colour that fills the canvas -> most cells
            found = max(cands, key=lambda d: counts[d["bg"]])
            break

    if found is None:
        ops.append(34)
        sels.append(submit_sel)
        return ops, sels

    bg = found["bg"]
    blockc = found["block"]
    rect = sorted(found["rect"])
    stem = sorted(found["stem"])
    dr, dc = found["step"]
    nsteps = found["nsteps"]

    # 1. the connector stem is scaffolding: erase it back to background
    ops.append(int(bg))
    sels.append(sel_of(stem))

    dst = [(r + dr * nsteps, c + dc * nsteps) for r, c in rect]

    if blockc != 0:
        # 2. slide the block along the stem until it touches the anchor rectangle
        move_op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        ops.append(move_op)
        sels.append(sel_of(rect))          # first Move grabs the block
        for _ in range(nsteps - 1):
            ops.append(move_op)
            sels.append(sel_of([]))        # empty -> keep the same object grabbed
        # 3. the block's original footprint is left at 0; restore background there
        hole = sorted(set(rect) - set(dst))
        if bg != 0 and hole:
            ops.append(int(bg))
            sels.append(sel_of(hole))
    else:
        # colour-0 blocks cannot be carried by ARCLE's object buffer: paint instead
        vacated = sorted(set(rect) - set(dst))
        if vacated:
            ops.append(int(bg))
            sels.append(sel_of(vacated))
        ops.append(0)
        sels.append(sel_of(dst))

    ops.append(34)
    sels.append(submit_sel)
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
