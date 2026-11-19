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
    from dsl import *
    from utils import *
except Exception:
    pass
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, objc = random.sample(range(10), 2)
    # the sliding block must be nonzero so ARCLE's object buffer can carry it
    otherc = random.choice([c for c in range(1, 10) if c not in (bgc, objc)])
    return {"bgc": bgc, "objc": objc, "otherc": otherc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, otherc=None) -> dict:
    if bgc is None or objc is None or otherc is None:
        c = sample_colors()
        bgc, objc, otherc = c["bgc"], c["objc"], c["otherc"]
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    objh = unifint(diff_lb, diff_ub, (2, h - 5))
    objw = unifint(diff_lb, diff_ub, (2, w - 5))
    loci = randint(0, h - objh)
    locj = randint(0, w - objw)
    obj = backdrop(frozenset({(loci, locj), (loci + objh - 1, locj + objw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, obj)
    margs = (h - (loci + objh), w - (locj + objw), loci, locj)
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


def _vertical(G, ca, cb):
    # block of colour cb sits above/below the solid rectangle of colour ca,
    # joined to it by a 1-wide stem; returns (block cells, stem cells, step, nsteps)
    A = np.argwhere(G == ca)
    B = np.argwhere(G == cb)
    if len(A) == 0 or len(B) == 0:
        return None
    ar0, ar1 = A[:, 0].min(), A[:, 0].max()
    br0, br1 = B[:, 0].min(), B[:, 0].max()
    if br0 > ar1:
        dr = -1
    elif br1 < ar0:
        dr = 1
    else:
        return None
    rows = {}
    for r, c in B:
        rows.setdefault(int(r), []).append(int(c))
    rect_rows = sorted(r for r, cs in rows.items() if len(cs) >= 2)
    stem_rows = sorted(r for r, cs in rows.items() if len(cs) == 1)
    if not rect_rows or not stem_rows:
        return None
    rect = [(r, c) for r in rect_rows for c in rows[r]]
    rr0, rr1 = rect_rows[0], rect_rows[-1]
    cs = sorted(rows[rr0])
    if rr1 - rr0 + 1 != len(rect_rows):
        return None
    for r in rect_rows:
        if sorted(rows[r]) != list(range(cs[0], cs[-1] + 1)):
            return None
    stem = [(r, rows[r][0]) for r in stem_rows]
    expect = list(range(ar1 + 1, rr0)) if dr == -1 else list(range(rr1 + 1, ar0))
    if stem_rows != expect or len({c for _, c in stem}) != 1:
        return None
    return rect, stem, (dr, 0), len(stem_rows)


def _analyze(I, ca, cb):
    res = _vertical(I, ca, cb)
    if res is not None:
        return res
    res = _vertical(I.T, ca, cb)
    if res is None:
        return None
    rect, stem, (dr, _), n = res
    return ([(c, r) for r, c in rect], [(c, r) for r, c in stem], (0, dr), n)


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    submit_sel = [0, 0, hi - 1, wi - 1]
    colors = sorted(set(I.flatten().tolist()))
    counts = Counter(I.flatten().tolist())

    found = None
    for ca in colors:
        A = np.argwhere(I == ca)
        r0, c0 = A.min(0)
        r1, c1 = A.max(0)
        if len(A) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue  # the anchor is a solid rectangle
        for cb in colors:
            if cb == ca:
                continue
            others = [c for c in colors if c not in (ca, cb)]
            if not others:
                continue
            res = _analyze(I, ca, cb)
            if res is None:
                continue
            bg = max(others, key=lambda c: counts[c])
            cand = (counts[bg], bg, cb, res)
            if found is None or cand[0] > found[0]:
                found = cand

    ops, sels = [], []
    if found is None:
        ops.append(34)
        sels.append(submit_sel)
        return ops, sels

    _, bg, blockc, (rect, stem, (dr, dc), nsteps) = found
    rect = sorted(rect)
    stem = sorted(stem)
    dst = [(r + dr * nsteps, c + dc * nsteps) for r, c in rect]
    leftover_stem = sorted(set(stem) - set(dst))   # stem cells the block never reaches
    hole = sorted(set(rect) - set(dst))            # block footprint it vacates

    if blockc != 0:
        # slide the block along its stem (stem length = steps) until it docks
        move_op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        ops.append(move_op)
        sels.append(sel_of(rect))          # first Move grabs the block
        for _ in range(nsteps - 1):
            ops.append(move_op)
            sels.append(sel_of([]))        # keep the same object grabbed
        # cleanup: vacated footprint (left at 0) + any stem the block did not cover
        clean = list(hole) if bg != 0 else []
        clean += leftover_stem
        if clean:
            ops.append(int(bg))
            sels.append(sel_of(sorted(set(clean))))
    else:
        clean = sorted(set(hole) | set(leftover_stem))
        if clean:
            ops.append(int(bg))
            sels.append(sel_of(clean))
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
