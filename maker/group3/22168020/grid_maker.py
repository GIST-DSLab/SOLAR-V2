"""
ARC Task: 22168020 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is a colour role that matters; the object colours
    # play no part in the rule, since each object is filled with its own colour.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _connect(a, b):
    (ai, aj), (bi, bj) = a, b
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    if ai == bi:
        return {(ai, j) for j in range(sj, ej + 1)}
    if aj == bj:
        return {(i, aj) for i in range(si, ei + 1)}
    if bi - ai == bj - aj:
        return {(i, j) for i, j in zip(range(si, ei + 1), range(sj, ej + 1))}
    if bi - ai == aj - bj:
        return {(i, j) for i, j in zip(range(si, ei + 1), range(ej, sj - 1, -1))}
    return set()


def _shoot_line(a, b):
    return _connect(a, b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return int(round(random.uniform(lo, hi_)))

    if bgc is None:
        bgc = random.randint(0, 9)
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    num = unifint(diff_lb, diff_ub, (1, max(1, min(9, (h * w) // 10))))
    remcols = [c for c in cols if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 6 * num

    def mirror(cells, mode):
        if mode == "identity":
            return set(cells)
        if mode == "dmirror":
            return {(j, i) for i, j in cells}
        if not cells:
            return set()
        if mode == "hmirror":
            mi = min(i for i, _ in cells) + max(i for i, _ in cells)
            return {(mi - i, j) for i, j in cells}
        if mode == "vmirror":
            mj = min(j for _, j in cells) + max(j for _, j in cells)
            return {(i, mj - j) for i, j in cells}
        if mode == "cmirror":
            # cmirror = vmirror(dmirror(vmirror(x)))
            return mirror(mirror(mirror(cells, "vmirror"), "dmirror"), "vmirror")
        return set(cells)

    while tr < maxtr and succ < num:
        d = unifint(diff_lb, diff_ub, (2, 5))
        ow = 2 * d
        if not inds:
            tr += 1
            continue
        loci, locj = random.choice(sorted(inds))
        io1 = _connect((loci, locj), (loci + d - 1, locj + d - 1))
        io2 = _connect((loci, locj + ow - 1), (loci + d - 1, locj + d))
        io = io1 | io2 | {(loci + d, locj + d - 1), (loci + d, locj + d)}
        oo = set()
        for a in io:
            for b in io:
                if a[0] == b[0]:
                    oo |= _connect(a, b)
        mode = random.choice(["identity", "dmirror", "cmirror", "hmirror", "vmirror"])
        # apply the mirror jointly so io and oo stay aligned
        allc = io | oo
        if mode in ("hmirror", "vmirror", "cmirror"):
            if mode == "hmirror":
                mi = min(i for i, _ in allc) + max(i for i, _ in allc)
                f = lambda p: (mi - p[0], p[1])
            elif mode == "vmirror":
                mj = min(j for _, j in allc) + max(j for _, j in allc)
                f = lambda p: (p[0], mj - p[1])
            else:
                mi = min(i for i, _ in allc) + max(i for i, _ in allc)
                mj = min(j for _, j in allc) + max(j for _, j in allc)
                # cmirror: reflect over anti-diagonal of the bbox
                ci, cj = min(i for i, _ in allc), min(j for _, j in allc)
                f = lambda p: (ci + (mj - p[1]) - cj, cj + (mi - p[0]) - ci)
            io = {f(p) for p in io}
            oo = {f(p) for p in oo}
        elif mode == "dmirror":
            io = {(j, i) for i, j in io}
            oo = {(j, i) for i, j in oo}
        col = random.choice(remcols)
        if oo.issubset(inds):
            for (i, j) in io:
                gi[i, j] = col
            for (i, j) in oo:
                go[i, j] = col
            succ += 1
            inds = inds - oo
            remcols = [c for c in remcols if c != col]
        tr += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Background: the canvas colour the generator paints first (dominant).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Unit of work: one whole object (= one colour) at a time.
    # Each object is a pair of arms; its region is the wedge they enclose,
    # i.e. every cell on a straight line joining two of the object's cells.
    # The whole wedge is painted in a single stroke of the object's colour.
    cur = I.copy()
    colors = sorted({int(v) for v in I.flatten().tolist() if int(v) != bgc})
    # order objects top-to-bottom, left-to-right by their first cell
    def first_cell(col):
        rr, cc = np.where(I == col)
        return (int(rr.min()), int(cc.min()))
    colors.sort(key=first_cell)

    for col in colors:
        cells = [(int(r), int(c)) for r, c in zip(*np.where(I == col))]
        wedge = set(cells)
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                wedge |= _connect(cells[i], cells[j])
        wedge = {(r, c) for r, c in wedge if 0 <= r < hi and 0 <= c < wi}
        if all(cur[r, c] == col for r, c in wedge):
            continue  # nothing to fill for this object
        ops.append(col)
        sels.append(sel_of(sorted(wedge)))
        for r, c in wedge:
            cur[r, c] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task 22168020"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22168020"
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
                                f"for task 22168020"
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
                    f"Failed to build a complete episode for task 22168020 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22168020-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
