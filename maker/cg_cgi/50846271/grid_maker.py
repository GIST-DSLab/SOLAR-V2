"""
ARC Task: 50846271 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 8]
    bgc, crossc, noisec = random.sample(cols, 3)
    return {"bgc": bgc, "crossc": crossc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, crossc=None, noisec=None, **kw) -> dict:
    from dsl import (remove, interval, connect, totuple, canvas, asindices, fill, shift,
                     mapply, neighbors, sfilter, recolor, toindices, paint)
    from utils import unifint
    from random import sample, randint, choice
    cols = remove(8, interval(0, 10, 1))
    if bgc is None or crossc is None or noisec is None:
        bgc, crossc, noisec = sample(cols, 3)
    cf1 = lambda d: {(d//2, 0), (d//2, d-1)} | set(sample(totuple(connect((d//2, 0), (d//2, d-1))), randint(1, d)))
    cf2 = lambda d: {(0, d//2), (d - 1, d//2)} | set(sample(totuple(connect((0, d//2), (d-1, d//2))), randint(1, d)))

    def cf3(d):
        # partial cross with marks on both arms; a single mark on each arm would
        # also fit a second, diagonally offset centre, so keep that case out
        while True:
            hs = randint(1, d - 1)
            vs = randint(1, d - 1)
            if hs > 1 or vs > 1:
                break
        return (set(sample(totuple(remove((d//2, d//2), connect((d//2, 0), (d//2, d-1)))), hs)) |
                set(sample(totuple(remove((d//2, d//2), connect((0, d//2), (d - 1, d//2)))), vs)))

    cf = lambda d: choice((cf1, cf2, cf3))(d)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    dim = unifint(diff_lb, diff_ub, (1, 3))
    dim = 2 * dim + 1
    cross = connect((dim//2, 0), (dim//2, dim - 1)) | connect((0, dim//2), (dim - 1, dim//2))
    gi = canvas(bgc, (h, w))
    namt = unifint(diff_lb, diff_ub, (int(0.35 * h * w), int(0.65 * h * w)))
    inds = asindices(gi)
    noise = sample(totuple(inds), namt)
    gi = fill(gi, noisec, noise)
    initcross = choice((cf1, cf2))(dim)
    loci = randint(0, h - dim)
    locj = randint(0, w - dim)
    delt = shift(cross - initcross, (loci, locj))
    gi = fill(gi, crossc, shift(initcross, (loci, locj)))
    gi = fill(gi, noisec, delt)
    go = fill(gi, 8, delt)
    plcd = shift(cross, (loci, locj))
    nbhs = mapply(neighbors, plcd)
    inds = (inds - plcd) - nbhs
    nbhs2 = mapply(neighbors, nbhs)
    inds = inds - nbhs2
    inds = inds - mapply(neighbors, nbhs2)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) / (10 * dim)))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - dim and ij[1] <= w - dim)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        marked = shift(cf(dim), loc)
        full = shift(cross, loc)
        unmarked = full - marked
        inobj = recolor(noisec, unmarked) | recolor(crossc, marked)
        outobj = recolor(8, unmarked) | recolor(crossc, marked)
        outobji = toindices(outobj)
        if outobji.issubset(inds):
            dnbhs = mapply(neighbors, outobji)
            dnbhs2 = mapply(neighbors, dnbhs)
            inds = (inds - outobji) - (dnbhs | dnbhs2 | mapply(neighbors, dnbhs2))
            succ += 1
            gi = paint(gi, inobj)
            go = paint(go, outobj)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    from collections import Counter
    from maker.sel_helpers import sel_of
    A = np.asarray(I, dtype=int)
    hi, wi = A.shape

    # The marker colour is the rarest colour; it draws partial plus shapes.
    cnt = Counter(A.flatten().tolist())
    crossc = min(cnt, key=lambda c: cnt[c])
    marks = {(r, c) for r in range(hi) for c in range(wi) if A[r, c] == crossc}

    # Colour carried by the hidden cross cells: read from the demonstrations
    # (the input colour of cells that the rule turns into 8).
    fill_cols = None
    if examples:
        fc = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            m = (eo == 8) & (ei != 8)
            fc.update(ei[m].tolist())
        if fc:
            nc = fc.most_common(1)[0][0]
            if nc in cnt and nc != crossc:
                fill_cols = [nc]
    if fill_cols is None:
        fill_cols = [c for c in cnt if c != crossc]

    def arms(r, c, hf):
        hor = [(r, c + k) for k in range(-hf, hf + 1)]
        ver = [(r + k, c) for k in range(-hf, hf + 1)]
        return hor, ver

    def solve(d, noisec):
        # candidate plus-centres of arm length d explaining the marks
        hf = d // 2
        allowed = lambda p: A[p] == crossc or A[p] == noisec
        cand = {}
        for (mr, mc) in marks:
            for k in range(-hf, hf + 1):
                for (r, c) in ((mr, mc + k), (mr + k, mc)):
                    if (r, c) in cand or r - hf < 0 or c - hf < 0 or r + hf >= hi or c + hf >= wi:
                        continue
                    hor, ver = arms(r, c, hf)
                    cells = set(hor) | set(ver)
                    if not all(allowed(p) for p in cells):
                        continue
                    on = marks & cells
                    # no foreign marks within 3 cells of this cross
                    near = {p for p in marks
                            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 3 for q in cells)}
                    if near != on:
                        continue
                    hs = set(hor); vs = set(ver)
                    typA = on <= hs and hor[0] in on and hor[-1] in on
                    typB = on <= vs and ver[0] in on and ver[-1] in on
                    typC = ((r, c) not in on and (on & (hs - {(r, c)})) and (on & (vs - {(r, c)})))
                    if typA or typB or typC:
                        cand[(r, c)] = (cells, on, bool(typA or typB))
        # exact cover of all marks by mutually separated crosses
        order = sorted(marks)
        sols = []

        def rec(i, chosen, covered):
            if len(sols) > 1:
                return
            while i < len(order) and order[i] in covered:
                i += 1
            if i == len(order):
                sols.append(list(chosen)); return
            p = order[i]
            for ctr, (cells, on, ab) in cand.items():
                if p not in on or on & covered:
                    continue
                ok = all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 4
                         for o in chosen for a in cand[o][0] for b in cells)
                if not ok:
                    continue
                chosen.append(ctr)
                rec(i + 1, chosen, covered | on)
                chosen.pop()
        rec(0, [], frozenset())
        sols = [s for s in sols if any(cand[c][2] for c in s)]
        return sols, cand

    result = None
    for d in (7, 5, 3):
        for nc in fill_cols:
            sols, cand = solve(d, nc)
            if sols:
                result = (d, sols[0], cand)
                break
        if result:
            break

    ops, sels = [], []
    if result:
        d, centers, cand = result
        hf = d // 2
        painted = set()
        for (r, c) in sorted(centers):
            hor, ver = arms(r, c, hf)
            # complete each arm of this plus: paint every hidden run of it with 8
            for arm in (ver, hor):
                run = []
                for p in arm + [None]:
                    if p is not None and p not in marks and p not in painted:
                        run.append(p)
                    else:
                        if run:
                            ops.append(8); sels.append(sel_of(run))
                            painted.update(run)
                        run = []
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
                        f"num_examples+1 ({num_examples + 1}) for task 50846271"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 50846271"
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
                                f"for task 50846271"
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
                    f"Failed to build a complete episode for task 50846271 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"50846271-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
