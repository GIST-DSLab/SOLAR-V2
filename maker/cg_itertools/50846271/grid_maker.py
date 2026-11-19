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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc, crossc, noisec = random.sample(cols, 3)
    return {"bgc": bgc, "crossc": crossc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, crossc, noisec) -> dict:
    cf1 = lambda d: {(d//2, 0), (d//2, d-1)} | set(sample(totuple(connect((d//2, 0), (d//2, d-1))), randint(1, d)))
    cf2 = lambda d: {(0, d//2), (d - 1, d//2)} | set(sample(totuple(connect((0, d//2), (d-1, d//2))), randint(1, d)))
    cf3 = lambda d: set(sample(totuple(remove((d//2, d//2), connect((d//2, 0), (d//2, d-1)))), randint(1, d-1))) | set(sample(totuple(remove((d//2, d//2), connect((0, d//2), (d - 1, d//2)))), randint(1, d-1)))
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
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    A = np.asarray(I, dtype=int)
    hi, wi = A.shape

    # Rule (the golf entry's reading): the input hides complete plus signs of one
    # size; each is partly drawn in the cross colour, the rest of its arms is
    # disguised as noise. Find the disjoint set of pluses that accounts for every
    # cross-coloured cell, then reveal each plus's disguised arm cells in 8.
    cnt = Counter(A.flatten().tolist())
    crossc = min(cnt, key=lambda c: cnt[c])

    # which colour disguises the hidden arm cells: read from the demonstrations
    noise_opts = []
    if examples:
        vc = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape == eo.shape:
                vc.update(ei[eo == 8].tolist())
        for c, _ in vc.most_common():
            if c != crossc and c in cnt and c not in noise_opts:
                noise_opts.append(c)
    for c, _ in cnt.most_common():
        if c != crossc and c not in noise_opts:
            noise_opts.append(c)

    marks = {(r, c) for r in range(hi) for c in range(wi) if A[r, c] == crossc}

    def plus(r, c, half):
        cells = [(r, c + k) for k in range(-half, half + 1)]
        cells += [(r + k, c) for k in range(-half, half + 1) if k != 0]
        return cells

    def valid_pattern(r, c, half, cells, noisec):
        if any(not (0 <= rr < hi and 0 <= cc < wi) for rr, cc in cells):
            return None
        if any(A[rr, cc] not in (crossc, noisec) for rr, cc in cells):
            return None
        hor = [(r, c + k) for k in range(-half, half + 1)]
        ver = [(r + k, c) for k in range(-half, half + 1)]
        mh = [p for p in hor if p in marks and p != (r, c)]
        mv = [p for p in ver if p in marks and p != (r, c)]
        cm = (r, c) in marks
        full = False
        if not mv and (r, c - half) in marks and (r, c + half) in marks:
            full = True          # horizontal arm drawn end to end
        elif not mh and (r - half, c) in marks and (r + half, c) in marks:
            full = True          # vertical arm drawn end to end
        elif mh and mv and not cm:
            pass                 # both arms partly drawn, centre hidden
        else:
            return None
        return full

    def solve(half, noisec):
        cand = {}
        info = {}
        for r in range(hi):
            for c in range(wi):
                cells = plus(r, c, half)
                f = valid_pattern(r, c, half, cells, noisec)
                if f is None:
                    continue
                info[(r, c)] = (frozenset(cells), f)
                for p in cells:
                    if p in marks:
                        cand.setdefault(p, []).append((r, c))
        if any(p not in cand for p in marks):
            return None

        def far(a, b):
            # placed pluses keep a gap of >= 3 cells (Chebyshev distance >= 4)
            for (r1, c1) in info[a][0]:
                for (r2, c2) in info[b][0]:
                    if max(abs(r1 - r2), abs(c1 - c2)) < 4:
                        return False
            return True

        budget = [0]

        def rec(chosen, covered):
            budget[0] += 1
            if budget[0] > 20000:
                return None
            left = [p for p in marks if p not in covered]
            if not left:
                return list(chosen)
            best = None
            for p in left:
                opts = [k for k in cand[p]
                        if not (info[k][0] & covered) and all(far(k, q) for q in chosen)
                        and (info[k][0] & marks) <= (marks - covered)]
                if best is None or len(opts) < len(best[1]):
                    best = (p, opts)
                if not opts:
                    return None
            for k in best[1]:
                res = rec(chosen + [k], covered | (info[k][0] & marks))
                if res is not None:
                    return res
            return None

        res = rec([], frozenset())
        if res is None or not any(info[k][1] for k in res):
            return None
        return res

    found = None
    for noisec in noise_opts:
        for half in (3, 2, 1):
            res = solve(half, noisec)
            if res is not None:
                found = (res, noisec, half)
                break
        if found is not None:
            break

    ops, sels = [], []
    if found is not None:
        centres, noisec, half = found
        for (r, c) in sorted(centres):
            # this plus's disguised arm cells: its cells that are noise-coloured in I
            hidden = [p for p in plus(r, c, half) if A[p] == noisec]
            if hidden:
                ops.append(8)
                sels.append(sel_of(hidden))

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
