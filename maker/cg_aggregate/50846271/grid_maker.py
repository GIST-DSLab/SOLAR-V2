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
    # reject grids whose crosses can be completed in more than one way
    # (e.g. one cell per arm, where the other diagonal centre also fits)
    ops, sels = derive_operations(gi, go, [(gi, go)])
    import numpy as _np
    G = _np.array(gi, dtype=int)
    for o, s in zip(ops, sels):
        if o == 8:
            for r, c in s['cells']:
                G[r, c] = 8
    if not (G == _np.array(go, dtype=int)).all():
        raise ValueError('ambiguous cross completion')
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    A = np.asarray(I, dtype=int)
    hi, wi = A.shape
    ho, wo = np.asarray(O).shape

    # Rule: the rarest colour marks fragments of plus-shaped crosses (all the same
    # odd size per grid). Every cross arm cell that is not cross-coloured is filled with 8.
    # Reading: treat each WHOLE plus as the unit -- find the set of plus centres that
    # explains every cross-coloured cell -- rather than gluing fragments together.
    cnt = Counter(A.flatten().tolist())
    crossc = min(cnt, key=lambda c: cnt[c])
    marks = {(r, c) for r in range(hi) for c in range(wi) if A[r, c] == crossc}

    # The demonstrations show which colour the missing arm cells carry (the noise colour).
    noisec = None
    if examples:
        votes = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            m = (eo == 8) & (ei != 8)
            votes.update(ei[m].tolist())
        if votes:
            noisec = votes.most_common(1)[0][0]
            if noisec not in cnt:
                noisec = None

    def cross_cells(ctr, half):
        r, c = ctr
        return ({(r + k, c) for k in range(-half, half + 1)} |
                {(r, c + k) for k in range(-half, half + 1)})

    def shape_ok(ctr, half, cov):
        r, c = ctr
        hor = {p for p in cov if p[0] == r and p != ctr}
        ver = {p for p in cov if p[1] == c and p != ctr}
        if all(p[0] == r for p in cov) and (r, c - half) in cov and (r, c + half) in cov:
            return True          # a full-span horizontal arm
        if all(p[1] == c for p in cov) and (r - half, c) in cov and (r + half, c) in cov:
            return True          # a full-span vertical arm
        return ctr not in cov and len(hor) > 0 and len(ver) > 0   # pieces of both arms

    def cheb_far(s1, s2):
        for (a, b) in s1:
            for (x, y) in s2:
                if abs(a - x) <= 3 and abs(b - y) <= 3:
                    return False
        return True

    def solve(half):
        cands = {}
        for r in range(half, hi - half):
            for c in range(half, wi - half):
                full = cross_cells((r, c), half)
                cov = full & marks
                if not cov:
                    continue
                if noisec is not None and any(A[p] not in (crossc, noisec) for p in full):
                    continue
                if not shape_ok((r, c), half, cov):
                    continue
                cands[(r, c)] = (frozenset(cov), full)
        by_cell = {}
        for ctr, (cov, full) in cands.items():
            for p in cov:
                by_cell.setdefault(p, []).append(ctr)
        if any(p not in by_cell for p in marks):
            return None
        chosen = []
        budget = [20000]

        def bt(left):
            if not left:
                return True
            budget[0] -= 1
            if budget[0] < 0:
                return False
            p = min(left, key=lambda q: len(by_cell[q]))
            for ctr in by_cell[p]:
                cov, full = cands[ctr]
                if not cov <= left:
                    continue
                if not all(cheb_far(full, cands[o][1]) for o in chosen):
                    continue
                chosen.append(ctr)
                if bt(left - cov):
                    return True
                chosen.pop()
            return False

        return sorted(chosen) if bt(frozenset(marks)) else None

    centers, half = None, 0
    for nc in ([noisec, None] if noisec is not None else [None]):
        noisec = nc
        for hf in range(1, max(hi, wi) // 2 + 1):
            res = solve(hf)
            if res is not None:
                centers, half = res, hf
                break
        if centers is not None:
            break

    ops, sels = [], []
    for (r, c) in centers or []:
        # one cross at a time: fill each gap run of the vertical arm, then the
        # horizontal arm (the centre, if missing, goes with the vertical arm)
        vert = [(r + k, c) for k in range(-half, half + 1)]
        horiz = [(r, c + k) for k in range(-half, half + 1) if k != 0]
        for line in (vert, horiz):
            run = []
            for p in line + [None]:
                if p is not None and p not in marks and (not run or abs(p[0] - run[-1][0]) + abs(p[1] - run[-1][1]) == 1):
                    run.append(p)
                    continue
                if run:
                    ops.append(8); sels.append(sel_of(run)); run = []
                if p is not None and p not in marks:
                    run = [p]
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
