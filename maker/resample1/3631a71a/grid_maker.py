"""
ARC Task: 3631a71a (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    patchcol = random.choice(cols)
    bgc = random.choice([c for c in cols if c != patchcol])
    return {"bgc": bgc, "patchcol": patchcol}


def _orbit(i, j, N):
    # The grid is the top-left NxN of an (N+2)x(N+2) grid that stays the same
    # under transpose, up/down mirroring and left/right mirroring.
    M = N + 2
    out = set()
    for a in (i, M - 1 - i):
        for b in (j, M - 1 - j):
            for p, q in ((a, b), (b, a)):
                if 0 <= p < N and 0 <= q < N:
                    out.add((p, q))
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, patchcol=None) -> dict:
    cols = interval(0, 10, 1)
    if patchcol is None:
        patchcol = choice(cols)
    if bgc is None or bgc == patchcol:
        bgc = choice(remove(patchcol, cols))
    hmax = min(15, (min(max_h, max_w) + 2) // 2)
    hmax = max(6, hmax)
    h = unifint(diff_lb, diff_ub, (6, hmax))
    w = h
    remcols = difference(cols, (bgc, patchcol))
    c = canvas(bgc, (h, w))
    inds = sfilter(asindices(c), lambda ij: ij[0] >= ij[1])
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    ncells = unifint(diff_lb, diff_ub, (1, len(inds)))
    cells = set(sample(totuple(inds), ncells))
    obj = {(choice(ccols), ij) for ij in cells}
    c = paint(dmirror(paint(c, obj)), obj)
    c = hconcat(c, vmirror(c))
    c = vconcat(c, hmirror(c))
    cutoff = 2
    go = dmirror(dmirror(c[:-cutoff])[:-cutoff])
    gi = tuple(e for e in go)
    forbidden = asindices(canvas(-1, (cutoff, cutoff)))
    dmirrareaL = shift(asindices(canvas(-1, (h*2-2*cutoff, cutoff))), (cutoff, 0))
    dmirrareaT = shift(asindices(canvas(-1, (cutoff, 2*w-2*cutoff))), (0, cutoff))
    inds1 = sfilter(asindices(gi), lambda ij: cutoff <= ij[0] < h and cutoff <= ij[1] < w and ij[0] >= ij[1])
    inds2 = dmirror(inds1)
    inds3 = shift(hmirror(inds1), (h-cutoff, 0))
    inds4 = shift(hmirror(inds2), (h-cutoff, 0))
    inds5 = shift(vmirror(inds1), (0, w-cutoff))
    inds6 = shift(vmirror(inds2), (0, w-cutoff))
    inds7 = shift(hmirror(vmirror(inds1)), (h-cutoff, w-cutoff))
    inds8 = shift(hmirror(vmirror(inds2)), (h-cutoff, w-cutoff))
    f1 = identity
    f2 = dmirror
    f3 = lambda x: hmirror(shift(x, invert((h-cutoff, 0))))
    f4 = lambda x: dmirror(hmirror(shift(x, invert((h-cutoff, 0)))))
    f5 = lambda x: vmirror(shift(x, invert((0, w-cutoff))))
    f6 = lambda x: dmirror(vmirror(shift(x, invert((0, w-cutoff)))))
    f7 = lambda x: vmirror(hmirror(shift(x, invert((h-cutoff, w-cutoff)))))
    f8 = lambda x: dmirror(vmirror(hmirror(shift(x, invert((h-cutoff, w-cutoff))))))
    indsarr = [inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8]
    farr = [f1, f2, f3, f4, f5, f6, f7, f8]
    ndist = unifint(diff_lb, diff_ub, (1, int((2*h*2*w) ** 0.5)))
    succ = 0
    tr = 0
    maxtr = 10 * ndist
    fullh, fullw = shape(gi)
    while succ < ndist and tr < maxtr:
        tr += 1
        oh = randint(2, h//2+1)
        ow = randint(2, w//2+1)
        loci = randint(0, fullh - oh)
        locj = randint(0, fullw - ow)
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        isleft = set()
        gi2 = fill(gi, patchcol, bd)
        if patchcol in palette(toobject(forbidden, gi2)):
            continue
        oo1 = toindices(sfilter(toobject(dmirrareaL, gi2), lambda cij: cij[0] != patchcol))
        oo2 = toindices(sfilter(toobject(dmirrareaT, gi2), lambda cij: cij[0] != patchcol))
        oo2 = frozenset({(ij[1], ij[0]) for ij in oo2})
        if oo1 | oo2 != dmirrareaL:
            continue
        for ii, ff in zip(indsarr, farr):
            oo = toobject(ii, gi2)
            rem = toindices(sfilter(oo, lambda cij: cij[0] != patchcol))
            if len(rem) > 0:
                isleft = isleft | ff(rem)
        if isleft != inds1:
            continue
        succ += 1
        gi = gi2
    # Reject the rare instance where a patch cell has no unpatched symmetric
    # partner, because its output colour cannot be recovered from the input.
    gi_np = np.array(gi)
    N = gi_np.shape[0]
    for i in range(N):
        for j in range(N):
            if gi_np[i, j] == patchcol and all(gi_np[p, q] == patchcol for p, q in _orbit(i, j, N)):
                raise ValueError("unrecoverable patch")
    return {'input': gi, 'output': go}


def _restore(I, pc):
    N = I.shape[0]
    R = I.copy()
    for i in range(N):
        for j in range(N):
            if I[i, j] == pc:
                vals = [I[p, q] for p, q in _orbit(i, j, N) if I[p, q] != pc]
                if not vals:
                    return None
                R[i, j] = vals[0]
    return R


def _symmetric(R):
    if not np.array_equal(R, R.T):
        return False
    S = R[2:, :]
    if not np.array_equal(S, S[::-1, :]):
        return False
    S = R[:, 2:]
    return np.array_equal(S, S[:, ::-1])


def _find_patchcol(I, examples=None):
    # Patch colour = the colour that, once replaced by its symmetric partners,
    # leaves a fully symmetric grid. Colours seen as patches in the
    # demonstrations are tried first.
    cands = []
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            if ei.shape != eo.shape:
                continue
            diff = ei != eo
            if diff.any():
                cs = set(ei[diff].tolist())
                if len(cs) == 1:
                    cands.append(cs.pop())
    order = []
    for c in cands + sorted(set(I.flatten().tolist())):
        if c not in order:
            order.append(c)
    for c in order:
        if c not in I:
            continue
        R = _restore(I, c)
        if R is not None and _symmetric(R):
            return c, R
    return None, I.copy()


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    N = I.shape[0]
    ops, sels = [], []
    pc, R = _find_patchcol(I, examples)
    if pc is not None:
        # Repair one patch (connected region of the patch colour) at a time:
        # each patch cell takes the colour of its symmetric partner in I.
        seen = np.zeros_like(I, dtype=bool)
        for r0 in range(N):
            for c0 in range(N):
                if I[r0, c0] != pc or seen[r0, c0]:
                    continue
                comp = []
                dq = deque([(r0, c0)]); seen[r0, c0] = True
                while dq:
                    r, c = dq.popleft(); comp.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < N and 0 <= cc < N and not seen[rr, cc] and I[rr, cc] == pc:
                            seen[rr, cc] = True; dq.append((rr, cc))
                bycol = {}
                for r, c in comp:
                    bycol.setdefault(int(R[r, c]), []).append((r, c))
                for col in sorted(bycol):
                    if col == pc:
                        continue
                    ops.append(col); sels.append(sel_of(bycol[col]))
    ops.append(34); sels.append([0, 0, N - 1, N - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3631a71a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3631a71a"
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
                                f"for task 3631a71a"
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
                    f"Failed to build a complete episode for task 3631a71a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3631a71a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
