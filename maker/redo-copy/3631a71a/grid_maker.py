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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    patchcol = random.choice(cols)                              # colour of the occluding rectangles
    bgc = random.choice([c for c in cols if c != patchcol])
    return {"bgc": bgc, "patchcol": patchcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, patchcol) -> dict:
    cols = interval(0, 10, 1)
    lim = min(max_h, max_w)
    hi_h = max(6, min(15, (lim + 2) // 2))          # canvas is (2h-2) x (2h-2)
    h = unifint(diff_lb, diff_ub, (6, hi_h))
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
    dmirrareaL = shift(asindices(canvas(-1, (h * 2 - 2 * cutoff, cutoff))), (cutoff, 0))
    dmirrareaT = shift(asindices(canvas(-1, (cutoff, 2 * w - 2 * cutoff))), (0, cutoff))
    inds1 = sfilter(asindices(gi), lambda ij: cutoff <= ij[0] < h and cutoff <= ij[1] < w and ij[0] >= ij[1])
    inds2 = dmirror(inds1)
    inds3 = shift(hmirror(inds1), (h - cutoff, 0))
    inds4 = shift(hmirror(inds2), (h - cutoff, 0))
    inds5 = shift(vmirror(inds1), (0, w - cutoff))
    inds6 = shift(vmirror(inds2), (0, w - cutoff))
    inds7 = shift(hmirror(vmirror(inds1)), (h - cutoff, w - cutoff))
    inds8 = shift(hmirror(vmirror(inds2)), (h - cutoff, w - cutoff))
    f1 = identity
    f2 = dmirror
    f3 = lambda x: hmirror(shift(x, invert((h - cutoff, 0))))
    f4 = lambda x: dmirror(hmirror(shift(x, invert((h - cutoff, 0)))))
    f5 = lambda x: vmirror(shift(x, invert((0, w - cutoff))))
    f6 = lambda x: dmirror(vmirror(shift(x, invert((0, w - cutoff)))))
    f7 = lambda x: vmirror(hmirror(shift(x, invert((h - cutoff, w - cutoff)))))
    f8 = lambda x: dmirror(vmirror(hmirror(shift(x, invert((h - cutoff, w - cutoff))))))
    indsarr = [inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8]
    farr = [f1, f2, f3, f4, f5, f6, f7, f8]
    ndist = unifint(diff_lb, diff_ub, (1, int((2 * h * 2 * w) ** 0.5)))
    succ = 0
    tr = 0
    maxtr = 10 * ndist
    fullh, fullw = shape(gi)
    while succ < ndist and tr < maxtr:
        tr += 1
        oh = randint(2, h // 2 + 1)
        ow = randint(2, w // 2 + 1)
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
    if gi == go:
        raise ValueError("no patch placed")
    # the original coverage test mirrors subsets inside their own bbox, so it can
    # let through instances where a whole symmetry orbit is occluded (information
    # destroyed).  Keep only instances that are actually solvable/identifiable.
    gia = np.array(gi, dtype=int)
    orbs = _symmetry_orbits(gia.shape[0])
    for g in orbs:
        rest = {int(gia[r, cc]) for r, cc in g if int(gia[r, cc]) != patchcol}
        if len(rest) != 1:
            raise ValueError("occluded beyond recovery")
    if _find_patchcol(gia, orbs) != patchcol:
        raise ValueError("patch colour not identifiable")
    return {'input': gi, 'output': go}


def _symmetry_orbits(n):
    """Cells (r,c) of an n x n grid grouped by this task's symmetry group.

    The hidden picture is a transpose-symmetric block mirrored right and down,
    with the last two rows/columns then dropped.  What survives on the n x n
    canvas is invariant under
      (r, c) -> (c, r)        main-diagonal mirror
      (r, c) -> (n+1-r, c)    row mirror about the axis shifted by 2 (r >= 2)
      (r, c) -> (r, n+1-c)    col mirror about the axis shifted by 2 (c >= 2)
    Cells sharing an orbit must all show the same colour.
    """
    parent = list(range(n * n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in range(n):
        for c in range(n):
            i = r * n + c
            union(i, c * n + r)
            if r >= 2:
                union(i, (n + 1 - r) * n + c)
            if c >= 2:
                union(i, r * n + (n + 1 - c))
    groups = {}
    for r in range(n):
        for c in range(n):
            groups.setdefault(find(r * n + c), []).append((r, c))
    return list(groups.values())


def _find_patchcol(I, orbits):
    """Colour of the occluding rectangles, measured from I alone.

    It never occurs in the top-left 2x2 corner, and once its cells are ignored
    every symmetry orbit must show exactly one remaining colour (and not become
    empty).  Any other colour breaks one of those two conditions.
    """
    corner = set(I[0:2, 0:2].flatten().tolist())
    cands = []
    for v in sorted(set(I.flatten().tolist())):
        if v in corner:
            continue
        good = True
        for g in orbits:
            rest = {int(I[r, c]) for r, c in g if int(I[r, c]) != v}
            if len(rest) != 1:
                good = False
                break
        if good:
            cands.append(v)
    if not cands:
        return None
    return min(cands, key=lambda v: int((I == v).sum()))


def _components(cells):
    """4-connected components (the individual blots), ordered by top-left cell."""
    todo = set(cells)
    comps = []
    while todo:
        seed = min(todo)
        stack = [seed]
        todo.discard(seed)
        comp = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in todo:
                    todo.discard((nr, nc))
                    stack.append((nr, nc))
        comps.append(sorted(comp))
    comps.sort(key=lambda comp: comp[0])
    return comps


def derive_operations(I, O):
    """Repair each occluding blot from the picture's own mirror symmetries.

    Nothing here is read out of O: the patch colour, the affected cells and the
    colour each of them must take are all measured from I plus the symmetry rule.
    """
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    ops, sels = [], []

    orbits = _symmetry_orbits(hi)
    patchcol = _find_patchcol(I, orbits)

    if patchcol is not None:
        # every occluded cell takes the colour its symmetry orbit still shows
        true_col = {}
        for g in orbits:
            rest = [int(I[r, c]) for r, c in g if int(I[r, c]) != patchcol]
            if not rest:
                continue
            col = rest[0]
            for r, c in g:
                if int(I[r, c]) == patchcol:
                    true_col[(r, c)] = col
        # repair one blot at a time, dominant colour of that blot first
        for comp in _components(true_col.keys()):
            by_col = {}
            for rc in comp:
                by_col.setdefault(true_col[rc], []).append(rc)
            for col in sorted(by_col, key=lambda k: (-len(by_col[k]), k)):
                ops.append(int(col))
                sels.append(sel_of(by_col[col]))

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
