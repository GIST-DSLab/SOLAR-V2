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
import numpy as np
from itertools import product
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    patchcol = random.choice(cols)
    bgc = random.choice([c for c in cols if c != patchcol])
    return {"bgc": bgc, "patchcol": patchcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, patchcol) -> dict:
    cols = interval(0, 10, 1)
    hub = min(15, (min(max_h, max_w) + 2) // 2)
    hub = max(6, hub)
    remcols = difference(cols, (bgc, patchcol))
    while True:
        h = unifint(diff_lb, diff_ub, (6, hub))
        w = h
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
        if succ > 0:
            return {'input': gi, 'output': go}


def _sim(I, ops, sels):
    """Local ARCLE model (Color / CopyI / CopyO / Paste / Flip / Rotate)."""
    G = np.asarray(I, dtype=int).copy()
    clip = None
    changed = []
    for op, sel in zip(ops, sels):
        before = G.copy()
        if isinstance(sel, dict):
            cells = [tuple(c) for c in sel["cells"]]
        else:
            r, c, h, w = sel
            cells = [(rr, cc) for rr in range(r, r + h + 1) for cc in range(c, c + w + 1)]
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        if 0 <= op <= 9:
            for (r, c) in cells:
                G[r, c] = op
        elif op == 29:
            clip = G[min(rs):max(rs) + 1, min(cs):max(cs) + 1].copy()
        elif op == 28:
            clip = np.asarray(I, dtype=int)[min(rs):max(rs) + 1, min(cs):max(cs) + 1].copy()
        elif op == 30:
            if clip is not None:
                r0, c0 = min(rs), min(cs)
                ch, cw = clip.shape
                for i in range(ch):
                    for j in range(cw):
                        if clip[i, j] != 0 and r0 + i < G.shape[0] and c0 + j < G.shape[1]:
                            G[r0 + i, c0 + j] = clip[i, j]
        elif op in (24, 25, 26, 27):
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            reg = G[r0:r1 + 1, c0:c1 + 1]
            if op == 24:
                reg2 = np.rot90(reg, 1)
            elif op == 25:
                reg2 = np.rot90(reg, 3)
            elif op == 26:
                reg2 = np.fliplr(reg)
            else:
                reg2 = np.flipud(reg)
            if reg2.shape == reg.shape:
                G[r0:r1 + 1, c0:c1 + 1] = reg2
        changed.append(not np.array_equal(before, G))
    return G, changed


def _patch_color(I, O):
    """The occluder colour: the solid rectangles that break the grid's mirrors."""
    N, M = I.shape
    cand = None
    for i in range(N):
        for j in range(M):
            pairs = [(j, i)] if (j < N and i < M) else []
            if 2 <= i <= N - 1 and 2 <= N + 1 - i <= N - 1:
                pairs.append((N + 1 - i, j))
            if 2 <= j <= M - 1 and 2 <= M + 1 - j <= M - 1:
                pairs.append((i, M + 1 - j))
            for (a, b) in pairs:
                if I[i, j] != I[a, b]:
                    s = {int(I[i, j]), int(I[a, b])}
                    cand = s if cand is None else (cand & s)
    if cand and len(cand) == 1:
        return cand.pop()
    extra = set(np.unique(I).tolist()) - set(np.unique(O).tolist())
    if len(extra) == 1:
        return extra.pop()
    diff = I != O
    if diff.any():
        return int(Counter(I[diff].tolist()).most_common(1)[0][0])
    return None


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    N, M = I.shape
    full = [0, 0, N - 1, M - 1]
    if np.array_equal(I, O):
        return [34], [full]

    patchcol = _patch_color(I, O)
    patches = [tuple(p) for p in np.argwhere(I == patchcol)]

    ops0, sels0 = [], []
    if patchcol != 0 and patches:
        # punch the occluding patches out: 0 means "unknown", and 0 is exactly
        # what Paste lets through, so a reflection shows through only there
        ops0.append(0)
        sels0.append(sel_of(patches))

    band_r = [2, 0, N - 3, M - 1]     # rows 2..N-1 : whole band, the up/down mirror
    band_c = [0, 2, N - 1, M - 3]     # cols 2..M-1 : whole band, the left/right mirror
    block = [2, 2, N - 3, M - 3]      # rows/cols 2.. : the anti-diagonal square

    # A pass = reflect one region of the grid, then lay the punched grid back on
    # top of it. Every bbox below is exactly the rectangle being reflected.
    PASSES = []
    if N == M:
        # reflection across the main diagonal = rot90 CCW then flip up/down
        PASSES.append(("T1", [29, 24, 27, 30], [full, full, full, [0, 0, 0, 0]]))
        PASSES.append(("T2", [29, 25, 26, 30], [full, full, full, [0, 0, 0, 0]]))
        # reflection across the anti-diagonal of the rows/cols 2.. square
        PASSES.append(("A1", [29, 24, 26, 30], [block, block, block, [2, 2, 0, 0]]))
        PASSES.append(("A2", [29, 25, 27, 30], [block, block, block, [2, 2, 0, 0]]))
    PASSES.append(("H", [29, 27, 30], [band_r, band_r, [2, 0, 0, 0]]))   # up/down
    PASSES.append(("V", [29, 26, 30], [band_c, band_c, [0, 2, 0, 0]]))   # left/right

    def clean(ops, sels):
        """Reaches O, every op visibly acts, and no single op can be dropped."""
        G, changed = _sim(I, ops, sels)
        if not np.array_equal(G, O):
            return False
        for k, o in enumerate(ops):
            if o not in (28, 29) and not changed[k]:
                return False
        for k in range(len(ops)):
            G2, _ = _sim(I, ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:])
            if G2.shape == O.shape and np.array_equal(G2, O):
                return False
        return True

    if N > 4 and M > 4:
        for L in (1, 2, 3):
            for seq in product(PASSES, repeat=L):
                ops = list(ops0)
                sels = list(sels0)
                for _, pops, psels in seq:
                    ops += pops
                    sels += psels
                if clean(ops, sels):
                    ops.append(34)
                    sels.append(full)
                    return ops, sels

    # Fallback: keep whichever reflections still reveal something, close any
    # hole that is left from its mirror value, then drop anything droppable.
    ops, sels = list(ops0), list(sels0)
    for _round in range(3):
        for _, pops, psels in PASSES:
            cand_ops, cand_sels = ops + pops, sels + psels
            if not np.array_equal(_sim(I, cand_ops, cand_sels)[0], _sim(I, ops, sels)[0]):
                ops, sels = cand_ops, cand_sels
    G, _ = _sim(I, ops, sels)
    rest = {}
    for r in range(N):
        for c in range(M):
            if G[r, c] != O[r, c]:
                rest.setdefault(int(O[r, c]), []).append((r, c))
    for col, cells in sorted(rest.items()):
        ops.append(col)
        sels.append(sel_of(cells))
    k = 0
    while k < len(ops):
        G2, _ = _sim(I, ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:])
        if G2.shape == O.shape and np.array_equal(G2, O):
            ops.pop(k)
            sels.pop(k)
            k = 0
        else:
            k += 1
    ops.append(34)
    sels.append(full)
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
