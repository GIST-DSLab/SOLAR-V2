"""
ARC Task: 0dfd9992 (RE-ARC) — LLM-generated grid_maker
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
    # The damage colour is the convention a solver must read from the
    # demonstrations, so it is fixed for the whole episode (bgc with it).
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or noisec is None:
        bgc, noisec = sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            loci = hp * a
            locj = wp * b
            locs.add((loci, locj))
            mf1 = identity if a % 2 == 0 else hmirror
            mf2 = identity if b % 2 == 0 else vmirror
            mf = compose(mf1, mf2)
            go = paint(go, shift(mf(pobj), (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, int((h * w) ** 0.5 // 2)))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        candset = apply(normalize, apply(rbind(toobject, gi2), places))
        if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
           len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2 and \
           (pobj in candset or hmirror(pobj) in candset or vmirror(pobj) in candset
                or hmirror(vmirror(pobj)) in candset):
            succ += 1
            gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """The grid is a wallpaper that repeats by translation; rectangular patches
    of one occluding colour hide parts of it.  Each occluded patch is restored
    from its orbit: CopyI an intact block of the wallpaper lying a whole number
    of periods away and Paste it over the patch (when no translate is intact,
    the mirror image across a fold line is pasted and flipped back in place).
    Paste carries no 0s, so where the copied block holds 0 the patch is first
    cleared with Color0.  Cells no single intact block can cover are painted
    from their orbit one at a time.  O is never consulted."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # ── the occluding colour: present in every demo input, gone from its output
    noise = None
    if examples:
        cand = None
        for a, b in examples:
            a = np.asarray(a, dtype=int); b = np.asarray(b, dtype=int)
            s = set(np.unique(a).tolist()) - set(np.unique(b).tolist())
            if s:
                cand = s if cand is None else (cand & s or cand)
        if cand:
            inI = [c for c in cand if (I == c).any()]
            if inI:
                noise = min(inI, key=lambda c: int((I == c).sum()))
            elif cand:
                noise = min(cand)
    if noise is None:
        # fallback: the colour whose removal leaves the tightest periodic pattern
        best = None
        for col in sorted(set(I.flatten().tolist())):
            m = (I == col)
            if (~m.any(axis=1)).sum() < 2 or (~m.any(axis=0)).sum() < 2:
                continue
            pr = _periods(I, ~m, 0); pc = _periods(I, ~m, 1)
            if not pr or not pc:
                continue
            key = (pr[0] * pc[0], int(m.sum()), col)
            if best is None or key < best:
                best = key
        noise = best[2] if best else None
    if noise is None or not (I == noise).any():
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    occ = (I == noise)
    known = ~occ

    # ── the wallpaper's symmetries, measured on the intact cells: translation
    #    by a period along each axis, plus (where the tiling is mirrored) the
    #    fold lines, which recur every period.
    def lines_for(P, axis):
        n = I.shape[axis]
        out = []
        for phase in range(P):
            ls, sup, good = [], 0, True
            for A in range(phase, 2 * n - 2, P):
                a0, a1 = max(0, A - (n - 1)), min(A, n - 1)
                if a1 - a0 < 1:
                    continue
                if axis == 0:
                    g, k = I[a0:a1 + 1, :], known[a0:a1 + 1, :]
                else:
                    g, k = I[:, a0:a1 + 1].T, known[:, a0:a1 + 1].T
                both = k & k[::-1]
                if (both & (g != g[::-1])).any():
                    good = False
                    break
                sup += int(both.sum())
                ls.append(A)
            if good and sup >= I.shape[1 - axis]:
                out += ls
        return sorted(set(out))

    def maps(r0, h, P, lines, n):
        # every place the run of lines r0..r0+h-1 is drawn again: (start, mirrored, cost)
        res = []
        for k in range(-(n // P) - 1, n // P + 2):
            y = r0 + k * P
            if 0 <= y and y + h <= n:
                res.append((y, False, abs(k)))
        for A in lines:
            y = A - (r0 + h - 1)
            if 0 <= y and y + h <= n and y != r0:
                res.append((y, True, 0.5 + abs(A - 2 * r0 - h + 1) / (2.0 * n)))
        return res

    def orbit_val(r, c, sym):
        Pr, Pc, Lr, Lc = sym
        for y, _, _ in maps(r, 1, Pr, Lr, hi):
            for x, _, _ in maps(c, 1, Pc, Lc, wi):
                if known[y, x]:
                    return int(I[y, x])
        return None

    prs = _periods(I, known, 0) + [hi]
    pcs = _periods(I, known, 1) + [wi]
    occ_cells = [tuple(p) for p in np.argwhere(occ)]
    sym = None
    for a, b in sorted(((a, b) for a in prs for b in pcs), key=lambda t: (t[0] * t[1], t)):
        cand = (a, b, lines_for(a, 0), lines_for(b, 1))
        if all(orbit_val(r, c, cand) is not None for r, c in occ_cells):
            sym = cand
            break
    if sym is None:
        sym = (prs[0], pcs[0], lines_for(prs[0], 0), lines_for(pcs[0], 1))
    Pr, Pc, Lr, Lc = sym

    # ── the occluded area as disjoint maximal rectangles (the patches) ──────
    left = occ.copy()
    rects = []
    for r in range(hi):
        for c in range(wi):
            if not left[r, c]:
                continue
            w = 0
            while c + w < wi and left[r, c + w]:
                w += 1
            h = 1
            while r + h < hi and left[r + h, c:c + w].all():
                h += 1
            left[r:r + h, c:c + w] = False
            rects.append((r, c, h, w))

    def source_for(r, c, h, w):
        # nearest intact copy of the block elsewhere in the wallpaper; plain
        # translates first, mirror images only when no translate is intact
        best = None
        for y, fy, cy in maps(r, h, Pr, Lr, hi):
            for x, fx, cx in maps(c, w, Pc, Lc, wi):
                if (y, x) == (r, c) and not fy and not fx:
                    continue
                if not known[y:y + h, x:x + w].all():
                    continue
                key = (int(fy) + int(fx), cy + cx, y, x)
                if best is None or key < best[0]:
                    best = (key, y, x, fy, fx)
        return None if best is None else best[1:]

    def restore(r, c, h, w):
        if h == 1 and w == 1:
            v = orbit_val(r, c, sym)
            if v is not None and v != noise:
                ops.append(v); sels.append(sel_of([(r, c)]))
            return
        src = source_for(r, c, h, w)
        if src is None:
            if h >= w:
                h1 = h // 2
                restore(r, c, h1, w); restore(r + h1, c, h - h1, w)
            else:
                w1 = w // 2
                restore(r, c, h, w1); restore(r, c + w1, h, w - w1)
            return
        y, x, fy, fx = src
        block = I[y:y + h, x:x + w]
        if noise != 0 and (block == 0).any():
            # Paste writes no 0s: clear the whole patch rectangle first so its 0 cells show
            ops.append(0); sels.append([r, c, h - 1, w - 1])
        if not (block != 0).any():
            return
        ops.append(28); sels.append([y, x, h - 1, w - 1])   # the intact copy (full rectangle)
        ops.append(30); sels.append([r, c, 0, 0])           # laid over the patch
        tgt = [r, c, h - 1, w - 1]                          # the whole patch rectangle
        if fx and w > 1 and not (block == block[:, ::-1]).all():
            ops.append(26); sels.append(tgt)                # mirrored across a column fold
        if fy and h > 1 and not (block == block[::-1, :]).all():
            ops.append(27); sels.append(tgt)                # mirrored across a row fold

    for (r, c, h, w) in rects:
        restore(r, c, h, w)

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
    return ops, sels


def _periods(I, ok, axis):
    n = I.shape[axis]
    out = []
    for p in range(1, n):
        if axis == 0:
            a, b, ka, kb = I[p:], I[:n - p], ok[p:], ok[:n - p]
        else:
            a, b, ka, kb = I[:, p:], I[:, :n - p], ok[:, p:], ok[:, :n - p]
        if not ((a != b) & ka & kb).any():
            out.append(p)
    return out


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
                        f"num_examples+1 ({num_examples + 1}) for task 0dfd9992"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0dfd9992"
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
                                f"for task 0dfd9992"
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
                    f"Failed to build a complete episode for task 0dfd9992 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0dfd9992-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
