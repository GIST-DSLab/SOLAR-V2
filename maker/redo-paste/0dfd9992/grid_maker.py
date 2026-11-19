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
import numpy as np

try:
    from maker.sel_helpers import sel_of
except Exception:                                    # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# =====================================================================
# 1. colors  (bgc is never visible - the tiling covers it - but the
#    intruder color and the tile palette are fixed for the episode)
# =====================================================================
def sample_colors(num_examples=None) -> dict:
    import random
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    remcols = [c for c in cols if c != noisec]
    numc = random.randint(2, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "noisec": noisec, "ccols": tuple(ccols)}


# =====================================================================
# 2. generator
# =====================================================================
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=0, noisec=5, ccols=(1, 2)) -> dict:
    max_h = max(10, max_h)
    max_w = max(10, max_w)
    ccols = list(ccols)
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
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
           (pobj in candset or hmirror(pobj) in candset or
            vmirror(pobj) in candset or hmirror(vmirror(pobj)) in candset):
            succ += 1
            gi = gi2
    rots = [identity, rot180]
    if h <= max_w and w <= max_h:
        rots += [rot90, rot270]
    rotf = choice(tuple(rots))
    return {'input': rotf(gi), 'output': rotf(go)}


# =====================================================================
# 3. derivation
#    The grid is one tile stamped over the canvas, mirrored on every
#    other row-block / column-block.  A few solid patches of one
#    intruder color hide part of it.  Every hidden block is restored by
#    copying the block of the pattern that the tiling maps onto it -
#    CopyI + Paste, plus a Flip when that correspondence is a mirror.
#    A block is cleared to 0 first when the copy has to carry 0-colored
#    pattern cells (Paste is transparent on 0).
# =====================================================================
def _min_period(I, nc, axis):
    """Tightest translation the visible (non-intruder) cells repeat under."""
    n = I.shape[axis]
    for p in range(1, n):
        a, b = (I[:n - p, :], I[p:, :]) if axis == 0 else (I[:, :n - p], I[:, p:])
        m = (a != nc) & (b != nc)
        if m.any() and np.array_equal(a[m], b[m]):
            return p
    return None


def _support(I, nc, axis, s, t):
    """None if the index map i -> s*i+t is contradicted by a visible pair,
    else the number of visible pairs that confirm it."""
    n = I.shape[axis]
    src = np.array([s * i + t for i in range(n)])
    idx = np.nonzero((src >= 0) & (src < n))[0]
    if len(idx) == 0:
        return None
    a, b = (I[idx, :], I[src[idx], :]) if axis == 0 else (I[:, idx], I[:, src[idx]])
    m = (a != nc) & (b != nc)
    if not np.array_equal(a[m], b[m]):
        return None
    return int(m.sum())


def _maps_for_axis(I, nc, axis):
    """The pattern's own symmetries along one axis: translations by the
    period, and its mirror axes."""
    n = I.shape[axis]
    p = _min_period(I, nc, axis)
    out = []
    if p is not None:
        for k in range(1, (n - 1) // p + 1):
            for d in (k * p, -k * p):
                sup = _support(I, nc, axis, 1, d)
                if sup is not None:
                    out.append((1, d, sup))
    # mirror axes i -> t - i.  Two mirror axes of one pattern always differ by
    # a whole period, so keep only the best supported family: an axis that
    # merely happens to agree where the damage hides the truth is discarded.
    axes = {}
    for t in range(0, 2 * n - 1):
        ov = min(t, n - 1) - max(0, t - n + 1) + 1
        if ov < 2:
            continue
        sup = _support(I, nc, axis, -1, t)
        if sup is not None:
            axes[t] = sup
    if axes:
        step = p if p else n
        classes = {}
        for t, sup in axes.items():
            classes.setdefault(t % step, []).append((sup, t))
        best = max(classes.values(),
                   key=lambda v: (max(x[0] for x in v), sum(x[0] for x in v)))
        for sup, t in sorted(best, key=lambda x: x[1]):
            out.append((-1, t, sup))
    return out


def _pair_masks(I, nc, h, w):
    """Every source->destination correspondence the tiling allows, together
    with the cells each one may legitimately restore.  Nearest first."""
    rmaps = [(1, 0, 10 ** 9)] + _maps_for_axis(I, nc, 0)
    cmaps = [(1, 0, 10 ** 9)] + _maps_for_axis(I, nc, 1)
    pairs = []
    for rs, rt, rsup in rmaps:
        for cs, ct, csup in cmaps:
            if (rs, rt) == (1, 0) and (cs, ct) == (1, 0):
                continue
            cost = (abs(rt) if rs == 1 else h) + (abs(ct) if cs == 1 else w)
            pairs.append((cost, -min(rsup, csup), rs, rt, cs, ct))
    pairs.sort()
    rows, cols = np.arange(h), np.arange(w)
    order, masks, srcs = [], {}, {}
    for _, _, rs, rt, cs, ct in pairs:
        sr, sc = rs * rows + rt, cs * cols + ct
        okr, okc = (sr >= 0) & (sr < h), (sc >= 0) & (sc < w)
        src = I[np.ix_(np.clip(sr, 0, h - 1), np.clip(sc, 0, w - 1))]
        m = okr[:, None] & okc[None, :] & (src != nc) & ((I == nc) | (I == src))
        key = (rs, rt, cs, ct)
        order.append(key); masks[key] = m; srcs[key] = src
    return order, masks, srcs


def _count_objects(I, col):
    h, w = I.shape
    seen = np.zeros((h, w), bool)
    n = 0
    for r in range(h):
        for c in range(w):
            if I[r, c] == col and not seen[r, c]:
                n += 1
                stack = [(r, c)]; seen[r, c] = True
                while stack:
                    x, y = stack.pop()
                    for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        u, v = x + a, y + b
                        if 0 <= u < h and 0 <= v < w and not seen[u, v] and I[u, v] == col:
                            seen[u, v] = True; stack.append((u, v))
    return n


def _infer_damage(I):
    """Which color is the intruder?  The one that, once each of its cells is
    restored from the pattern's own symmetry, leaves a grid that is exactly
    doubly periodic - nothing anomalous left over - under the simplest
    reading of the grid (least damage assumed + smallest repeating tile)."""
    h, w = I.shape
    passing, fallback = [], []
    for col in sorted(set(I.flatten().tolist())):
        pv, ph = _min_period(I, col, 0), _min_period(I, col, 1)
        if pv is None or ph is None:
            continue
        order, masks, srcs = _pair_masks(I, col, h, w)
        R = I.copy()
        todo = (I == col)
        for k in order:
            sel = todo & masks[k]
            R[sel] = srcs[k][sel]
            todo &= ~sel
        key = (int((I == col).sum()) + pv * ph, int((I == col).sum()),
               _count_objects(I, col), col)
        if todo.any():
            continue
        if pv <= h - 2 and ph <= w - 2 and \
           np.array_equal(R[:h - pv, :], R[pv:, :]) and \
           np.array_equal(R[:, :w - ph], R[:, ph:]):
            passing.append((key, col, order, masks))
        else:
            fallback.append((key, col, order, masks))
    pool = passing if passing else fallback
    if pool:
        pool.sort(key=lambda e: e[0])
        return pool[0][1], pool[0][2], pool[0][3]
    col = int(I.flatten()[0])
    order, masks, _ = _pair_masks(I, col, h, w)
    return col, order, masks


def _run(I, ops, sels):
    """Replay of the ops actually used here (Color, FlipH/V, CopyI, Paste)."""
    h, w = I.shape
    G = I.copy()
    clip = None
    for op, sel in zip(ops, sels):
        if isinstance(sel, dict):
            cells = sel["cells"]
            if not cells:
                continue
            rr = [c[0] for c in cells]; cc = [c[1] for c in cells]
            r1, r2, c1, c2 = min(rr), max(rr), min(cc), max(cc)
            m = np.zeros((h, w), bool)
            for r, c in cells:
                m[r, c] = True
        else:
            r, c, hh, ww = sel
            r1, c1, r2, c2 = r, c, r + hh, c + ww
            m = np.zeros((h, w), bool); m[r1:r2 + 1, c1:c2 + 1] = True
        if 0 <= op <= 9:
            G[m] = op
        elif op in (26, 27):
            sub = np.where(m[r1:r2 + 1, c1:c2 + 1], G[r1:r2 + 1, c1:c2 + 1], 0)
            bg = G[r1:r2 + 1, c1:c2 + 1].copy(); bg[m[r1:r2 + 1, c1:c2 + 1]] = 0
            sub = sub[:, ::-1] if op == 26 else sub[::-1, :]
            G[r1:r2 + 1, c1:c2 + 1] = np.where(sub != 0, sub, bg)
        elif op == 28:
            clip = I[r1:r2 + 1, c1:c2 + 1].copy()
        elif op == 30 and clip is not None:
            ch, cw = clip.shape
            eh, ew = min(ch, h - r1), min(cw, w - c1)
            tgt = G[r1:r1 + eh, c1:c1 + ew]; src = clip[:eh, :ew]
            G[r1:r1 + eh, c1:c1 + ew] = np.where(src != 0, src, tgt)
    return G


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    nc, order, masks = _infer_damage(I)
    W = I.copy()
    rem = (I == nc)
    clip = None

    while rem.any():
        r0, c0 = [int(x) for x in np.argwhere(rem)[0]]
        pref = np.cumsum(np.cumsum(rem.astype(int), axis=0), axis=1)

        def cnt(r1, c1, hh, ww):
            r2, c2 = r1 + hh - 1, c1 + ww - 1
            tot = pref[r2, c2]
            if r1: tot -= pref[r1 - 1, c2]
            if c1: tot -= pref[r2, c1 - 1]
            if r1 and c1: tot += pref[r1 - 1, c1 - 1]
            return int(tot)

        # biggest block of damage one correspondence of the tiling can restore
        best = None
        for key in order:
            m = masks[key]
            if not m[r0, c0]:
                continue
            curmax = w - c0
            for hh in range(1, h - r0 + 1):
                r = r0 + hh - 1
                run = 0
                while run < curmax and m[r, c0 + run]:
                    run += 1
                curmax = min(curmax, run)
                if curmax == 0:
                    break
                for ww in range(1, curmax + 1):
                    k = (cnt(r0, c0, hh, ww), hh * ww)
                    if best is None or k > best[0]:
                        best = (k, key, hh, ww)
        if best is None:
            break
        _, (rs, rt, cs, ct), hh, ww = best
        # trim the block to the damage it actually covers
        rr, cc = np.nonzero(rem[r0:r0 + hh, c0:c0 + ww])
        r1, r2 = r0 + int(rr.min()), r0 + int(rr.max())
        c1, c2 = c0 + int(cc.min()), c0 + int(cc.max())
        hh, ww = int(r2 - r1 + 1), int(c2 - c1 + 1)
        sr0 = int(min(rs * r + rt for r in range(r1, r2 + 1)))
        sc0 = int(min(cs * c + ct for c in range(c1, c2 + 1)))
        patch = I[sr0:sr0 + hh, sc0:sc0 + ww]

        cur = W[r1:r1 + hh, c1:c1 + ww]
        # clear the block first, so the copy can also carry 0-colored cells
        if ((patch == 0) & (cur != 0)).any():
            ops.append(0); sels.append([r1, c1, hh - 1, ww - 1])      # exact rectangle
            W[r1:r1 + hh, c1:c1 + ww] = 0
            cur = W[r1:r1 + hh, c1:c1 + ww]
        after = np.where(patch != 0, patch, cur)
        if not np.array_equal(after, cur):
            if clip is None or not np.array_equal(clip, patch):
                ops.append(28); sels.append([sr0, sc0, hh - 1, ww - 1])  # exact rectangle
                clip = patch.copy()
            ops.append(30); sels.append([r1, c1, 0, 0])                  # paste origin
            W[r1:r1 + hh, c1:c1 + ww] = after
        if rs == -1:                       # the tiling mirrors it top/bottom
            blk = W[r1:r1 + hh, c1:c1 + ww]
            if not np.array_equal(blk, blk[::-1, :]):
                ops.append(27); sels.append([r1, c1, hh - 1, ww - 1])    # exact rectangle
                W[r1:r1 + hh, c1:c1 + ww] = blk[::-1, :].copy()
        if cs == -1:                       # ... and/or left/right
            blk = W[r1:r1 + hh, c1:c1 + ww]
            if not np.array_equal(blk, blk[:, ::-1]):
                ops.append(26); sels.append([r1, c1, hh - 1, ww - 1])    # exact rectangle
                W[r1:r1 + hh, c1:c1 + ww] = blk[:, ::-1].copy()
        rem[r1:r1 + hh, c1:c1 + ww] = (W[r1:r1 + hh, c1:c1 + ww] == nc)

    # safety net: only fires if the pattern genuinely cannot reach a cell
    if not np.array_equal(W, O):
        for col in sorted(set(O[W != O].tolist())):
            cells = [(int(r), int(c)) for r, c in np.argwhere((W != O) & (O == col))]
            ops.append(int(col)); sels.append(sel_of(cells))
            for r, c in cells:
                W[r, c] = col

    # drop any step that turns out not to matter
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ops):
            t_ops, t_sels = ops[:i] + ops[i + 1:], sels[:i] + sels[i + 1:]
            if np.array_equal(_run(I, t_ops, t_sels), O):
                ops, sels = t_ops, t_sels
                changed = True
            else:
                i += 1

    ops.append(34); sels.append([0, 0, h - 1, w - 1])   # whole grid
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
