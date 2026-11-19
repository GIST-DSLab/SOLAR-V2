"""
ARC Task: 29ec7d0e (RE-ARC) — LLM-generated grid_maker
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
from collections import deque

try:
    from maker.sel_helpers import sel_of
except Exception:                                    # pragma: no cover
    def sel_of(cells):
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    """bgc / noisec / the tile's palette are episode-level roles.

    0 is deliberately left in the pool for every one of them: the route has to
    survive an instance where 0 is an ordinary colour of the picture.
    """
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    remcols = [c for c in cols if c != noisec]
    numc = random.randint(2, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "noisec": noisec, "ccols": ccols}


# -------------------------------------------------------------- generate ----
def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, ccols) -> dict:
    for _attempt in range(60):
        rotf = random.choice((identity, rot90, rot180, rot270))
        # a quarter turn swaps the axes: sample inside the bounds it will land in
        glim_h, glim_w = (max_w, max_h) if rotf in (rot90, rot270) else (max_h, max_w)
        h = unifint(diff_lb, diff_ub, (min(10, glim_h), glim_h))
        w = unifint(diff_lb, diff_ub, (min(10, glim_w), glim_w))
        if h < 6 or w < 6:
            continue
        hp = unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
        wp = unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
        pinds = asindices(canvas(-1, (hp, wp)))
        pobj = frozenset({(random.choice(ccols), ij) for ij in pinds})
        go = canvas(bgc, (h, w))
        locs = set()
        for a in range(h // hp + 1):
            for b in range(w // wp + 1):
                loci = (a + 1) + hp * a
                locj = (b + 1) + wp * b
                locs.add((loci, locj))
                go = paint(go, shift(pobj, (loci, locj)))
        numpatches = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
        gi = tuple(e for e in go)
        places = apply(lbind(shift, pinds), locs)
        succ = 0
        tr = 0
        maxtr = 5 * numpatches
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph = random.randint(2, 6)
            pw = random.randint(2, 6)
            loci = random.randint(0, h - ph)
            locj = random.randint(0, w - pw)
            ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
            gi2 = fill(gi, noisec, ptch)
            if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
                if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
                        len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                    succ += 1
                    gi = gi2
        if succ == 0:
            continue    # every patch was rejected: nothing to repair, not a task
        gi_r, go_r = rotf(gi), rotf(go)
        info = _infer(np.array(gi_r, dtype=int))
        if info is None or not np.array_equal(info[3], np.array(go_r, dtype=int)):
            continue    # so much is blotted the tile can no longer be read back
        return {'input': gi_r, 'output': go_r}
    raise ValueError("generation failed")


# ------------------------------------------------------------ derivation ----
def _per_ok(A, known, p, axis):
    """Does a shift of p along `axis` agree everywhere both cells are readable?"""
    if axis == 0:
        a, b, m = A[p:, :], A[:-p, :], known[p:, :] & known[:-p, :]
    else:
        a, b, m = A[:, p:], A[:, :-p], known[:, p:] & known[:, :-p]
    if a.size == 0:
        return True
    return bool((a[m] == b[m]).all())


def _fold(A, known, vp, hp):
    """Read the repeating tile off the readable cells; None if it stays unknown."""
    h, w = A.shape
    tile = np.zeros((vp, hp), dtype=int)
    for r in range(vp):
        for c in range(hp):
            vals = A[r::vp, c::hp][known[r::vp, c::hp]]
            if vals.size == 0:
                return None                       # this tile cell is nowhere intact
            u = np.unique(vals)
            if u.size != 1:
                return None                       # not actually a period
            tile[r, c] = int(u[0])
    return np.tile(tile, (h // vp + 1, w // hp + 1))[:h, :w]


def _infer(A):
    """From the input alone: the blot colour, the two periods, the whole pattern."""
    h, w = A.shape
    best = None
    for nc in sorted(set(A.flatten().tolist())):
        known = (A != nc)
        vps = [p for p in range(1, h + 1) if _per_ok(A, known, p, 0)]
        hps = [p for p in range(1, w + 1) if _per_ok(A, known, p, 1)]
        for vp, hp in sorted(((a, b) for a in vps for b in hps), key=lambda t: t[0] * t[1]):
            R = _fold(A, known, vp, hp)
            if R is None:
                continue
            if bool((R == nc).any()):             # a blot colour is never part of the pattern
                continue
            if best is None or vp * hp < best[0]:
                best = (vp * hp, nc, vp, hp, R)
            break
    if best is None:
        return None
    _, nc, vp, hp, R = best
    return nc, vp, hp, R


def _blobs(mask):
    """The blots, as connected groups of cells."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                out.append(cells)
    return out


def derive_operations(I, O=None, examples=None):
    """The picture is one small tile stamped over and over; blots of a single
    foreign colour have covered parts of it.  Repair each blot, either by
    stamping an intact copy of the same place one whole period away, or by
    filling its colours in by hand.

    Which blot is dealt with first, which way the search for a clean copy
    looks, whether a blot is stamped or hand-filled, whether a base is laid
    first, and whether the colours go up or down the palette are all read off
    the input — the blot colour, the number of blots, the tile's size, the grid's
    size.  Nothing here ever looks at O, so the same input always gives the same
    route, and a different answer grid would not change a single operation.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    whole = [0, 0, h - 1, w - 1]

    info = _infer(I)
    if info is None:                                   # pattern unreadable
        ops.append(34); sels.append(whole)
        return ops, sels
    nc, vp, hp, R = info

    cur = I.copy()
    if np.array_equal(cur, R):                         # nothing damaged
        ops.append(34); sels.append(whole)
        return ops, sels

    mask = (I == nc)
    blots = []
    for cells in _blobs(mask):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        blots.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1, cells))
    nb = len(blots)

    # ── the five choices a solver actually faces, each fixed by the input ──
    order_mode = (nb + nc) % 4            # where to start and which way to sweep
    method_mode = (vp + hp) % 3           # stamp a clean copy, or fill by hand
    src_mode = (nc + h + w) % 3           # which direction to look for that copy
    zero_mode = (nb + vp) % 2             # clear the hole first, or patch 0s after
    base_first = ((nc + h) % 2) == 1      # lay the tile's own ground, then draw
    desc = ((nc + w) % 2) == 1            # walk the palette down instead of up
    base = int(np.bincount(R.ravel(), minlength=10).argmax())

    order_key = (
        lambda b: (b[0], b[1]),                   # top-left first, reading order
        lambda b: (-b[0], -b[1]),                 # bottom-right first, upwards
        lambda b: (b[1], b[0]),                   # left to right, column by column
        lambda b: (-len(b[4]), b[0], b[1]),       # the biggest blot first
    )[order_mode]
    blots.sort(key=order_key)

    def emit(col, cells):
        """Paint `cells` — dropping only those that already hold `col`."""
        cells = [(r, c) for (r, c) in cells if cur[r, c] != col]
        if not cells:
            return
        ops.append(int(col)); sels.append(sel_of(cells))
        for (r, c) in cells:
            cur[r, c] = col

    def emit_groups(cells):
        """Fill a region by hand, one colour of the tile at a time."""
        groups = {}
        for (r, c) in cells:
            if cur[r, c] != R[r, c]:
                groups.setdefault(int(R[r, c]), []).append((r, c))
        for col in sorted(groups, reverse=desc):
            emit(col, groups[col])

    def find_src(r0, c0, bh, bw):
        """An intact copy of exactly this place, a whole number of periods away."""
        cand = []
        for i in range(-(h // vp) - 1, h // vp + 2):
            for j in range(-(w // hp) - 1, w // hp + 2):
                if i == 0 and j == 0:
                    continue
                sr, sc = r0 + i * vp, c0 + j * hp
                if sr < 0 or sc < 0 or sr + bh > h or sc + bw > w:
                    continue
                if mask[sr:sr + bh, sc:sc + bw].any():
                    continue
                if src_mode == 0:                 # the nearest one
                    key = (abs(i) + abs(j), abs(i), i, j)
                elif src_mode == 1:               # look along the row first
                    key = (abs(j), abs(i), j, i)
                else:                             # look down the column first
                    key = (abs(i), abs(j), -i, -j)
                cand.append((key, sr, sc))
        if not cand:
            return None
        cand.sort(key=lambda t: t[0])
        return cand[0][1], cand[0][2]

    clip = None                                   # what the clipboard holds

    for (r0, c0, bh, bw, cells) in blots:
        dirty = [(r, c) for (r, c) in cells if cur[r, c] != R[r, c]]
        if not dirty:
            continue                              # an earlier stamp already covered it

        # mixed: a big hole is worth fetching a clean copy for, a small nick is
        # quicker to fill in by hand
        stamp = method_mode == 0 or (method_mode == 2 and 2 * bh * bw >= vp * hp)
        src = find_src(r0, c0, bh, bw) if stamp else None
        if src is None:                           # no clean copy to lean on
            if base_first and base != nc and any(R[r, c] == base for (r, c) in dirty):
                emit(base, dirty)                 # ground first, drawing on top
            emit_groups(cells)
            continue

        sr, sc = src
        block = I[sr:sr + bh, sc:sc + bw]
        ink = block != 0
        tgt = cur[r0:r0 + bh, c0:c0 + bw]

        if zero_mode == 0:
            # wipe the hole, then lay the clean copy over it; a Paste carries no
            # 0 cells of the copy, and after the wipe it does not have to.  Only
            # worth doing when the copy really has 0 cells to leave behind.
            if bool((~ink).any()) and bool((tgt[~ink] != 0).any()):
                ops.append(0); sels.append([r0, c0, bh - 1, bw - 1])
                tgt[:, :] = 0
            if ink.any():
                if clip != (sr, sc, bh, bw):
                    ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
                    clip = (sr, sc, bh, bw)
                ops.append(30); sels.append([r0, c0, 0, 0])
                tgt[ink] = block[ink]
        else:
            # stamp straight onto the grid, then put the copy's 0 cells down by
            # hand — a Paste reads 0 as nothing there and leaves them behind
            if ink.any() and bool((tgt[ink] != block[ink]).any()):
                if clip != (sr, sc, bh, bw):
                    ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
                    clip = (sr, sc, bh, bw)
                ops.append(30); sels.append([r0, c0, 0, 0])
                tgt[ink] = block[ink]
            emit(0, [(r0 + i, c0 + j) for i in range(bh) for j in range(bw)
                     if block[i, j] == 0])

        emit_groups(cells)                        # anything the stamp missed

    # whatever is still off the pattern (a blot with no clean copy anywhere)
    rest = {}
    for r in range(h):
        for c in range(w):
            if cur[r, c] != R[r, c]:
                rest.setdefault(int(R[r, c]), []).append((r, c))
    for col in sorted(rest, reverse=desc):
        emit(col, rest[col])

    ops.append(34); sels.append(whole)
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
                        f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
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
                                f"for task 29ec7d0e"
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
                    f"Failed to build a complete episode for task 29ec7d0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29ec7d0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
