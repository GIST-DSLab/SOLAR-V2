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
import random

import numpy as np

from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    """The generator's three colour roles, fixed once for the whole episode:
    the canvas colour, the colour the blots are painted with, and the palette
    the repeating tile is drawn from.

    0 stays in every pool on purpose — ARCLE reads 0 as "nothing there", so the
    route has to cope with instances where 0 is an ordinary colour of the
    wallpaper (or of the blots themselves)."""
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    remcols = [c for c in cols if c != noisec]
    numc = random.randint(2, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "noisec": noisec, "ccols": ccols}


# -------------------------------------------------------------- generate ----
def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, ccols) -> dict:
    for _attempt in range(80):
        rotf = random.choice((identity, rot90, rot180, rot270))
        # a quarter turn swaps the axes: sample inside the bounds it lands in
        lim_h, lim_w = (max_w, max_h) if rotf in (rot90, rot270) else (max_h, max_w)
        h = unifint(diff_lb, diff_ub, (min(10, lim_h), lim_h))
        w = unifint(diff_lb, diff_ub, (min(10, lim_w), lim_w))
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
        succ, tr, maxtr = 0, 0, 5 * numpatches
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
            continue        # every blot was rejected — nothing to repair
        # every cell of the tile has to survive somewhere, or the wallpaper
        # cannot be read back off the input at all
        A = np.array(gi, dtype=int)
        clean = A != noisec
        vper, hper = hp + 1, wp + 1
        readable = True
        for r in range(vper):
            for c in range(hper):
                v = A[r::vper, c::hper][clean[r::vper, c::hper]]
                if v.size == 0 or np.unique(v).size != 1:
                    readable = False
                    break
            if not readable:
                break
        if not readable:
            continue
        return {'input': rotf(gi), 'output': rotf(go)}
    raise ValueError("generation failed")


# ------------------------------------------------------------ derivation ----
def derive_operations(I, O=None, examples=None):
    """The picture is one small tile stamped edge to edge; blots of a single
    foreign colour have covered parts of it.  Read the tile back off whatever
    is still intact, then repair each blot — either by stamping the very same
    place from one whole period away, or by filling its colours in by hand.

    Which blot is repaired first, which way the hunt for a clean copy looks,
    whether a blot is stamped or hand-filled, whether the ground is laid first
    and drawn over, and whether the palette is walked up or down are all read
    off the input (the blot colour, how many blots there are, the tile's size,
    the grid's size), so one instance always gives one route and the variety
    comes from the instances differing.  O is never looked at: handed a
    different answer grid this returns exactly the same operations.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    whole = [0, 0, h - 1, w - 1]

    # ── read the wallpaper: its blot colour, its two periods, the tile ──────
    def axis_periods(A, clean, axis):
        n = A.shape[axis]
        out = []
        for p in range(1, n // 2 + 1):
            if axis == 0:
                a, b, m = A[p:, :], A[:-p, :], clean[p:, :] & clean[:-p, :]
            else:
                a, b, m = A[:, p:], A[:, :-p], clean[:, p:] & clean[:, :-p]
            if bool((a[m] == b[m]).all()):
                out.append(p)
        return out

    def fold(A, clean, vp, hp):
        tile = np.zeros((vp, hp), dtype=int)
        for r in range(vp):
            for c in range(hp):
                v = A[r::vp, c::hp][clean[r::vp, c::hp]]
                if v.size == 0:
                    return None                  # nowhere intact: unreadable
                u = np.unique(v)
                if u.size != 1:
                    return None                  # not actually a period
                tile[r, c] = int(u[0])
        return np.tile(tile, (h // vp + 1, w // hp + 1))[:h, :w]

    best = None
    for cand in sorted(set(I.ravel().tolist())):
        clean = (I != cand)
        if not clean.any():
            continue
        pairs = [(a, b) for a in axis_periods(I, clean, 0)
                 for b in axis_periods(I, clean, 1)]
        for vp, hp in sorted(pairs, key=lambda t: (t[0] * t[1], t[0])):
            R = fold(I, clean, vp, hp)
            if R is None:
                continue
            key = (vp * hp, int(clean.size - clean.sum()))
            if best is None or key < best[0]:
                best = (key, cand, vp, hp, R)
            break
    if best is None:                                  # unreadable
        ops.append(34); sels.append(whole)
        return ops, sels
    _, nc, vp, hp, R = best

    cur = I.copy()
    if np.array_equal(cur, R):                        # nothing is damaged
        ops.append(34); sels.append(whole)
        return ops, sels

    # ── the blots, as connected groups of cells ─────────────────────────────
    blotted = (I == nc)
    seen = np.zeros_like(blotted)
    blots = []
    for r in range(h):
        for c in range(w):
            if blotted[r, c] and not seen[r, c]:
                stack, cells = [(r, c)], []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and blotted[ny, nx] \
                                and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                blots.append((min(rs), min(cs), max(rs) - min(rs) + 1,
                              max(cs) - min(cs) + 1, cells))
    nb = len(blots)

    # ── the choices a solver actually faces, each one fixed by the input ────
    order_mode = (nb + nc) % 4          # where to start, which way to sweep
    method_mode = (vp + hp) % 3         # stamp a clean copy, or fill by hand
    src_mode = (nc + h + w) % 3         # which way to look for that copy
    zero_mode = (nb + vp) % 2           # wipe the hole first, or patch 0s after
    base_first = ((nc + h) % 2) == 1    # lay the ground, then draw on it
    desc = ((nc + w) % 2) == 1          # walk the palette down instead of up
    ground = int(np.bincount(R.ravel(), minlength=10).argmax())

    blots.sort(key=(
        lambda b: (b[0], b[1]),                  # reading order, top-left first
        lambda b: (-b[0], -b[1]),                # bottom-right first, upwards
        lambda b: (b[1], b[0]),                  # column by column
        lambda b: (-len(b[4]), b[0], b[1]),      # the biggest blot first
    )[order_mode])

    def emit(col, cells):
        """Paint `cells` col, dropping any that already hold it."""
        cells = [(r, c) for (r, c) in cells if cur[r, c] != col]
        if not cells:
            return
        ops.append(int(col)); sels.append(sel_of(cells))
        for (r, c) in cells:
            cur[r, c] = col

    def by_hand(cells):
        """Fill a region in by hand, one colour of the tile at a time."""
        groups = {}
        for (r, c) in cells:
            if cur[r, c] != R[r, c]:
                groups.setdefault(int(R[r, c]), []).append((r, c))
        for col in sorted(groups, reverse=desc):
            emit(col, groups[col])

    def clean_copy(r0, c0, bh, bw):
        """An intact copy of exactly this place, a whole number of periods away."""
        cand = []
        for i in range(-(h // vp) - 1, h // vp + 2):
            for j in range(-(w // hp) - 1, w // hp + 2):
                if i == 0 and j == 0:
                    continue
                sr, sc = r0 + i * vp, c0 + j * hp
                if sr < 0 or sc < 0 or sr + bh > h or sc + bw > w:
                    continue
                if blotted[sr:sr + bh, sc:sc + bw].any():
                    continue
                if src_mode == 0:                      # simply the nearest
                    cand.append(((abs(i) + abs(j), abs(i), i, j), sr, sc))
                elif src_mode == 1:                    # along the row first
                    cand.append(((abs(j), abs(i), j, i), sr, sc))
                else:                                  # up the column first
                    cand.append(((abs(i), abs(j), -i, -j), sr, sc))
        if not cand:
            return None
        cand.sort(key=lambda t: t[0])
        return cand[0][1], cand[0][2]

    held = None                                        # what the clipboard holds

    for (r0, c0, bh, bw, cells) in blots:
        if all(cur[r, c] == R[r, c] for (r, c) in cells):
            continue                                   # an earlier stamp got it

        # a hole worth fetching a clean copy for, or a nick quicker done by hand
        stamp = method_mode == 0 or (method_mode == 2 and 2 * bh * bw >= vp * hp)
        src = clean_copy(r0, c0, bh, bw) if stamp else None

        if src is None:                                # nothing clean to lean on
            if base_first and any(R[r, c] == ground for (r, c) in cells
                                  if cur[r, c] != R[r, c]):
                emit(ground, [(r, c) for (r, c) in cells if cur[r, c] != R[r, c]])
            by_hand(cells)
            continue

        sr, sc = src
        block = I[sr:sr + bh, sc:sc + bw]
        ink = block != 0                               # what a Paste can carry
        tgt = cur[r0:r0 + bh, c0:c0 + bw]              # a view: edits land in cur

        if zero_mode == 0:
            # wipe the hole, then lay the clean copy over it — a Paste carries
            # none of the copy's 0 cells, and after the wipe it need not.  Worth
            # doing only when the copy really has 0 cells to leave behind.
            if bool((~ink).any()) and bool((tgt[~ink] != 0).any()):
                ops.append(0); sels.append([r0, c0, bh - 1, bw - 1])
                tgt[:, :] = 0
            if ink.any() and bool((tgt[ink] != block[ink]).any()):
                if held != (sr, sc, bh, bw):
                    ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
                    held = (sr, sc, bh, bw)
                ops.append(30); sels.append([r0, c0, 0, 0])
                tgt[ink] = block[ink]
        else:
            # stamp straight onto the grid, then put the copy's 0 cells down by
            # hand: a Paste reads 0 as nothing there and leaves them behind
            if ink.any() and bool((tgt[ink] != block[ink]).any()):
                if held != (sr, sc, bh, bw):
                    ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
                    held = (sr, sc, bh, bw)
                ops.append(30); sels.append([r0, c0, 0, 0])
                tgt[ink] = block[ink]
            emit(0, [(r0 + i, c0 + j) for i in range(bh) for j in range(bw)
                     if block[i, j] == 0])

        by_hand(cells)                                 # whatever the stamp missed

    # anything still off the wallpaper (a blot with no clean copy anywhere)
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
