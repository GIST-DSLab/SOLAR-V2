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
# ─────────────────────────────────────────────────────────────────────────────
# The grid is a wallpaper: one hp x wp tile stamped over the whole canvas, every
# other stamp mirrored (rows flipped on odd tile-rows, columns flipped on odd
# tile-columns).  That construction makes the grid symmetric about a family of
# horizontal mirror lines spaced hp apart and a family of vertical ones spaced
# wp apart.  Solid rectangles of one colour the wallpaper never uses are then
# painted over it, and the whole picture may be given a quarter turn.
#
# Repairing a damaged rectangle is a reflection: the intact block that sits
# across one of those mirror lines is copied, stamped over the damage and
# flipped in place.  Everything the reflection needs — which colour marks the
# damage, where the mirror lines fall, how far the counterpart block sits, which
# axis it is mirrored about — is read off the input.  O is never consulted.
# ─────────────────────────────────────────────────────────────────────────────
import random
from collections import Counter, deque

import numpy as np


def _blobs(mask):
    """8-connected components of a boolean mask — one damaged splotch each."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            out.append(comp)
    return out


def _damage_color(I):
    """The colour the patches are painted in: it breaks into fewer separate
    pieces than any wallpaper colour (those are scattered over the whole grid),
    and of the colours tying for fewest pieces it covers the fewest cells."""
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces = Counter()
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(I[r, c])
            pieces[col] += 1
            seen[r, c] = True
            q = deque([(r, c)])
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
    fewest = min(pieces.values())
    tied = [col for col, n in pieces.items() if n == fewest]
    return int(min(tied, key=lambda col: (int((I == col).sum()), col)))


def _symmetry(A, clean):
    """The repeat and the mirror lines the intact cells of A show along its rows.

    A line halfway between rows q-1 and q reflects row r onto row 2q-1-r, so
    name the line by that constant a = 2q-1.  The wallpaper repeats every
    `period` rows and its mirror lines are evenly spaced `period` apart in a;
    where the first one falls is measured too, since a quarter turn of the whole
    picture leaves the tiling out of step with row 0."""
    n, m = A.shape
    same = np.zeros((n, n), dtype=bool)
    seen = np.zeros((n, n), dtype=int)
    for r1 in range(n):
        for r2 in range(r1, n):
            sel = clean[r1] & clean[r2]          # only cells the damage spares
            seen[r1, r2] = seen[r2, r1] = int(sel.sum())
            agree = bool(np.array_equal(A[r1][sel], A[r2][sel]))
            same[r1, r2] = same[r2, r1] = agree
    need = max(2 * m, int(clean.sum()) // 4)     # enough evidence to believe it

    for period in range(1, n):
        evid, ok = 0, True
        for step in range(period, n, period):            # the repeat itself
            for r in range(n - step):
                if not same[r, r + step]:
                    ok = False
                    break
                evid += seen[r, r + step]
            if not ok:
                break
        if not ok:
            continue
        for first in range(period):                      # where the lines fall
            lines, total, fits = [], evid, True
            for a in range(first, 2 * n - 1, period):
                pairs = 0
                for r in range(max(0, a - n + 1), (a + 1) // 2):
                    if not same[r, a - r]:
                        fits = False
                        break
                    pairs += seen[r, a - r]
                if not fits:
                    break
                if pairs:
                    lines.append(a)
                total += pairs
            if fits and lines and total >= need:
                return period, lines
    return None, []


def _sources(start, size, limit, period, lines):
    """Where along one axis an intact copy of a damaged span can be read from:
    reflected across one of the mirror lines, or carried a whole period along
    (which is what two of those reflections compose to)."""
    opts = [(start, False, 0)]
    for a in lines:
        s = a - (start + size - 1)
        if 0 <= s and s + size <= limit:
            opts.append((s, True, abs(s - start)))
    if period:
        for t in range(1, limit // period + 2):
            for s in (start + period * t, start - period * t):
                if 0 <= s and s + size <= limit:
                    opts.append((s, False, abs(s - start)))
    return opts


def _mirrored(block, fv, fh):
    out = block[::-1] if fv else block
    return out[:, ::-1] if fh else out


def _simulate(I, ops, sels):
    """What ARCLE makes of these ops — the Colour / CopyI / CopyO / Paste / Flip
    subset this maker emits, each carrying a full rectangle."""
    src_grid = np.asarray(I, dtype=int)
    G = src_grid.copy()
    h, w = G.shape
    clip = None
    for op, sel in zip(ops, sels):
        op = int(op)
        if op == 34:
            continue
        r, c, dh, dw = sel
        hh, ww = dh + 1, dw + 1
        if op < 10:
            G[r:r + hh, c:c + ww] = op
        elif op in (28, 29):
            clip = (src_grid if op == 28 else G)[r:r + hh, c:c + ww].copy()
        elif op == 30 and clip is not None:
            ch, cw = min(clip.shape[0], h - r), min(clip.shape[1], w - c)
            blk = clip[:ch, :cw]
            np.copyto(G[r:r + ch, c:c + cw], blk, where=blk > 0)
        elif op == 26:
            G[r:r + hh, c:c + ww] = G[r:r + hh, c:c + ww][:, ::-1]
        elif op == 27:
            G[r:r + hh, c:c + ww] = G[r:r + hh, c:c + ww][::-1]
    return G


def _plan(I):
    """Read the wallpaper off I and redraw every damaged rectangle from the
    intact block that mirrors onto it."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    noisec = _damage_color(I)
    hurt = (I == noisec)
    if not hurt.any():
        return ops, sels, I.copy()
    clean = ~hurt
    rper, rlines = _symmetry(I, clean)              # mirror lines across the rows
    cper, clines = _symmetry(I.T, clean.T)          # mirror lines down the columns

    G = I.copy()
    left = hurt.copy()
    clip = [None]                                   # what ARCLE's clipboard holds

    def redraw(r0, c0, hh, ww):
        """Stamp the intact block that mirrors onto this rectangle over it."""
        best = None
        for rs, fv, dr in _sources(r0, hh, h, rper, rlines):
            for cs, fh, dc in _sources(c0, ww, w, cper, clines):
                if rs == r0 and cs == c0 and not fv and not fh:
                    continue
                if left[rs:rs + hh, cs:cs + ww].any():
                    continue                        # that block is damaged too
                blk = G[rs:rs + hh, cs:cs + ww]
                # a mirror image that differs from the block is the reflection
                # actually on show: prefer it, and prefer the nearest block
                shows = (fv or fh) and not np.array_equal(_mirrored(blk, fv, fh), blk)
                key = (0 if shows else 1, dr + dc, rs, cs)
                if best is None or key < best[0]:
                    best = (key, rs, cs, fv, fh)
        if best is None:
            return False

        _, rs, cs, fv, fh = best
        block = G[rs:rs + hh, cs:cs + ww].copy()
        dst = G[r0:r0 + hh, c0:c0 + ww]
        held = clip[0]
        if held is None or held.shape != block.shape or not np.array_equal(held, block):
            # CopyI while that block still stands untouched in the input, CopyO
            # once it is one this trajectory has already redrawn.  The selection
            # is exactly the intact block — a full rectangle of the wallpaper.
            untouched = np.array_equal(I[rs:rs + hh, cs:cs + ww], block)
            ops.append(28 if untouched else 29)
            sels.append([rs, cs, hh - 1, ww - 1])
            clip[0] = block
        if ((block == 0) & (dst != 0)).any():
            # Paste never writes a 0, so lay the block's 0s down as a base first
            ops.append(0)
            sels.append([r0, c0, hh - 1, ww - 1])   # the damaged rectangle, whole
            dst[:, :] = 0
        ops.append(30)
        sels.append([r0, c0, 0, 0])                 # stamp it over the damage
        np.copyto(dst, block, where=block > 0)
        if fv and not np.array_equal(dst[::-1], dst):
            ops.append(27)
            sels.append([r0, c0, hh - 1, ww - 1])   # mirror it top <-> bottom
            dst[:, :] = dst[::-1]
        if fh and not np.array_equal(dst[:, ::-1], dst):
            ops.append(26)
            sels.append([r0, c0, hh - 1, ww - 1])   # mirror it left <-> right
            dst[:, :] = dst[:, ::-1]
        left[r0:r0 + hh, c0:c0 + ww] = False
        return True

    for _sweep in range(4):
        if not left.any():
            break
        moved = False
        for blob in _blobs(left):
            cells = [(r, c) for r, c in blob if left[r, c]]
            if not cells:
                continue                             # redrawn along with another
            r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
            c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
            if redraw(r0, c0, r1 - r0 + 1, c1 - c0 + 1):
                moved = True
                continue
            for r in range(r0, r1 + 1):              # a splotch whose counterpart
                band = [c for c in range(c0, c1 + 1) if left[r, c]]
                if not band:                         # is itself damaged comes
                    continue                         # back band by band,
                if redraw(r, min(band), 1, max(band) - min(band) + 1):
                    moved = True
                    continue
                for c in band:                       # and finally cell by cell
                    if redraw(r, c, 1, 1):
                        moved = True
        if not moved:
            break
    return ops, sels, G


def _prune(I, ops, sels, mended):
    """Drop anything the wallpaper makes unnecessary — a block already on the
    clipboard needs no second pick-up, and a rectangle a later stamp covers
    whole needs no stamp of its own.  Measured against the grid this plan builds
    from I, not against the answer."""
    k = 0
    while k < len(ops):
        thin_ops, thin_sels = ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:]
        if np.array_equal(_simulate(I, thin_ops, thin_sels), mended):
            ops, sels = thin_ops, thin_sels
            k = 0
        else:
            k += 1
    return ops, sels


# ── 1. colours ───────────────────────────────────────────────────────────────

def sample_colors(num_examples=None) -> dict:
    # noisec : the patch colour.  It is what marks the damage, so it has to stay
    #          the same across the episode for the test to be readable.
    # ccols  : the wallpaper palette (never contains noisec); an instance uses a
    #          prefix of it, so every grid of the episode is drawn in one scheme.
    # bgc    : the canvas the tiling is stamped onto — covered completely.
    cols = list(range(10))
    noisec = random.choice(cols)
    rest = [c for c in cols if c != noisec]
    bgc = random.choice(rest)
    ccols = random.sample(rest, len(rest))
    return {"bgc": bgc, "noisec": noisec, "ccols": ccols}


# ── 2. instances ─────────────────────────────────────────────────────────────

def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, ccols) -> dict:
    for _attempt in range(40):
        rot = choice((0, 1, 2, 3))
        turned = rot % 2 == 1                        # a quarter turn swaps h and w
        hcap = min(30, max_w if turned else max_h)
        wcap = min(30, max_h if turned else max_w)
        if hcap < 10 or wcap < 10:
            raise ValueError("grid cap too small for this task")

        h = unifint(diff_lb, diff_ub, (10, hcap))
        w = unifint(diff_lb, diff_ub, (10, wcap))
        hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
        wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
        pinds = asindices(canvas(-1, (hp, wp)))
        numc = unifint(diff_lb, diff_ub, (2, len(ccols)))
        pobj = frozenset({(choice(ccols[:numc]), ij) for ij in pinds})
        go = canvas(bgc, (h, w))
        locs = set()
        for a in range(h // hp + 1):
            for b in range(w // wp + 1):
                loci, locj = hp * a, wp * b
                locs.add((loci, locj))
                mf1 = identity if a % 2 == 0 else hmirror
                mf2 = identity if b % 2 == 0 else vmirror
                go = paint(go, shift(compose(mf1, mf2)(pobj), (loci, locj)))

        numpatches = unifint(diff_lb, diff_ub, (1, int((h * w) ** 0.5 // 2)))
        gi = tuple(e for e in go)
        places = apply(lbind(shift, pinds), locs)
        succ, tr, maxtr = 0, 0, 5 * numpatches
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph, pw = randint(2, 6), randint(2, 6)
            loci, locj = randint(0, h - ph), randint(0, w - pw)
            ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
            gi2 = fill(gi, noisec, ptch)
            candset = apply(normalize, apply(rbind(toobject, gi2), places))
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
               len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2 and \
               (pobj in candset or hmirror(pobj) in candset or
                    vmirror(pobj) in candset or hmirror(vmirror(pobj)) in candset):
                succ += 1
                gi = gi2
        if succ == 0:
            continue

        rotf = (identity, rot90, rot180, rot270)[rot]
        gi, gout = rotf(gi), rotf(go)

        # keep only instances whose damage the wallpaper's own mirrors can put
        # back, and that show the reflection rather than a bare repeat
        A, B = np.array(gi, dtype=int), np.array(gout, dtype=int)
        ops, sels = derive_operations(A, B)
        if not any(int(o) in (26, 27) for o in ops):
            continue
        if not np.array_equal(_simulate(A, ops, sels), B):
            continue
        return {"input": gi, "output": gout}

    raise ValueError("could not build an instance of 0dfd9992")


# ── 3. trajectory ────────────────────────────────────────────────────────────

def derive_operations(I, O):
    """Redraw every damaged block by reflecting the wallpaper's intact
    counterpart onto it: CopyI the block that sits across the mirror line, Paste
    it over the damage, FlipV / FlipH it in place.  The damage colour, the
    mirror lines, the counterpart block and the axis are all measured from I."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels, mended = _plan(I)
    ops, sels = _prune(I, ops, sels, mended)
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])                # the whole grid, as submitted
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
