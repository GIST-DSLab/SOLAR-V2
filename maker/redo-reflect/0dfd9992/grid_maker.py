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
# The task.  I is a wallpaper: a random hp x wp tile stamped over the whole
# canvas, every other stamp mirrored (rows flip on odd tile-rows, columns flip
# on odd tile-columns).  That construction makes the whole grid symmetric about
# EVERY horizontal line r = k*hp and EVERY vertical line c = k*wp.  Solid
# rectangular patches of one colour (never part of the palette) are then painted
# over it.  O is the wallpaper with those patches gone.
#
# The repair is therefore a reflection: a damaged block is re-drawn by mirroring
# an intact block of the wallpaper across one of those symmetry lines —
# CopyO the intact block, Paste it over the damaged one, FlipV / FlipH it in
# place.  (When the only usable counterpart sits a whole period away rather than
# across a line, the same copy/paste without a flip is the translation form of
# the very same symmetry.)  Paste is transparent to 0, so when the block being
# stamped carries 0s the damaged rectangle is first laid down as a 0 base.
# ─────────────────────────────────────────────────────────────────────────────
import random
from collections import Counter, deque

import numpy as np

from maker.sel_helpers import sel_of


def _blobs(mask):
    """8-connected components of a boolean mask (one damaged splotch each)."""
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


def _symmetries(I, clean):
    """Mirror lines and translation periods the intact cells of I actually show.

    A horizontal mirror line y means row r and row 2y-1-r agree everywhere both
    are intact; a row shift d means row r and row r+d agree.  Only candidates
    backed by a decent amount of evidence are kept.
    """
    h, w = I.shape
    min_r, min_c = max(6, 2 * w), max(6, 2 * h)
    row_lines, col_lines, row_sh, col_sh = [], [], [], []

    for y in range(1, h):
        cnt, ok = 0, True
        for r in range(h):
            r2 = 2 * y - 1 - r
            if r2 <= r or r2 >= h:
                continue
            m = clean[r] & clean[r2]
            if m.any():
                if not np.array_equal(I[r][m], I[r2][m]):
                    ok = False
                    break
                cnt += int(m.sum())
        if ok and cnt >= min_r:
            row_lines.append(y)
    for x in range(1, w):
        cnt, ok = 0, True
        for c in range(w):
            c2 = 2 * x - 1 - c
            if c2 <= c or c2 >= w:
                continue
            m = clean[:, c] & clean[:, c2]
            if m.any():
                if not np.array_equal(I[:, c][m], I[:, c2][m]):
                    ok = False
                    break
                cnt += int(m.sum())
        if ok and cnt >= min_c:
            col_lines.append(x)
    for d in range(1, h):
        m = clean[:h - d] & clean[d:]
        if int(m.sum()) >= min_r and np.array_equal(I[:h - d][m], I[d:][m]):
            row_sh.append(d)
    for d in range(1, w):
        m = clean[:, :w - d] & clean[:, d:]
        if int(m.sum()) >= min_c and np.array_equal(I[:, :w - d][m], I[:, d:][m]):
            col_sh.append(d)
    return row_lines, col_lines, row_sh, col_sh


def _rect(sel):
    r, c, dh, dw = sel
    return r, c, dh + 1, dw + 1


def _simulate(I, ops, sels):
    """ARCLE's effect of these ops on I (only the ops this maker emits).

    Paste writes just the clipboard's nonzero cells; Flip zeroes the grid under
    its selection before compositing the flipped block back, so a full-rectangle
    Flip is exactly a flip of that rectangle, 0s included.
    """
    G = np.asarray(I, dtype=int).copy()
    h, w = G.shape
    clip = None
    for op, sel in zip(ops, sels):
        op = int(op)
        if op == 34:
            continue
        if isinstance(sel, dict):
            for r, c in sel["cells"]:
                G[r, c] = op
            continue
        r, c, hh, ww = _rect(sel)
        if op < 10:
            G[r:r + hh, c:c + ww] = op
        elif op == 29:
            clip = G[r:r + hh, c:c + ww].copy()
        elif op == 30 and clip is not None:
            ch, cw = clip.shape
            ch, cw = min(ch, h - r), min(cw, w - c)
            blk, m = clip[:ch, :cw], clip[:ch, :cw] != 0
            G[r:r + ch, c:c + cw][m] = blk[m]
        elif op == 26:
            G[r:r + hh, c:c + ww] = G[r:r + hh, c:c + ww][:, ::-1]
        elif op == 27:
            G[r:r + hh, c:c + ww] = G[r:r + hh, c:c + ww][::-1]
    return G


def _prune(I, O, ops, sels):
    """Drop any op the wallpaper makes unnecessary — a block already sitting on
    the clipboard can be stamped again, so its second pick-up would do nothing."""
    changed = True
    while changed:
        changed = False
        for k in range(len(ops)):
            if np.array_equal(_simulate(I, ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:]), O):
                del ops[k]
                del sels[k]
                changed = True
                break
    return ops, sels


def _flipped(block, fv, fh):
    out = block[::-1] if fv else block
    return out[:, ::-1] if fh else out


def _axis_options(start, size, limit, lines, shifts, cap=24):
    """Where an intact copy of a damaged span can be read from, along one axis.

    Each option is (source start, flip?, distance): either the span translated
    by a whole period, or the span reflected across one of the symmetry lines.
    """
    opts = [(start, False, 0)]
    for d in shifts:
        for s in (start + d, start - d):
            if 0 <= s and s + size <= limit:
                opts.append((s, False, abs(s - start)))
    for y in lines:
        s = 2 * y - 1 - (start + size - 1)
        if 0 <= s and s + size <= limit:
            opts.append((s, True, abs(s - start)))
    opts.sort(key=lambda o: (0 if o[1] else 1, o[2]))
    return opts[:cap]


def _repair_rect(G, O, remaining, r0, c0, hh, ww, ops, sels, sym, clip):
    """Redraw the rectangle (r0,c0,hh,ww) from an intact mirror/period copy.

    CopyO the source block -> Paste it over the damaged block -> flip it in
    place along whichever axes were mirrored.  Returns True when a source that
    reproduces the wallpaper was found.
    """
    h, w = G.shape
    row_lines, col_lines, row_sh, col_sh = sym
    row_opts = _axis_options(r0, hh, h, row_lines, row_sh)
    col_opts = _axis_options(c0, ww, w, col_lines, col_sh)
    want = O[r0:r0 + hh, c0:c0 + ww]

    best = None
    for rs, fv, dr in row_opts:
        for cs, fh, dc in col_opts:
            if rs == r0 and cs == c0 and not fv and not fh:
                continue
            if remaining[rs:rs + hh, cs:cs + ww].any():
                continue
            if not np.array_equal(_flipped(G[rs:rs + hh, cs:cs + ww], fv, fh), want):
                continue
            key = (0 if (fv or fh) else 1, dr + dc)   # a mirror first, then the nearest
            if best is None or key < best[0]:
                best = (key, rs, cs, fv, fh)
    if best is None:
        return False

    _, rs, cs, fv, fh = best
    src = G[rs:rs + hh, cs:cs + ww].copy()
    held = clip[0]
    reuse = (held is not None and held.shape == src.shape
             and np.array_equal(_flipped(held, fv, fh), want))
    # whole rectangles: the selection IS exactly the block being copied / redrawn.
    # When the block already on the clipboard is one that mirrors onto this damage
    # just as well, stamp it again instead of picking up a second copy of it.
    if not reuse:
        ops.append(29)
        sels.append([rs, cs, hh - 1, ww - 1])        # CopyO the intact block
        clip[0] = src
    if (clip[0] == 0).any():
        # Paste never writes a 0, so lay the block's 0s down as a base first
        ops.append(0)
        sels.append([r0, c0, hh - 1, ww - 1])
    ops.append(30)
    sels.append([r0, c0, 0, 0])                      # Paste it over the damage
    G[r0:r0 + hh, c0:c0 + ww] = clip[0]
    if fv:
        blk = G[r0:r0 + hh, c0:c0 + ww][::-1].copy()
        if not np.array_equal(blk, G[r0:r0 + hh, c0:c0 + ww]):
            ops.append(27)
            sels.append([r0, c0, hh - 1, ww - 1])    # mirror it top<->bottom
            G[r0:r0 + hh, c0:c0 + ww] = blk
    if fh:
        blk = G[r0:r0 + hh, c0:c0 + ww][:, ::-1].copy()
        if not np.array_equal(blk, G[r0:r0 + hh, c0:c0 + ww]):
            ops.append(26)
            sels.append([r0, c0, hh - 1, ww - 1])    # mirror it left<->right
            G[r0:r0 + hh, c0:c0 + ww] = blk
    remaining[r0:r0 + hh, c0:c0 + ww] = False
    return True


def _plan(I, O):
    """(ops, sels, every_patch_done_geometrically) for the repair of I into O."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ops, sels = [], []
    diff = (I != O)
    if not diff.any():
        return ops, sels, True

    # the occluder colour: it is what the damaged cells carry, and it never
    # occurs in the wallpaper, so every cell of it is damage
    noisec = int(Counter(I[diff].tolist()).most_common(1)[0][0])
    remaining = (I == noisec) | diff
    clean = ~remaining
    G = I.copy()
    geometric = True
    sym = _symmetries(I, clean)
    clip = [None]                                    # what ARCLE's clipboard holds

    for blob in _blobs(remaining):
        cells = [(r, c) for r, c in blob if remaining[r, c]]
        if not cells:
            continue                                 # already redrawn with a neighbour
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if _repair_rect(G, O, remaining, r0, c0, r1 - r0 + 1, c1 - c0 + 1,
                        ops, sels, sym, clip):
            continue
        # the splotch as a whole has no intact counterpart in the grid; take it
        # apart row band by row band, and only paint what stays unreachable
        for r in range(r0, r1 + 1):
            row_cells = [c for c in range(c0, c1 + 1) if remaining[r, c]]
            if not row_cells:
                continue
            a, b = min(row_cells), max(row_cells)
            if _repair_rect(G, O, remaining, r, a, 1, b - a + 1, ops, sels, sym, clip):
                continue
            geometric = False
            groups = {}
            for c in row_cells:
                groups.setdefault(int(O[r, c]), []).append((r, c))
            for col in sorted(groups):
                ops.append(col)
                sels.append(sel_of(groups[col]))
                for cell in groups[col]:
                    G[cell] = col
                    remaining[cell] = False
    ops, sels = _prune(I, O, ops, sels)
    return ops, sels, geometric


# ── 1. colours ───────────────────────────────────────────────────────────────

def sample_colors(num_examples=None) -> dict:
    # noisec : the occluder colour.  It marks the damaged cells, so it must be
    #          the same in every instance of the episode or the test is unreadable.
    # ccols  : the wallpaper palette (never contains noisec).
    # bgc    : the canvas colour the tiling paints over (invisible in the end).
    cols = list(range(10))
    noisec = random.choice(cols)
    rest = [c for c in cols if c != noisec]
    bgc = random.choice(rest)
    ccols = random.sample(rest, random.randint(2, len(rest)))
    return {"bgc": bgc, "noisec": noisec, "ccols": ccols}


# ── 2. instances ─────────────────────────────────────────────────────────────

def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, ccols) -> dict:
    randint, choice = random.randint, random.choice

    def _unif(bounds):                       # re-arc's unifint, inlined
        a, b = bounds
        b = max(a, b)
        d = random.uniform(diff_lb, diff_ub)
        return min(max(a, round(a + (b - a) * d)), b)

    for _attempt in range(60):
        rot = choice((0, 1, 2, 3))
        swaps = rot % 2 == 1
        hcap = min(30, max_w if swaps else max_h)
        wcap = min(30, max_h if swaps else max_w)
        if hcap < 10 or wcap < 10:
            raise ValueError("grid cap too small for this task")

        h = _unif((10, hcap))
        w = _unif((10, wcap))
        hp = _unif((2, h // 2 - 1))
        wp = _unif((2, w // 2 - 1))

        tile = np.array([[choice(ccols) for _ in range(wp)] for _ in range(hp)], dtype=int)
        go = np.full((h, w), bgc, dtype=int)
        for a in range(h // hp + 1):
            for b in range(w // wp + 1):
                blk = tile
                if a % 2:                            # every other stamp is mirrored
                    blk = blk[::-1, :]
                if b % 2:
                    blk = blk[:, ::-1]
                r0, c0 = hp * a, wp * b
                if r0 >= h or c0 >= w:
                    continue
                r1, c1 = min(r0 + hp, h), min(c0 + wp, w)
                go[r0:r1, c0:c1] = blk[:r1 - r0, :c1 - c0]

        numpatches = _unif((1, max(1, int((h * w) ** 0.5 // 2))))
        gi = go.copy()
        succ, tr, maxtr = 0, 0, 5 * numpatches
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph, pw = randint(2, 6), randint(2, 6)
            loci, locj = randint(0, h - ph), randint(0, w - pw)
            gi2 = gi.copy()
            gi2[loci:loci + ph, locj:locj + pw] = noisec
            # keep two intact rows and two intact columns, as the original does
            if sum(1 for r in range(h) if noisec not in gi2[r]) < 2:
                continue
            if sum(1 for c in range(w) if noisec not in gi2[:, c]) < 2:
                continue
            gi = gi2
            succ += 1
        if succ == 0:
            continue

        gi = np.rot90(gi, rot)
        go = np.rot90(go, rot)
        # every patch must be reconstructible by mirroring an intact block of the
        # wallpaper — otherwise the instance does not show the rule
        _o, _s, geometric = _plan(gi, go)
        if not geometric or not np.array_equal(_simulate(gi, _o, _s), go):
            continue
        if not any(int(o) in (26, 27) for o in _o):
            continue                                 # the reflection must be on show
        return {"input": gi.tolist(), "output": go.tolist()}

    raise ValueError("could not build an instance")


# ── 3. trajectory ────────────────────────────────────────────────────────────

def derive_operations(I, O):
    """Repair each damaged block by reflecting an intact block of the wallpaper
    onto it: CopyO the counterpart across a symmetry line, Paste it over the
    damage, FlipV / FlipH it in place."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    ops, sels, _ = _plan(I, O)
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
