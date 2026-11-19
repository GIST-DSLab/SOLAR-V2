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
from collections import deque

import numpy as np

from maker.sel_helpers import sel_of

# The rule's offset lattice: neighbours-of-neighbours of the origin, i.e. every
# (k, l) with |k| <= 2 and |l| <= 2.  A damaged cell is repaired from the intact
# cell k vertical periods and l horizontal periods away.
_MULTS = sorted({(k, l) for k in (-2, -1, 0, 1, 2) for l in (-2, -1, 0, 1, 2)},
                key=lambda kl: (abs(kl[0]) + abs(kl[1]), abs(kl[0]), abs(kl[1])))


# ── helpers shared by generate() and derive_operations() ─────────────────────

def _components(grid, color):
    """4-connected components of `color` in a 2-D numpy int array."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append(comp)
    return comps


def _detect_noise_color(grid):
    """The occluder colour: fewest connected components, ties broken by fewest cells.

    Returns (colour, is_unique) — the same choice the task's rule makes.
    """
    stats = []
    for col in sorted(set(grid.flatten().tolist())):
        ncomp = len(_components(grid, col))
        ncell = int((grid == col).sum())
        stats.append((ncomp, ncell, col))
    stats.sort()
    best = stats[0]
    unique = len(stats) == 1 or (stats[1][0], stats[1][1]) != (best[0], best[1])
    return best[2], unique


def _row_period(mat):
    """Smallest p with mat[:, j] == mat[:, j-p] for every j >= p (else the width)."""
    w = mat.shape[1]
    for p in range(1, w):
        if np.array_equal(mat[:, p:], mat[:, :-p]):
            return p
    return w


def _periods(grid, noisec):
    """(vertical, horizontal) period, measured on the rows / columns free of noise."""
    clean_rows = [r for r in range(grid.shape[0]) if noisec not in grid[r]]
    clean_cols = [c for c in range(grid.shape[1]) if noisec not in grid[:, c]]
    if not clean_rows or not clean_cols:
        return None, None
    hp = _row_period(grid[clean_rows, :])
    vp = _row_period(grid[:, clean_cols].T)
    return vp, hp


def _predict(grid, noisec, vp, hp):
    """For every damaged cell, the colour periodicity dictates (absent if unreachable)."""
    h, w = grid.shape
    out = {}
    for r in range(h):
        for c in range(w):
            if grid[r, c] != noisec:
                continue
            for k, l in _MULTS:
                sr, sc = r - k * vp, c - l * hp
                if 0 <= sr < h and 0 <= sc < w and grid[sr, sc] != noisec:
                    out[(r, c)] = int(grid[sr, sc])
                    break
    return out


# ── 1. colours ───────────────────────────────────────────────────────────────

def sample_colors(num_examples=None) -> dict:
    # bgc   : canvas colour the generator paints before tiling (the tiles cover it entirely)
    # noisec: the occluder colour -- it is what marks the damaged cells, so it MUST be the
    #         same in every instance of the episode or the test instance is unreadable
    # cpool : the pattern palette (noisec is never part of it)
    cols = list(range(10))
    bgc = random.choice(cols)
    noisec = random.choice([c for c in cols if c != bgc])
    cpool = [c for c in cols if c != noisec]
    random.shuffle(cpool)
    return {"bgc": bgc, "noisec": noisec, "cpool": cpool}


# ── 2. instances ─────────────────────────────────────────────────────────────

def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, cpool) -> dict:
    randint, choice = random.randint, random.choice

    def _unif(bounds):                       # re-arc's unifint, inlined
        a, b = bounds
        d = random.uniform(diff_lb, diff_ub)
        return min(max(a, round(a + (b - a) * d)), b)

    for _attempt in range(40):
        rot = choice((0, 1, 2, 3))
        swaps = rot % 2 == 1
        hcap = max(10, min(30, max_w if swaps else max_h))
        wcap = max(10, min(30, max_h if swaps else max_w))
        if hcap < 10 or wcap < 10:
            raise ValueError("grid cap too small for this task")

        h = _unif((10, hcap))
        w = _unif((10, wcap))
        hp = _unif((2, h // 2 - 1))
        wp = _unif((2, w // 2 - 1))

        numc = _unif((2, 9))
        ccols = cpool[:numc]
        block = np.array([[choice(ccols) for _ in range(wp)] for _ in range(hp)], dtype=int)

        # mirror-tile the block over the whole canvas
        go = np.full((h, w), bgc, dtype=int)
        for a in range(h // hp + 1):
            for b in range(w // wp + 1):
                blk = block
                if a % 2:
                    blk = blk[::-1, :]
                if b % 2:
                    blk = blk[:, ::-1]
                r0, c0 = hp * a, wp * b
                r1, c1 = min(r0 + hp, h), min(c0 + wp, w)
                if r0 < h and c0 < w:
                    go[r0:r1, c0:c1] = blk[:r1 - r0, :c1 - c0]

        # tile origins fully inside the grid: candidate windows for the "a tile survived" test
        locs = [(hp * a, wp * b)
                for a in range(h // hp + 1) for b in range(w // wp + 1)
                if hp * (a + 1) <= h and wp * (b + 1) <= w]
        variants = [block, block[::-1, :], block[:, ::-1], block[::-1, ::-1]]

        numpatches = _unif((1, int((h * w) ** 0.5 // 2)))
        gi = go.copy()
        succ, tr, maxtr = 0, 0, 5 * numpatches
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph, pw = randint(2, 6), randint(2, 6)
            loci, locj = randint(0, h - ph), randint(0, w - pw)
            gi2 = gi.copy()
            gi2[loci:loci + ph, locj:locj + pw] = noisec

            # at least two intact rows and two intact columns
            if sum(1 for r in range(h) if noisec not in gi2[r]) < 2:
                continue
            if sum(1 for c in range(w) if noisec not in gi2[:, c]) < 2:
                continue
            # at least one whole tile survived untouched
            if not any(any(np.array_equal(gi2[r0:r0 + hp, c0:c0 + wp], v) for v in variants)
                       for r0, c0 in locs):
                continue
            # the occluder must still be the colour the rule singles out, the measured
            # periods must be genuine periods of the pattern, and every damaged cell must
            # have an intact counterpart a whole number of periods away -- otherwise the
            # rule would leave that cell occluded and the output would not be the pattern
            col, uniq = _detect_noise_color(gi2)
            if col != noisec or not uniq:
                continue
            vp, hpd = _periods(gi2, noisec)
            if not vp or not hpd or vp >= h or hpd >= w:
                continue
            if not np.array_equal(go[vp:, :], go[:-vp, :]):
                continue
            if not np.array_equal(go[:, hpd:], go[:, :-hpd]):
                continue
            pred = _predict(gi2, noisec, vp, hpd)
            if len(pred) != int((gi2 == noisec).sum()):
                continue
            if any(go[r, c] != v for (r, c), v in pred.items()):
                continue
            succ += 1
            gi = gi2

        if succ == 0:
            continue

        return {"input": np.rot90(gi, rot).tolist(),
                "output": np.rot90(go, rot).tolist()}

    raise ValueError("could not build an instance")


# ── 3. trajectory ────────────────────────────────────────────────────────────

def derive_operations(I, O):
    """I is a doubly periodic, mirror-tiled wallpaper with solid rectangular patches
    painted over it in one occluder colour (that colour never occurs in the pattern, so
    the damaged cells are plainly visible in I).  Repair every patch: each damaged cell
    takes the colour of the intact cell a whole number of periods away, the periods being
    measured on the rows and columns that no patch touches.  One Color op per
    (patch, colour); patches are repaired one at a time, in raster order of their
    top-left cell, and each op's selection is exactly that patch's cells of that colour."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    diff = [(r, c) for r in range(h) for c in range(w) if I[r, c] != O[r, c]]
    if diff:
        noisec = int(I[diff[0]])                  # the colour the damaged cells carry
        vp, hp = _periods(I, noisec)              # pattern periods, read off the clean lines
        pred = _predict(I, noisec, vp, hp) if vp and hp else {}
        for cell in diff:                         # safety net: the rule must explain O
            if pred.get(cell) != int(O[cell]):
                pred[cell] = int(O[cell])

        comps = [sorted(cmp) for cmp in _components(I, noisec)]
        comps.sort(key=lambda cmp: cmp[0])
        for comp in comps:                        # finish one patch before starting the next
            groups = {}
            for cell in comp:
                tgt = pred.get(cell)
                if tgt is None or tgt == noisec:  # nothing the periodicity can restore here
                    continue
                groups.setdefault(tgt, []).append(cell)
            for tgt in sorted(groups, key=lambda t: groups[t][0]):
                ops.append(tgt)
                sels.append(sel_of(groups[tgt]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
