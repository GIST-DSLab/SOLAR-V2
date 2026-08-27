"""
ARC Task: 6aa20dc0 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of


# ─────────────────────────────────────────────────────────────────────────────
# The rule of 6aa20dc0
#   The grid holds ONE complete "template": a small square object whose two
#   opposite corners carry two marker colours (c1, c2) and whose remaining cells
#   are the fill colour fgc (the template is the only component with 3 colours).
#   Scattered elsewhere are INCOMPLETE copies: only the two corner markers
#   survive, upscaled by some factor 1..4 and mirrored by some element of
#   {identity, dmirror, cmirror, hmirror, vmirror}, everything around those
#   markers being background.  The transformation completes every such copy by
#   painting its missing fgc cells (interior background holes stay background).
# ─────────────────────────────────────────────────────────────────────────────

# object = set of (colour, (row, col)) cells
_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _ul(obj):
    return (min(ij[0] for _, ij in obj), min(ij[1] for _, ij in obj))


def _lr(obj):
    return (max(ij[0] for _, ij in obj), max(ij[1] for _, ij in obj))


def _shift(obj, d):
    di, dj = d
    return frozenset((v, (ij[0] + di, ij[1] + dj)) for v, ij in obj)


def _normalize(obj):
    di, dj = _ul(obj)
    return _shift(obj, (-di, -dj))


def _identity(obj):
    return frozenset(obj)


def _dmirror(obj):
    a, b = _ul(obj)
    return frozenset((v, (ij[1] - b + a, ij[0] - a + b)) for v, ij in obj)


def _hmirror(obj):
    d = _ul(obj)[0] + _lr(obj)[0]
    return frozenset((v, (d - ij[0], ij[1])) for v, ij in obj)


def _vmirror(obj):
    d = _ul(obj)[1] + _lr(obj)[1]
    return frozenset((v, (ij[0], d - ij[1])) for v, ij in obj)


def _cmirror(obj):
    return _vmirror(_dmirror(_vmirror(obj)))


_MIRRORS = (_identity, _dmirror, _cmirror, _hmirror, _vmirror)


def _upscale(obj, f):
    if f == 1:
        return frozenset(obj)
    di, dj = _ul(obj)
    out = set()
    for v, ij in obj:
        i = (ij[0] - di) * f + di
        j = (ij[1] - dj) * f + dj
        for a in range(f):
            for b in range(f):
                out.add((v, (i + a, j + b)))
    return frozenset(out)


def _components(grid, bgc):
    """8-connected, multi-colour, background-excluding components."""
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] == bgc:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                i, j = stack.pop()
                cells.append((grid[i][j], (i, j)))
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        a, b = i + di, j + dj
                        if 0 <= a < h and 0 <= b < w and not seen[a][b] \
                                and grid[a][b] != bgc:
                            seen[a][b] = True
                            stack.append((a, b))
            comps.append(cells)
    return comps


def _occurrences(grid, patt, bgc):
    """Every top-left placement at which `patt` matches `grid` exactly."""
    h, w = len(grid), len(grid[0])
    mi = min(ij[0] for _, ij in patt)
    mj = min(ij[1] for _, ij in patt)
    cells = [(v, ij[0] - mi, ij[1] - mj) for v, ij in patt]
    cells.sort(key=lambda t: (t[0] == bgc, t[1], t[2]))   # markers first (early exit)
    oh = max(t[1] for t in cells) + 1
    ow = max(t[2] for t in cells) + 1
    res = []
    for i in range(h - oh + 1):
        for j in range(w - ow + 1):
            ok = True
            for v, a, b in cells:
                if grid[i + a][j + b] != v:
                    ok = False
                    break
            if ok:
                res.append((i, j))
    return res


def _rule(grid):
    """Read the rule off the input alone.

    Returns (bgc, fgc, {completed object (absolute cells): placement}).
    A copy is recognised where the template's non-fgc cells (its two corner
    markers), in some mirroring at some upscale, sit on the grid with pure
    background in the 4-neighbour ring around them — matched on a
    background-padded canvas so copies touching the border still count.
    """
    h, w = len(grid), len(grid[0])
    bgc = Counter(v for row in grid for v in row).most_common(1)[0][0]
    comps = _components(grid, bgc)
    if not comps:
        return None
    ncols = [len({v for v, _ in cm}) for cm in comps]
    mx = max(ncols)
    key = [cell for cm, nc in zip(comps, ncols) if nc == mx for cell in cm]
    rs = [ij[0] for _, ij in key]
    cs = [ij[1] for _, ij in key]
    tmpl = frozenset(
        (grid[r][c], (r, c))
        for r in range(min(rs), max(rs) + 1)
        for c in range(min(cs), max(cs) + 1)
        if grid[r][c] != bgc
    )
    if not tmpl:
        return None
    fgc = Counter(v for v, _ in tmpl).most_common(1)[0][0]
    base = _normalize(tmpl)
    padded = [[bgc] * (w + 2) for _ in range(h + 2)]
    for r in range(h):
        prow = padded[r + 1]
        for c in range(w):
            prow[c + 1] = grid[r][c]
    placements = {}
    for fac in (1, 2, 3, 4):
        up = _upscale(base, fac)
        for mf in _MIRRORS:
            stamp = _normalize(mf(up))
            vis = [cell for cell in stamp if cell[0] != fgc]
            if not vis:
                continue
            visn = _normalize(frozenset(vis))
            visidx = {ij for _, ij in visn}
            ring = set()
            for i, j in visidx:
                for di, dj in _DIRS4:
                    p = (i + di, j + dj)
                    if p not in visidx:
                        ring.add(p)
            patt = list(visn) + [(bgc, p) for p in ring]
            # the pattern's ring reaches (-1,-1), so normalising it shifts by
            # (1,1) — which cancels the (1,1) padding offset exactly
            for loc in _occurrences(padded, patt, bgc):
                placements[frozenset(_shift(stamp, loc))] = loc
    return bgc, fgc, placements


def _apply_rule(grid):
    res = _rule(grid)
    if res is None:
        return None
    bgc, fgc, placements = res
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    done = {}
    for obj in placements:
        for v, (i, j) in obj:
            if not (0 <= i < h and 0 <= j < w):
                continue
            if done.get((i, j), v) != v:
                return None                       # ambiguous overlap
            if grid[i][j] != bgc and grid[i][j] != v:
                return None                       # would overwrite other content
            done[(i, j)] = v
            out[i][j] = v
    return out


# ── 1. colours ───────────────────────────────────────────────────────────────
def sample_colors(num_examples=None) -> dict:
    # the generator samples 4 distinct colours: background, fill colour, and the
    # two corner-marker colours.  The rule (which colour gets painted) depends on
    # the fill colour role, so all four are fixed for the whole episode.
    # No discrete structural variants exist: upscale factor and mirroring vary
    # per copy INSIDE every instance, so every case is already demonstrated.
    bgc, fgc, c1, c2 = random.sample(range(10), 4)
    return {"bgc": bgc, "fgc": fgc, "c1": c1, "c2": c2}


# ── 2. generator ─────────────────────────────────────────────────────────────
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _attempt(diff_lb, diff_ub, max_h, max_w, bgc, fgc, c1, c2, nocc_cap):
    h = _unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = _unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    od = _unifint(diff_lb, diff_ub, (2, 4))

    ncellsextra = random.randint(1, max(1, (od ** 2 - 2) // 2))
    pool = [(i, j) for i in range(od) for j in range(od)
            if (i, j) not in ((0, 0), (od - 1, od - 1))]
    extracells = set(random.sample(pool, ncellsextra))
    extracells.add(random.choice([(0, 1), (1, 0)]))              # dneighbour of (0,0)
    extracells.add(random.choice([(od - 2, od - 1), (od - 1, od - 2)]))

    obj = {(c1, (0, 0)), (c2, (od - 1, od - 1))}
    obj |= {(fgc, ij) for ij in extracells}
    obj = frozenset(obj) | _dmirror(frozenset(obj))              # diagonal symmetry
    if random.choice((True, False)):
        obj = _hmirror(obj)

    # the template must read as ONE 3-coloured component, else it is not the
    # unique max-colour object the rule keys on
    tgrid = [[bgc] * od for _ in range(od)]
    for v, (i, j) in obj:
        tgrid[i][j] = v
    tcomps = _components(tgrid, bgc)
    if len(tcomps) != 1 or len({v for v, _ in tcomps[0]}) != 3:
        return None

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    loci = random.randint(0, h - od)
    locj = random.randint(0, w - od)
    for v, (i, j) in _shift(obj, (loci, locj)):
        gi[i][j] = v
        go[i][j] = v

    inds = {(i, j) for i in range(h) for j in range(w)}
    inds -= {(i, j)
             for i in range(loci - 1, loci + od + 1)
             for j in range(locj - 1, locj + od + 1)}

    nocc = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (od ** 2 * 2))))
    if nocc_cap is not None:
        nocc = min(nocc, nocc_cap)
    succ, tr, maxtr = 0, 0, 4 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = random.randint(1, 4)
        mf1 = random.choice(_MIRRORS)
        mf2 = random.choice(_MIRRORS)
        cobj = _normalize(_upscale(mf2(mf1(obj)), fac))
        ohx = _lr(cobj)[0] + 1
        owx = _lr(cobj)[1] + 1
        cands = [ij for ij in inds if ij[0] <= h - ohx and ij[1] <= w - owx]
        if not cands:
            continue
        locc = random.choice(sorted(cands))
        cobjo = _shift(cobj, locc)
        cobjoi = {ij for _, ij in cobjo}
        if not cobjoi <= inds:
            continue
        succ += 1
        r0 = min(ij[0] for ij in cobjoi) - 1
        r1 = max(ij[0] for ij in cobjoi) + 1
        c0 = min(ij[1] for ij in cobjoi) - 1
        c1b = max(ij[1] for ij in cobjoi) + 1
        inds -= {(i, j) for i in range(r0, r1 + 1) for j in range(c0, c1b + 1)}
        for v, (i, j) in cobjo:
            go[i][j] = v
            if v != fgc:                    # only the corner markers survive in I
                gi[i][j] = v
    if succ < 1:
        return None

    # background must be the grid's unambiguous majority colour
    cnt = Counter(v for row in gi for v in row).most_common(2)
    if cnt[0][0] != bgc or (len(cnt) > 1 and cnt[0][1] == cnt[1][1]):
        return None
    # the generator's own output must coincide with what the rule produces:
    # markers of separate copies can accidentally line up into an extra readable
    # copy, and such instances are ambiguous — resample them away
    if _apply_rule(gi) != go:
        return None
    return {"input": gi, "output": go}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, c1, c2) -> dict:
    max_h = max(10, min(int(max_h), 30))
    max_w = max(10, min(int(max_w), 30))
    for k in range(400):
        cap = None if k < 120 else (3 if k < 240 else 1)
        inst = _attempt(diff_lb, diff_ub, max_h, max_w, bgc, fgc, c1, c2, cap)
        if inst is not None:
            return inst
    raise ValueError("6aa20dc0: could not sample a consistent instance")


# ── 3. operations ────────────────────────────────────────────────────────────
def derive_operations(I, O):
    """One Color(fgc) op per incomplete copy: paint that copy's missing cells.

    Everything is measured from I: the template is the 3-coloured component, fgc
    is the colour dominating it, and each copy is located by matching the
    template's markers (mirrored / upscaled, surrounded by background).  Copies
    are completed one whole object at a time, in reading order.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    grid = I.tolist()
    cur = [row[:] for row in grid]
    ops, sels = [], []

    res = _rule(grid)
    if res is not None:
        bgc, fgc, placements = res
        for obj, loc in sorted(placements.items(), key=lambda kv: kv[1]):
            cells = [(i, j) for v, (i, j) in obj
                     if v == fgc and 0 <= i < h and 0 <= j < w and cur[i][j] != fgc]
            if not cells:                 # already completed by an overlapping copy
                continue
            for i, j in cells:
                cur[i][j] = fgc
            ops.append(int(fgc))
            sels.append(sel_of(cells))

    # safety net: emits nothing for instances this generate() produces (the rule
    # above reproduces O exactly there); present only so a foreign (I, O) pair
    # from the raw RE-ARC generator still yields a correct trajectory
    rest = [(r, c) for r in range(h) for c in range(w) if cur[r][c] != O[r, c]]
    while rest:
        col = int(O[rest[0][0], rest[0][1]])
        grp = [(r, c) for r, c in rest if int(O[r, c]) == col]
        for r, c in grp:
            cur[r][c] = col
        ops.append(col)
        sels.append(sel_of(grp))
        rest = [(r, c) for r, c in rest if int(O[r, c]) != col]

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])     # full-grid rectangle: Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
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
                                f"for task 6aa20dc0"
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
                    f"Failed to build a complete episode for task 6aa20dc0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6aa20dc0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
