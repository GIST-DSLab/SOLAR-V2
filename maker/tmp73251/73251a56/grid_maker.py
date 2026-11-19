"""
ARC Task: 00d62c1b (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def _mirror_pos(mode, n, i, j):
    """Position of (i, j) reflected across the grid's symmetry diagonal."""
    return (j, i) if mode == "main" else (n - 1 - j, n - 1 - i)


def _solve_from_input(G):
    """Recover the clean, diagonal-symmetric fan from a noisy grid.

    The task's grids are square colour fans that are exact mirror images of
    themselves across one diagonal, with the diagonal itself painted a single
    colour, and then vandalised by axis-aligned rectangles of ONE noise colour
    that never lands symmetrically on both sides of that diagonal.

    So for a candidate (noise colour c, diagonal `mode`): copy every non-c cell
    onto its mirror position, repaint the diagonal with the colour that sits at
    the diagonal's untouched corner, and demand that the result be perfectly
    mirror-symmetric.  Among the candidates that pass, the true noise colour is
    the rarest one.  Returns (count, colour, mode, repaired_grid) or None.
    """
    G = [list(r) for r in G]
    n = len(G)
    if n == 0 or len(G[0]) != n:
        return None
    best = None
    pal = sorted({v for row in G for v in row})
    for mode in ("main", "anti"):
        if mode == "main":
            diag = [(k, k) for k in range(n)]
            dcol = G[0][0]
        else:
            diag = [(k, n - 1 - k) for k in range(n)]
            dcol = G[0][n - 1]
        for c in pal:
            R = [row[:] for row in G]
            for i in range(n):
                Gi = G[i]
                for j in range(n):
                    v = Gi[j]
                    if v != c:
                        pi, pj = _mirror_pos(mode, n, i, j)
                        R[pi][pj] = v
            for (i, j) in diag:
                R[i][j] = dcol
            ok = True
            for i in range(n):
                for j in range(n):
                    pi, pj = _mirror_pos(mode, n, i, j)
                    if R[i][j] != R[pi][pj]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cnt = sum(row.count(c) for row in G)
                if best is None or cnt < best[0]:
                    best = (cnt, c, mode, R)
    return best


def _components(cells):
    """4-connected components of a set of (r, c) cells."""
    todo = set(cells)
    out = []
    while todo:
        seed = min(todo)
        todo.discard(seed)
        comp = [seed]
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in todo:
                    todo.discard((nr, nc))
                    comp.append((nr, nc))
                    stack.append((nr, nc))
        out.append(sorted(comp))
    return out


VARIANTS = [{"axis": "main"}, {"axis": "anti"}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    noisec = random.choice(cols)
    palette_order = [c for c in cols if c != noisec]
    random.shuffle(palette_order)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"noisec": noisec, "palette_order": palette_order, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, noisec, palette_order, axis=None) -> dict:
    remcols = [c for c in palette_order if c != noisec]
    maxd = min(30, int(max_h), int(max_w))
    maxd = max(8, maxd)
    lod = min(10, maxd)
    attempt = 0
    while True:
        attempt += 1
        d = unifint(diff_lb, diff_ub, (lod, maxd))
        h, w = d, d
        nsl = unifint(diff_lb, diff_ub, (2, max(2, min(9, h // 2))))
        nsl = max(2, min(nsl, len(remcols), h - 1))
        slopes = [0] + sorted(sample(interval(1, h - 1, 1), nsl - 1))
        ccols = remcols[:nsl]
        gi = canvas(-1, (h, w))
        inds = asindices(gi)
        for col, hdelt in zip(ccols, slopes):
            slope = hdelt / w
            locs = sfilter(inds, lambda ij: slope * ij[1] <= ij[0])
            gi = fill(gi, col, locs)
        ln = connect((0, 0), (d - 1, d - 1))
        gi = fill(gi, ccols[-2], ln)
        obj = asobject(gi)
        obj = sfilter(obj, lambda cij: cij[1][1] >= cij[1][0])
        gi = paint(gi, dmirror(obj))
        go = tuple(e for e in gi)
        cf1 = lambda g: ccols[-2] in palette(toobject(ln, g))
        cf2 = lambda g: len((ofcolor(g, noisec) & frozenset(
            {ij[::-1] for ij in ofcolor(g, noisec)})) - ln) == 0
        nhi = max(1, (h * w) // 15)
        ndist = unifint(diff_lb, diff_ub, (1, nhi))
        if attempt > 40:
            ndist = min(ndist, 1 + attempt % 3)
        tr, succ = 0, 0
        maxtr = 10 * ndist
        while tr < maxtr and succ < ndist:
            tr += 1
            oh = randint(1, min(5, h - 2))
            ow = randint(1, min(5, w - 2))
            loci = randint(1, h - oh - 1)
            locj = randint(1, w - ow - 1)
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            gi2 = fill(gi, noisec, bd)
            if cf1(gi2) and cf2(gi2):
                succ += 1
                gi = gi2
        if gi == go:
            continue
        if axis == "main":
            rotf = choice((identity, rot180))
        elif axis == "anti":
            rotf = choice((rot90, rot270))
        else:
            rotf = choice((identity, rot90, rot180, rot270))
        gi = rotf(gi)
        go = rotf(go)
        best = _solve_from_input(gi)
        if best is None or best[1] != noisec:
            continue
        if tuple(tuple(r) for r in best[3]) != tuple(tuple(int(v) for v in r) for r in go):
            continue
        return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    G = [[int(v) for v in row] for row in I]

    ops, sels = [], []
    best = _solve_from_input(G)
    if best is not None:
        _cnt, noise, _mode, R = best
        # every cell of the noise colour whose mirror/diagonal reconstruction
        # gives it a different colour is a vandalised cell
        dirty = [(r, c) for r in range(hi) for c in range(wi)
                 if G[r][c] == noise and R[r][c] != noise]
        # repair one noise blob at a time, band by band inside the blob
        for comp in sorted(_components(dirty), key=lambda cc: cc[0]):
            bands = {}
            for (r, c) in comp:
                bands.setdefault(R[r][c], []).append((r, c))
            for col in sorted(bands, key=lambda k: bands[k][0]):
                ops.append(int(col))
                sels.append(sel_of(bands[col]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
    return ops, sels
