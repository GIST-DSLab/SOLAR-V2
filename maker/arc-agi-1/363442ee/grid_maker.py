"""
ARC Task: 363442ee (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

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


def sample_colors() -> dict:
    cols = list(range(10))
    bgc = 0
    remcols = [c for c in cols if c != bgc]
    barcol = random.choice(remcols)
    remcols2 = [c for c in remcols if c != barcol]
    dotcol = random.choice(remcols2)
    return {"bgc": bgc, "barcol": barcol, "dotcol": dotcol}


def _apply_mf(g, name):
    if name == 'identity':
        return g
    if name == 'vmirror':
        return np.fliplr(g).copy()
    if name == 'hmirror':
        return np.flipud(g).copy()
    if name == 'rot90':
        return np.rot90(g, k=3).copy()
    if name == 'rot180':
        return np.rot90(g, k=2).copy()
    if name == 'rot270':
        return np.rot90(g, k=1).copy()
    if name == 'dmirror':
        return g.T.copy()
    if name == 'cmirror':
        return np.rot90(g.T, k=2).copy()
    return g


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, dotcol) -> dict:
    cols = list(range(10))
    lim = min(max_h, max_w)

    valid_hw = []
    for hb in range(1, 4):
        for wb in range(1, 4):
            hh = hb * 2 + 1
            ww = wb * 2 + 1
            if lim // hh >= 2 and (lim - ww - 1) // ww >= 2:
                valid_hw.append((hh, ww))
    if not valid_hw:
        valid_hw = [(3, 3)]

    h, w = random.choice(valid_hw)

    max_nremh = max(2, lim // h)
    max_nremw = max(2, (lim - w - 1) // w)
    nremh = random.randint(2, max_nremh)
    nremw = random.randint(2, max_nremw)

    rsh = nremh * h
    rsw = nremw * w

    remcols = [c for c in cols if c != bgc]
    remcols2 = [c for c in remcols if c != barcol]
    pool = remcols2
    nfullremcols = random.randint(1, len(pool))
    fullremcols = random.sample(pool, nfullremcols)

    ulc = np.array([[random.choice(fullremcols) for _ in range(w)] for _ in range(h)], dtype=int)

    total_h = rsh
    total_w = w + 1 + rsw

    gi = np.full((total_h, total_w), bgc, dtype=int)
    go = np.full((total_h, total_w), bgc, dtype=int)

    gi[:h, :w] = ulc
    go[:h, :w] = ulc

    gi[:, w] = barcol
    go[:, w] = barcol

    dotcands = [(i, j) for i in range(0, rsh, h) for j in range(0, rsw, w)]
    dev = random.randint(1, max(1, len(dotcands) // 2))
    ndots = random.choice([dev, len(dotcands) - dev])
    ndots = min(max(1, ndots), len(dotcands))
    dots = random.sample(dotcands, ndots)

    osi, osj = h // 2, w // 2
    rs_start = w + 1

    for (di, dj) in dots:
        gi[osi + di, rs_start + osj + dj] = dotcol
        go[di:di + h, rs_start + dj:rs_start + dj + w] = ulc

    mfs_options = ['identity', 'vmirror', 'hmirror', 'rot90', 'rot180', 'rot270', 'dmirror', 'cmirror']
    nmfs = random.choice([1, 2])
    chosen = random.sample(mfs_options, nmfs)
    for name in chosen:
        gi = _apply_mf(gi, name)
        go = _apply_mf(go, name)

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    frontier_mask = np.zeros_like(I, dtype=bool)
    for r in range(hi):
        row = I[r]
        if int(row[0]) != bgc and bool(np.all(row == row[0])):
            frontier_mask[r, :] = True
    for c in range(wi):
        col = I[:, c]
        if int(col[0]) != bgc and bool(np.all(col == col[0])):
            frontier_mask[:, c] = True

    active = (I != bgc) & (~frontier_mask)

    labeled = np.zeros_like(I, dtype=int)
    cur = 0
    for r in range(hi):
        for c in range(wi):
            if active[r, c] and labeled[r, c] == 0:
                cur += 1
                stack = [(r, c)]
                while stack:
                    x, y = stack.pop()
                    if 0 <= x < hi and 0 <= y < wi and active[x, y] and labeled[x, y] == 0:
                        labeled[x, y] = cur
                        stack.append((x + 1, y))
                        stack.append((x - 1, y))
                        stack.append((x, y + 1))
                        stack.append((x, y - 1))

    ops, sels = [], []

    if cur == 0:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    sizes = [(l, int(np.sum(labeled == l))) for l in range(1, cur + 1)]
    stamp_label = max(sizes, key=lambda x: x[1])[0]

    stamp_rs, stamp_cs = np.where(labeled == stamp_label)
    sr0, sr1 = int(stamp_rs.min()), int(stamp_rs.max())
    sc0, sc1 = int(stamp_cs.min()), int(stamp_cs.max())
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1

    dots = []
    for l, sz in sizes:
        if l == stamp_label:
            continue
        rs, cs = np.where(labeled == l)
        dr = int(round(float(rs.mean())))
        dc = int(round(float(cs.mean())))
        dots.append((dr, dc))

    ops.append(28)
    sels.append([sr0, sc0, sh - 1, sw - 1])

    for (dr, dc) in dots:
        tr = dr - sh // 2
        tc = dc - sw // 2
        if 0 <= tr and tr + sh <= hi and 0 <= tc and tc + sw <= wi:
            ops.append(30)
            sels.append([tr, tc, 0, 0])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
            pr_in:  List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in:  List[NDArray] = []
            ex_out: List[NDArray] = []
            ops:  List[int]       = []
            sels: List[List[int]] = []

            # sample color roles once per episode → consistent across all instances
            colors = sample_colors()

            j = 0
            while j < num_examples + 1:
                ok = False
                for _ in range(10):
                    try:
                        r = generate(
                            random.uniform(0.2, 0.5),
                            random.uniform(0.5, 0.8),
                            max_h, max_w,
                            **colors,
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
                    j += 1
                    continue
                if j == num_examples:
                    pr_in.append(I)
                    pr_out.append(O)
                    ops, sels = derive_operations(I, O)
                else:
                    ex_in.append(I)
                    ex_out.append(O)
                j += 1

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"363442ee-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
