"""
ARC Task: 3345333e (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, objc, occcol = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "occcol": occcol}


def _unifint(lb, ub, bounds):
    a, b = bounds
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, occcol=None) -> dict:
    if bgc is None or objc is None or occcol is None:
        bgc, objc, occcol = random.sample(range(10), 3)
    T = {
        'identity': lambda g: [row[:] for row in g],
        'dmirror': lambda g: [list(r) for r in zip(*g)],
        'cmirror': lambda g: [list(r) for r in zip(*[row[::-1] for row in g[::-1]])],
        'vmirror': lambda g: [row[::-1] for row in g],
        'hmirror': lambda g: [row[:] for row in g[::-1]],
        'rot90': lambda g: [list(r) for r in zip(*g[::-1])],
        'rot180': lambda g: [row[::-1] for row in g[::-1]],
        'rot270': lambda g: [list(r) for r in zip(*g)][::-1],
    }
    fns = random.sample(list(T), random.choice((1, 2)))
    swaps = sum(f in ('dmirror', 'cmirror', 'rot90', 'rot270') for f in fns) % 2 == 1
    mh, mw = (max_w, max_h) if swaps else (max_h, max_w)
    mh = max(10, min(30, mh)); mw = max(10, min(30, mw))
    h = _unifint(diff_lb, diff_ub, (10, mh))
    w = _unifint(diff_lb, diff_ub, (10, mw))
    oh = _unifint(diff_lb, diff_ub, (4, h - 2))
    ow = _unifint(diff_lb, diff_ub, (4, (w - 2) // 2))
    nc = _unifint(diff_lb, diff_ub, (min(oh, ow), (oh * ow) // 3 * 2))
    shp = {(0, 0)}
    bounds = {(i, j) for i in range(oh) for j in range(ow)}

    def grow():
        cand = set()
        for (i, j) in shp:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        cand.add((i + di, j + dj))
        cand = sorted((cand & bounds) - shp)
        shp.add(random.choice(cand))

    for _ in range(nc):
        grow()
    while (max(i for i, _ in shp) - min(i for i, _ in shp) + 1 < 3
           or max(j for _, j in shp) - min(j for _, j in shp) + 1 < 3):
        grow()
    W0 = max(j for _, j in shp) + 1
    vm = {(i, W0 - 1 - j) for i, j in shp}
    if random.choice((True, False)):
        vm = {(i, j) for i, j in vm if j != W0 - 1}
    vw = max(j for _, j in vm) - min(j for _, j in vm) + 1
    comb = shp | {(i, j - vw) for i, j in vm}
    mi = min(i for i, _ in comb); mj = min(j for _, j in comb)
    shp2 = {(i - mi, j - mj) for i, j in comb}
    oh = max(i for i, _ in shp2) + 1; ow = max(j for _, j in shp2) + 1
    loci = random.randint(1, h - oh - 1)
    locj = random.randint(1, w - ow - 1)
    go = [[bgc] * w for _ in range(h)]
    for i, j in shp2:
        go[i + loci][j + locj] = objc
    boxh = _unifint(diff_lb, diff_ub, (2, oh - 1))
    boxw = _unifint(diff_lb, diff_ub, (2, ow // 2))
    ulci = random.randint(loci - 1, loci + oh - boxh + 1)
    ulcj = random.randint(locj + ow // 2 + 1, locj + ow - boxw + 1)
    gi = [row[:] for row in go]
    for i in range(ulci, ulci + boxh):
        for j in range(ulcj, ulcj + boxw):
            gi[i][j] = occcol
    for f in fns:
        gi = T[f](gi); go = T[f](go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """A solid rectangle hides part of a mirror-symmetric blob; restore the hidden piece.

    The code-golf reading agrees with this one: each occluder cell takes the value of its
    mirror cell across the blob's symmetry axis. The unit that changes is the occluding
    rectangle, and it becomes the mirrored copy of its intact counterpart. So the route is
    the same: CopyI the intact mirror region, Paste it over the rectangle, Flip it in place,
    then use one Color0 for the colour-0 cells that Copy/Paste cannot carry.
    Everything is measured from I alone.
    """
    import numpy as np
    from maker.sel_helpers import sel_of

    def _analyze(g):
        H = len(g); W = len(g[0])
        border = ([g[0][j] for j in range(W)] + [g[H - 1][j] for j in range(W)]
                  + [g[i][0] for i in range(H)] + [g[i][W - 1] for i in range(H)])
        bg = max(sorted(set(border)), key=border.count)
        cellsof = {}
        for i in range(H):
            for j in range(W):
                cellsof.setdefault(g[i][j], []).append((i, j))
        occc = None; occ_area = None
        for col in sorted(cellsof):
            if col == bg:
                continue
            st = cellsof[col]
            rs = [p[0] for p in st]; cs = [p[1] for p in st]
            area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if len(st) == area and (occ_area is None or area < occ_area):
                occc = col; occ_area = area
        if occc is None:
            return None
        others = [col for col in sorted(cellsof) if col != bg and col != occc]
        if len(others) != 1:
            return None
        oc = others[0]
        V = cellsof[oc]; B = cellsof[occc]
        r0 = min(p[0] for p in B); r1 = max(p[0] for p in B)
        c0 = min(p[1] for p in B); c1 = max(p[1] for p in B)
        comb = V + B
        hh = max(p[0] for p in comb) - min(p[0] for p in comb) + 1
        ww = max(p[1] for p in comb) - min(p[1] for p in comb) + 1
        k = max(hh // 2 + 1, ww // 2 + 1)
        stride = W + 2 * k + 2

        def bits(cells):
            v = 0
            for (r, c) in cells:
                v |= 1 << ((r + k) * stride + (c + k))
            return v

        Vb = bits(V); BGb = bits(cellsof[bg])
        vD = min(p[1] for p in V) + max(p[1] for p in V)
        hD = min(p[0] for p in V) + max(p[0] for p in V)
        cands = (('v', bits([(r, vD - c) for r, c in V])),
                 ('h', bits([(hD - r, c) for r, c in V])))
        best = None; bestsc = -1
        for mt, Mb in cands:
            for di in range(-k, k + 1):
                for dj in range(-k, k + 1):
                    sh = di * stride + dj
                    Sb = (Mb << sh) if sh >= 0 else (Mb >> (-sh))
                    if Sb & BGb:
                        continue
                    sc = bin(Sb & Vb).count('1')
                    if sc > bestsc:
                        bestsc = sc; best = (mt, di, dj)
        if best is None:
            return None
        mt, di, dj = best
        if mt == 'v':
            sr0, sr1 = r0 - di, r1 - di
            sc0, sc1 = vD + dj - c1, vD + dj - c0
        else:
            sr0, sr1 = hD + di - r1, hD + di - r0
            sc0, sc1 = c0 - dj, c1 - dj
        if sr0 < 0 or sc0 < 0 or sr1 >= H or sc1 >= W:
            return None
        return {'bgc': bg, 'objc': oc, 'occc': occc, 'mt': mt,
                'occ': (r0, r1, c0, c1), 'src': (sr0, sr1, sc0, sc1)}

    G = np.asarray(I, dtype=int)
    H, W = G.shape
    g = [[int(x) for x in row] for row in G]
    info = _analyze(g)
    r0, r1, c0, c1 = info['occ']
    sr0, sr1, sc0, sc1 = info['src']
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    flip_op = 26 if info['mt'] == 'v' else 27   # 26 = left<->right, 27 = up<->down

    ops = []
    sels = []
    # 1. copy the intact mirror-image region out of the input (a full rectangle)
    src_cells = [(r, c) for r in range(sr0, sr1 + 1) for c in range(sc0, sc1 + 1)]
    ops.append(28); sels.append(sel_of(src_cells))
    # 2. lay it over the occluding rectangle, anchored at its top-left corner
    ops.append(30); sels.append(sel_of([(r0, c0)]))
    work = [row[:] for row in g]
    for a in range(bh):
        for b in range(bw):
            v = g[sr0 + a][sc0 + b]
            if v != 0:
                work[r0 + a][c0 + b] = v
    # 3. mirror the freshly laid block in place (full rectangle = the occluder's footprint)
    dst_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    blk = [[work[r0 + a][c0 + b] for b in range(bw)] for a in range(bh)]
    if flip_op == 26:
        fl = [list(reversed(row)) for row in blk]
    else:
        fl = [row[:] for row in reversed(blk)]
    if fl != blk:
        ops.append(flip_op); sels.append(sel_of(dst_cells))
        for a in range(bh):
            for b in range(bw):
                work[r0 + a][c0 + b] = fl[a][b]
    # 4. colour-0 source cells never travel with Copy/Paste; lay them at their mirrored spots
    zeros = []
    for a in range(bh):
        for b in range(bw):
            if g[sr0 + a][sc0 + b] == 0:
                d = (r0 + a, c0 + bw - 1 - b) if flip_op == 26 else (r0 + bh - 1 - a, c0 + b)
                if work[d[0]][d[1]] != 0:
                    zeros.append(d)
    if zeros:
        ops.append(0); sels.append(sel_of(sorted(set(zeros))))
    ops.append(34); sels.append(sel_of([(r, c) for r in range(H) for c in range(W)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 3345333e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3345333e"
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
                                f"for task 3345333e"
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
                    f"Failed to build a complete episode for task 3345333e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3345333e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
