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
import numpy as np


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo = max(a, min(lo, b)); hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    bgc, objc, occcol = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "occcol": occcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, occcol=None) -> dict:
    if bgc is None or objc is None or occcol is None:
        bgc, objc, occcol = random.sample(range(10), 3)
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (10, max_h))
    w = _unifint(diff_lb, diff_ub, (10, max_w))
    oh = _unifint(diff_lb, diff_ub, (4, h - 2))
    ow = _unifint(diff_lb, diff_ub, (4, (w - 2) // 2))
    nc = _unifint(diff_lb, diff_ub, (min(oh, ow), (oh * ow) // 3 * 2))

    def nbrs(p):
        i, j = p
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    shp = {(0, 0)}

    def grow():
        cand = set()
        for p in shp:
            cand |= nbrs(p)
        cand = sorted((cand & bounds) - shp)
        if cand:
            shp.add(random.choice(cand))
            return True
        return False

    for _ in range(nc):
        if not grow():
            break

    def hgt(s):
        return max(p[0] for p in s) - min(p[0] for p in s) + 1

    def wid(s):
        return max(p[1] for p in s) - min(p[1] for p in s) + 1

    while hgt(shp) < 3 or wid(shp) < 3:
        if not grow():
            break
    ws = wid(shp)
    dsum = min(p[1] for p in shp) + max(p[1] for p in shp)
    vm = {(i, dsum - j) for i, j in shp}
    if random.choice((True, False)):
        vm = {p for p in vm if p[1] != ws - 1}
    wv = wid(vm)
    comb = shp | {(i, j - wv) for i, j in vm}
    mi = min(p[0] for p in comb); mj = min(p[1] for p in comb)
    shp = {(i - mi, j - mj) for i, j in comb}
    oh, ow = hgt(shp), wid(shp)
    loci = random.randint(1, h - oh - 1)
    locj = random.randint(1, w - ow - 1)
    shp = {(i + loci, j + locj) for i, j in shp}
    go = np.full((h, w), bgc, dtype=int)
    for i, j in shp:
        go[i, j] = objc
    boxh = _unifint(diff_lb, diff_ub, (2, oh - 1))
    boxw = _unifint(diff_lb, diff_ub, (2, ow // 2))
    ulci = random.randint(loci - 1, loci + oh - boxh + 1)
    ulcj = random.randint(locj + ow // 2 + 1, locj + ow - boxw + 1)
    gi = go.copy()
    for i in range(ulci, ulci + boxh):
        for j in range(ulcj, ulcj + boxw):
            if 0 <= i < h and 0 <= j < w:
                gi[i, j] = occcol

    mfs = {
        "id": lambda g: g,
        "dm": lambda g: g.T,
        "cm": lambda g: np.rot90(g, 2).T,
        "vm": lambda g: np.fliplr(g),
        "hm": lambda g: np.flipud(g),
        "r90": lambda g: np.rot90(g, -1),
        "r180": lambda g: np.rot90(g, 2),
        "r270": lambda g: np.rot90(g, 1),
    }
    transposing = {"dm", "cm", "r90", "r270"}
    names = list(mfs)
    if h > max_w or w > max_h:
        names = [n for n in names if n not in transposing]
    for n in random.sample(names, random.choice((1, 2))):
        gi = mfs[n](gi); go = mfs[n](go)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
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
    bgc, objc, occc = info['bgc'], info['objc'], info['occc']
    r0, r1, c0, c1 = info['occ']
    sr0, sr1, sc0, sc1 = info['src']
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    vert = info['mt'] == 'v'          # mirror axis is vertical -> left<->right
    flip_op = 26 if vert else 27

    def partner(a, b):
        """Input cell that rectangle cell (r0+a, c0+b) mirrors onto."""
        return (sr0 + a, sc0 + bw - 1 - b) if vert else (sr0 + bh - 1 - a, sc0 + b)

    ops, sels = [], []
    rect = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    work = [row[:] for row in g]

    # 1. The rectangle is treated as "possibly object": it joins the blob.
    #    One flood fill of the single-coloured block turns it into object colour.
    ops.append(10 + objc); sels.append(sel_of([(r0, c0)]))
    for r, c in rect:
        work[r][c] = objc

    # 2. A cell stays object only if its mirror partner is non-background too.
    #    Carve back the rectangle cells whose partner (in the intact half) is background.
    src_bg = [(r, c) for r in range(sr0, sr1 + 1) for c in range(sc0, sc1 + 1)
              if g[r][c] == bgc]
    if src_bg and bgc != 0:
        # copy the partner region's background cells, lay them on the block, mirror the block
        ops.append(28); sels.append(sel_of(src_bg))
        # the clipboard spans only the copied cells' bounding box: anchor it accordingly
        mr = min(r for r, _ in src_bg); mc = min(c for _, c in src_bg)
        ops.append(30); sels.append(sel_of([(r0 + mr - sr0, c0 + mc - sc0)]))
        for (r, c) in src_bg:
            work[r0 + r - sr0][c0 + c - sc0] = bgc
        blk = [[work[r0 + a][c0 + b] for b in range(bw)] for a in range(bh)]
        fl = [list(reversed(row)) for row in blk] if vert else [row[:] for row in reversed(blk)]
        if fl != blk:
            ops.append(flip_op); sels.append(sel_of(rect))
            for a in range(bh):
                for b in range(bw):
                    work[r0 + a][c0 + b] = fl[a][b]
    elif src_bg:
        # background is colour 0, which Copy/Paste cannot carry: paint the carved cells
        carve = [(r0 + a, c0 + b) for a in range(bh) for b in range(bw)
                 if g[partner(a, b)[0]][partner(a, b)[1]] == bgc]
        ops.append(bgc); sels.append(sel_of(carve))

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
