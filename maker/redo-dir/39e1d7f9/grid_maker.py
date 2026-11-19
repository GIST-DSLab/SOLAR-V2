"""
ARC Task: 39e1d7f9 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc, dotc = random.sample(cols, 3)
    remcols = [c for c in cols if c not in (bgc, linc, dotc)]
    ncols = random.randint(1, 7)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "linc": linc, "dotc": dotc, "ccols": ccols}


# ---------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc, ccols) -> dict:

    def attempt():
        h_ub = max(5, min(10, (max_h + 1) // 2))
        w_ub = max(5, min(10, (max_w + 1) // 2))
        h = unifint(diff_lb, diff_ub, (5, h_ub))
        w = unifint(diff_lb, diff_ub, (5, w_ub))
        gi = [[bgc] * w for _ in range(h)]
        loci = random.randint(1, h - 2)
        locj = random.randint(1, w - 2)
        if h == 5:
            loci = random.choice((1, h - 2))
        if w == 5:
            locj = random.choice((1, w - 2))
        npix = unifint(diff_lb, diff_ub, (1, 8))
        ring = {(loci + dr, locj + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        ring.discard((loci, locj))
        pixs = {(loci, locj)}
        for _ in range(npix):
            cands = set()
            for (r, c) in pixs:
                for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (r + dr, c + dc)
                    if q in ring and q not in pixs:
                        cands.add(q)
            if not cands:
                break
            pixs.add(random.choice(sorted(cands)))
        pixs.discard((loci, locj))
        tmpl = {(r - loci, c - locj): random.choice(ccols) for (r, c) in sorted(pixs)}
        gi[loci][locj] = dotc
        for (dr, dc), col in tmpl.items():
            gi[loci + dr][locj + dc] = col
        go = [row[:] for row in gi]

        offs = list(tmpl.keys()) + [(0, 0)]
        minr = min(o[0] for o in offs); maxr = max(o[0] for o in offs)
        minc = min(o[1] for o in offs); maxc = max(o[1] for o in offs)
        bbox = [(r, c) for r in range(minr, maxr + 1) for c in range(minc, maxc + 1)]

        # every stamp keeps a 1-cell clearance, so occurrences never collide
        inds = {(r, c) for r in range(h) for c in range(w) if gi[r][c] == bgc}
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(tmpl) + 1))))
        succ, tr, maxtr = 0, 0, 6 * noccs + 10
        tried = set()
        while (tr < maxtr and succ < noccs) or succ == 0:
            cands = sorted(inds - tried)
            if not cands:
                break
            tr += 1
            li, lj = random.choice(cands)
            bb = {(li + r, lj + c) for (r, c) in bbox}
            if any(not (0 <= r < h and 0 <= c < w) for (r, c) in bb):
                tried.add((li, lj))
                continue
            dil = {(r + dr, c + dc) for (r, c) in bb
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            dil = {(r, c) for (r, c) in dil if 0 <= r < h and 0 <= c < w}
            if dil <= inds:
                inds -= bb
                succ += 1
                gi[li][lj] = dotc
                go[li][lj] = dotc
                for (dr, dc), col in tmpl.items():
                    go[li + dr][lj + dc] = col
            else:
                tried.add((li, lj))
        if succ == 0:
            return None

        hfac = unifint(diff_lb, diff_ub, (1, max(1, (max_h - h + 1) // h)))
        wfac = unifint(diff_lb, diff_ub, (1, max(1, (max_w - w + 1) // w)))
        fullh = hfac * h + h - 1
        fullw = wfac * w + w - 1
        gi2 = [[linc] * fullw for _ in range(fullh)]
        go2 = [[linc] * fullw for _ in range(fullh)]
        for a in range(h):
            for b in range(w):
                for rr in range(a * (hfac + 1), a * (hfac + 1) + hfac):
                    for cc in range(b * (wfac + 1), b * (wfac + 1) + wfac):
                        gi2[rr][cc] = gi[a][b]
                        go2[rr][cc] = go[a][b]
        return {'input': tuple(tuple(r) for r in gi2),
                'output': tuple(tuple(r) for r in go2)}

    for _ in range(30):
        res = attempt()
        if res is not None:
            return res
    raise ValueError('39e1d7f9: no valid stamp placement found')


# ---------------------------------------------------------------- operations
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    submit_sel = [0, 0, H - 1, W - 1]

    # the picture is a cell grid blown up by hfac x wfac with 1-px separator lines
    const_rows = [r for r in range(H) if len(set(I[r].tolist())) == 1]
    const_cols = [c for c in range(W) if len(set(I[:, c].tolist())) == 1]
    hfac = const_rows[0] if const_rows else H
    wfac = const_cols[0] if const_cols else W
    ph, pw = hfac + 1, wfac + 1
    h = (H + 1) // ph
    w = (W + 1) // pw

    C = [[int(I[a * ph, b * pw]) for b in range(w)] for a in range(h)]
    CO = [[int(O[a * ph, b * pw]) for b in range(w)] for a in range(h)]
    bgc = Counter(x for row in C for x in row).most_common(1)[0][0]

    def blk(a, b):
        return [(r, c) for r in range(a * ph, a * ph + hfac)
                for c in range(b * pw, b * pw + wfac)]

    def inb(a, b):
        return 0 <= a < h and 0 <= b < w

    # the one complete stamp in I: a dot with a coloured pattern around it.
    # every other cell of that dot colour is a bare dot that receives the stamp.
    found = None
    for p0 in [(a, b) for a in range(h) for b in range(w) if C[a][b] != bgc]:
        dc0 = C[p0[0]][p0[1]]
        T = {}
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) == (0, 0):
                    continue
                a, b = p0[0] + dr, p0[1] + dc
                if inb(a, b) and C[a][b] != bgc:
                    T[(dr, dc)] = C[a][b]
        if not T:
            continue
        S = [(a, b) for a in range(h) for b in range(w) if C[a][b] == dc0]
        ok = True
        sim = [row[:] for row in C]
        for p in S:
            for (dr, dc), col in T.items():
                a, b = p[0] + dr, p[1] + dc
                if not inb(a, b):
                    ok = False
                    break
                sim[a][b] = col
            if not ok:
                break
        if ok and sim == CO:
            found = (dc0, p0, T, S)
            break

    ops, sels = [], []
    G = I.copy()

    def stamp_by_hand(p, T):
        bycol = {}
        for (dr, dc), col in T.items():
            bycol.setdefault(col, []).extend(blk(p[0] + dr, p[1] + dc))
        for col in sorted(bycol):
            cells = bycol[col]
            if any(G[r, c] != col for (r, c) in cells):
                ops.append(int(col))
                sels.append(sel_of(cells))
                for (r, c) in cells:
                    G[r, c] = col

    if found is not None:
        dotc, src, T, S = found
        offs = list(T.keys()) + [(0, 0)]
        minr = min(o[0] for o in offs); maxr = max(o[0] for o in offs)
        minc = min(o[1] for o in offs); maxc = max(o[1] for o in offs)
        rect = [(dr, dc) for dr in range(minr, maxr + 1)
                for dc in range(minc, maxc + 1)]
        srcval = {o: T.get(o, dotc if o == (0, 0) else bgc) for o in rect}

        dests = [p for p in S
                 if any(C[p[0] + dr][p[1] + dc] != col for (dr, dc), col in T.items())]
        owner = {}
        for p in dests:
            for o in list(T.keys()) + [(0, 0)]:
                owner[(p[0] + o[0], p[1] + o[1])] = p

        # a Paste drops the whole stamp rectangle: note which foreign cells it buries
        damage = {}
        for p in dests:
            bad = []
            for o in rect:
                v = srcval[o]
                if v == 0:                      # Paste never writes colour-0 cells
                    continue
                a, b = p[0] + o[0], p[1] + o[1]
                if inb(a, b) and CO[a][b] != v:
                    bad.append((a, b))
            damage[p] = bad

        unsafe = {p for p in dests if any(owner.get(q) is None for q in damage[p])}
        pasted = [p for p in dests if p not in unsafe]
        pset = set(pasted)
        succ = {p: set() for p in pasted}
        indeg = {p: 0 for p in pasted}
        for p in pasted:                        # stamp buried by another -> paste it later
            for q in damage[p]:
                o = owner.get(q)
                if o is not None and o in pset and o != p and o not in succ[p]:
                    succ[p].add(o)
                    indeg[o] += 1
        order, ready = [], [p for p in pasted if indeg[p] == 0]
        while ready:
            p = ready.pop(0)
            order.append(p)
            for q in sorted(succ[p]):
                indeg[q] -= 1
                if indeg[q] == 0:
                    ready.append(q)
        if len(order) != len(pasted):
            unsafe |= {p for p in pasted if p not in set(order)}
        by_hand = [p for p in dests if p in unsafe]

        sh = (maxr - minr + 1) * ph - 1
        sw = (maxc - minc + 1) * pw - 1
        sr = (src[0] + minr) * ph
        sc = (src[1] + minc) * pw
        patch = I[sr:sr + sh, sc:sc + sw]
        zero_offs = sorted([o for o, col in T.items() if col == 0])
        has_nz = any(col != 0 for col in T.values())

        if order and has_nz:
            # bbox format on purpose: the copied region IS exactly this full rectangle
            # - the stamp together with the background/separator cells inside its bbox
            ops.append(28); sels.append([sr, sc, sh - 1, sw - 1])
            for p in order:
                dr0 = (p[0] + minr) * ph
                dc0 = (p[1] + minc) * pw
                ops.append(30); sels.append([dr0, dc0, 0, 0])
                tgt = G[dr0:dr0 + sh, dc0:dc0 + sw]
                G[dr0:dr0 + sh, dc0:dc0 + sw] = np.where(patch != 0, patch, tgt)
                if zero_offs:                   # the stamp's colour-0 cells Paste skips
                    cells = []
                    for o in zero_offs:
                        cells.extend(blk(p[0] + o[0], p[1] + o[1]))
                    if any(G[r, c] != 0 for (r, c) in cells):
                        ops.append(0); sels.append(sel_of(cells))
                        for (r, c) in cells:
                            G[r, c] = 0
        else:
            by_hand = list(dests)
        for p in by_hand:
            stamp_by_hand(p, T)

    if not np.array_equal(G, O):
        ops, sels = [], []
        G = I.copy()
        rem = {(a, b) for a in range(h) for b in range(w) if C[a][b] != CO[a][b]}
        while rem:
            stack = [min(rem)]
            comp = set()
            while stack:
                q = stack.pop()
                if q in rem:
                    rem.discard(q)
                    comp.add(q)
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            n = (q[0] + dr, q[1] + dc)
                            if n in rem:
                                stack.append(n)
            bycol = {}
            for (a, b) in sorted(comp):
                bycol.setdefault(CO[a][b], []).extend(blk(a, b))
            for col in sorted(bycol):
                ops.append(int(col)); sels.append(sel_of(bycol[col]))
                for (r, c) in bycol[col]:
                    G[r, c] = col

    ops.append(34); sels.append(submit_sel)
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
                        f"num_examples+1 ({num_examples + 1}) for task 39e1d7f9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 39e1d7f9"
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
                                f"for task 39e1d7f9"
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
                    f"Failed to build a complete episode for task 39e1d7f9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"39e1d7f9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
