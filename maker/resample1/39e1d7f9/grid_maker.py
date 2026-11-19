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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    bgc, linc, dotc = random.sample(range(10), 3)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc) -> dict:
    cols = list(range(10))
    remcols = [c for c in cols if c not in (bgc, linc, dotc)]
    hmax = max(5, min(10, (max_h + 1) // 2))
    wmax = max(5, min(10, (max_w + 1) // 2))
    h = _unifint(diff_lb, diff_ub, (5, hmax))
    w = _unifint(diff_lb, diff_ub, (5, wmax))
    gi = [[bgc] * w for _ in range(h)]
    loci = random.randint(1, h - 2)
    locj = random.randint(1, w - 2)
    if h == 5:
        loci = random.choice((1, h - 2))
    if w == 5:
        locj = random.choice((1, w - 2))
    npix = _unifint(diff_lb, diff_ub, (1, 8))
    ncols = _unifint(diff_lb, diff_ub, (1, 7))
    ccols = random.sample(remcols, ncols)

    def dn(p):
        i, j = p
        return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

    def nb(p):
        i, j = p
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    cands = nb((loci, locj))
    pixs = {(loci, locj)}
    for _ in range(npix):
        adj = set()
        for p in pixs:
            adj |= dn(p)
        opts = sorted((adj & cands) - pixs)
        pixs.add(random.choice(opts))
    pixs = sorted(pixs - {(loci, locj)})
    obj = [(random.choice(ccols), ij) for ij in pixs]
    gi[loci][locj] = dotc
    for c, (i, j) in obj:
        gi[i][j] = c
    go = [row[:] for row in gi]
    noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(pixs) + 1))))
    succ = 0
    tr = 0
    maxtr = 6 * noccs
    excl = set()
    for p in nb((loci, locj)):
        excl |= dn(p)
    inds = {(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc} - excl
    objn = [(c, (i - loci, j - locj)) for c, (i, j) in obj]
    tf = set()
    while (tr < maxtr and succ < noccs) or succ == 0:
        lop = sorted(inds - tf)
        if not lop:
            break
        tr += 1
        li, lj = random.choice(lop)
        plcd = [(c, (i + li, j + lj)) for c, (i, j) in objn]
        plcdi = {ij for _, ij in plcd}
        if plcdi.issubset(inds):
            inds = inds - (plcdi | {(li, lj)})
            succ += 1
            gi[li][lj] = dotc
            go[li][lj] = dotc
            for c, (i, j) in plcd:
                go[i][j] = c
        else:
            tf.add((li, lj))
    hfac = _unifint(diff_lb, diff_ub, (1, max(1, (max_h - h + 1) // h)))
    wfac = _unifint(diff_lb, diff_ub, (1, max(1, (max_w - w + 1) // w)))
    fh = hfac * h + h - 1
    fw = wfac * w + w - 1

    def up(g):
        out = [[linc] * fw for _ in range(fh)]
        for a in range(h):
            for b in range(w):
                for x in range(hfac):
                    for y in range(wfac):
                        out[a * (hfac + 1) + x][b * (wfac + 1) + y] = g[a][b]
        return out

    return {"input": up(gi), "output": up(go)}


def _structure(I):
    H, W = I.shape
    for hf in range(1, H + 1):
        if (H + 1) % (hf + 1):
            continue
        hrows = list(range(hf, H, hf + 1))
        for wf in range(1, W + 1):
            if (W + 1) % (wf + 1):
                continue
            wcols = list(range(wf, W, wf + 1))
            if not hrows or not wcols:
                continue
            L = I[hrows[0], 0]
            if not all((I[r, :] == L).all() for r in hrows):
                continue
            if not all((I[:, c] == L).all() for c in wcols):
                continue
            h, w = len(hrows) + 1, len(wcols) + 1
            ok = True
            C = np.zeros((h, w), dtype=int)
            for a in range(h):
                for b in range(w):
                    blk = I[a * (hf + 1):a * (hf + 1) + hf, b * (wf + 1):b * (wf + 1) + wf]
                    if not (blk == blk[0, 0]).all():
                        ok = False
                        break
                    C[a, b] = blk[0, 0]
                if not ok:
                    break
            if ok:
                return hf, wf, C
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []
    st = _structure(I)
    if st is None:
        return [34], [[0, 0, H - 1, W - 1]]
    hf, wf, C = st
    h, w = C.shape
    bgc = Counter(C.flatten().tolist()).most_common(1)[0][0]
    # 4-connected multicolour components of non-background cells (compressed grid)
    seen = set()
    comps = []
    for i in range(h):
        for j in range(w):
            if C[i, j] == bgc or (i, j) in seen:
                continue
            stack = [(i, j)]
            seen.add((i, j))
            comp = []
            while stack:
                p = stack.pop()
                comp.append(p)
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (p[0] + di, p[1] + dj)
                    if 0 <= q[0] < h and 0 <= q[1] < w and q not in seen and C[q] != bgc:
                        seen.add(q)
                        stack.append(q)
            comps.append(comp)
    if not comps:
        return [34], [[0, 0, H - 1, W - 1]]
    tmpl = max(comps, key=lambda c: (len({C[p] for p in c}), len(c)))
    others = [p for c in comps if c is not tmpl for p in c]
    if not others:
        return [34], [[0, 0, H - 1, W - 1]]
    dotc = Counter(C[p] for p in others).most_common(1)[0][0]
    centers = [p for p in tmpl if C[p] == dotc]
    if len(centers) != 1:
        return [34], [[0, 0, H - 1, W - 1]]
    ci, cj = centers[0]
    offs = {}
    for p in tmpl:
        if p == (ci, cj):
            continue
        offs.setdefault(int(C[p]), []).append((p[0] - ci, p[1] - cj))
    dots = sorted(p for p in others if C[p] == dotc)

    def block(a, b):
        return [(a * (hf + 1) + x, b * (wf + 1) + y) for x in range(hf) for y in range(wf)]

    cur = I.copy()
    # stamp the template's neighbourhood around each lone dot, one dot at a time,
    # one Color op per template colour
    for (di, dj) in dots:
        for col in sorted(offs):
            cells = []
            for (oi, oj) in offs[col]:
                a, b = di + oi, dj + oj
                if 0 <= a < h and 0 <= b < w:
                    cells += [rc for rc in block(a, b) if cur[rc] != col]
            if cells:
                ops.append(col); sels.append(sel_of(cells))
                for rc in cells:
                    cur[rc] = col
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
