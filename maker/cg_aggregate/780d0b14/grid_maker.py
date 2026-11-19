"""
ARC Task: 780d0b14 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule only reads "one non-background colour per block"; block colours are free.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unifint(lo, hi):
        a, b = lo, hi
        if b < a:
            b = a
        d = random.uniform(diff_lb, diff_ub)
        return int(round(a + (b - a) * d))
    if bgc is None:
        bgc = random.randint(0, 9)
    nh = unifint(2, 6)
    nw = unifint(2, 6)
    nh = min(nh, max(2, (max_h + 1) // 3))
    nw = min(nw, max(2, (max_w + 1) // 3))
    remcols = [c for c in range(10) if c != bgc]
    ncols = unifint(3, 9)
    ccols = random.sample(remcols, ncols)

    def rnd():
        return [[random.choice(ccols) for _ in range(nw)] for _ in range(nh)]

    def ok(g):
        rows = [tuple(r) for r in g]
        cols = [tuple(g[i][j] for i in range(nh)) for j in range(nw)]
        dr = sum(1 for k in range(nh) if k == 0 or rows[k] != rows[k - 1])
        dc = sum(1 for k in range(nw) if k == 0 or cols[k] != cols[k - 1])
        return dr >= nh and dc >= nw
    go = rnd()
    while not ok(go):
        go = rnd()
    h = unifint(3 * nh - 1, max_h)
    w = unifint(3 * nw - 1, max_w)
    hdist = [2] * nh
    for _ in range(h - 3 * nh + 1):
        hdist[random.randint(0, nh - 1)] += 1
    wdist = [2] * nw
    for _ in range(w - 3 * nw + 1):
        wdist[random.randint(0, nw - 1)] += 1
    gi = np.full((h, w), bgc, dtype=int)
    blocks = []
    r = 0
    for i in range(nh):
        c = 0
        for j in range(nw):
            gi[r:r + hdist[i], c:c + wdist[j]] = go[i][j]
            blocks.append((r, c, hdist[i], wdist[j]))
            c += wdist[j] + 1
        r += hdist[i] + 1
    for (r, c, bh, bw) in blocks:
        cells = [(r + a, c + b) for a in range(bh) for b in range(bw)]
        k = unifint(1, len(cells) // 2)
        for (a, b) in random.sample(cells, k):
            gi[a, b] = bgc
    return {"input": gi.tolist(), "output": [list(x) for x in go]}


def _simulate(I, ops, sels):
    g = I.copy()
    clip = None
    for op, s in zip(ops, sels):
        cells = s["cells"]
        if op <= 9:
            for r, c in cells:
                g[r, c] = op
        elif op == 29:
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            r0, c0 = min(rs), min(cs)
            clip = g[r0:max(rs) + 1, c0:max(cs) + 1].copy()
        elif op == 30:
            if clip is None:
                return None
            r0, c0 = cells[0]
            ch, cw = clip.shape
            for a in range(ch):
                for b in range(cw):
                    if clip[a, b] != 0 and r0 + a < g.shape[0] and c0 + b < g.shape[1]:
                        g[r0 + a, c0 + b] = clip[a, b]
        elif op == 33:
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            g = g[min(rs):max(rs) + 1, min(cs):max(cs) + 1].copy()
    return g


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The grid is cut by uniform lines of the background colour into a mosaic of
    # blocks; each block is one colour with some background holes.
    frow = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    fcol = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    bgc = int(I[frow[0], 0]) if frow else int(I[0, fcol[0]])

    def bands(n, fl):
        res, start = [], None
        for i in range(n):
            if i in fl:
                if start is not None:
                    res.append((start, i - 1)); start = None
            elif start is None:
                start = i
        if start is not None:
            res.append((start, n - 1))
        return res

    rb = bands(hi, set(frow))
    cb = bands(wi, set(fcol))
    nh, nw = len(rb), len(cb)
    fg = [[int(next(iter(set(I[r0:r1 + 1, c0:c1 + 1].flatten().tolist()) - {bgc})))
           for (c0, c1) in cb] for (r0, r1) in rb]

    plan = []
    # A. merge each row band into its first row: repair the holes of that row,
    #    block segment by block segment (0 cannot travel through Copy/Paste, skip it).
    for i, (r0, r1) in enumerate(rb):
        for j, (c0, c1) in enumerate(cb):
            if fg[i][j] == 0:
                continue
            holes = [(r0, c) for c in range(c0, c1 + 1) if I[r0, c] != fg[i][j]]
            if holes:
                plan.append((fg[i][j], sel_of(holes)))
    # B. collapse rows: the merged row of band i is copied up to row i.
    for i, (r0, r1) in enumerate(rb):
        if r0 != i:
            plan.append((29, sel_of([(r0, c) for c in range(wi)])))   # whole row
            plan.append((30, sel_of([(i, 0)])))
    # C. collapse columns: the first column of band j is copied left to column j.
    for j, (c0, c1) in enumerate(cb):
        if c0 != j:
            plan.append((29, sel_of([(r, c0) for r in range(nh)])))   # column strip
            plan.append((30, sel_of([(0, j)])))
    # D. blocks whose colour is 0 (Paste cannot write 0) are painted directly.
    cur = _simulate(I, [p[0] for p in plan], [p[1] for p in plan])
    zc = [(i, j) for i in range(nh) for j in range(nw) if fg[i][j] == 0 and cur[i, j] != 0]
    if zc:
        plan.append((0, sel_of(zc)))
    # E. keep only the nh x nw block grid.
    plan.append((33, sel_of([(r, c) for r in range(nh) for c in range(nw)])))

    # drop any op of the plan whose removal leaves the plan's own result unchanged
    def res(p):
        return _simulate(I, [x[0] for x in p], [x[1] for x in p])
    target = res(plan)
    changed = True
    while changed:
        changed = False
        for k in range(len(plan) - 1):
            cands = [plan[:k] + plan[k + 1:]]
            if plan[k][0] == 30 and k > 0 and plan[k - 1][0] == 29:
                cands.append(plan[:k - 1] + plan[k + 1:])
            for q in cands:
                r = res(q)
                if r is not None and r.shape == target.shape and (r == target).all():
                    plan = q; changed = True
                    break
            if changed:
                break
    ops = [p[0] for p in plan] + [34]
    sels = [p[1] for p in plan] + [sel_of([(r, c) for r in range(nh) for c in range(nw)])]
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
                        f"num_examples+1 ({num_examples + 1}) for task 780d0b14"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 780d0b14"
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
                                f"for task 780d0b14"
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
                    f"Failed to build a complete episode for task 780d0b14 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"780d0b14-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
