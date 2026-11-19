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
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    nh = unifint(diff_lb, diff_ub, (2, 6))
    nw = unifint(diff_lb, diff_ub, (2, 6))
    nh = min(nh, (max_h + 1) // 3)
    nw = min(nw, (max_w + 1) // 3)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (3, 9))
    ccols = sample(remcols, ncols)
    go = canvas(-1, (nh, nw))
    obj = {(choice(ccols), ij) for ij in asindices(go)}
    go = paint(go, obj)
    while len(dedupe(go)) < nh or len(dedupe(dmirror(go))) < nw:
        obj = {(choice(ccols), ij) for ij in asindices(go)}
        go = paint(go, obj)
    h = unifint(diff_lb, diff_ub, (2 * nh + nh - 1, max(2 * nh + nh - 1, max_h)))
    w = unifint(diff_lb, diff_ub, (2 * nw + nw - 1, max(2 * nw + nw - 1, max_w)))
    hdist = [2 for k in range(nh)]
    for k in range(h - 2 * nh - nh + 1):
        idx = randint(0, nh - 1)
        hdist[idx] += 1
    wdist = [2 for k in range(nw)]
    for k in range(w - 2 * nw - nw + 1):
        idx = randint(0, nw - 1)
        wdist[idx] += 1
    gi = merge(tuple(repeat(r, c) + (repeat(bgc, nw),) for r, c in zip(go, hdist)))[:-1]
    gi = dmirror(merge(tuple(repeat(r, c) + (repeat(bgc, h),) for r, c in zip(dmirror(gi), wdist)))[:-1])
    objs = objects(gi, T, F, F)
    bgobjs = colorfilter(objs, bgc)
    objs = objs - bgobjs
    for obj in objs:
        gi = fill(gi, bgc, sample(totuple(toindices(obj)), unifint(diff_lb, diff_ub, (1, len(obj) // 2))))
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    # Reading (two-pass collapse): first every row-band (rows between the
    # uniform bgc divider rows) collapses into ONE row: canvas row i gets,
    # segment by segment, the colour each block of band i holds. Then every
    # column-band collapses into ONE column: the first column of band j (which
    # now carries band j's colours in rows 0..nh-1) is copied into column j.
    # Finally the canvas is cut down to the nh x nw corner.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    frow = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    fcol = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    if frow:
        bgc = int(I[frow[0], 0])
    else:
        bgc = int(I[0, fcol[0]])
    frow = [r for r in frow if I[r, 0] == bgc]
    fcol = [c for c in fcol if I[0, c] == bgc]

    def bands(n, fl):
        fl = set(fl)
        res, start = [], None
        for i in range(n):
            if i in fl:
                if start is not None:
                    res.append((start, i - 1))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            res.append((start, n - 1))
        return res

    rbands = bands(hi, frow)
    cbands = bands(wi, fcol)

    def bcol(rb, cb):
        s = set(I[rb[0]:rb[1] + 1, cb[0]:cb[1] + 1].flatten().tolist()) - {bgc}
        return int(next(iter(s))) if s else None

    # a line of removed pixels can look like a divider; such a false divider
    # splits one band into two with identical colour profiles (the generator
    # guarantees neighbouring bands differ), so glue those back together.
    def glue(bl, other, rows):
        changed = True
        while changed:
            changed = False
            for k in range(len(bl) - 1):
                a = [bcol(bl[k], o) if rows else bcol(o, bl[k]) for o in other]
                b = [bcol(bl[k + 1], o) if rows else bcol(o, bl[k + 1]) for o in other]
                if all(x is None or y is None or x == y for x, y in zip(a, b)):
                    bl = bl[:k] + [(bl[k][0], bl[k + 1][1])] + bl[k + 2:]
                    changed = True
                    break
        return bl

    for _ in range(3):
        rbands = glue(rbands, cbands, True)
        cbands = glue(cbands, rbands, False)
    nh, nw = len(rbands), len(cbands)

    def block_color(i, j):
        return bcol(rbands[i], cbands[j])

    cand = []
    # pass 1: collapse each row band into canvas row i (one segment per column band)
    for i in range(nh):
        for j in range(nw):
            c0, c1 = cbands[j]
            cand.append((block_color(i, j), [(i, c) for c in range(c0, c1 + 1)]))

    def simulate(row_ops):
        g = I.copy()
        out = []
        for op, cells in row_ops:
            new = g.copy()
            for r, c in cells:
                new[r, c] = op
            if not np.array_equal(new, g):
                out.append((op, sel_of(cells)))
                g = new
        # pass 2: column band j collapses into column j (copy its first column)
        for j in range(1, nw):
            src = cbands[j][0]
            col_src = g[0:nh, src].copy()
            nz_mis = [r for r in range(nh) if col_src[r] != 0 and g[r, j] != col_src[r]]
            if nz_mis:
                out.append((29, sel_of([(r, src) for r in range(nh)])))   # CopyO band j's collapsed column
                out.append((30, sel_of([(0, j)])))                        # Paste into column j
                for r in range(nh):
                    if col_src[r] != 0:
                        g[r, j] = col_src[r]
            zfix = [(r, j) for r in range(nh) if col_src[r] == 0 and g[r, j] != 0]
            if zfix:
                out.append((0, sel_of(zfix)))   # colour 0 is invisible to Paste: write it explicitly
                for r, c in zfix:
                    g[r, c] = 0
        return out

    # Build the full trajectory, then drop any op whose removal leaves the
    # collapsed result (computed from I's blocks) unchanged, by replaying on I.
    target = np.array([[block_color(i, j) for j in range(nw)] for i in range(nh)])
    seq = simulate(cand)
    seq = seq + [(33, [0, 0, nh - 1, nw - 1])]   # crop to the collapsed nh x nw corner (full rectangle)

    def replay(seq):
        g = I.copy()
        clip = None
        for op, s in seq:
            if isinstance(s, dict):
                cells = [tuple(x) for x in s["cells"]]
            else:
                r, c, h, w = s
                cells = [(rr, cc) for rr in range(r, r + h + 1) for cc in range(c, c + w + 1)]
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            if op < 10:
                for r, c in cells:
                    g[r, c] = op
            elif op == 29:
                clip = g[min(rs):max(rs) + 1, min(cs):max(cs) + 1].copy()
            elif op == 30 and clip is not None:
                r0, c0 = min(rs), min(cs)
                for a in range(clip.shape[0]):
                    for b in range(clip.shape[1]):
                        if clip[a, b] != 0 and r0 + a < g.shape[0] and c0 + b < g.shape[1]:
                            g[r0 + a, c0 + b] = clip[a, b]
            elif op == 33:
                ng = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=int)
                sub = g[min(rs):max(rs) + 1, min(cs):max(cs) + 1]
                ng[:sub.shape[0], :sub.shape[1]] = sub
                g = ng
        return g

    changed = True
    while changed:
        changed = False
        for k in range(len(seq) - 1):
            trial = seq[:k] + seq[k + 1:]
            res = replay(trial)
            if res.shape == target.shape and np.array_equal(res, target):
                seq = trial
                changed = True
                break
    ops = [o for o, _ in seq]
    sels = [s for _, s in seq]
    ops.append(34); sels.append([0, 0, nh - 1, nw - 1])
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
