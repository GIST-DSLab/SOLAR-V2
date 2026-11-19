"""
ARC Task: 7df24a62 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, sqc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, sqc=None) -> dict:
    from dsl import (canvas, backdrop, inbox, asindices, totuple, shape, fill,
                     asobject, shift, paint, ofcolor, toindices, mapply, dneighbors,
                     identity, dmirror, cmirror, hmirror, vmirror, normalize,
                     sfilter, replace, lbind, occurrences, trim)
    from utils import unifint
    if bgc is None or noisec is None or sqc is None:
        bgc, noisec, sqc = random.sample(range(10), 3)
    # the grid is trimmed by one cell on each side at the end
    H = max(12, min(32, max_h + 2))
    W = max(12, min(32, max_w + 2))
    h = unifint(diff_lb, diff_ub, (12, H))
    w = unifint(diff_lb, diff_ub, (12, W))
    oh = unifint(diff_lb, diff_ub, (3, min(7, h // 3)))
    ow = unifint(diff_lb, diff_ub, (3, min(7, w // 3)))
    tmpg = canvas(sqc, (oh, ow))
    inbounds = backdrop(inbox(asindices(tmpg)))
    obj = {random.choice(totuple(inbounds))}
    while shape(obj) != (oh - 2, ow - 2):
        obj.add(random.choice(totuple(inbounds - obj)))
    pat = fill(tmpg, noisec, obj)
    targ = asobject(fill(canvas(bgc, (oh, ow)), noisec, obj))
    sour = asobject(pat)
    gi = canvas(bgc, (h, w))
    loci = random.randint(1, h - oh - 1)
    locj = random.randint(1, w - ow - 1)
    plcddd = shift(sour, (loci, locj))
    gi = paint(gi, plcddd)
    inds = ofcolor(gi, bgc) & shift(asindices(canvas(-1, (h - 2, w - 2))), (1, 1))
    inds = inds - (toindices(plcddd) | mapply(dneighbors, toindices(plcddd)))
    namt = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // 4)))
    noise = random.sample(totuple(inds), namt)
    gi = fill(gi, noisec, noise)
    targs, sours = [], []
    for fn1 in (identity, dmirror, cmirror, hmirror, vmirror):
        for fn2 in (identity, dmirror, cmirror, hmirror, vmirror):
            targs.append(normalize(fn1(fn2(targ))))
            sours.append(normalize(fn1(fn2(sour))))
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (oh * ow * 4))))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        t = random.choice(targs)
        hh, ww = shape(t)
        cands = sfilter(inds, lambda ij: 1 <= ij[0] <= h - hh - 1 and 1 <= ij[1] <= w - ww - 1)
        if len(cands) == 0:
            continue
        loc = random.choice(totuple(cands))
        tp = shift(t, loc)
        tpi = toindices(tp)
        if tpi.issubset(inds):
            succ += 1
            inds = inds - tpi
            gi = paint(gi, tp)
    go = replace(gi, sqc, bgc)
    go = paint(go, plcddd)
    res = set()
    for t, s in zip(targs, sours):
        res |= mapply(lbind(shift, s), occurrences(go, t))
    go = paint(go, res)
    return {"input": trim(gi), "output": trim(go)}


def derive_operations(I, O=None, examples=None):
    # Paint, not copy.  Wherever the box's pattern recurs, its noise cells are
    # already in place in I; only the box body is missing.  So each recurrence
    # (any of the 8 orientations) is completed by colouring the background
    # cells of its own rectangle with the box colour.  O is never read.
    from collections import Counter
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []
    cnt = Counter(I.flatten().tolist())
    cols = sorted(cnt)
    bgc = cnt.most_common(1)[0][0]

    def extent(c):
        rr, cc = np.nonzero(I == c)
        return max(rr.max() - rr.min() + 1, cc.max() - cc.min() + 1)

    if len(cols) >= 3:
        # box colour: the colour whose cells span the smallest extent
        sqc = min(cols, key=extent)
        rest = [c for c in cols if c != sqc]
        # noise colour: the rarer of the remaining colours
        noisec = min(rest, key=lambda c: cnt[c])
        rr, cc = np.nonzero(I == sqc)
        model = I[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        # the pattern by itself: noise cells kept, box body read as background
        targ = np.where(model == noisec, noisec, bgc)
        tfs = []
        for t in (targ, targ.T, targ[::-1, ::-1].T):
            for u in (t, t[::-1, :], t[:, ::-1], t[::-1, ::-1]):
                if not any(x.shape == u.shape and np.array_equal(x, u) for x in tfs):
                    tfs.append(u)
        # a recurrence may touch the border: pad by one background cell
        P = np.full((H + 2, W + 2), bgc)
        P[1:-1, 1:-1] = I
        occ = set()
        for t in tfs:
            th, tw = t.shape
            for r in range(H + 3 - th):
                for c in range(W + 3 - tw):
                    if np.array_equal(P[r:r + th, c:c + tw], t):
                        occ.add((r - 1, c - 1, th, tw))
        G = I.copy()
        for (r, c, th, tw) in sorted(occ):
            # this recurrence's rectangle, its background cells as they stand now
            cells = [(a, b) for a in range(max(r, 0), min(r + th, H))
                     for b in range(max(c, 0), min(c + tw, W)) if G[a, b] == bgc]
            if not cells:
                continue
            ops.append(int(sqc)); sels.append(sel_of(cells))
            for p in cells:
                G[p] = sqc

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 7df24a62"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7df24a62"
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
                                f"for task 7df24a62"
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
                    f"Failed to build a complete episode for task 7df24a62 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7df24a62-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
