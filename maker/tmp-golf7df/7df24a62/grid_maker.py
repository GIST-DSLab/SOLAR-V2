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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, sqc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, noisec=4, sqc=1) -> dict:
    # the untrimmed canvas carries a one-cell border that trim() removes
    h = unifint(diff_lb, diff_ub, (12, min(32, max_h + 2)))
    w = unifint(diff_lb, diff_ub, (12, min(32, max_w + 2)))
    oh = unifint(diff_lb, diff_ub, (3, min(7, h // 3)))
    ow = unifint(diff_lb, diff_ub, (3, min(7, w // 3)))
    tmpg = canvas(sqc, (oh, ow))
    inbounds = backdrop(inbox(asindices(tmpg)))
    obj = {choice(totuple(inbounds))}
    while shape(obj) != (oh - 2, ow - 2):
        obj.add(choice(totuple(inbounds - obj)))
    pat = fill(tmpg, noisec, obj)
    targ = asobject(fill(canvas(bgc, (oh, ow)), noisec, obj))
    sour = asobject(pat)
    gi = canvas(bgc, (h, w))
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    plcddd = shift(sour, (loci, locj))
    gi = paint(gi, plcddd)
    inds = ofcolor(gi, bgc) & shift(asindices(canvas(-1, (h - 2, w - 2))), (1, 1))
    inds = inds - (toindices(plcddd) | mapply(dneighbors, toindices(plcddd)))
    namt = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // 4)))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, noisec, noise)
    targs, sours = [], []
    for fn1 in (identity, dmirror, cmirror, hmirror, vmirror):
        for fn2 in (identity, dmirror, cmirror, hmirror, vmirror):
            targs.append(normalize(fn1(fn2(targ))))
            sours.append(normalize(fn1(fn2(sour))))
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (oh * ow * 4))))
    succ, tr, maxtr = 0, 0, 5 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        t = choice(targs)
        hh, ww = shape(t)
        cands = sfilter(inds, lambda ij: 1 <= ij[0] <= h - hh - 1 and 1 <= ij[1] <= w - ww - 1)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
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


# ARCLE in-place ops on a square-or-not rectangle, as numpy
_APPLY = {
    24: lambda a: np.rot90(a, 1),   # Rotate90 = CCW (square only)
    25: lambda a: np.rot90(a, 3),   # Rotate270 = CW (square only)
    26: lambda a: np.fliplr(a),     # FlipH
    27: lambda a: np.flipud(a),     # FlipV
}


def _orientations(model):
    """The eight ways the box can lie, duplicates removed (in a fixed order)."""
    cands = [model, np.fliplr(model), np.flipud(model), np.rot90(model, 2),
             model.T, np.rot90(model, 1), np.rot90(model, 3), np.rot90(model.T, 2)]
    out = []
    for a in cands:
        a = np.ascontiguousarray(a)
        if not any(a.shape == b.shape and np.array_equal(a, b) for b in out):
            out.append(a)
    return out


def _turns(model, target):
    """Shortest chain of in-place Flip/Rotate ops taking model to target.
    Rotations are only offered when the box is square (ARCLE's rotate needs it)."""
    allowed = [26, 27] + ([24, 25] if model.shape[0] == model.shape[1] else [])
    frontier = [(model, [])]
    seen = [model]
    for _ in range(4):
        nxt = []
        for a, path in frontier:
            if a.shape == target.shape and np.array_equal(a, target):
                return path
            for op in allowed:
                b = _APPLY[op](a)
                if any(b.shape == s.shape and np.array_equal(b, s) for s in seen):
                    continue
                seen.append(b)
                nxt.append((b, path + [op]))
        frontier = nxt
    for a, path in frontier:
        if a.shape == target.shape and np.array_equal(a, target):
            return path
    return None


def derive_operations(I, O=None, examples=None):
    # O is never read. Reading: the one solid box is a stamp; its pattern is
    # sought, in every way the box can lie, with the grid seen through a
    # one-cell frame of background (so a pattern may hug the edge). Each way
    # of lying is built ONCE and then stamped on every place it recurs.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    cnt = {int(v): int((I == v).sum()) for v in np.unique(I)}
    bgc = max(cnt, key=lambda v: cnt[v])

    def span(v):
        rr, cc = np.nonzero(I == v)
        return max(rr.max() - rr.min() + 1, cc.max() - cc.min() + 1)

    sqc = min(cnt, key=lambda v: (span(v), cnt[v]))
    rest = [v for v in cnt if v not in (bgc, sqc)]
    if not rest:
        ops.append(34); sels.append([0, 0, H - 1, W - 1])
        return ops, sels
    noisec = min(rest, key=lambda v: cnt[v])

    rr, cc = np.nonzero(I == sqc)
    br, bc = int(rr.min()), int(cc.min())
    oh, ow = int(rr.max()) - br + 1, int(cc.max()) - bc + 1
    model = I[br:br + oh, bc:bc + ow].copy()

    P = np.full((H + 2, W + 2), bgc, dtype=int)
    P[1:-1, 1:-1] = I

    def occurrences(T):
        th, tw = T.shape
        want = np.where(T == noisec, noisec, bgc)
        found = []
        for r in range(H + 2 - th + 1):
            for c in range(W + 2 - tw + 1):
                if np.array_equal(P[r:r + th, c:c + tw], want):
                    found.append((r - 1, c - 1))
        return found

    G = I.copy()
    pastable = (sqc != 0 and noisec != 0)   # 0 would be dropped by Copy/Paste
    clip = None                               # what the clipboard holds

    def inside(r, c, th, tw):
        return r >= 0 and c >= 0 and r + th <= H and c + tw <= W

    def paint_body(r, c, T):
        # the box's body around a pattern that is already there
        th, tw = T.shape
        cells = [(r + i, c + j) for i in range(th) for j in range(tw)
                 if T[i, j] == sqc and 0 <= r + i < H and 0 <= c + j < W
                 and G[r + i, c + j] != sqc]
        if cells:
            ops.append(int(sqc)); sels.append(sel_of(cells))
            for p in cells:
                G[p] = sqc
        nz = [(r + i, c + j) for i in range(th) for j in range(tw)
              if T[i, j] == noisec and 0 <= r + i < H and 0 <= c + j < W
              and G[r + i, c + j] != noisec]
        if nz:   # only if an earlier stamp covered part of this pattern
            ops.append(int(noisec)); sels.append(sel_of(nz))
            for p in nz:
                G[p] = noisec

    for k, T in enumerate(_orientations(model)):
        th, tw = T.shape
        places = occurrences(T)
        built = None          # top-left of this orientation's first full copy
        for (r, c) in places:
            if inside(r, c, th, tw) and np.array_equal(G[r:r + th, c:c + tw], T):
                continue
            if not inside(r, c, th, tw) or not pastable:
                paint_body(r, c, T)
                continue
            if built is None:
                path = _turns(model, T)
                if path is None:
                    # a non-square box lying on its side: draw it directly
                    paint_body(r, c, T)
                else:
                    if clip != "model":   # the box itself, its own rectangle
                        ops.append(28); sels.append([br, bc, oh - 1, ow - 1])
                        clip = "model"
                    ops.append(30); sels.append([r, c, 0, 0])
                    cur = model.copy()
                    G[r:r + oh, c:c + ow] = cur
                    for op in path:   # turn it, exactly its own rectangle
                        ops.append(op)
                        sels.append([r, c, cur.shape[0] - 1, cur.shape[1] - 1])
                        cur = _APPLY[op](cur)
                        G[r:r + th, c:c + tw] = cur
                built = (r, c)
                continue
            if clip != k:
                if k == 0:
                    if clip != "model":
                        ops.append(28); sels.append([br, bc, oh - 1, ow - 1])
                        clip = "model"
                else:   # pick up the copy just built, exactly its rectangle
                    ops.append(29); sels.append([built[0], built[1], th - 1, tw - 1])
                    clip = k
            ops.append(30); sels.append([r, c, 0, 0])
            G[r:r + th, c:c + tw] = T
        if clip == k and k == 0:
            clip = "model"

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
