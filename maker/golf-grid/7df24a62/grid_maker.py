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


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sqc) -> dict:
    h = unifint(diff_lb, diff_ub, (12, max(12, min(32, max_h + 2))))
    w = unifint(diff_lb, diff_ub, (12, max(12, min(32, max_w + 2))))
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
    gi = trim(gi)
    go = trim(go)
    return {"input": gi, "output": go}


# Whole-grid turns: the selection is always the full working canvas.
_GRID_OPS = {
    26: lambda a: np.fliplr(a),
    27: lambda a: np.flipud(a),
    24: lambda a: np.rot90(a, 1),
    25: lambda a: np.rot90(a, 3),
}


def _roles(I):
    """bgc = most common colour; sqc = the colour whose cells span the
    smallest extent (the box); noisec = the rarest remaining colour."""
    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])
    best = None
    for v in vals:
        rr, cc = np.where(I == v)
        ext = max(rr.max() - rr.min() + 1, cc.max() - cc.min() + 1)
        if best is None or ext < best[0]:
            best = (ext, int(v))
    sqc = best[1]
    rest = [(int(c), int(v)) for v, c in zip(vals, cnts) if int(v) not in (bgc, sqc)]
    noisec = min(rest)[1] if rest else None
    return bgc, sqc, noisec


def _states(CH, CW, h, w, gens):
    """Every way the whole canvas can be turned with the given ops (BFS from
    the untouched canvas). Each state is an index map: canvas cell -> flat
    index of the original input cell lying there (or -1 for padding)."""
    idx0 = np.full((CH, CW), -1, dtype=int)
    idx0[:h, :w] = np.arange(h * w).reshape(h, w)
    order = [idx0]
    seen = {idx0.tobytes(): 0}
    k = 0
    while k < len(order):
        cur = order[k]
        for op in gens:
            nxt = np.ascontiguousarray(_GRID_OPS[op](cur))
            key = nxt.tobytes()
            if key not in seen:
                seen[key] = len(order)
                order.append(nxt)
        k += 1
    return order, seen


def _plan(I, model, bgc, sqc, CH, CW, gens):
    """For each turn of the grid, find where the box's pattern reads exactly
    as the key does (pattern cells noisy, the box's body cells bare), and
    record which original cells the key's body would cover there."""
    h, w = I.shape
    oh, ow = model.shape
    T = np.where(model == sqc, bgc, model)
    body = (model == sqc)
    order, seen = _states(CH, CW, h, w, gens)
    flat = I.flatten()
    claimed = set()
    stamps = [[] for _ in order]
    for s, idx in enumerate(order):
        rows = np.where((idx >= 0).any(axis=1))[0]
        cols = np.where((idx >= 0).any(axis=0))[0]
        Jidx = idx[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        J = flat[Jidx]
        jh, jw = J.shape
        P = np.full((jh + 2, jw + 2), bgc, dtype=int)
        P[1:-1, 1:-1] = J
        Pidx = np.full((jh + 2, jw + 2), -1, dtype=int)
        Pidx[1:-1, 1:-1] = Jidx
        for r in range(jh + 3 - oh):
            for c in range(jw + 3 - ow):
                if np.array_equal(P[r:r + oh, c:c + ow], T):
                    cells = frozenset(int(v) for v in Pidx[r:r + oh, c:c + ow][body] if v >= 0)
                    if cells and cells not in claimed:
                        claimed.add(cells)
                        stamps[s].append(cells)
    return order, seen, stamps


def _path(src_key, dst_key, seen, order, gens):
    """Shortest sequence of whole-grid turns from one state to another."""
    from collections import deque
    prev = {src_key: None}
    dq = deque([src_key])
    while dq:
        k = dq.popleft()
        if k == dst_key:
            break
        cur = order[seen[k]]
        for op in gens:
            nk = np.ascontiguousarray(_GRID_OPS[op](cur)).tobytes()
            if nk not in prev:
                prev[nk] = (k, op)
                dq.append(nk)
    seq = []
    k = dst_key
    while prev[k] is not None:
        k, op = prev[k]
        seq.append(op)
    return seq[::-1]


def derive_operations(I, O=None, examples=None):
    # O is never read. Reading of the task (turn the WORLD, not the box):
    # the key box is read in one fixed way only. The whole grid is turned
    # through its orientations; in each one, wherever the pattern reads
    # exactly as the key does, the key's body is laid around it. Then the
    # grid is turned back.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    bgc, sqc, noisec = _roles(I)
    if noisec is None:
        return [34], [[0, 0, h - 1, w - 1]]
    rr, cc = np.where(I == sqc)
    br, bc = int(rr.min()), int(cc.min())
    oh, ow = int(rr.max()) - br + 1, int(cc.max()) - bc + 1
    model = I[br:br + oh, bc:bc + ow].copy()

    sq = max(h, w)
    order, seen, stamps = _plan(I, model, bgc, sqc, sq, sq, [26, 27, 24, 25])
    resize = False
    if h != w:
        needs_turn = any(stamps[s] and (order[s] >= 0).any(axis=1).sum() != h
                         for s in range(len(order)))
        if needs_turn:
            resize = True
            CH = CW = sq
            gens = [26, 27, 24, 25]
        else:
            CH, CW = h, w
            gens = [26, 27]
            order, seen, stamps = _plan(I, model, bgc, sqc, CH, CW, gens)
    else:
        CH = CW = h
        gens = [26, 27, 24, 25]

    S = np.zeros((CH, CW), dtype=int)
    S[:h, :w] = I
    idx = order[0].copy()
    home = order[0].tobytes()
    full = [0, 0, CH - 1, CW - 1]   # the whole canvas
    targets = [s for s in range(len(order)) if stamps[s]]
    if not targets:
        return [34], [[0, 0, h - 1, w - 1]]
    if resize:
        # make room to turn a non-square grid a quarter: square canvas
        ops.append(33); sels.append(full)

    def turn_to(key):
        nonlocal S, idx
        for op in _path(idx.tobytes(), key, seen, order, gens):
            nS = _GRID_OPS[op](S)
            idx = np.ascontiguousarray(_GRID_OPS[op](idx))
            if np.array_equal(nS, S):
                continue          # the grid looks the same either way
            ops.append(op); sels.append(full)
            S = nS

    for s in targets:
        turn_to(order[s].tobytes())
        flat = idx.flatten()
        for cells in stamps[s]:
            # lay the key's body around this pattern, read as the key reads
            pos = [divmod(int(p), CW) for p in np.where(np.isin(flat, list(cells)))[0]]
            pos = [p for p in pos if S[p] != sqc]
            if pos:
                ops.append(int(sqc)); sels.append(sel_of(pos))
                for p in pos:
                    S[p] = sqc
    turn_to(home)
    if resize:
        ops.append(33); sels.append([0, 0, h - 1, w - 1])   # back to the grid's own size
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
