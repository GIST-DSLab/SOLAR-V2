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


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, sqc=None) -> dict:
    if bgc is None or noisec is None or sqc is None:
        bgc, noisec, sqc = random.sample(range(10), 3)
    hub = max(12, min(32, max_h + 2))
    wub = max(12, min(32, max_w + 2))
    h = unifint(diff_lb, diff_ub, (12, hub))
    w = unifint(diff_lb, diff_ub, (12, wub))
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
    targs = []
    sours = []
    for fn1 in (identity, dmirror, cmirror, hmirror, vmirror):
        for fn2 in (identity, dmirror, cmirror, hmirror, vmirror):
            targs.append(normalize(fn1(fn2(targ))))
            sours.append(normalize(fn1(fn2(sour))))
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // ((oh * ow * 4)))))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
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


def _roles(I):
    # bgc: the colour the canvas is drawn on (majority); sqc: the colour packed
    # into the one small rectangle (smallest extent); noisec: the remaining colour.
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])
    best = None
    for v in vals:
        rr, cc = np.nonzero(I == v)
        ext = max(rr.max() - rr.min() + 1, cc.max() - cc.min() + 1)
        if best is None or ext < best[0]:
            best = (ext, int(v))
    sqc = best[1]
    rest = [(int(c), int(v)) for v, c in zip(vals, counts) if int(v) != sqc]
    rest = [x for x in rest if x[1] != bgc] or rest
    noisec = min(rest)[1]
    return bgc, sqc, noisec


def _orientations(M):
    out, seen = [], set()
    for k in range(4):
        R = np.rot90(M, k)
        for T in (R, np.fliplr(R)):
            key = (T.shape, T.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(T)
    return out


def derive_operations(I, O=None, examples=None):
    # O is never read.  Reading: the scattered noise dots already ARE the
    # pattern of every new box; wherever the box's dot pattern recurs (in any
    # rotation / reflection) on clean background, only the box's body is
    # missing.  Each such place is filled with the box colour on exactly the
    # cells the box's body (its sqc cells, turned the way this copy lies)
    # covers -- the dots are left where they are.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []
    bgc, sqc, noisec = _roles(I)
    rr, cc = np.nonzero(I == sqc)
    model = I[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    P = np.full((H + 2, W + 2), bgc, dtype=int)   # a box may hang 1 past the edge
    P[1:H + 1, 1:W + 1] = I
    G = I.copy()
    done = set()
    for T in _orientations(model):
        th, tw = T.shape
        targ = np.where(T == sqc, bgc, T)          # the box with its body unpainted
        body = [(i, j) for i in range(th) for j in range(tw) if T[i, j] == sqc]
        for pr in range(H + 3 - th):
            for pc in range(W + 3 - tw):
                if not np.array_equal(P[pr:pr + th, pc:pc + tw], targ):
                    continue
                r0, c0 = pr - 1, pc - 1
                cells = [(r0 + i, c0 + j) for i, j in body
                         if 0 <= r0 + i < H and 0 <= c0 + j < W]
                key = frozenset(cells)
                if key in done:
                    continue
                done.add(key)
                cells = [p for p in cells if G[p] != sqc]
                if not cells:
                    continue
                ops.append(int(sqc)); sels.append(sel_of(cells))   # the box's body here
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
