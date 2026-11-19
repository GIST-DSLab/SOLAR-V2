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

# ── LLM-generated ──
import random
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, sqc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sqc) -> dict:
    # the generator trims a 1-cell frame, so the raw canvas may be max+2
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


# The eight ways the box can lie, each with the in-place ARCLE turns that make
# it out of a freshly stamped copy of the box (quarter turns need a square).
_ORIENT = [
    (lambda a: a,                       []),
    (lambda a: np.fliplr(a),            [26]),
    (lambda a: np.flipud(a),            [27]),
    (lambda a: np.flipud(np.fliplr(a)), [26, 27]),
    (lambda a: np.rot90(a, 3),          [25]),
    (lambda a: np.rot90(a, 1),          [24]),
    (lambda a: a.T,                     [25, 26]),
    (lambda a: np.rot90(a, 2).T,        [24, 26]),
]
_APPLY = {
    24: lambda a: np.rot90(a, 1), 25: lambda a: np.rot90(a, 3),
    26: lambda a: np.fliplr(a), 27: lambda a: np.flipud(a),
}


def _roles(I):
    """bgc: the most common colour.  sqc: the colour whose bounding box has an
    unbroken border of itself (the box's frame).  noisec: the other one."""
    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])
    sqc = None
    for v in vals:
        v = int(v)
        if v == bgc:
            continue
        rr, cc = np.where(I == v)
        r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = I[r0:r1 + 1, c0:c1 + 1]
        if (sub[0] == v).all() and (sub[-1] == v).all() and \
           (sub[:, 0] == v).all() and (sub[:, -1] == v).all():
            sqc = v
            break
    if sqc is None:
        return bgc, None, None
    rest = [int(v) for v in vals if int(v) not in (bgc, sqc)]
    return bgc, sqc, (rest[0] if rest else None)


def derive_operations(I, O=None, examples=None):
    # O is never read.  Reading (as in the golf entry): sweep the grid position
    # by position; wherever the box's pattern lies bare in one of its eight
    # orientations (the generator's 1-cell background frame around the grid
    # counts as background), stamp the box in that orientation over it.  Each
    # stamp is one unit: a Paste of that oriented box, taken from wherever an
    # identical oriented box already stands on the grid; the first box of a new
    # orientation is stamped from the original box and turned in place.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []
    bgc, sqc, noisec = _roles(I)
    if sqc is None or noisec is None:
        return [34], [[0, 0, H - 1, W - 1]]
    rr, cc = np.where(I == sqc)
    br, bc = int(rr.min()), int(cc.min())
    oh, ow = int(rr.max()) - br + 1, int(cc.max()) - bc + 1
    model = I[br:br + oh, bc:bc + ow].copy()
    pastable = bool((model != 0).all())      # 0 never pastes in ARCLE

    P = np.full((H + 2, W + 2), bgc, dtype=int)
    P[1:H + 1, 1:W + 1] = I

    # every (position, oriented box) where the bare pattern lies, raster order
    stamps = {}
    for f, tops in _ORIENT:
        U = np.ascontiguousarray(f(model))
        T = np.where(U == sqc, bgc, U)
        th, tw = T.shape
        for r in range(H + 3 - th):
            for c in range(W + 3 - tw):
                if np.array_equal(P[r:r + th, c:c + tw], T):
                    key = (r - 1, c - 1, U.shape, U.tobytes())
                    if key not in stamps or len(tops) < len(stamps[key][1]):
                        stamps[key] = (U, tops)
    order = sorted(stamps.items(), key=lambda kv: (kv[0][0], kv[0][1]))

    G = I.copy()
    clip = None          # the array currently on the clipboard
    placed = [((br, bc), model)]   # oriented boxes standing on the grid

    def paint_cells(r, c, U):
        # colour the box onto the cells of the window that are on the grid
        for colour in (sqc, noisec):
            cells = [(i, j) for i in range(max(r, 0), min(r + U.shape[0], H))
                     for j in range(max(c, 0), min(c + U.shape[1], W))
                     if U[i - r, j - c] == colour and G[i, j] != colour]
            if cells:
                ops.append(int(colour)); sels.append(sel_of(cells))
                for p in cells:
                    G[p] = colour

    for (r, c, _s, _b), (U, tops) in order:
        th, tw = U.shape
        inside = r >= 0 and c >= 0 and r + th <= H and c + tw <= W
        if inside and np.array_equal(G[r:r + th, c:c + tw], U):
            continue                         # already standing here
        if not inside or not pastable:
            paint_cells(r, c, U)             # a box over the edge: colour it on
            if inside:
                placed.append(((r, c), U))
            continue
        if clip is None or clip.shape != U.shape or not np.array_equal(clip, U):
            src = None
            for (pr, pc), V in placed:
                if V.shape == U.shape and np.array_equal(V, U) and \
                   np.array_equal(G[pr:pr + th, pc:pc + tw], U):
                    src = (pr, pc)
                    break
            if src is not None:
                # copy an identical oriented box that already stands here
                if src == (br, bc) and np.array_equal(I[br:br + oh, bc:bc + ow], model):
                    ops.append(28)
                else:
                    ops.append(29)
                sels.append([src[0], src[1], th - 1, tw - 1])
                clip = U
            elif oh == ow or all(op in (26, 27) for op in tops):
                # new orientation: stamp the original box and turn it in place
                if clip is None or clip.shape != model.shape or not np.array_equal(clip, model):
                    ops.append(28); sels.append([br, bc, oh - 1, ow - 1])
                    clip = model
                ops.append(30); sels.append([r, c, 0, 0])
                cur = model.copy()
                G[r:r + oh, c:c + ow] = cur
                for op in tops:
                    nxt = _APPLY[op](cur)
                    if np.array_equal(nxt, cur):
                        continue
                    # the stamped box's own full rectangle
                    ops.append(op); sels.append([r, c, cur.shape[0] - 1, cur.shape[1] - 1])
                    cur = nxt
                    G[r:r + th, c:c + tw] = cur
                placed.append(((r, c), U))
                continue
            else:
                # a quarter turn of a non-square box cannot be done in place:
                # colour its body around the pattern already lying there
                paint_cells(r, c, U)
                placed.append(((r, c), U))
                continue
        ops.append(30); sels.append([r, c, 0, 0])     # stamp it
        G[r:r + th, c:c + tw] = U
        placed.append(((r, c), U))

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
