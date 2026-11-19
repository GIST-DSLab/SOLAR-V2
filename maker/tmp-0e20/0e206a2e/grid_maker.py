"""
ARC Task: 0e206a2e (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, acol, bcol, ccol, Dcol = random.sample(cols, 5)
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "ccol": ccol, "Dcol": Dcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol, **kwargs) -> dict:
    # Reject draws where a coincidental marker triplet makes the rule ambiguous
    # (the generator's output disagrees with the declarative rule).
    from verifiers import verify_0e206a2e
    res = None
    for _ in range(30):
        res = _generate_raw(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol)
        if verify_0e206a2e(res["input"]) == res["output"]:
            return res
    raise RuntimeError("could not draw an unambiguous instance")


def _generate_raw(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    nsrcs = unifint(diff_lb, diff_ub, (1, min(h, w) // 5))
    srcs = []
    abclist = []
    maxtrforsrc = 5 * nsrcs
    trforsrc = 0
    srcsucc = 0
    while trforsrc < maxtrforsrc and srcsucc < nsrcs:
        trforsrc += 1
        objsize = unifint(diff_lb, diff_ub, (5, 20))
        bb = asindices(canvas(-1, (7, 7)))
        sp = choice(totuple(bb))
        bb = remove(sp, bb)
        shp = {sp}
        for k in range(objsize - 1):
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while 1 in shape(shp):
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while len(set([x - y for x, y in shp])) == 1 or len(set([x + y for x, y in shp])) == 1:
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        shp = normalize(shp)
        shp = list(shp)
        shuffle(shp)
        a, b, c = shp[:3]
        while 1 in shape({a, b, c}) or (len(set([x - y for x, y in {a, b, c}])) == 1 or len(set([x + y for x, y in {a, b, c}])) == 1):
            shuffle(shp)
            a, b, c = shp[:3]
        if sorted(shape({a, b, c})) in abclist:
            continue
        D = shp[3:]
        markers = {(acol, a), (bcol, b), (ccol, c)}
        obj = markers | {(Dcol, ij) for ij in D}
        obj = frozenset(obj)
        opts = sfilter(inds, lambda ij: shift(set(shp), ij).issubset(inds))
        if len(opts) == 0:
            continue
        loc = choice(totuple(opts))
        srcsucc += 1
        gi = paint(gi, shift(obj, loc))
        shpplcd = shift(set(shp), loc)
        go = fill(go, -1, shpplcd)
        inds = (inds - shpplcd) - mapply(neighbors, shpplcd)
        srcs.append((obj, markers))
        abclist.append(sorted(shape({a, b, c})))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 30))
    maxtrials = 10 * num
    tr = 0
    succ = 0
    while succ < num and tr < maxtrials:
        mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
        fn = choice(mfs)
        gi = fn(gi)
        go = fn(go)
        aigo = asindices(go)
        fullinds = ofcolor(go, bgc) - mapply(neighbors, aigo - ofcolor(go, bgc))
        obj, markers = choice(srcs)
        shp = toindices(obj)
        if len(fullinds) == 0:
            break
        loctr = choice(totuple(fullinds))
        xx = shift(shp, loctr)
        if xx.issubset(fullinds):
            succ += 1
            gi = paint(gi, shift(markers, loctr))
            go = paint(go, shift(obj, loctr))
        tr += 1
    go = replace(go, -1, bgc)
    return {"input": gi, "output": go}


_TFS = [
    lambda r, c: (r, c), lambda r, c: (c, r), lambda r, c: (-c, -r),
    lambda r, c: (r, -c), lambda r, c: (-r, c), lambda r, c: (c, -r),
    lambda r, c: (-r, -c), lambda r, c: (-c, r),
]


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 1. Source objects: 4-connected multicolour components with exactly 4 colours.
    seen = np.zeros_like(I, dtype=bool)
    sources = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            comp, stack = [], [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != bgc:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len({int(I[y, x]) for y, x in comp}) == 4:
                sources.append(comp)
    src_cells_all = set(cell for s in sources for cell in s)
    G = I.copy()
    for (y, x) in src_cells_all:
        G[y, x] = bgc

    # 2. Find placements: for every source and symmetry, every position where
    #    the 3 marker colours occur in the grid with sources removed.
    placements = []  # (src_index, is_identity, offset, obj cells {cell: colour})
    keys = set()
    for si, comp in enumerate(sources):
        cnt = Counter(int(I[y, x]) for y, x in comp)
        dcol = cnt.most_common(1)[0][0]
        for ti, tf in enumerate(_TFS):
            pts = {tf(y, x): int(I[y, x]) for y, x in comp}
            mk = [p for p, v in pts.items() if v != dcol]
            r0 = min(p[0] for p in mk); c0 = min(p[1] for p in mk)
            pts = {(p[0] - r0, p[1] - c0): v for p, v in pts.items()}
            mk = [(p, v) for p, v in pts.items() if v != dcol]
            for oy in range(-7, hi + 7):
                for ox in range(-7, wi + 7):
                    ok = True
                    for (py, px), v in mk:
                        y, x = py + oy, px + ox
                        if not (0 <= y < hi and 0 <= x < wi) or G[y, x] != v:
                            ok = False
                            break
                    if not ok:
                        continue
                    cells = {(py + oy, px + ox): v for (py, px), v in pts.items()
                             if 0 <= py + oy < hi and 0 <= px + ox < wi}
                    key = frozenset(cells.items())
                    if key in keys:
                        continue
                    keys.add(key)
                    # identity translation of the source?
                    shift_vec = None
                    trans = {(y, x): int(I[y, x]) for y, x in comp}
                    ys = min(y for y, x in comp); xs = min(x for y, x in comp)
                    yd = min(y for y, x in cells); xd = min(x for y, x in cells)
                    dy, dx = yd - ys, xd - xs
                    if len(cells) == len(comp) and all(
                            cells.get((y + dy, x + dx)) == v for (y, x), v in trans.items()):
                        shift_vec = (dy, dx)
                    placements.append((si, shift_vec, cells, dcol))

    ops, sels = [], []
    grid = I.copy()
    moved = set()
    used_pl = set()
    # 3. A source with a pure-translation target is slid there with Moves.
    for si, comp in enumerate(sources):
        dcol = Counter(int(I[y, x]) for y, x in comp).most_common(1)[0][0]
        if dcol == 0:
            continue
        for pi, (psi, sv, cells, _) in enumerate(placements):
            if psi != si or sv is None or pi in used_pl:
                continue
            dy, dx = sv
            cur = list(comp)
            first = True
            steps = [(20 if dy < 0 else 21, (-1 if dy < 0 else 1, 0))] * abs(dy) + \
                    [(23 if dx < 0 else 22, (0, -1 if dx < 0 else 1))] * abs(dx)
            for op, (sy, sx) in steps:
                ops.append(op)
                sels.append(sel_of(cur) if first else sel_of([]))
                first = False
                cur = [(y + sy, x + sx) for y, x in cur]
            for (y, x) in comp:
                grid[y, x] = 0
            for (y, x), v in cells.items():
                grid[y, x] = v
            hole = sorted(set(comp) - set(cur))
            if hole and bgc != 0:
                ops.append(bgc); sels.append(sel_of(hole))
            for cell in hole:
                grid[cell] = bgc
            moved.add(si); used_pl.add(pi)
            break
    # 4. Erase the remaining sources.
    for si, comp in enumerate(sources):
        if si in moved:
            continue
        ops.append(bgc); sels.append(sel_of(comp))
        for cell in comp:
            grid[cell] = bgc
    # 5. Grow the object body around each remaining marker triplet.
    for pi, (si, sv, cells, dcol) in enumerate(placements):
        if pi in used_pl:
            continue
        body = [cell for cell, v in cells.items() if v == dcol and grid[cell] != dcol]
        if body:
            ops.append(dcol); sels.append(sel_of(body))
            for cell in body:
                grid[cell] = dcol

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 0e206a2e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0e206a2e"
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
                                f"for task 0e206a2e"
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
                    f"Failed to build a complete episode for task 0e206a2e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0e206a2e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
