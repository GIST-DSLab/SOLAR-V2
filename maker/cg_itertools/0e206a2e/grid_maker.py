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
import random
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, acol, bcol, ccol, Dcol = random.sample(range(10), 5)
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "ccol": ccol, "Dcol": Dcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, acol=None, bcol=None,
             ccol=None, Dcol=None, **kwargs) -> dict:
    from random import choice, shuffle
    if None in (bgc, acol, bcol, ccol, Dcol):
        bgc, acol, bcol, ccol, Dcol = random.sample(range(10), 5)

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        return random.randint(min(lo, hi), max(lo, hi))

    def shape_of(pts):
        rs = [p[0] for p in pts]; cs = [p[1] for p in pts]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    def normalize(pts):
        mr = min(p[0] for p in pts); mc = min(p[1] for p in pts)
        return {(r - mr, c - mc) for r, c in pts}

    def dnb(p):
        r, c = p
        return {(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)}

    def nb(p):
        r, c = p
        return {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    # dihedral transforms on grids (list of lists)
    def T(g): return [list(x) for x in zip(*g)]
    def dmirror(g): return T(g)
    def hmirror(g): return [list(r) for r in g[::-1]]
    def vmirror(g): return [list(r[::-1]) for r in g]
    def cmirror(g): return vmirror(dmirror(vmirror(g)))
    def rot90(g): return [list(r) for r in zip(*g[::-1])]
    def rot180(g): return [list(r[::-1]) for r in g[::-1]]
    def rot270(g): return [list(r) for r in zip(*g)][::-1]
    def identity(g): return [list(r) for r in g]

    lo_h = min(10, max_h); lo_w = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (lo_h, max_h))
    w = unifint(diff_lb, diff_ub, (lo_w, max_w))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    nsrcs = unifint(diff_lb, diff_ub, (1, max(1, min(h, w) // 5)))
    srcs = []
    abclist = []
    maxtr = 5 * nsrcs
    tr = 0
    succ = 0
    while tr < maxtr and succ < nsrcs:
        tr += 1
        objsize = unifint(diff_lb, diff_ub, (5, 20))
        bb = {(i, j) for i in range(7) for j in range(7)}
        sp = choice(sorted(bb))
        bb.discard(sp)
        shp = {sp}
        for _ in range(objsize - 1):
            cand = set()
            for p in shp:
                cand |= dnb(p)
            cand = (bb - shp) & cand
            shp.add(choice(sorted(cand)))
        while 1 in shape_of(shp):
            cand = set()
            for p in shp:
                cand |= dnb(p)
            shp.add(choice(sorted((bb - shp) & cand)))
        while len({x - y for x, y in shp}) == 1 or len({x + y for x, y in shp}) == 1:
            cand = set()
            for p in shp:
                cand |= dnb(p)
            shp.add(choice(sorted((bb - shp) & cand)))
        shp = list(normalize(shp))
        shuffle(shp)
        a, b, c = shp[:3]
        guard = 0
        while 1 in shape_of({a, b, c}) or len({x - y for x, y in {a, b, c}}) == 1 \
                or len({x + y for x, y in {a, b, c}}) == 1:
            shuffle(shp)
            a, b, c = shp[:3]
            guard += 1
            if guard > 500:
                break
        if guard > 500:
            continue
        if sorted(shape_of({a, b, c})) in abclist:
            continue
        D = shp[3:]
        markers = {(acol, a), (bcol, b), (ccol, c)}
        obj = markers | {(Dcol, ij) for ij in D}
        opts = [ij for ij in inds
                if all((p[0] + ij[0], p[1] + ij[1]) in inds for p in shp)]
        if not opts:
            continue
        loc = choice(sorted(opts))
        succ += 1
        for v, (r, cc) in obj:
            gi[r + loc[0]][cc + loc[1]] = v
        placed = {(r + loc[0], cc + loc[1]) for r, cc in shp}
        for r, cc in placed:
            go[r][cc] = -1
        rem = set(placed)
        for p in placed:
            rem |= nb(p)
        inds = inds - rem
        srcs.append((obj, markers))
        abclist.append(sorted(shape_of({a, b, c})))
    if not srcs:
        raise ValueError("no source")

    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 30)))
    maxtrials = 10 * num
    tr = 0
    succ = 0
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    while succ < num and tr < maxtrials:
        fn = choice(mfs)
        gi = fn(gi)
        go = fn(go)
        H, W = len(go), len(go[0])
        nonbg = {(i, j) for i in range(H) for j in range(W) if go[i][j] != bgc}
        bad = set()
        for p in nonbg:
            bad |= nb(p)
        fullinds = {(i, j) for i in range(H) for j in range(W) if go[i][j] == bgc} - bad
        obj, markers = choice(srcs)
        shp = {p for _, p in obj}
        if not fullinds:
            break
        loctr = choice(sorted(fullinds))
        xx = {(r + loctr[0], cc + loctr[1]) for r, cc in shp}
        if xx.issubset(fullinds):
            succ += 1
            for v, (r, cc) in markers:
                gi[r + loctr[0]][cc + loctr[1]] = v
            for v, (r, cc) in obj:
                go[r + loctr[0]][cc + loctr[1]] = v
        tr += 1
    go = [[bgc if v == -1 else v for v in row] for row in go]
    if succ == 0:
        raise ValueError("no targets")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (template = one rigid stamp):
      * Each 4-connected component with exactly four colours is a TEMPLATE:
        a body colour (majority) plus three single-cell markers.
      * All templates are erased first (one Color-bgc op per template object).
      * Then, template by template: the template is treated as a rigid stamp
        under all 8 dihedral orientations; wherever its three markers land
        exactly on a matching bare marker triple, the stamp's body cells are
        painted (one Color op per stamp placement). The selection is just
        "the cells the transformed template lands on". O is never read.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    ops, sels = [], []

    # 1. 4-connected non-bg components
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb_ = a + da, b + db
                        if 0 <= na < h and 0 <= nb_ < w and not seen[na, nb_] \
                                and I[na, nb_] != bgc:
                            seen[na, nb_] = True
                            stack.append((na, nb_))
                comps.append(sorted(cells))

    # 2. templates
    templates = []
    for cells in comps:
        cols = [int(I[r, c]) for r, c in cells]
        if len(set(cols)) != 4:
            continue
        body_col = Counter(cols).most_common(1)[0][0]
        markers = [(int(I[r, c]), (r, c)) for r, c in cells if int(I[r, c]) != body_col]
        body = [(r, c) for r, c in cells if int(I[r, c]) == body_col]
        if len(markers) == 3 and body:
            templates.append((cells, int(body_col), markers, body))

    g = I.copy()

    # 3. erase every template object
    for cells, _, _, _ in templates:
        ops.append(bgc)
        sels.append(sel_of(cells))
        for r, c in cells:
            g[r, c] = bgc

    ge = g.copy()  # erased grid on which marker triples are searched

    MAPS = [
        lambda r, c: (r, c), lambda r, c: (c, r),
        lambda r, c: (-r, c), lambda r, c: (r, -c),
        lambda r, c: (-r, -c), lambda r, c: (-c, r),
        lambda r, c: (c, -r), lambda r, c: (-c, -r),
    ]

    # 4. stamp each template, template by template
    for cells, body_col, markers, body in templates:
        done = set()
        for M in MAPS:
            mk = [(v, M(*p)) for v, p in markers]
            bd = [M(*p) for p in body]
            v0, (r0, c0) = mk[0]
            for pr in range(h):
                for pc in range(w):
                    if ge[pr, pc] != v0:
                        continue
                    tr_, tc_ = pr - r0, pc - c0
                    ok = True
                    for v, (mr, mc) in mk[1:]:
                        rr, cc = mr + tr_, mc + tc_
                        if not (0 <= rr < h and 0 <= cc < w) or ge[rr, cc] != v:
                            ok = False
                            break
                    if not ok:
                        continue
                    stamp = sorted({(br + tr_, bc + tc_) for br, bc in bd
                                    if 0 <= br + tr_ < h and 0 <= bc + tc_ < w})
                    key = tuple(stamp)
                    if key in done:
                        continue
                    done.add(key)
                    todo = [p for p in stamp if int(g[p[0], p[1]]) != body_col]
                    if not todo:
                        continue
                    ops.append(body_col)
                    sels.append(sel_of(todo))
                    for r, c in todo:
                        g[r, c] = body_col

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
