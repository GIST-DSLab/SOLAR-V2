"""
ARC Task: 3e980e27 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, rcol, gcol = random.sample(cols, 3)
    return {"bgc": bgc, "rcol": rcol, "gcol": gcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rcol, gcol, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    def nbrs(p):
        r, c = p
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}

    def normalize(obj):
        mr = min(i for _, (i, j) in obj)
        mc = min(j for _, (i, j) in obj)
        return {(v, (i - mr, j - mc)) for v, (i, j) in obj}

    def shp(obj):
        rs = [i for _, (i, j) in obj]
        cs = [j for _, (i, j) in obj]
        return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    def shift(obj, d):
        return {(v, (i + d[0], j + d[1])) for v, (i, j) in obj}

    def vmirror(obj):
        _, w = shp(obj)
        mc = min(j for _, (i, j) in obj)
        return {(v, (i, mc + w - 1 - (j - mc))) for v, (i, j) in obj}

    H = max(11, min(30, max_h))
    W = max(11, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (11, H))
    w = unifint(diff_lb, diff_ub, (11, W))
    objs = []
    for fixc, remc in ((2, rcol), (3, gcol)):
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        obj = {random.choice(sorted(bounds))}
        ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = random.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(2, ncells), oh * ow)
        for _ in range(ncells - 1):
            nb = set()
            for p in obj:
                nb |= nbrs(p)
            obj.add(random.choice(sorted((bounds - obj) & nb)))
        mr = min(i for i, j in obj)
        mc = min(j for i, j in obj)
        obj = {(i - mr, j - mc) for i, j in obj}
        fixp = random.choice(sorted(obj))
        o = {(fixc, fixp)} | {(remc, p) for p in obj if p != fixp}
        objs.append(o)
    robj, gobj = objs
    obj1, obj2 = random.sample(objs, 2)
    h1, w1 = shp(obj1)
    h2, w2 = shp(obj2)
    loci1 = random.randint(0, h - h1 - h2 - 1)
    locj1 = random.randint(0, w - w1)
    loci2 = random.randint(loci1 + h1 + 1, h - h2)
    locj2 = random.randint(0, w - w2)
    gi = [[bgc] * w for _ in range(h)]

    def paint(g, obj):
        g = [list(row) for row in g]
        for v, (i, j) in obj:
            if 0 <= i < h and 0 <= j < w:
                g[i][j] = v
        return g

    obj1p = shift(obj1, (loci1, locj1))
    obj2p = shift(obj2, (loci2, locj2))
    gi = paint(gi, obj1p)
    gi = paint(gi, obj2p)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // int(1.5 * (len(robj) + len(gobj)))))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    robj = vmirror(robj)
    excl = set()
    for _, p in obj1p | obj2p:
        excl |= nbrs(p)
    inds = {(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc} - excl
    go = [list(r) for r in gi]
    objopts = [robj, gobj]
    while tr < maxtr and succ < noccs:
        tr += 1
        obj = random.choice(objopts)
        oh, ow = shp(obj)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        plcd = shift(obj, loc)
        plcdi = {p for _, p in plcd}
        if plcdi.issubset(inds):
            succ += 1
            nb = set()
            for p in plcdi:
                nb |= nbrs(p)
            inds = (inds - plcdi) - nb
            gi = paint(gi, {(v, p) for v, p in plcd if v in (2, 3)})
            go = paint(go, plcd)
    if unifint(diff_lb, diff_ub, (1, 100)) < 30:
        c = random.choice((2, 3))

        def comps(g):
            seen = set()
            res = []
            for i in range(h):
                for j in range(w):
                    if g[i][j] == bgc or (i, j) in seen:
                        continue
                    st = [(i, j)]
                    seen.add((i, j))
                    comp = []
                    while st:
                        p = st.pop()
                        comp.append(p)
                        for q in nbrs(p):
                            if 0 <= q[0] < h and 0 <= q[1] < w and q not in seen and g[q[0]][q[1]] != bgc:
                                seen.add(q)
                                st.append(q)
                    res.append(comp)
            return res

        for g in (gi, go):
            for comp in comps(g):
                if any(g[i][j] == c for i, j in comp):
                    for i, j in comp:
                        g[i][j] = bgc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # background: the canvas colour, always the majority colour in this task
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected multicolour components of non-background cells
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            st = [(r, c)]
            seen[r, c] = True
            comp = []
            while st:
                a, b = st.pop()
                comp.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        x, y = a + da, b + db
                        if 0 <= x < h and 0 <= y < w and not seen[x, y] and I[x, y] != bgc:
                            seen[x, y] = True
                            st.append((x, y))
            comps.append(comp)

    ops, sels = [], []
    # 3-anchored template copied as is; 2-anchored template copied mirrored left-right
    for anchor, mirror in ((3, False), (2, True)):
        cand = [cp for cp in comps if any(I[p] == anchor for p in cp)]
        if not cand:
            continue
        tmpl = max(cand, key=lambda cp: len({int(I[p]) for p in cp}))
        if len({int(I[p]) for p in tmpl}) < 2:
            continue
        ar, ac = [p for p in tmpl if I[p] == anchor][0]
        tset = set(tmpl)
        targets = [(r, c) for r in range(h) for c in range(w)
                   if I[r, c] == anchor and (r, c) not in tset]
        for pr, pc in targets:
            bycol = {}
            for (r, c) in tmpl:
                if (r, c) == (ar, ac):
                    continue
                dc = -(c - ac) if mirror else (c - ac)
                rr, cc = pr + r - ar, pc + dc
                if 0 <= rr < h and 0 <= cc < w:
                    bycol.setdefault(int(I[r, c]), []).append((rr, cc))
            # build this copy around its anchor, one colour at a time
            for col in sorted(bycol):
                cells = sorted(set(bycol[col]))
                if cells:
                    ops.append(col)
                    sels.append(sel_of(cells))
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
                        f"num_examples+1 ({num_examples + 1}) for task 3e980e27"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3e980e27"
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
                                f"for task 3e980e27"
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
                    f"Failed to build a complete episode for task 3e980e27 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3e980e27-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
