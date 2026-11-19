"""
ARC Task: 36d67576 (RE-ARC) — LLM-generated grid_maker
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
try:
    from maker.sel_helpers import sel_of
except Exception:
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in sorted(set(cells))]}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, mainc, markerc = random.sample(cols, 3)
    remcols = [c for c in cols if c not in (bgc, mainc, markerc)]
    ncols = random.randint(1, len(remcols))
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc, "ccols": ccols}


def _nbrs(ij):
    i, j = ij
    return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}


def _normo(obj):
    mi = min(p[0] for _, p in obj); mj = min(p[1] for _, p in obj)
    return frozenset((v, (p[0] - mi, p[1] - mj)) for v, p in obj)


def _T(name, obj):
    # 'i' identity, 'd' transpose, 'c' anti-transpose, 'h' flip up/down, 'v' flip left/right
    if name == 'i':
        return _normo(obj)
    if name == 'd':
        return _normo([(v, (j, i)) for v, (i, j) in obj])
    if name == 'c':
        return _normo([(v, (-j, -i)) for v, (i, j) in obj])
    if name == 'h':
        return _normo([(v, (-i, j)) for v, (i, j) in obj])
    return _normo([(v, (i, -j)) for v, (i, j) in obj])


def generate(diff_lb, diff_ub, max_h, max_w, bgc, mainc, markerc, ccols, **kw) -> dict:
    for _ in range(500):
        h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        gi = [[bgc] * w for _ in range(h)]
        oh = _unifint(diff_lb, diff_ub, (2, 5))
        ow = _unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if random.choice((True, False)):
            oh, ow = ow, oh
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        ncells = _unifint(diff_lb, diff_ub, (4, len(bounds)))
        cells = {random.choice(sorted(bounds))}
        for _k in range(ncells - 1):
            fr = set()
            for c in cells:
                fr |= _nbrs(c)
            cand = sorted((bounds - cells) & fr)
            cells.add(random.choice(cand))
        mi = min(r for r, _ in cells); mj = min(c for _, c in cells)
        cells = {(r - mi, c - mj) for r, c in cells}
        oh = max(r for r, _ in cells) + 1; ow = max(c for _, c in cells) + 1
        ntocompc = _unifint(diff_lb, diff_ub, (1, ncells - 3))
        markercell = random.choice(sorted(cells))
        remobj = sorted(cells - {markercell})
        tocompc = set(random.sample(remobj, ntocompc))
        mainpart = (cells - {markercell}) - tocompc
        obj = frozenset([(random.choice(ccols), ij) for ij in tocompc] +
                        [(mainc, ij) for ij in mainpart] + [(markerc, markercell)])
        smobjn = _normo([x for x in obj if x[0] in (mainc, markerc)])
        objn = _normo(obj)
        if any(_T(f, smobjn) == smobjn and _T(f, objn) != objn for f in 'dchv'):
            continue
        loci = random.randint(0, h - oh); locj = random.randint(0, w - ow)
        plcd = {(v, (i + loci, j + locj)) for v, (i, j) in obj}
        for v, (i, j) in plcd:
            gi[i][j] = v
        go = [row[:] for row in gi]
        plcdi = {p for _, p in plcd}
        allidx = {(i, j) for i in range(h) for j in range(w)}
        bad = set(plcdi)
        for p in plcdi:
            bad |= _nbrs(p)
        inds = allidx - bad
        noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        succ = tr = 0
        while tr < noccs * 5 and succ < noccs:
            tr += 1
            outobj = _T(random.choice('idchv'), _T(random.choice('idchv'), obj))
            ooh = max(p[0] for _, p in outobj) + 1; oow = max(p[1] for _, p in outobj) + 1
            cands = sorted(ij for ij in inds if ij[0] <= h - ooh and ij[1] <= w - oow)
            if not cands:
                continue
            li, lj = random.choice(cands)
            outp = {(v, (i + li, j + lj)) for v, (i, j) in outobj}
            outi = {p for _, p in outp}
            if outi <= inds:
                succ += 1
                b2 = set(outi)
                for p in outi:
                    b2 |= _nbrs(p)
                inds = inds - b2
                for v, (i, j) in outp:
                    go[i][j] = v
                    if v in (mainc, markerc):
                        gi[i][j] = v
        if succ == 0:
            continue
        # keep only instances the rule reproduces unambiguously
        Ia = np.array(gi); G = Ia.copy()
        ops, sels = derive_operations(Ia, np.array(go), None)
        for o, s in zip(ops[:-1], sels[:-1]):
            for r, c in s["cells"]:
                G[r, c] = o
        if not (G == np.array(go)).all():
            continue
        return {"input": gi, "output": go}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    """
    Reading taken from the golf entry, where it differs from the maker: the
    template is the component with the MOST DISTINCT COLOURS (main + marker +
    completion colours), not just the largest component. The golf entry
    hard-codes the key colours as {2,4}. Here they are measured instead: the
    palette of all the other (partial) components.
    Where the two readings agree, nothing changes. Each of the 8 mirror/rotation
    variants of the template is stamped at every place where its main+marker key
    appears in I. Stamping goes one site at a time: one Color op per template
    colour, covering the stamp cells that do not already hold that colour on
    the current grid.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    submit = [0, 0, h - 1, w - 1]

    # background: the colour every demonstration input is drawn on
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ei, _eo in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            st = [(r, c)]; seen[r, c] = True; cells = []
            while st:
                a, b = st.pop(); cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bgc:
                            seen[na, nb] = True; st.append((na, nb))
            comps.append(cells)
    if len(comps) < 2:
        return [34], [submit]

    ncol = lambda cc: len({int(I[r, c]) for r, c in cc})
    tidx = max(range(len(comps)), key=lambda k: (ncol(comps[k]), len(comps[k])))
    key_pal = set()
    for k, cc in enumerate(comps):
        if k != tidx:
            key_pal |= {int(I[r, c]) for r, c in cc}
    template = frozenset((int(I[r, c]), (r, c)) for r, c in comps[tidx])
    if not any(v in key_pal for v, _ in template):
        return [34], [submit]

    variants, sv = [], set()
    for f1 in 'idchv':
        for f2 in 'idchv':
            t = _T(f1, _T(f2, template))
            if t not in sv:
                sv.add(t); variants.append(t)

    sites = []
    for var in variants:
        key = [(v, p) for v, p in var if v in key_pal]
        if not key:
            continue
        sr = min(p[0] for _, p in key); sc = min(p[1] for _, p in key)
        akey = [(v, (p[0] - sr, p[1] - sc)) for v, p in key]
        afull = [(v, (p[0] - sr, p[1] - sc)) for v, p in var]
        ma = max(p[0] for _, p in akey); mb = max(p[1] for _, p in akey)
        for i in range(h - ma):
            for j in range(w - mb):
                if all(I[i + a, j + b] == v for v, (a, b) in akey):
                    sites.append(((i, j), afull))
    sites.sort(key=lambda t: t[0])

    ops, sels = [], []
    G = I.copy()
    for (i, j), afull in sites:
        bycol = {}
        for v, (a, b) in afull:
            rr, cc = i + a, j + b
            if 0 <= rr < h and 0 <= cc < w and G[rr, cc] != v:
                bycol.setdefault(v, []).append((rr, cc))
        for v in sorted(bycol):
            ops.append(int(v)); sels.append(sel_of(bycol[v]))
            for rr, cc in bycol[v]:
                G[rr, cc] = v
    ops.append(34); sels.append(submit)
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
                        f"num_examples+1 ({num_examples + 1}) for task 36d67576"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 36d67576"
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
                                f"for task 36d67576"
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
                    f"Failed to build a complete episode for task 36d67576 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"36d67576-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
