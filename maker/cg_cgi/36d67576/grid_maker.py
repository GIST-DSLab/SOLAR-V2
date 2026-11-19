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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

# ---------------------------------------------------------------------------
# Comparing the maker's reading with the code-golf reading: they agree.
#   * The thing treated as one unit is the same in both: the single complete
#     template, i.e. the connected group that has the completion colours.
#   * The match is the same: a partial is the template's main+marker cells,
#     found under any of the 8 mirror/rotation symmetries.
#   * The output is the same: each match is finished by laying the transformed
#     template's completion cells next to it.
# The golf entry rotates the grid instead of the template and re-stamps cells
# that are already correct. Neither changes which cells are painted. So the
# grouping stays: each occurrence is finished as a whole object, one Color op
# per completion colour. Each selection is "the cells the transformed template
# copy lands on", which can be read from I alone.
# ---------------------------------------------------------------------------


def _unif(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    lo = a + (b - a) * min(lb, ub)
    hi = a + (b - a) * max(lb, ub)
    v = int(round(random.uniform(lo, hi)))
    return max(a, min(b, v))


def _nbrs(ij):
    i, j = ij
    return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}


def _norm(obj):
    mi = min(p[0] for _, p in obj)
    mj = min(p[1] for _, p in obj)
    return frozenset((v, (p[0] - mi, p[1] - mj)) for v, p in obj)


def _m_id(o): return _norm(o)
def _m_d(o): return _norm(frozenset((v, (j, i)) for v, (i, j) in o))
def _m_c(o): return _norm(frozenset((v, (-j, -i)) for v, (i, j) in o))
def _m_h(o): return _norm(frozenset((v, (-i, j)) for v, (i, j) in o))
def _m_v(o): return _norm(frozenset((v, (i, -j)) for v, (i, j) in o))


_MIRRORS = [_m_id, _m_d, _m_c, _m_h, _m_v]


def sample_colors(num_examples=None) -> dict:
    bgc, mainc, markerc = random.sample(range(10), 3)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, mainc, markerc) -> dict:
    remcols = [c for c in range(10) if c not in (bgc, mainc, markerc)]
    while True:
        h = _unif(diff_lb, diff_ub, (min(10, max_h), max_h))
        w = _unif(diff_lb, diff_ub, (min(10, max_w), max_w))
        oh = _unif(diff_lb, diff_ub, (2, 5))
        ow = _unif(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if random.choice((True, False)):
            oh, ow = ow, oh
        if oh > h or ow > w:
            continue
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        ncells = _unif(diff_lb, diff_ub, (4, len(bounds)))
        cells = {random.choice(sorted(bounds))}
        for _ in range(ncells - 1):
            fr = set()
            for ij in cells:
                fr |= _nbrs(ij)
            cand = sorted((bounds - cells) & fr)
            cells.add(random.choice(cand))
        mi = min(i for i, _ in cells); mj = min(j for _, j in cells)
        cells = {(i - mi, j - mj) for i, j in cells}
        oh = max(i for i, _ in cells) + 1
        ow = max(j for _, j in cells) + 1
        ntocompc = _unif(diff_lb, diff_ub, (1, ncells - 3))
        markercell = random.choice(sorted(cells))
        rem = sorted(cells - {markercell})
        tocompc = set(random.sample(rem, ntocompc))
        mainpart = (cells - {markercell}) - tocompc
        mainobj = {(mainc, ij) for ij in mainpart}
        markobj = {(markerc, markercell)}
        compobj = {(random.choice(remcols), ij) for ij in tocompc}
        obj = frozenset(compobj | mainobj | markobj)
        smobjn = _norm(frozenset(mainobj | markobj))
        objn = _norm(obj)
        fake = False
        for f in (_m_d, _m_c, _m_h, _m_v):
            if f(smobjn) == smobjn and f(objn) != objn:
                fake = True
                break
        if fake:
            continue

        gi = [[bgc] * w for _ in range(h)]
        loci = random.randint(0, h - oh)
        locj = random.randint(0, w - ow)
        plcdi = set()
        for v, (i, j) in obj:
            gi[i + loci][j + locj] = v
            plcdi.add((i + loci, j + locj))
        nb = set()
        for ij in plcdi:
            nb |= _nbrs(ij)
        inds = {(i, j) for i in range(h) for j in range(w)} - plcdi - nb
        noccs = _unif(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        go = [row[:] for row in gi]
        succ, tr, maxtr = 0, 0, noccs * 5
        while tr < maxtr and succ < noccs:
            tr += 1
            f1 = random.choice(_MIRRORS); f2 = random.choice(_MIRRORS)
            outobj = f1(f2(objn))
            inobj = [(v, p) for v, p in outobj if v in (mainc, markerc)]
            ohh = max(p[0] for _, p in outobj) + 1
            oww = max(p[1] for _, p in outobj) + 1
            cands = sorted(ij for ij in inds if ij[0] <= h - ohh and ij[1] <= w - oww)
            if not cands:
                continue
            li, lj = random.choice(cands)
            opi = {(i + li, j + lj) for _, (i, j) in outobj}
            if opi.issubset(inds):
                succ += 1
                nb = set()
                for ij in opi:
                    nb |= _nbrs(ij)
                inds = (inds - opi) - nb
                for v, (i, j) in inobj:
                    gi[i + li][j + lj] = v
                for v, (i, j) in outobj:
                    go[i + li][j + lj] = v
        return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    submit = [0, 0, O.shape[0] - 1, O.shape[1] - 1]

    # background: the canvas colour, which is the dominant colour in these sparse grids
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected non-background components
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bgc:
                            seen[na, nb] = True
                            stack.append((na, nb))
            comps.append(cells)

    if len(comps) < 2:
        ops.append(34); sels.append(submit)
        return ops, sels

    # template = the complete (largest) object; partial palette = colours of the rest
    tidx = max(range(len(comps)), key=lambda k: len(comps[k]))
    sub_palette = set()
    for k, cc in enumerate(comps):
        if k == tidx:
            continue
        for (r, c) in cc:
            sub_palette.add(int(I[r, c]))
    template = frozenset((int(I[r, c]), (r, c)) for (r, c) in comps[tidx])
    if not any(v in sub_palette for v, _ in template):
        ops.append(34); sels.append(submit)
        return ops, sels

    base = _norm(template)
    variants, seen_v = [], set()
    for f1 in _MIRRORS:
        for f2 in _MIRRORS:
            t = f1(f2(base))
            if t not in seen_v:
                seen_v.add(t)
                variants.append(t)

    # every place in I where a transformed template's main+marker part occurs
    placements = []
    for var in variants:
        sub = [(v, p) for v, p in var if v in sub_palette]
        if not sub:
            continue
        sr = min(p[0] for _, p in sub)
        sc = min(p[1] for _, p in sub)
        a_sub = [(v, (p[0] - sr, p[1] - sc)) for v, p in sub]
        a_full = [(v, (p[0] - sr, p[1] - sc)) for v, p in var]
        maxa = max(p[0] for _, p in a_sub)
        maxb = max(p[1] for _, p in a_sub)
        for i in range(0, h - maxa):
            for j in range(0, w - maxb):
                if all(I[i + a, j + b] == v for v, (a, b) in a_sub):
                    placements.append(((i, j), a_full))
    placements.sort(key=lambda t: t[0])

    # finish each occurrence as a whole object: one Color op per completion colour,
    # on the cells where the transformed template copy lands
    G = I.copy()
    for (i, j), a_full in placements:
        bycol = {}
        for v, (a, b) in a_full:
            rr, cc = i + a, j + b
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if G[rr, cc] == v:
                continue
            bycol.setdefault(v, []).append((rr, cc))
        for v in sorted(bycol):
            cells = sorted(bycol[v])
            ops.append(int(v))
            sels.append(sel_of(cells))
            for (rr, cc) in cells:
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
