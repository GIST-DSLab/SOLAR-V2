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


# ----------------------------------------------------------------------------
# small DSL helpers (pure python, objects = set of (colour, (i, j)))
# ----------------------------------------------------------------------------
def _nbrs8(ij):
    i, j = ij
    return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}


def _inds(obj):
    return {p for _, p in obj}


def _norm(obj):
    mi = min(p[0] for _, p in obj)
    mj = min(p[1] for _, p in obj)
    return frozenset((v, (p[0] - mi, p[1] - mj)) for v, p in obj)


def _corners(obj):
    rs = [p[0] for _, p in obj]
    cs = [p[1] for _, p in obj]
    return min(rs), min(cs), max(rs), max(cs)


def _hm(obj):
    t, l, b, r = _corners(obj)
    return frozenset((v, (t + b - i, j)) for v, (i, j) in obj)


def _vm(obj):
    t, l, b, r = _corners(obj)
    return frozenset((v, (i, l + r - j)) for v, (i, j) in obj)


def _dm(obj):
    t, l, b, r = _corners(obj)
    return frozenset((v, (j - l + t, i - t + l)) for v, (i, j) in obj)


def _cm(obj):
    return _vm(_dm(_vm(obj)))


def _id(obj):
    return frozenset(obj)


def sample_colors(num_examples=None) -> dict:
    bgc, mainc, markerc = random.sample(range(10), 3)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, mainc=None, markerc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    cols = list(range(10))
    if bgc is None or mainc is None or markerc is None:
        bgc, mainc, markerc = random.sample(cols, 3)
    while True:
        h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        remcols = [c for c in cols if c not in (bgc, mainc, markerc)]
        gi = [[bgc] * w for _ in range(h)]
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if random.choice((True, False)):
            oh, ow = ow, oh
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        ncells = unifint(diff_lb, diff_ub, (4, len(bounds)))
        objs = {random.choice(sorted(bounds))}
        for _ in range(ncells - 1):
            fr = set()
            for p in objs:
                fr |= _nbrs8(p)
            objs.add(random.choice(sorted((bounds - objs) & fr)))
        mi = min(p[0] for p in objs); mj = min(p[1] for p in objs)
        objs = {(p[0] - mi, p[1] - mj) for p in objs}
        oh = max(p[0] for p in objs) + 1
        ow = max(p[1] for p in objs) + 1
        ntocompc = unifint(diff_lb, diff_ub, (1, ncells - 3))
        markercell = random.choice(sorted(objs))
        remobj = objs - {markercell}
        tocompc = set(random.sample(sorted(remobj), ntocompc))
        mainpart = remobj - tocompc
        mainpartobj = {(mainc, p) for p in mainpart}
        markerobj = {(markerc, markercell)}
        tocompcobj = {(random.choice(remcols), p) for p in tocompc}
        obj = frozenset(tocompcobj | mainpartobj | markerobj)
        smobjn = _norm(mainpartobj | markerobj)
        fake = False
        for f in (_dm, _cm, _hm, _vm):
            if f(smobjn) == smobjn and f(obj) != obj:
                fake = True
                break
        if fake:
            continue
        loci = random.randint(0, h - oh)
        locj = random.randint(0, w - ow)
        for v, (i, j) in obj:
            gi[i + loci][j + locj] = v
        plcdi = {(i + loci, j + locj) for _, (i, j) in obj}
        allin = {(i, j) for i in range(h) for j in range(w)}
        nb = set()
        for p in plcdi:
            nb |= _nbrs8(p)
        inds = allin - plcdi - nb
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        succ, tr, maxtr = 0, 0, noccs * 5
        go = [row[:] for row in gi]
        funcs = (_id, _dm, _cm, _hm, _vm)
        while tr < maxtr and succ < noccs:
            tr += 1
            f1 = random.choice(funcs); f2 = random.choice(funcs)
            outobj = _norm(f1(f2(obj)))
            inobj = [(v, p) for v, p in outobj if v in (mainc, markerc)]
            ohh = max(p[0] for _, p in outobj) + 1
            oww = max(p[1] for _, p in outobj) + 1
            cands = [ij for ij in inds if ij[0] <= h - ohh and ij[1] <= w - oww]
            if not cands:
                continue
            loc = random.choice(sorted(cands))
            outpi = {(i + loc[0], j + loc[1]) for _, (i, j) in outobj}
            if outpi.issubset(inds):
                succ += 1
                nb = set()
                for p in outpi:
                    nb |= _nbrs8(p)
                inds = inds - outpi - nb
                for v, (i, j) in inobj:
                    gi[i + loc[0]][j + loc[1]] = v
                for v, (i, j) in outobj:
                    go[i + loc[0]][j + loc[1]] = v
        break
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule (read from I alone):
      - template = largest 8-connected non-bg component (main colour + marker
        + completion colours); partials = all other components, palette P.
      - For each of the mirror compositions of the template, find every place
        in I where its P-coloured sub-shape occurs, and complete that partial
        object by painting the missing completion cells, one Color op per
        completion colour, finishing one partial object before the next.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    submit = [0, 0, O.shape[0] - 1, O.shape[1] - 1]

    bgc = None
    if examples:
        cnt = Counter()
        for ex in examples:
            cnt.update(np.asarray(ex[0]).flatten().tolist())
        cnt.update(I.flatten().tolist())
        bgc = cnt.most_common(1)[0][0]
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

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

    tidx = max(range(len(comps)), key=lambda k: len(comps[k]))
    sub_palette = set()
    for k, cc in enumerate(comps):
        if k != tidx:
            for (r, c) in cc:
                sub_palette.add(int(I[r, c]))

    template = frozenset((int(I[r, c]), (r, c)) for (r, c) in comps[tidx])
    if not any(v in sub_palette for v, _ in template):
        ops.append(34); sels.append(submit)
        return ops, sels

    funcs = [_id, _dm, _cm, _hm, _vm]
    base = _norm(template)
    variants, seen_v = [], set()
    for f1 in funcs:
        for f2 in funcs:
            t = _norm(f1(f2(base)))
            if t not in seen_v:
                seen_v.add(t)
                variants.append(t)

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
