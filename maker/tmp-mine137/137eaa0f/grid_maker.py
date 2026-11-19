"""
ARC Task: 137eaa0f (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # the marker dot is the registration point every fragment carries; keep it
    # a colour ARCLE can lift (0 is "empty" to object ops)
    dotc = random.choice([c for c in cols if c != bgc and c != 0])
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if dotc is None:
        dotc = choice([c for c in cols if c != bgc and c != 0])
    h = unifint(diff_lb, diff_ub, (2, 4))
    w = unifint(diff_lb, diff_ub, (2, 4))
    remcols = [c for c in cols if c not in (bgc, dotc)]
    go = canvas(dotc, (h, w))
    inds = totuple(asindices(go))
    loc = choice(inds)
    reminds = remove(loc, inds)
    nc = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    choscols = sample(remcols, nc)
    cd = {c: set() for c in choscols}
    for c in choscols:
        ij = choice(reminds)
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(go, c, idxes)
    lo = min(h, w) * 2
    gih = unifint(diff_lb, diff_ub, (lo, max(lo, max_h)))
    giw = unifint(diff_lb, diff_ub, (lo, max(lo, max_w)))
    objs = tuple(
        normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]})))
        for c in choscols
    )
    maxtr = min(h, w) * 2
    for _attempt in range(200):
        succ = True
        gi = canvas(bgc, (gih, giw))
        inds = asindices(gi)
        for obj in objs:
            oh, ow = shape(obj)
            succ2 = False
            tr = 0
            while tr < maxtr and not succ2:
                loci = randint(0, gih - oh)
                locj = randint(0, giw - ow)
                plcd = shift(obj, (loci, locj))
                tr += 1
                if toindices(plcd).issubset(inds):
                    succ2 = True
            if succ2:
                gi = paint(gi, plcd)
                inds = difference(inds, toindices(plcd))
                inds = difference(inds, mapply(neighbors, toindices(plcd)))
            else:
                succ = False
                break
        if succ:
            return {'input': gi, 'output': go}
        maxtr = int(maxtr * 1.5)
        gih = randint(gih, max(gih, max_h))
        giw = randint(giw, max(giw, max_w))
    raise ValueError("could not place fragments")


def _bg(I, examples):
    # the canvas colour the fragments are scattered over: the input's majority
    # colour (the generator paints the whole canvas with it and adds only a few
    # small fragments)
    I = np.asarray(I, dtype=int)
    return Counter(I.flatten().tolist()).most_common(1)[0][0]


def _assign(I, bgc, dotc):
    """Read the fragments: each dot of colour dotc is one fragment's
    registration mark; every other colour is the body of exactly one fragment.
    The fragments, laid dot-on-dot, must tile one rectangle of at most 4x4
    without overlap.  Returns (dots, {dot: [colours]}, body) or None."""
    dots = [tuple(map(int, p)) for p in zip(*np.where(I == dotc))]
    body = {}
    for c in np.unique(I).tolist():
        if c in (bgc, dotc):
            continue
        body[c] = [tuple(map(int, p)) for p in zip(*np.where(I == c))]
    if not dots or len(body) != len(dots):
        return None
    cols = list(body)

    def dist(cells, d):
        return min(abs(r - d[0]) + abs(q - d[1]) for r, q in cells)

    cand = {}
    for c in cols:
        opts = []
        for d in dots:
            rel = [(r - d[0], q - d[1]) for r, q in body[c]]
            rr = [r for r, _ in rel] + [0]; qq = [q for _, q in rel] + [0]
            if max(rr) - min(rr) < 4 and max(qq) - min(qq) < 4:
                opts.append((dist(body[c], d), d, rel))
        opts.sort()
        if not opts:
            return None
        cand[c] = opts
    cols.sort(key=lambda c: len(cand[c]))
    total = 1 + sum(len(body[c]) for c in cols)
    res = []

    def rec(i, taken, used, assign):
        if res:
            return
        if i == len(cols):
            if len(used) != len(dots):
                return
            rs = [r for r, _ in taken]; qs = [q for _, q in taken]
            if (max(rs) - min(rs) + 1) * (max(qs) - min(qs) + 1) == total:
                res.append(dict(assign))
            return
        c = cols[i]
        for _, d, rel in cand[c]:
            if any(p in taken for p in rel):
                continue
            # each dot carries exactly one fragment colour
            if d in used:
                continue
            nt = taken | set(rel)
            rs = [r for r, _ in nt]; qs = [q for _, q in nt]
            if max(rs) - min(rs) >= 4 or max(qs) - min(qs) >= 4:
                continue
            assign[c] = d
            rec(i + 1, nt, used | {d}, assign)
            del assign[c]

    rec(0, {(0, 0)}, set(), {})
    if not res:
        return None
    frags = {d: [] for d in dots}
    for c, d in res[0].items():
        frags[d].append(c)
    return dots, frags, body


def _read(I, examples):
    I = np.asarray(I, dtype=int)
    bgc = _bg(I, examples)
    present = [c for c in np.unique(I).tolist() if c != bgc]
    # the marker colour: the one colour every demonstrated answer holds exactly
    # once (all the dots collapse onto one cell)
    pref = []
    if examples:
        cand = None
        for _ei, eo in examples:
            vals, cnts = np.unique(np.asarray(eo, dtype=int), return_counts=True)
            once = {int(v) for v, n in zip(vals, cnts) if n == 1}
            cand = once if cand is None else cand & once
        pref = sorted((cand or set()) & set(present))
    rest = sorted(present, key=lambda c: -int((I == c).sum()))
    for dotc in pref + [c for c in rest if c not in pref]:
        got = _assign(I, bgc, dotc)
        if got is not None:
            return bgc, dotc, got
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    read = _read(I, examples)
    if read is None:
        raise ValueError("137eaa0f: could not read the fragments out of the input")
    bgc, dotc, (dots, frags, body) = read

    # each fragment = its dot plus its body, read relative to its own dot
    F = []
    for d in dots:
        bcells = [p for c in frags[d] for p in body[c]]
        cells = [d] + bcells
        F.append({"dot": d, "colors": frags[d], "body": bcells, "cells": cells,
                  "rel": [(r - d[0], q - d[1]) for r, q in cells]})
    allrel = set(p for f in F for p in f["rel"])
    r0 = min(r for r, _ in allrel); r1 = max(r for r, _ in allrel)
    c0 = min(q for _, q in allrel); c1 = max(q for _, q in allrel)
    th, tw = r1 - r0 + 1, c1 - c0 + 1

    owner = {}
    for k, f in enumerate(F):
        for p in f["cells"]:
            owner[p] = k

    def schedule(meet, anchor):
        """Order in which to carry the fragments to `meet` so that nothing
        lands on a fragment still waiting to be picked up."""
        movers = [k for k in range(len(F)) if k != anchor]
        must_before = {k: set() for k in movers}   # k must move after these
        for k in movers:
            dr, dc = meet[0] - F[k]["dot"][0], meet[1] - F[k]["dot"][1]
            for r, q in F[k]["cells"]:
                o = owner.get((r + dr, q + dc))
                if o is not None and o != k:
                    if o == anchor:
                        if (r + dr, q + dc) != meet:
                            return None
                        continue
                    must_before[k].add(o)
        order, done = [], set()
        while len(order) < len(movers):
            ready = [k for k in movers if k not in done and must_before[k] <= done]
            if not ready:
                return None
            k = min(ready, key=lambda k: (abs(F[k]["dot"][0] - meet[0]) +
                                          abs(F[k]["dot"][1] - meet[1]), F[k]["dot"]))
            order.append(k); done.add(k)
        return order, sum(len(v) for v in must_before.values())

    # meeting point: preferably one fragment's own dot (that fragment stays
    # put); otherwise any spot on the canvas where the whole frame fits
    best = None
    metas = [(F[k]["dot"], k) for k in range(len(F))]
    metas += [((mr, mc), None) for mr in range(hi) for mc in range(wi)]
    for meet, anchor in metas:
        tr, tc = meet[0] + r0, meet[1] + c0
        if tr < 0 or tc < 0 or tr + th > hi or tc + tw > wi:
            continue
        s = schedule(meet, anchor)
        if s is None:
            continue
        order, ndeps = s
        travel = sum(abs(meet[0] - F[k]["dot"][0]) + abs(meet[1] - F[k]["dot"][1])
                     for k in order)
        key = (ndeps > 0, anchor is None, travel)
        if best is None or key < best[0]:
            best = (key, meet, anchor, order, (tr, tc))
    if best is None:
        raise ValueError("137eaa0f: no meeting frame available")
    _, meet, anchor, order, (tr, tc) = best

    used = set(np.unique(I).tolist())
    spare = next((c for c in range(1, 10) if c not in used),
                 next(c for c in range(1, 10) if c != bgc))

    ops, sels = [], []
    for k in order:
        f = F[k]
        dr, dc = meet[0] - f["dot"][0], meet[1] - f["dot"][1]
        if dr == 0 and dc == 0:
            continue                      # already sitting on the meeting point
        zb = [p for p in f["body"] if I[p] == 0]
        if zb:
            # 0 is "empty" to ARCLE's object ops: make that body liftable first
            ops.append(spare); sels.append(sel_of(zb))
        # grab the whole fragment once, then keep sliding it with an empty
        # selection until its dot sits on the meeting point
        steps = [20 if dr < 0 else 21] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
        ops.append(steps[0]); sels.append(sel_of(f["cells"]))
        for st in steps[1:]:
            ops.append(st); sels.append(sel_of([]))
        if zb:
            ops.append(0); sels.append(sel_of([(r + dr, q + dc) for r, q in zb]))

    # the fragments now tile one frame around the meeting point: that is the answer
    ops.append(33); sels.append([tr, tc, th - 1, tw - 1])
    ops.append(34); sels.append([0, 0, th - 1, tw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
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
                                f"for task 137eaa0f"
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
                    f"Failed to build a complete episode for task 137eaa0f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"137eaa0f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
