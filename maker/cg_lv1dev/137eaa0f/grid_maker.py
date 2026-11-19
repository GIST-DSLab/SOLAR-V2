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

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dotc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if dotc is None:
        dotc = choice(remove(bgc, cols))
    lim = min(max_h, max_w)
    h = unifint(diff_lb, diff_ub, (2, max(2, min(4, lim // 2))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(4, lim // 2))))
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
    for _ in range(60):
        succ = True
        gi = canvas(bgc, (gih, giw))
        inds = asindices(gi)
        for obj in objs:
            oh, ow = shape(obj)
            succ2 = False
            tr = 0
            while tr < maxtr and not succ2:
                if gih - oh < 0 or giw - ow < 0:
                    break
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
        maxtr = min(int(maxtr * 1.5) + 1, 500)
        gih = randint(gih, max(gih, max_h))
        giw = randint(giw, max(giw, max_w))
    raise ValueError("placement failed")


def _assign(dots, groups):
    """Pair every colour group with its own dot so that, laid over one another
    on their dots, the groups tile one rectangle (at most 4x4) exactly once."""
    order = sorted(range(len(groups)), key=lambda k: -len(groups[k]))
    best = [None]

    def near(cells):
        return sorted(dots, key=lambda d: (min(abs(d[0] - r) + abs(d[1] - c)
                                                for r, c in cells), d))

    def rec(i, used, taken, pick):
        if best[0] is not None:
            return
        if i == len(order):
            rs = [p[0] for p in taken]; cs = [p[1] for p in taken]
            hh = max(rs) - min(rs) + 1; ww = max(cs) - min(cs) + 1
            if hh * ww == len(taken):
                best[0] = dict(pick)
            return
        k = order[i]
        for d in near(groups[k]):
            if d in used:
                continue
            rel = {(r - d[0], c - d[1]) for r, c in groups[k]}
            if rel & taken:
                continue
            allc = taken | rel
            rs = [p[0] for p in allc]; cs = [p[1] for p in allc]
            if max(rs) - min(rs) >= 4 or max(cs) - min(cs) >= 4:
                continue
            pick[k] = d
            rec(i + 1, used | {d}, allc, pick)
            del pick[k]

    rec(0, frozenset(), {(0, 0)}, {})
    return best[0]


def _read_fragments(I):
    """Background, marker colour, and one fragment per other colour, each
    hanging from its own marker dot."""
    H, W = I.shape
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    fg = sorted(set(cnt) - {bgc})
    for dotc in sorted(fg, key=lambda c: cnt[c]):
        if cnt[dotc] != len(fg) - 1:
            continue
        dots = [(r, c) for r in range(H) for c in range(W) if I[r, c] == dotc]
        cols = [c for c in fg if c != dotc]
        groups = [[(r, c) for r in range(H) for c in range(W) if I[r, c] == col]
                  for col in cols]
        pick = _assign(dots, groups)
        if pick is None:
            continue
        frags = [{"color": cols[k], "cells": groups[k], "dot": pick[k]}
                 for k in range(len(cols))]
        return bgc, dotc, frags
    raise ValueError("137eaa0f: could not pair the colours with their dots")


def _order(movers, grab, dest):
    """Order the slides so no fragment lands on one that has not left yet."""
    left = list(movers)
    out = []
    while left:
        for k in left:
            if not any(set(dest[k]) & set(grab[j]) for j in left if j != k):
                out.append(k)
                left.remove(k)
                break
        else:
            return None
    return out


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    bgc, dotc, frags = _read_fragments(I)

    # laid on top of each other at their dots, the fragments fill one frame;
    # measure it relative to the dot
    rel = [(0, 0)]
    for f in frags:
        dr0, dc0 = f["dot"]
        rel += [(r - dr0, c - dc0) for r, c in f["cells"]]
    r_lo = min(p[0] for p in rel); r_hi = max(p[0] for p in rel)
    c_lo = min(p[1] for p in rel); c_hi = max(p[1] for p in rel)
    fh, fw = r_hi - r_lo + 1, c_hi - c_lo + 1

    # a fragment painted 0 is "nothing" in the overlay (and to ARCLE): it is
    # never lifted; its place in the frame is simply left empty.
    zero = [k for k, f in enumerate(frags) if f["color"] == 0]
    solid = [k for k, f in enumerate(frags) if f["color"] != 0]

    # meeting point: prefer a dot that is already there (the empty fragment's
    # dot first, so nothing of it is left stray), otherwise any free spot
    cands = []
    for k in zero + solid:
        if frags[k]["dot"] not in cands:
            cands.append(frags[k]["dot"])
    rest = [(r, c) for r in range(H) for c in range(W) if (r, c) not in cands]
    rest.sort(key=lambda m: sum(abs(m[0] - frags[k]["dot"][0]) + abs(m[1] - frags[k]["dot"][1])
                                for k in solid))
    cands += rest

    plan = None
    for m in cands:
        tr, tc = m[0] + r_lo, m[1] + c_lo
        if tr < 0 or tc < 0 or tr + fh > H or tc + fw > W:
            continue
        frame = {(r, c) for r in range(tr, tr + fh) for c in range(tc, tc + fw)}
        # a dot of the empty fragment that stays put must not sit in the frame
        if dotc != 0 and any(frags[k]["dot"] != m and frags[k]["dot"] in frame
                             for k in zero):
            continue
        movers = [k for k in solid if frags[k]["dot"] != m]
        grab, dest, shift = {}, {}, {}
        for k in movers:
            f = frags[k]
            cells = list(f["cells"]) + ([f["dot"]] if dotc != 0 else [])
            dr, dc = m[0] - f["dot"][0], m[1] - f["dot"][1]
            grab[k] = cells
            dest[k] = [(r + dr, c + dc) for r, c in cells]
            shift[k] = (dr, dc)
        order = _order(movers, grab, dest)
        if order is None:
            continue
        plan = (tr, tc, order, grab, shift)
        break
    if plan is None:
        raise ValueError("137eaa0f: no place to assemble the frame")
    tr, tc, order, grab, shift = plan

    ops, sels = [], []
    for k in order:
        dr, dc = shift[k]
        steps = [20 if dr < 0 else 21] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
        # grab the fragment (colour cells + its dot) once, then keep sliding it
        ops.append(steps[0]); sels.append(sel_of(grab[k]))
        for st in steps[1:]:
            ops.append(st); sels.append(sel_of([]))

    # track the current grid to know which frame cells still show background
    g = I.copy()
    for k in order:
        dr, dc = shift[k]
        vals = [(r, c, g[r, c]) for r, c in grab[k]]
        for r, c, _ in vals:
            g[r, c] = 0
        for r, c, v in vals:
            g[r + dr, c + dc] = v
    # whatever of the frame no fragment covers is empty in the answer: the
    # background still showing there is emptied
    bg_in_frame = [(r, c) for r in range(tr, tr + fh) for c in range(tc, tc + fw)
                   if g[r, c] == bgc and bgc != 0]
    if bg_in_frame:
        ops.append(0); sels.append(sel_of(bg_in_frame))

    # the assembled frame is the answer
    ops.append(33); sels.append([tr, tc, fh - 1, fw - 1])
    ops.append(34); sels.append([0, 0, fh - 1, fw - 1])
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
