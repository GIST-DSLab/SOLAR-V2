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
    remcols = [c for c in cols if c not in (bgc, dotc)]
    h = unifint(diff_lb, diff_ub, (2, 4))
    w = unifint(diff_lb, diff_ub, (2, 4))
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
    gih = unifint(diff_lb, diff_ub, (lo, max_h))
    giw = unifint(diff_lb, diff_ub, (lo, max_w))
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
        maxtr = int(maxtr * 1.5)
        gih = randint(gih, max_h)
        giw = randint(giw, max_w)
    raise ValueError("could not place fragments")


# ── reading the input ─────────────────────────────────────────────────────────

def _background(I, examples):
    # the canvas colour the fragments are scattered on; every demonstration
    # input shares it, so take it from them when they are available
    if examples:
        votes = Counter()
        for ei, _ in examples:
            ei = np.asarray(ei, dtype=int)
            votes[Counter(ei.flatten().tolist()).most_common(1)[0][0]] += 1
        c = votes.most_common(1)[0][0]
        if (I == c).any():
            return int(c)
    return int(Counter(I.flatten().tolist()).most_common(1)[0][0])


def _dot_candidates(G, bgc):
    # the marker colour: one marker per fragment, so it occurs exactly as often
    # as there are other (fragment) colours, and every marker is a lone cell
    cnt = Counter(int(v) for v in G.flatten().tolist() if v != bgc)
    out = []
    for c, n in cnt.items():
        if n != len(cnt) - 1:
            continue
        cells = list(zip(*np.where(G == c)))
        lone = all(not any(abs(r - r2) + abs(cc - c2) == 1 for r2, c2 in cells)
                   for r, cc in cells)
        out.append((0 if lone else 1, c))
    return [c for _, c in sorted(out)]


def _dot_colours(I, bgc, examples):
    cands = _dot_candidates(I, bgc)
    if examples and len(cands) > 1:
        # the marker colour is fixed for the episode: the demonstrations agree on it
        for ei, _ in examples:
            ei = np.asarray(ei, dtype=int)
            ec = _dot_candidates(ei, _background(ei, None))
            keep = [c for c in cands if c in ec]
            if keep:
                cands = keep + [c for c in cands if c not in keep]
    return cands


def _fragments(I, bgc, dotc):
    hi, wi = I.shape
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]
    colours = sorted({int(v) for v in I.flatten().tolist()} - {bgc, dotc})
    cells = {k: [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == k]
             for k in colours}
    # every fragment carries exactly one marker, and seen from their markers
    # all fragments interlock into one solid rectangle: search for the pairing
    # that achieves this, trying nearer markers first
    pref = {k: sorted(dots, key=lambda d: min(abs(r - d[0]) + abs(c - d[1])
                                              for r, c in cells[k]))
            for k in colours}
    colours.sort(key=lambda k: len(cells[k]), reverse=True)
    pick = {}

    def rel(k, d):
        return {(r - d[0], c - d[1]) for r, c in cells[k]}

    def dfs(i, used, taken):
        if i == len(colours):
            allc = used | {(0, 0)}
            rs = [a for a, _ in allc]; cs = [b for _, b in allc]
            area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            return area == len(allc)
        k = colours[i]
        for d in pref[k]:
            if d in taken:
                continue
            rk = rel(k, d)
            if (0, 0) in rk or rk & used:
                continue
            pick[k] = d
            if dfs(i + 1, used | rk, taken | {d}):
                return True
        return False

    if len(dots) != len(colours) or not dfs(0, set(), set()):
        raise ValueError("137eaa0f: fragments and markers do not pair up")
    return [{"color": k, "dot": pick[k], "body": cells[k],
             "cells": cells[k] + [pick[k]]} for k in sorted(colours)]


def _plan(I, bgc, dotc):
    hi, wi = I.shape
    frags = _fragments(I, bgc, dotc)
    # every fragment, seen from its own marker, is a piece of one frame
    rel = [(r - f["dot"][0], c - f["dot"][1]) for f in frags for r, c in f["cells"]]
    r0 = min(a for a, _ in rel); c0 = min(b for _, b in rel)
    th = max(a for a, _ in rel) - r0 + 1
    tw = max(b for _, b in rel) - c0 + 1
    lr, lc = -r0, -c0                      # marker position inside the frame
    best = None
    for tr in range(0, hi - th + 1):
        for tc in range(0, wi - tw + 1):
            mr, mc = tr + lr, tc + lc       # where the markers meet
            info = []
            for f in frags:
                dr, dc = mr - f["dot"][0], mc - f["dot"][1]
                dst = [(r + dr, c + dc) for r, c in f["cells"]]
                info.append((dr, dc, dst))
            # a fragment must leave before another lands on top of it
            n = len(frags)
            moving = [i for i in range(n) if (info[i][0], info[i][1]) != (0, 0)]
            before = {i: set() for i in moving}
            for g in moving:
                dg = set(info[g][2])
                for f in moving:
                    if f != g and dg & set(frags[f]["cells"]):
                        before[g].add(f)
            order, done, okk = [], set(), True
            rest = sorted(moving, key=lambda i: abs(info[i][0]) + abs(info[i][1]))
            while rest:
                nxt = next((i for i in rest if before[i] <= done), None)
                if nxt is None:
                    okk = False
                    break
                order.append(nxt); done.add(nxt); rest.remove(nxt)
            if not okk:
                continue
            still = n - len(moving)
            zero_moves = sum(1 for i in moving if frags[i]["color"] == 0)
            travel = sum(abs(info[i][0]) + abs(info[i][1]) for i in moving)
            key = (zero_moves, -still, travel, tr, tc)
            if best is None or key < best[0]:
                best = (key, tr, tc, order, info)
    if best is None:
        raise ValueError("137eaa0f: no frame position to assemble in")
    _, tr, tc, order, info = best
    return frags, (tr, tc, th, tw), order, info


def _move_object(grid, cells, dr, dc):
    """Mirror of ARCLE's grab-once slide: the grabbed cells are lifted from a
    snapshot and pasted (non-zero only) at their shifted place."""
    g = grid.copy()
    snap = grid.copy()
    vals = [(r, c, grid[r, c]) for r, c in cells]
    for r, c, _ in vals:
        snap[r, c] = 0
    g[:, :] = snap
    for r, c, v in vals:
        if v != 0:
            nr, nc = r + dr, c + dc
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                g[nr, nc] = v
    return g


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    bgc = _background(I, examples)
    plan = None
    for dotc in _dot_colours(I, bgc, examples):
        try:
            plan = _plan(I, bgc, dotc)
            break
        except ValueError:
            continue
    if plan is None:
        raise ValueError("137eaa0f: could not read the fragments out of the input")
    frags, (tr, tc, th, tw), order, info = plan

    ops, sels = [], []
    cur = I.copy()
    for i in order:
        f = frags[i]
        dr, dc = info[i][0], info[i][1]
        # 0 is "empty" to ARCLE's object ops: only the fragment's coloured cells
        # can be carried; its 0-coloured cells are painted where they land
        visible = [p for p in f["cells"] if cur[p[0], p[1]] != 0]
        zero = [p for p in f["cells"] if I[p[0], p[1]] == 0]
        if f["color"] == 0 and len(frags) > 1:
            # only the marker of a black fragment is visible, and the other
            # fragments already bring a marker to the meeting point
            visible = []
        steps = [20 if dr < 0 else 21] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
        if visible and steps:
            # grab the fragment once, then keep sliding it with empty selections
            ops.append(steps[0]); sels.append(sel_of(visible))
            for st in steps[1:]:
                ops.append(st); sels.append(sel_of([]))
            cur = _move_object(cur, visible, dr, dc)
        if zero:
            landing = [(r + dr, c + dc) for r, c in zero]
            todo = [p for p in landing if cur[p[0], p[1]] != 0]
            if todo:
                ops.append(0); sels.append(sel_of(landing))
                for r, c in landing:
                    cur[r, c] = 0

    # the fragments now fill one frame around the shared marker — keep only it
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
