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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """One colour scheme for the whole episode.

    bgc and the marker colour (dotc, the colour every fragment carries a copy
    of) are the two roles the rule depends on, so both are fixed here.  The
    fragment colours are drawn from a fixed shuffled palette prefix, and the
    tile shape / fragment count of each instance are planned up front, so the
    test instance always repeats an example's shape and colour set.
    """
    bgc = random.choice(list(range(10)))
    dotc = random.choice([c for c in range(1, 10) if c != bgc])
    palette = [c for c in range(1, 10) if c != bgc and c != dotc]
    random.shuffle(palette)

    n_ex = num_examples if num_examples else 3
    plan = []
    for _ in range(n_ex):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        nc = random.randint(1, min(h * w - 1, 8, len(palette)))
        plan.append({"h": h, "w": w, "nc": nc})
    plan.append(dict(random.choice(plan)))
    return {"bgc": bgc, "dotc": dotc, "palette": palette, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, dotc: int, palette=None,
             h=None, w=None, nc=None) -> dict:
    if palette is None:
        palette = [c for c in range(1, 10) if c != bgc and c != dotc]
    hcap = max(2, min(4, max_h // 2))
    wcap = max(2, min(4, max_w // 2))
    if h is None:
        h = unifint(diff_lb, diff_ub, (2, 4))
    if w is None:
        w = unifint(diff_lb, diff_ub, (2, 4))
    h = max(2, min(h, hcap))
    w = max(2, min(w, wcap))
    # on a small canvas only so many fragments can be laid down apart from one
    # another; on a 30x30 canvas this never binds
    room = max(1, (max_h * max_w) // ((h + 1) * (w + 1)))
    ncmax = min(h * w - 1, 8, len(palette), room)
    if nc is None:
        nc = unifint(diff_lb, diff_ub, (1, ncmax))
    nc = max(1, min(nc, ncmax))

    go = canvas(dotc, (h, w))
    inds = totuple(asindices(go))
    loc = choice(inds)
    reminds = remove(loc, inds)
    choscols = tuple(palette[:nc])
    cd = {c: set() for c in choscols}
    for c in choscols:
        ij = choice(reminds)
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(go, c, idxes)

    lo_h = min(min(h, w) * 2, max_h)
    lo_w = min(min(h, w) * 2, max_w)
    gih = unifint(diff_lb, diff_ub, (lo_h, max_h))
    giw = unifint(diff_lb, diff_ub, (lo_w, max_w))
    objs = tuple(
        normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]})))
        for c in choscols
    )
    maxtr = min(h, w) * 2
    for _attempt in range(40):
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
            # keep only placements this route can actually assemble: the
            # fragment/marker pairing must be readable from the input alone and
            # the fragments must be able to converge without one landing on
            # another before that one has moved.
            arr = np.array([list(row) for row in gi], dtype=int)
            plan = _plan_assembly(arr)
            if plan is not None:
                tile = np.array([list(row) for row in go], dtype=int)
                if plan["tile"].shape == tile.shape and np.array_equal(plan["tile"], tile):
                    return {'input': gi, 'output': go}
        maxtr = min(200, int(maxtr * 1.5) + 1)
        gih = randint(gih, max_h)
        giw = randint(giw, max_w)
    raise ValueError("could not place fragments")


# ── derive_operations ─────────────────────────────────────────────────────────
#
# Reading of the task, taken from the input alone:
#   The grid holds a handful of fragments.  Every fragment is one cell of the
#   marker colour plus a few cells of one other colour; the marker colour is the
#   one that turns up in the most separate pieces.  Laid on top of one another
#   with their marker cells superimposed, the fragments fill one small frame,
#   and that frame is the answer.
#
#   So: slide every fragment until its marker cell sits on one chosen marker
#   cell, then crop the canvas to the frame they now fill.  Which cell they
#   meet at, which way each one travels and how far, are all measured from the
#   input — O is never read.

_NBRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    """Number of 4-connected pieces in a set of cells."""
    todo = set(cells)
    n = 0
    while todo:
        n += 1
        stack = [todo.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in _NBRS4:
                p = (r + dr, c + dc)
                if p in todo:
                    todo.discard(p)
                    stack.append(p)
    return n


def _read_input(I):
    """Background colour and, per non-background colour, its cells and pieces."""
    H, W = I.shape
    cnt = Counter(int(v) for v in I.flatten().tolist())
    bgc = max(sorted(cnt), key=lambda c: cnt[c])
    cells = {}
    for r in range(H):
        for c in range(W):
            v = int(I[r, c])
            if v != bgc:
                cells.setdefault(v, []).append((r, c))
    if not cells:
        return None
    ncomp = {c: _components(v) for c, v in cells.items()}
    return bgc, cells, ncomp


def _pair_with_markers(dots, groups):
    """Give each fragment the marker cell it belongs with.

    A fragment goes with a marker cell nearest to it — that is the relation the
    fragments were drawn with.  When two marker cells are equally near, the tie
    is settled by which choice makes the fragments interlock: the frame they
    span is as small as it can be, and they claim as few cells twice as
    possible.  Two fragments may only claim the same cell when one of them is
    painted 0, the colour that is indistinguishable from an empty cell — a
    visible fragment covered by another would leave its own cell blank, which
    no tile of this task does.  Returns (frame area, cells claimed twice,
    fragment colour -> marker).
    """
    colors = sorted(groups, key=lambda c: (-len(groups[c]), c))
    cands = {}
    for c in colors:
        g = groups[c]
        dist = {d: min(abs(d[0] - r) + abs(d[1] - cc) for r, cc in g)
                for d in dots}
        near = min(dist.values())
        cands[c] = [d for d in sorted(dots) if dist[d] == near]

    best = [None]
    budget = [30000]

    def dfs(i, occupied, box, coll, chosen):
        if budget[0] <= 0:
            return
        budget[0] -= 1
        area = (box[1] - box[0] + 1) * (box[3] - box[2] + 1)
        if best[0] is not None and (area, coll) >= best[0][:2]:
            return
        if i == len(colors):
            best[0] = (area, coll, dict(chosen))
            return
        c = colors[i]
        for d in cands[c]:
            offs = [(r - d[0], cc - d[1]) for r, cc in groups[c]]
            over = [o for o in offs if o in occupied]
            if any(c != 0 and occupied[o] != 0 for o in over):
                continue
            nocc = dict(occupied)
            for o in offs:
                if o not in nocc or c != 0:
                    nocc[o] = c
            nbox = (min(box[0], min(o[0] for o in offs)),
                    max(box[1], max(o[0] for o in offs)),
                    min(box[2], min(o[1] for o in offs)),
                    max(box[3], max(o[1] for o in offs)))
            chosen[c] = d
            dfs(i + 1, nocc, nbox, coll + len(over), chosen)
            del chosen[c]

    dfs(0, {(0, 0): -1}, (0, 0, 0, 0), 0, {})
    return best[0]


def _order_moves(frags, star, dotc, moving_dot, first=()):
    """Order the slides so no fragment is overwritten before it has moved.

    A fragment that lands on cells another fragment still occupies has to wait
    for that one.  Returns an order, or None if no order can satisfy that.
    """
    n = len(frags)
    paint = []
    need = []
    for f in frags:
        p = set(f["dst"])
        if not moving_dot:
            p.discard(star)
        paint.append(p)
        nd = set(f["cells"])
        if not moving_dot:
            nd.discard(f["dot"])
        need.append(nd)
    preds = [set() for _ in range(n)]
    for y, x in first:            # a fragment that is covered has to arrive first
        preds[x].add(y)
    for x in range(n):
        for y in range(n):
            if x == y:
                continue
            inter = paint[x] & need[y]
            if frags[y]["dot"] == star:
                inter = inter - {star}
            if inter:
                preds[x].add(y)
    order = []
    done = set()
    while len(order) < n:
        ready = [k for k in range(n) if k not in done and preds[k] <= done]
        if not ready:
            return None
        ready.sort(key=lambda k: (frags[k]["steps"], k))
        order.append(ready[0])
        done.add(ready[0])
    return order


def _plan_for_marker(I, bgc, dotc, dots, groups, pairing):
    """Work out where the fragments meet, and in what order they may travel."""
    H, W = I.shape
    offs = {c: [(r - pairing[c][0], cc - pairing[c][1]) for r, cc in groups[c]]
            for c in groups}
    allo = [(0, 0)] + [o for c in offs for o in offs[c]]
    rmin = min(o[0] for o in allo)
    rmax = max(o[0] for o in allo)
    cmin = min(o[1] for o in allo)
    cmax = max(o[1] for o in allo)
    th, tw = rmax - rmin + 1, cmax - cmin + 1

    # a cell claimed by two fragments shows the visible one: the 0-painted
    # fragment goes down first and the other covers it
    order_of_paint = sorted(offs, key=lambda c: c != 0)
    tile = np.zeros((th, tw), dtype=int)
    tile[-rmin, -cmin] = dotc
    for c in order_of_paint:
        for (dr, dc) in offs[c]:
            tile[dr - rmin, dc - cmin] = c

    index_of = {c: i for i, c in enumerate(sorted(groups))}
    claims = {}
    for c in offs:
        for o in offs[c]:
            claims.setdefault(o, []).append(c)
    first = [(index_of[0], index_of[c]) for o, cs in claims.items()
             if len(cs) > 1 for c in cs if c != 0]

    moving_dot = (dotc != 0)      # ARCLE cannot pick up a 0-valued cell
    # Where the fragments meet: a marker cell that already has room for the
    # whole tile around it — that fragment then stays put and the others come
    # to it.  Only if no marker cell has the room does any free position do
    # (and then the marker colour has to be one ARCLE can carry).
    best = None
    for cand in ([sorted(dots)] +
                 ([[(r, c) for r in range(H) for c in range(W)]] if moving_dot else [])):
        for star in cand:
            if star[0] + rmin < 0 or star[0] + rmax >= H:
                continue
            if star[1] + cmin < 0 or star[1] + cmax >= W:
                continue
            frags = []
            for c in sorted(groups):
                d = pairing[c]
                dr, dc = star[0] - d[0], star[1] - d[1]
                body = [(r, cc) for r, cc in groups[c]]
                body_dst = [(r + dr, cc + dc) for r, cc in body]
                frags.append({
                    "color": c,
                    "body": body,
                    "body_dst": body_dst,
                    "cells": body + [d],
                    "dot": d,
                    "dst": body_dst + [star],
                    "shift": (dr, dc),
                    "steps": abs(dr) + abs(dc),
                })
            order = _order_moves(frags, star, dotc, moving_dot, first)
            if order is None:
                continue
            cost = sum(f["steps"] for f in frags)
            if best is None or cost < best[0]:
                best = (cost, star, frags, order)
            if cost == 0:
                break
        if best is not None:
            break
    if best is None:
        return None
    cost, star, frags, order = best
    return {
        "bgc": bgc, "dotc": dotc, "tile": tile, "star": star,
        "top": (star[0] + rmin, star[1] + cmin), "th": th, "tw": tw,
        "frags": frags, "order": order, "moving_dot": moving_dot,
    }


def _plan_assembly(I):
    """Everything the trajectory needs, measured from the input alone.

    The marker colour is the one that turns up in the most separate pieces, but
    a colour is only taken as the marker when the reading holds together: one
    marker cell per fragment, and the fragments, stacked on their markers,
    interlocking.  Of the readings that hold together, the one whose frame is
    smallest is the tile.
    """
    I = np.asarray(I, dtype=int)
    info = _read_input(I)
    if info is None:
        return None
    bgc, cells, ncomp = info
    readings = []
    for dotc in cells:
        dots = sorted(cells[dotc])
        groups = {c: sorted(v) for c, v in cells.items() if c != dotc}
        if not groups or len(dots) != len(groups):
            continue
        got = _pair_with_markers(dots, groups)
        if got is None:
            continue
        area, coll, pairing = got
        readings.append((area, coll, -ncomp[dotc], len(cells[dotc]), dotc,
                         dots, groups, pairing))
    readings.sort(key=lambda t: t[:5])
    for area, _cl, _nc, _n, dotc, dots, groups, pairing in readings:
        plan = _plan_for_marker(I, bgc, dotc, dots, groups, pairing)
        if plan is not None:
            return plan
    return None


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    plan = _plan_assembly(I)
    if plan is None:
        raise ValueError("137eaa0f: could not read the fragments out of the input")

    bgc = plan["bgc"]
    th, tw = plan["th"], plan["tw"]
    tr, tc = plan["top"]
    ops, sels = [], []

    # a colour ARCLE cannot pick up: 0 is "empty" to every object operation, so
    # a fragment painted 0 is lifted with a temporary colour and put back to 0
    # once it has arrived.
    used = set(np.unique(I).tolist())
    spare = next((c for c in range(1, 10) if c not in used),
                 next(c for c in range(1, 10) if c != bgc))

    for k in plan["order"]:
        f = plan["frags"][k]
        dr, dc = f["shift"]
        if dr == 0 and dc == 0:
            continue                      # this fragment is already the meeting point
        if f["color"] == 0:
            # 0 is "empty" to every ARCLE object op, so make this fragment
            # visible before lifting it and put it back to 0 once it lands
            ops.append(spare); sels.append(sel_of(f["body"]))
        # grab the whole fragment once, then keep sliding it with an empty
        # selection so ARCLE restores every cell it glides over
        steps = [20 if dr < 0 else 21] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
        ops.append(steps[0]); sels.append(sel_of(f["cells"]))
        for st in steps[1:]:
            ops.append(st); sels.append(sel_of([]))
        if f["color"] == 0:
            ops.append(0); sels.append(sel_of(f["body_dst"]))

    # any cell of the frame no fragment reaches is empty in the answer; on a
    # non-black background it still shows the background and has to be emptied
    sim = I.copy()
    covered = set()
    for k in plan["order"]:
        f = plan["frags"][k]
        covered.update(f["dst"])
        if f["shift"] == (0, 0):
            continue
        for p in f["cells"]:
            sim[p[0], p[1]] = 0
        for p in f["body_dst"]:
            sim[p[0], p[1]] = f["color"]
        if plan["moving_dot"]:
            sim[plan["star"][0], plan["star"][1]] = plan["dotc"]
    empty = [(r, c) for r in range(tr, tr + th) for c in range(tc, tc + tw)
             if (r, c) not in covered and sim[r, c] != 0]
    if empty:
        ops.append(0); sels.append(sel_of(empty))

    # the fragments now fill one frame — that frame is the answer
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
