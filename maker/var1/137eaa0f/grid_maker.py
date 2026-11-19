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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """One colour scheme for the whole episode.

    bgc and the marker colour (dotc, the colour every fragment carries a single
    cell of) are the two roles the rule depends on, so both are fixed here.  The
    fragment colours come from one shuffled palette, and each instance's tile
    shape / fragment count is planned up front, so the test instance always
    repeats the shape and colour set of one of the examples.
    """
    bgc = random.choice(list(range(10)))
    dotc = random.choice([c for c in range(10) if c != bgc])
    palette = [c for c in range(10) if c != bgc and c != dotc]
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
        palette = [c for c in range(10) if c != bgc and c != dotc]
    hcap = max(2, min(4, max_h // 2))
    wcap = max(2, min(4, max_w // 2))
    if h is None:
        h = unifint(diff_lb, diff_ub, (2, 4))
    if w is None:
        w = unifint(diff_lb, diff_ub, (2, 4))
    h = max(2, min(h, hcap))
    w = max(2, min(w, wcap))
    # on a small canvas only so many fragments can be laid down clear of one
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
            # fragment/marker pairing has to be readable from the input alone,
            # and the fragments have to be able to converge without one landing
            # on another before that one has moved.
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
#   The grid holds a handful of fragments lying clear of one another.  Each is a
#   few cells of one colour plus a single cell of the marker colour — the one
#   colour that turns up once in every fragment.  Stacked with their marker cells
#   superimposed, the fragments tile one small rectangle exactly: no gap, no
#   overlap.  That rectangle is the answer.
#
#   So: pick one marker cell as the meeting point, slide every other fragment
#   until its own marker lands on it, and crop to the rectangle they now fill.
#
#   Where they meet, in which order they are brought in and by which route are
#   the three choices a person actually makes here, and two people make them
#   differently.  Each is settled below by a feature of the input — how many
#   fragments there are, what colour marks them, how big the rectangle is — so
#   one grid always gives one route, and different grids give different routes.
#   O is never looked at.

_N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _pieces(cells):
    """The 8-connected pieces of a set of cells."""
    todo = set(cells)
    out = []
    while todo:
        seed = todo.pop()
        comp, stack = {seed}, [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in _N8:
                p = (r + dr, c + dc)
                if p in todo:
                    todo.discard(p)
                    comp.add(p)
                    stack.append(p)
        out.append(comp)
    return out


def _match(dots, groups, area, forced):
    """Give every fragment colour the marker cell that belongs to it.

    The fragments were cut out of one rectangle, so laid back on their markers
    they have to cover it exactly — every offset used once, the whole frame
    filled.  That is what picks the pairing out; nearness alone does not, since
    a fragment's own marker can lie further off than a neighbour's.
    """
    colours = sorted(groups, key=lambda c: (-len(groups[c]), c))
    chosen, answer = {}, []

    def near(d, c):
        return min(abs(d[0] - r) + abs(d[1] - cc) for r, cc in groups[c])

    def dfs(i, free, occ, box):
        r0, r1, c0, c1 = box
        if (r1 - r0 + 1) * (c1 - c0 + 1) > area:
            return
        if i == len(colours):
            if len(occ) == area:
                answer.append(dict(chosen))
            return
        c = colours[i]
        cand = [forced[c]] if c in forced else sorted(free, key=lambda d: (near(d, c), d))
        for d in cand:
            if d not in free:
                continue
            offs = [(r - d[0], cc - d[1]) for r, cc in groups[c]]
            if any(o in occ for o in offs):
                continue
            chosen[c] = d
            dfs(i + 1, free - {d},
                occ | set(offs),
                (min(r0, min(o[0] for o in offs)), max(r1, max(o[0] for o in offs)),
                 min(c0, min(o[1] for o in offs)), max(c1, max(o[1] for o in offs))))
            del chosen[c]
            if answer:
                return

    dfs(0, frozenset(dots), {(0, 0)}, (0, 0, 0, 0))
    return answer[0] if answer else None


def _read_fragments(I):
    """Background, marker colour and each fragment's own marker, read from I."""
    H, W = I.shape
    cnt = Counter(int(v) for v in I.flatten().tolist())
    for bgc in sorted(cnt, key=lambda c: (-cnt[c], c)):
        cells = {}
        for r in range(H):
            for c in range(W):
                v = int(I[r, c])
                if v != bgc:
                    cells.setdefault(v, []).append((r, c))
        if len(cells) < 2:
            continue
        total = sum(len(v) for v in cells.values())
        pieces = _pieces([p for v in cells.values() for p in v])
        for dotc in sorted(cells):
            dots = sorted(cells[dotc])
            groups = {c: sorted(v) for c, v in cells.items() if c != dotc}
            # one marker cell per fragment, and the rectangle they tile holds
            # every cell of every fragment with the markers stacked into one
            if len(dots) != len(groups):
                continue
            area = total - len(groups) + 1
            if area < 4:
                continue
            # a fragment lies clear of every other one, so a marker touching a
            # patch of colour is that patch's own marker
            forced = {}
            for comp in pieces:
                ds = [p for p in comp if int(I[p]) == dotc]
                cs = {int(I[p]) for p in comp} - {dotc}
                if len(ds) == 1 and len(cs) == 1:
                    forced[cs.pop()] = ds[0]
            for use in ([forced, {}] if forced else [{}]):
                pairing = _match(dots, groups, area, use)
                if pairing is not None:
                    return bgc, dotc, dots, groups, pairing
    return None


def _assemble(I):
    """The pairing turned into the rectangle the fragments tile."""
    got = _read_fragments(I)
    if got is None:
        return None
    bgc, dotc, dots, groups, pairing = got
    offs = {c: [(r - pairing[c][0], cc - pairing[c][1]) for r, cc in groups[c]]
            for c in groups}
    allo = [(0, 0)] + [o for c in offs for o in offs[c]]
    rmin, rmax = min(o[0] for o in allo), max(o[0] for o in allo)
    cmin, cmax = min(o[1] for o in allo), max(o[1] for o in allo)
    tile = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=int)
    tile[-rmin, -cmin] = dotc
    for c in offs:
        for dr, dc in offs[c]:
            tile[dr - rmin, dc - cmin] = c
    return {"bgc": bgc, "dotc": dotc, "pairing": pairing, "groups": groups,
            "tile": tile, "th": tile.shape[0], "tw": tile.shape[1],
            "rmin": rmin, "cmin": cmin}


def _fragments_at(read, star):
    """Every fragment with the place it ends up if they all meet at `star`."""
    frags = []
    for c in sorted(read["groups"]):
        d = read["pairing"][c]
        dr, dc = star[0] - d[0], star[1] - d[1]
        body = list(read["groups"][c])
        body_dst = [(r + dr, cc + dc) for r, cc in body]
        frags.append({
            "color": c, "dot": d, "body": body, "cells": body + [d],
            "body_dst": body_dst, "dst": body_dst + [star],
            "shift": (dr, dc), "steps": abs(dr) + abs(dc),
        })
    return frags


def _travel_order(frags, tie):
    """The order the fragments are dealt with.

    A fragment may only be laid down once every fragment whose own cells it
    would bury has left; the order is forced that far.  `tie` settles the rest —
    which of the fragments that are free to go, goes now.
    """
    acts = [k for k in range(len(frags)) if frags[k]["steps"]]
    movers = [k for k in acts if frags[k]["color"] != 0]
    preds = {}
    for k in acts:
        # a slide writes the fragment's whole footprint, a marked-out one (see
        # derive_operations) only its body
        w = set(frags[k]["dst"] if frags[k]["color"] != 0 else frags[k]["body_dst"])
        preds[k] = {j for j in movers if j != k and w & set(frags[j]["cells"])}
    order, done = [], set()
    while len(order) < len(acts):
        ready = [k for k in acts if k not in done and preds[k] <= done]
        if not ready:
            return None
        ready.sort(key=lambda k: tie(frags[k]))
        order.append(ready[0])
        done.add(ready[0])
    return order


def _route(dr, dc, mode):
    """One fragment's path: rows first, columns first, or a diagonal."""
    ver = [20 if dr < 0 else 21] * abs(dr)
    hor = [22 if dc > 0 else 23] * abs(dc)
    if mode == 0:
        return ver + hor
    if mode == 1:
        return hor + ver
    out, i, j = [], 0, 0
    while i < len(ver) or j < len(hor):
        if j >= len(hor) or (i < len(ver) and (i + 1) * len(hor) <= (j + 1) * len(ver)):
            out.append(ver[i]); i += 1
        else:
            out.append(hor[j]); j += 1
    return out


def _plan_assembly(I):
    """Where the fragments meet, in what order they travel, by which route."""
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    read = _assemble(I)
    if read is None:
        return None
    th, tw, rmin, cmin = read["th"], read["tw"], read["rmin"], read["cmin"]
    nfrag = len(read["groups"])

    def fits(star):
        return (0 <= star[0] + rmin and star[0] + rmin + th <= H
                and 0 <= star[1] + cmin and star[1] + cmin + tw <= W)

    # which fragment stays put and lets the others come to it
    markers = [read["pairing"][c] for c in sorted(read["groups"])]
    cands = [d for d in markers if fits(d)]
    rule = (nfrag + read["dotc"]) % 4
    if rule == 0:      # the biggest piece is the one worth not moving
        size = {read["pairing"][c]: len(read["groups"][c]) for c in read["groups"]}
        cands.sort(key=lambda d: (-size[d], d))
    elif rule == 1:    # the piece nearest the middle of the grid
        cands.sort(key=lambda d: (abs(2 * d[0] - H + 1) + abs(2 * d[1] - W + 1), d))
    elif rule == 2:    # the first piece you read
        cands.sort()
    else:              # the last piece you read
        cands.sort(reverse=True)
    if not cands:
        # no marker cell has room for the whole rectangle around it, so the
        # fragments have to meet on empty ground instead
        cands = sorted((s for s in ((r, c) for r in range(H) for c in range(W)) if fits(s)),
                       key=lambda s: (sum(abs(s[0] - d[0]) + abs(s[1] - d[1]) for d in markers), s))

    # which of the fragments that are free to go, goes first
    ties = [lambda f: (f["steps"], f["dot"]),          # the nearest one
            lambda f: (-f["steps"], f["dot"]),         # the furthest one
            lambda f: (min(f["cells"]), f["dot"]),     # the highest one on the grid
            lambda f: (min(f["dst"]), f["dot"])]       # fill the frame in reading order
    tie = ties[(nfrag + th) % 4]
    mode = (read["dotc"] + tw) % 3

    for star in cands:
        frags = _fragments_at(read, star)
        order = _travel_order(frags, tie)
        if order is None:
            continue
        return {"read": read, "bgc": read["bgc"], "dotc": read["dotc"],
                "tile": read["tile"], "th": th, "tw": tw, "star": star,
                "top": (star[0] + rmin, star[1] + cmin),
                "frags": frags, "order": order, "mode": mode}
    return None


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    plan = _plan_assembly(I)
    if plan is None:
        raise ValueError("137eaa0f: could not read the fragments out of the input")

    th, tw = plan["th"], plan["tw"]
    tr, tc = plan["top"]
    star = plan["star"]
    cur = I.copy()            # the grid as it stands, step by step
    ops, sels = [], []

    for k in plan["order"]:
        f = plan["frags"][k]
        if f["color"] == 0:
            # 0 reads as empty to every ARCLE object operation, so this fragment
            # cannot be picked up at all: its cells are marked out where they
            # belong instead of being carried there, and only those of them that
            # are not already empty need marking
            late = [q for q in f["body_dst"] if cur[q] != 0]
            if late:
                ops.append(0); sels.append(sel_of(late))
                for q in late:
                    cur[q] = 0
            continue
        # grab the fragment once, then keep sliding it with an empty selection:
        # ARCLE keeps the same piece in hand and restores every cell it glides
        # over, so only the cells it started on are left empty behind it
        steps = _route(f["shift"][0], f["shift"][1], plan["mode"])
        ops.append(steps[0]); sels.append(sel_of(f["cells"]))
        for st in steps[1:]:
            ops.append(st); sels.append(sel_of([]))
        carried = {p: int(cur[p]) for p in f["cells"]}
        for p in f["cells"]:
            cur[p] = 0
        for p, q in zip(f["cells"], f["dst"]):
            if carried[p]:
                cur[q] = carried[p]

    if int(cur[star]) != plan["dotc"]:
        # the fragments met on empty ground and their marker colour is one ARCLE
        # cannot carry, so the meeting cell itself is set by hand
        ops.append(plan["dotc"]); sels.append(sel_of([star]))
        cur[star] = plan["dotc"]

    if not np.array_equal(cur[tr:tr + th, tc:tc + tw], plan["tile"]):
        raise ValueError("137eaa0f: the fragments did not assemble")

    # the fragments now fill one rectangle — that rectangle is the answer
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
