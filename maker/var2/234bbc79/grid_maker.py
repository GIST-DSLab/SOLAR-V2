"""
ARC Task: 234bbc79 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for (r, c) in cells]}


# ----------------------------------------------------------------------------- helpers
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lb = int(a + (b - a) * diff_lb)
    ub = int(a + (b - a) * diff_ub)
    lb = max(a, min(b, lb))
    ub = max(a, min(b, ub))
    if ub < lb:
        lb, ub = ub, lb
    return random.randint(lb, ub)


def _components(I, bgc):
    """4-connected components of non-background cells."""
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def _plan(I):
    """Everything is measured from I alone.

    The snake was cut into segments and the pieces were scattered rightwards.
    Each piece carries marker cells (the marker colour appears exactly once or
    twice in EVERY piece and is the rarest such colour).  Re-assembly: walk the
    pieces left to right; piece k has to sit so that its leftmost marker lands
    one column right of the trailing marker of everything assembled so far; the
    two markers that then touch become ordinary body cells of their own piece.

    The offsets do NOT depend on the order the pieces are actually dragged in:
    every destination is fixed here, from I, before a single operation is
    emitted.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    flat = I.reshape(-1).tolist()
    cnt = Counter(flat)
    ranked = cnt.most_common()
    if not ranked:
        return None
    bgc = ranked[0][0]
    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
        return None
    comps = _components(I, bgc)
    n = len(comps)
    if n < 2:
        return None
    comps.sort(key=lambda cells: min(c for (_, c) in cells))

    # marker colour: present 1 or 2 times in every piece, rarest overall
    cands = []
    for col in sorted(set(flat) - {bgc}):
        ok = True
        for cells in comps:
            k = sum(1 for p in cells if I[p] == col)
            if k not in (1, 2):
                ok = False
                break
        if ok:
            cands.append(col)
    if not cands:
        return None
    dotc = min(cands, key=lambda col: (cnt[col], col))

    pieces = [[(int(r), int(c)) for (r, c) in cells] for cells in comps]
    acc = {(int(r), int(c)): int(I[r, c]) for (r, c) in pieces[0]}
    steps = []
    for k in range(1, n):
        dots = [p for p, v in acc.items() if v == dotc]
        if len(dots) != 1:
            return None
        dpos = dots[0]
        best = None
        bestcols = set()
        for p, v in acc.items():
            if p == dpos:
                continue
            d = abs(p[0] - dpos[0]) + abs(p[1] - dpos[1])
            if best is None or d < best:
                best, bestcols = d, {v}
            elif d == best:
                bestcols.add(v)
        if len(bestcols) != 1:
            return None
        acccol = bestcols.pop()

        cells = pieces[k]
        objdots = sorted([p for p in cells if I[p] == dotc], key=lambda p: (p[1], p[0]))
        if not objdots:
            return None
        if len(objdots) > 1 and objdots[0][1] == objdots[1][1]:
            return None
        head = objdots[0]
        best = None
        bestcols = set()
        for p in cells:
            if I[p] == dotc:
                continue
            d = abs(p[0] - head[0]) + abs(p[1] - head[1])
            if best is None or d < best:
                best, bestcols = d, {int(I[p])}
            elif d == best:
                bestcols.add(int(I[p]))
        if len(bestcols) != 1:
            return None
        headcol = bestcols.pop()

        dr = dpos[0] - head[0]
        dc = dpos[1] - head[1] + 1
        dst = [(r + dr, c + dc) for (r, c) in cells]
        if any(p in acc for p in dst):
            return None
        if any(not (0 <= r < h) or dcol < 0 for (r, dcol) in dst):
            return None

        steps.append({
            "idx": k,
            "cells": cells,
            "dr": int(dr), "dc": int(dc),
            "dpos": dpos, "acccol": int(acccol),
            "headdst": (int(head[0] + dr), int(head[1] + dc)),
            "headcol": int(headcol),
        })
        acc[dpos] = int(acccol)
        for (r, c) in cells:
            acc[(r + dr, c + dc)] = int(I[r, c])
        acc[(head[0] + dr, head[1] + dc)] = int(headcol)

    mnc = min(c for (_, c) in acc)
    mxc = max(c for (_, c) in acc)
    W = mxc - mnc + 1
    out = np.full((h, W), bgc, dtype=int)
    for (r, c), v in acc.items():
        if 0 <= r < h and 0 <= c < W:
            out[r, c] = v
    return {"bgc": int(bgc), "dotc": int(dotc), "W": int(W),
            "pieces": pieces, "steps": steps, "out": out}


# ----------------------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    # 0 is an ordinary colour of the picture here: it may fall to the markers or
    # to a piece's body, and the route has to cope with ARCLE reading it as
    # "nothing there".
    cols = list(range(10))
    bgc = random.choice(cols)
    rest = [c for c in cols if c != bgc]
    dotc = random.choice(rest)
    remcols = [c for c in rest if c != dotc]
    k = random.randint(1, len(remcols))
    pool = random.sample(remcols, k)
    return {"bgc": bgc, "dotc": dotc, "pool": pool}


# ----------------------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, pool=None, **kwargs) -> dict:
    if bgc is None or dotc is None or pool is None:
        ck = sample_colors()
        bgc, dotc, pool = ck["bgc"], ck["dotc"], ck["pool"]
    pool = list(pool)

    hmax = max(5, min(30, int(max_h)))
    wlim = max(8, min(30, int(max_w)))
    wcap = min(20, wlim - 2)
    if wcap < 6:
        wcap = 6

    for _attempt in range(6000):
        h = _unifint(diff_lb, diff_ub, (5, hmax))
        w = _unifint(diff_lb, diff_ub, (6, wcap))

        # --- grow the snake (right / up / down, never revisiting) ---
        spi = random.randint(0, h - 1)
        snek = [(spi, 0)]
        occ = {(spi, 0)}
        while True:
            pi, pj = snek[-1]
            if pj == w - 1 and random.choice((True, False, False)):
                break
            options = []
            if pi < h - 1 and (pi + 1, pj) not in occ:
                options.append((pi + 1, pj))
            if pi > 0 and (pi - 1, pj) not in occ:
                options.append((pi - 1, pj))
            if pj < w - 1:
                options.append((pi, pj + 1))
            if not options:
                break
            loc = random.choice(options)
            snek.append(loc)
            occ.add(loc)

        # --- cut into segments at rightward steps ---
        objs, cobj = [], []
        for idx, cel in enumerate(snek):
            cw = (max(c for (_, c) in cobj) - min(c for (_, c) in cobj)) if cobj else 0
            if len(cobj) > 2 and cw > 0 and idx > 0 and snek[idx - 1] == (cel[0], cel[1] - 1):
                objs.append(cobj)
                cobj = [cel]
            else:
                cobj.append(cel)
        if not objs:
            continue
        objs[-1] = objs[-1] + cobj
        nobjs = len(objs)
        if nobjs < 2:
            continue
        ntokeep = _unifint(diff_lb, diff_ub, (2, nobjs))
        for _ in range(nobjs - ntokeep):
            idx = random.randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx + 1]] + objs[idx + 2:]
        n = len(objs)
        if n < 2:
            continue

        ok = True
        for ob in objs:
            cs = [c for (_, c) in ob]
            if max(cs) - min(cs) < 1 or ob[0][1] != min(cs) or ob[-1][1] != max(cs):
                ok = False
                break
        if not ok:
            continue

        sw = max(c for (_, c) in snek) + 1
        minw = sw + (n - 1)
        wtop = min(30, int(max_w))
        if minw > wtop or sw > wtop or h > int(max_h):
            continue
        fullw = _unifint(diff_lb, diff_ub, (minw, wtop))
        extra = fullw - minw
        spacings = [1] * (n - 1)
        for _ in range(extra):
            i = random.randint(0, n - 1)
            if i < n - 1:
                spacings[i] += 1

        # --- colours per segment ---
        ncols = _unifint(diff_lb, diff_ub, (1, len(pool)))
        ccols = random.sample(pool, ncols)
        segcols = [random.choice(ccols) for _ in objs]

        # --- scatter the segments over the input canvas ---
        gi = [[bgc] * fullw for _ in range(h)]
        nextcol = 0
        placed_fail = False
        for i, ob in enumerate(objs):
            colr = segcols[i]
            cells = {}
            for j, p in enumerate(ob):
                if j == 0 and i > 0:
                    cells[p] = dotc
                elif j == len(ob) - 1 and i < n - 1:
                    cells[p] = dotc
                else:
                    cells[p] = colr
            rs = [r for (r, _) in cells]
            cs = [c for (_, c) in cells]
            minr, minc = min(rs), min(cs)
            hh, ww = max(rs) - minr + 1, max(cs) - minc + 1
            if i == 0:
                ulr, ulc = minr, 0
            else:
                if h - hh < 0:
                    placed_fail = True
                    break
                ulr = random.randint(0, h - hh)
                ulc = nextcol + spacings[i - 1]
            if ulc + ww > fullw:
                placed_fail = True
                break
            nextcol = ulc + ww
            for (pr, pc), v in cells.items():
                gi[ulr + pr - minr][ulc + pc - minc] = v
        if placed_fail:
            continue

        # --- reference reassembly (the original snake, markers absorbed) ---
        go = [[bgc] * sw for _ in range(h)]
        for i, ob in enumerate(objs):
            for (pr, pc) in ob:
                go[pr][pc] = segcols[i]

        gi_np = np.array(gi, dtype=int)
        go_np = np.array(go, dtype=int)
        if Counter(gi_np.reshape(-1).tolist()).most_common(1)[0][0] != bgc:
            continue
        pl = _plan(gi_np)
        if pl is None or pl["out"].shape != go_np.shape:
            continue
        if not np.array_equal(pl["out"], go_np):
            continue

        return {
            "input": tuple(tuple(int(v) for v in row) for row in gi),
            "output": tuple(tuple(int(v) for v in row) for row in go),
        }

    raise RuntimeError("234bbc79: could not generate an instance")


# ----------------------------------------------------------------------------- trajectory
def _style(I, plan):
    """The four choices a solver faces here, each settled by something in I.

    Nothing below changes WHAT is done — only the route through it — and every
    switch reads a feature of the input grid, so one instance always gives one
    route and two different instances usually give different ones.
    """
    pieces = plan["pieces"]
    n = len(pieces)
    sizes = [len(p) for p in pieces]
    heaviest = max(range(n), key=lambda k: (sizes[k], -k))
    if heaviest == 0:
        order_mode = "chain"      # the anchor is the big one: just walk along the chain
    elif heaviest == n - 1:
        order_mode = "far"        # the big one is at the far end: fetch from that end
    else:
        order_mode = "heavy"      # otherwise deal with the biggest piece first
    long_leg_first = (plan["dotc"] % 2 == 0)       # a colour: which leg of the trip goes first
    seams_at_end = (n % 2 == 0)                    # a count: fix each seam, or all of them at the end
    top0 = min(r for (r, _) in pieces[0])
    sweep_at_end = (2 * top0 >= I.shape[0])        # a position: sweep scraps now, or once at the end
    return order_mode, long_leg_first, seams_at_end, sweep_at_end


def _drag_order(plan, order_mode):
    """An order for dragging the pieces home.

    A piece may only be grabbed while it still sits untouched in its input
    place, so piece i has to go before piece j whenever j's destination lies on
    top of i.  Left-to-right always satisfies that; so do other orders, and
    which one is used is decided by `order_mode`.
    """
    pieces = plan["pieces"]
    movers = [st["idx"] for st in plan["steps"]]
    src = {k: set(pieces[k]) for k in movers}
    dst = {}
    for st in plan["steps"]:
        dr, dc = st["dr"], st["dc"]
        dst[st["idx"]] = {(r + dr, c + dc) for (r, c) in st["cells"]}
    need = {j: {i for i in movers if i != j and (dst[j] & src[i])} for j in movers}

    done, order = set(), []
    while len(order) < len(movers):
        ready = [k for k in movers if k not in done and need[k] <= done]
        if not ready:                      # cannot happen: left-to-right is always legal
            ready = [min(k for k in movers if k not in done)]
        if order_mode == "far":
            pick = max(ready)
        elif order_mode == "heavy":
            pick = max(ready, key=lambda k: (len(src[k]), -k))
        else:
            pick = min(ready)
        order.append(pick)
        done.add(pick)
    return order


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    plan = _plan(I)                      # measured from I only
    if plan is None:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    bgc = plan["bgc"]
    dotc = plan["dotc"]
    W = plan["W"]
    pieces = plan["pieces"]
    step = {st["idx"]: st for st in plan["steps"]}

    # every destination, fixed up front from I
    dst_of = {}
    for k, st in step.items():
        dr, dc = st["dr"], st["dc"]
        dst_of[k] = [(r + dr, c + dc) for (r, c) in st["cells"]]
    assembled = set(pieces[0])           # cells the finished snake occupies
    for k in dst_of:
        assembled |= set(dst_of[k])

    order_mode, long_leg_first, seams_at_end, sweep_at_end = _style(I, plan)
    order = _drag_order(plan, order_mode)

    grid = I.copy()                      # what the grid looks like after the ops so far
    history = {grid.tobytes()}           # every picture the route has already shown

    def paint(colour, cells):
        """Colour these cells, leaving out any that already hold that colour."""
        todo = sorted({p for p in cells if int(grid[p]) != int(colour)})
        if not todo:
            return
        ops.append(int(colour))
        sels.append(sel_of(todo))
        for p in todo:
            grid[p] = int(colour)
        history.add(grid.tobytes())

    placed = {0}
    seam_done = set()
    seam_queue = []                      # (cell, colour) when the seams are left to the end
    scraps = []                          # vacated footprints when they are swept at the end

    def close_seam(k):
        """The two markers that now touch turn into body cells of their piece."""
        seam_done.add(k)
        st = step[k]
        pairs = [(st["dpos"], st["acccol"]), (st["headdst"], st["headcol"])]
        if seams_at_end:
            seam_queue.extend(pairs)
        elif pairs[0][1] == pairs[1][1]:
            paint(pairs[0][1], [pairs[0][0], pairs[1][0]])
        else:
            for cell, colour in pairs:
                paint(colour, [cell])

    for k in order:
        st = step[k]
        src = st["cells"]
        dr, dc = st["dr"], st["dc"]
        # What the drag would actually carry into the finished picture: cells
        # ARCLE can pick up (non-zero) that keep their colour there (a marker
        # does not — the seam gives it the body colour of its piece).  When a
        # piece's body is colour 0 there is nothing to pick up at all, so the
        # piece is laid down at its destination instead of dragged.
        carry = [p for p in src if int(grid[p]) != 0 and int(I[p]) != dotc]

        if carry and (dr or dc):
            # one grab, then empty selections: ARCLE keeps the piece held and
            # re-lays every cell it glides over, so the path needs no repair
            behind = grid.copy()         # the snapshot ARCLE takes at the grab
            for p in src:
                behind[p] = 0
            held = [(p, int(grid[p])) for p in src if int(grid[p]) != 0]
            vstep = (1 if dr > 0 else -1, 0)
            hstep = (0, 1 if dc > 0 else -1)
            vop = 21 if dr > 0 else 20
            hop = 22 if dc > 0 else 23
            V, Hn = abs(dr), abs(dc)

            frames = {}

            def at(i, jj, _behind=behind, _held=held, _frames=frames):
                if (i, jj) not in _frames:
                    f = _behind.copy()
                    for (r, c), v in _held:
                        f[r + i * vstep[0], c + jj * hstep[1]] = v
                    _frames[(i, jj)] = f
                return _frames[(i, jj)]

            # Down-then-across and across-then-down land the piece in the same
            # place, so the way round is free.  Pick one on which every step
            # shows: a step that slides the piece onto cells of its own colour
            # can leave the picture exactly as it was.
            vertical_is_longer = V >= Hn
            prefer = ["v", "h"] if (vertical_is_longer == long_leg_first) else ["h", "v"]

            def walk(i, jj, used, acc, budget):
                if budget[0] <= 0:
                    return None
                budget[0] -= 1
                if i == V and jj == Hn:
                    return list(acc)
                for a in prefer:
                    ni, nj = (i + 1, jj) if a == "v" else (i, jj + 1)
                    if ni > V or nj > Hn:
                        continue
                    mark = at(ni, nj).tobytes()
                    if mark in used:
                        continue
                    used.add(mark)
                    acc.append(a)
                    got = walk(ni, nj, used, acc, budget)
                    if got is not None:
                        return got
                    acc.pop()
                    used.discard(mark)
                return None

            route = walk(0, 0, set(history) | {at(0, 0).tobytes()}, [], [40000])
            if route is None:            # no way round on which every step shows
                route = ["v"] * V + ["h"] * Hn
                if prefer[0] == "h":
                    route.reverse()

            cur = list(src)
            i = jj = 0
            grabbed = False
            for a in route:
                if a == "v":
                    sr, sc = vstep
                    ops.append(vop)
                    i += 1
                else:
                    sr, sc = hstep
                    ops.append(hop)
                    jj += 1
                # the first step grabs the piece, the rest keep hold of it
                sels.append(sel_of([]) if grabbed else sel_of(cur))
                grabbed = True
                cur = [(r + sr, c + sc) for (r, c) in cur]
                history.add(at(i, jj).tobytes())
            grid = at(V, Hn).copy()

        # cells of this piece that are colour 0 never travel with the grab
        # (ARCLE reads 0 as nothing there), so lay them down where they land.
        # Markers are left out: they are not body colour, the seam gives them theirs.
        zeros = [(p[0] + dr, p[1] + dc) for p in src
                 if int(I[p]) == 0 and int(I[p]) != dotc]
        if zeros:
            paint(0, zeros)

        # Tidy the place the piece came from — but only the part of it that is
        # still inside the finished picture (columns 0..W-1); the rest is cut
        # away by the crop at the end, so scrubbing it would leave no trace.
        vacated = sorted(p for p in set(src) - assembled if p[1] < W)
        if vacated:
            if sweep_at_end:
                scraps.extend(vacated)
            else:
                paint(bgc, vacated)

        placed.add(k)
        for seam in (k, k + 1):          # a seam can be closed once both its pieces are home
            if seam in step and seam not in seam_done and seam - 1 in placed and seam in placed:
                close_seam(seam)

    if scraps:
        paint(bgc, scraps)

    if seam_queue:                       # one pass per colour, over every seam at once
        colours = []
        for _cell, colour in seam_queue:
            if colour not in colours:
                colours.append(colour)
        for colour in colours:
            paint(colour, [cell for cell, col in seam_queue if col == colour])

    if W != wi:
        # full rectangle: the assembled snake occupies columns 0..W-1
        ops.append(33)
        sels.append([0, 0, hi - 1, W - 1])

    ops.append(34)
    sels.append([0, 0, hi - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 234bbc79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 234bbc79"
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
                                f"for task 234bbc79"
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
                    f"Failed to build a complete episode for task 234bbc79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"234bbc79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
