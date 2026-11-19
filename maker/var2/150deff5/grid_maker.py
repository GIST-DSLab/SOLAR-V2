"""
ARC Task: 150deff5 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# reading the picture: which foreground cells form which placed object
# ----------------------------------------------------------------------------
# Two shapes are ever placed: a 2x2 block and a 3-long line (either way round).
# Where two of them touch, the same blob of foreground can be read more than one
# way, so an object is only read off where its own surroundings settle it:
#   block : 2x2 of fg, clear background above / left / upper-right — contact
#           allowed only at its lower-right corner
#   line  : a fg cell with background above, left and right — the head of a
#           3-long line running away from it
# Everything read is set aside and the sweep runs again, so an object that was
# unreadable only because a neighbour touched it becomes readable once that
# neighbour is gone. Four orientations per sweep, five sweeps.
_BLOCK_FG = ((1, 1), (1, 2), (2, 1), (2, 2))
_BLOCK_BG = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0), (3, 0), (3, 1))
_LINE_FG = ((1, 1),)
_LINE_BG = ((0, 1), (1, 0), (1, 2))
_LINE_PAINT = ((1, 1), (2, 1), (3, 1))


def _sweep(grid, idx, fgc, bgc, pat_fg, pat_bg, pat_paint, kind, found):
    """one shape, four orientations; grid/idx come back in the same orientation"""
    ext = pat_fg + pat_bg
    hh = max(r for r, _ in ext)
    ww = max(c for _, c in ext)
    for _turn in range(4):
        H, W = grid.shape
        nh, nw = H - hh, W - ww
        room = np.ones((max(nh, 0), max(nw, 0)), dtype=bool)
        for (r, c) in pat_fg:
            room &= grid[r:r + nh, c:c + nw] == fgc
        for (r, c) in pat_bg:
            room &= grid[r:r + nh, c:c + nw] == bgc
        for (i, j) in np.argwhere(room):
            i, j = int(i), int(j)
            cells = []
            for (r, c) in pat_paint:
                a, b = i + r, j + c
                if 0 <= a < H and 0 <= b < W:
                    grid[a, b] = -1 if kind == "block" else -2
                    if idx[a, b] >= 0:
                        cells.append(int(idx[a, b]))
            if cells:
                found.append((kind, cells))
        grid = np.ascontiguousarray(np.rot90(grid, 3))
        idx = np.ascontiguousarray(np.rot90(idx, 3))
    return grid, idx


def _cover(comp):
    """last resort: tile a blob with blocks and lines, first tiling that fits"""
    index = {}
    for (r, c) in sorted(comp):
        shapes = (("block", ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))),
                  ("line", ((r, c), (r, c + 1), (r, c + 2))),
                  ("line", ((r, c), (r + 1, c), (r + 2, c))))
        for kind, cs in shapes:
            if all(p in comp for p in cs):
                index.setdefault((r, c), []).append((kind, cs))
    dead = set()

    def rec(rem):
        if not rem:
            return []
        key = frozenset(rem)
        if key in dead:
            return None
        cell = min(rem)
        for kind, cs in index.get(cell, ()):
            cset = set(cs)
            if cset <= rem:
                sub = rec(rem - cset)
                if sub is not None:
                    return [(kind, cs)] + sub
        dead.add(key)
        return None

    return rec(set(comp))


def _blobs(fg):
    """4-connected clumps of foreground"""
    out, seen = [], set()
    for cell in sorted(fg):
        if cell in seen:
            continue
        blob, dq = set(), deque([cell])
        seen.add(cell)
        while dq:
            r, c = dq.popleft()
            blob.add((r, c))
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in fg and nb not in seen:
                    seen.add(nb)
                    dq.append(nb)
        out.append(blob)
    return out


def read_objects(grid, mend=True):
    """(objects, bgc, fgc); objects = [(kind, ((r, c), ...)), ...]"""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cnt = Counter(g.flatten().tolist())
    bgc = max(cnt.items(), key=lambda kv: kv[1])[0]   # the canvas colour
    fgc = min(cnt.items(), key=lambda kv: kv[1])[0]   # the colour the shapes wear
    if fgc == bgc:
        return [], bgc, fgc
    # a one-cell background margin, so an object against the edge still has the
    # clear surroundings that make it readable
    pad = np.full((h + 2, w + 2), bgc, dtype=int)
    pad[1:h + 1, 1:w + 1] = g
    idx = np.full((h + 2, w + 2), -1, dtype=int)
    idx[1:h + 1, 1:w + 1] = np.arange(h * w).reshape(h, w)

    found = []
    for _round in range(5):
        before = len(found)
        pad, idx = _sweep(pad, idx, fgc, bgc, _BLOCK_FG, _BLOCK_BG,
                          _BLOCK_FG, "block", found)
        pad, idx = _sweep(pad, idx, fgc, bgc, _LINE_FG, _LINE_BG,
                          _LINE_PAINT, "line", found)
        pad = np.where(pad < 0, bgc, pad)             # set aside what was read
        if len(found) == before:
            break

    objs, taken = [], set()
    for kind, flat in found:
        cells = tuple(sorted((f // w, f % w) for f in flat))
        if any(p in taken for p in cells):
            continue
        taken.update(cells)
        objs.append((kind, cells))
    fg = {(r, c) for r in range(h) for c in range(w) if g[r, c] != bgc}
    rest = fg - taken
    if rest and mend:
        for blob in _blobs(rest):
            sol = _cover(blob)
            if sol:
                objs.extend((k, tuple(sorted(cs))) for k, cs in sol)
    return objs, bgc, fgc


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # generator: cols = interval(0,10,1) minus (2, 8); bgc, fgc = sample(cols, 2)
    # either role may land on colour 0 — the route never carries bgc or fgc
    # through a Copy/Paste, it only paints the two recolour colours, so 0 is safe
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        ies = [round((b - a) * f + a) for f in (lb, ub)]
        lo, hi = min(ies), max(ies)
        lo = max(lo, a)
        hi = min(hi, b)
        if lo > hi:
            lo = hi = a
        return random.randint(lo, hi)

    bo = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
    ro1 = frozenset({(0, 0), (0, 1), (0, 2)})
    ro2 = frozenset({(0, 0), (1, 0), (2, 0)})

    h = unifint(diff_lb, diff_ub, (min(8, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(8, max_w), max_w))
    noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 10)))

    for attempt in range(40):
        gi = [[bgc for _ in range(w)] for _ in range(h)]
        go = [[bgc for _ in range(w)] for _ in range(h)]
        inds = {(i, j) for i in range(h) for j in range(w)}
        boforb, reforb = set(), set()
        for _ in range(max(2, noccs - attempt // 8)):
            if random.choice((0, 1)) == 0:
                obj, col = bo, 8
            else:
                obj, col = random.choice((ro1, ro2)), 2
            oh = max(r for r, _ in obj) + 1
            ow = max(c for _, c in obj) + 1
            forb = boforb if col == 8 else reforb
            cands = []
            for (i, j) in inds:
                if i > h - oh or j > w - ow or (i, j) in forb:
                    continue
                if all((i + dr, j + dc) in inds for (dr, dc) in obj):
                    cands.append((i, j))
            if not cands:
                break
            li, lj = random.choice(sorted(cands))
            if col == 8:
                boforb.update({(li - 2, lj), (li + 2, lj), (li, lj + 2), (li, lj - 2)})
            elif obj == ro1:
                reforb.update({(li, lj + 3), (li, lj - 3)})
            else:
                reforb.update({(li + 1, lj), (li - 1, lj)})
            plcd = {(li + dr, lj + dc) for (dr, dc) in obj}
            for (r, c) in plcd:
                gi[r][c] = fgc
                go[r][c] = col
            inds -= plcd

        # keep only pictures whose objects can still be told apart: where shapes
        # touch, one blob of foreground sometimes reads both ways and no solver,
        # mine included, could know which placement was meant
        objs, _bg, _fg = read_objects(gi, mend=False)
        told = [[bgc for _ in range(w)] for _ in range(h)]
        for kind, cells in objs:
            for (r, c) in cells:
                told[r][c] = 8 if kind == "block" else 2
        if told == go:
            return {"input": gi, "output": go}

    raise ValueError("no unambiguous placement found")


# ----------------------------------------------------------------------------
# 3. derive operations
# ----------------------------------------------------------------------------
def _convention(examples):
    """which colour each shape turns into — read off the worked examples"""
    block, line = Counter(), Counter()
    for pair in (examples or []):
        try:
            ei = np.asarray(pair[0], dtype=int)
            eo = np.asarray(pair[1], dtype=int)
        except Exception:
            continue
        if ei.shape != eo.shape:
            continue
        objs, _bg, _fg = read_objects(ei)
        for kind, cells in objs:
            vals = {int(eo[r, c]) for (r, c) in cells}
            if len(vals) == 1:
                (block if kind == "block" else line)[vals.pop()] += 1
    return (block.most_common(1)[0][0] if block else 8,
            line.most_common(1)[0][0] if line else 2)


def derive_operations(I, O, examples=None):
    """Recolour every placed shape: 2x2 block -> one colour, 3-line -> the other.

    The rule is fixed; the route through it is not. Where to start, which way to
    sweep, whether to flood a tangled clump and put the odd shape back on top,
    whether to stamp a copy of the first lone block over the rest — each is read
    off this input (the big clump's corner, the grid's shape, how many clumps are
    tangled, how the two kinds are split), so one input always gives one route
    and the variety comes from the inputs differing.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    block_col, line_col = _convention(examples)
    objs, bgc, fgc = read_objects(I)
    colour_of = {"block": int(block_col), "line": int(line_col)}

    fg = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc}
    blobs = _blobs(fg)
    home = {}
    for k, blob in enumerate(blobs):
        for p in blob:
            home[p] = k
    units = [[set(blob), []] for blob in blobs]
    for kind, cells in objs:
        units[home[cells[0]]][1].append((kind, cells))
    units = [u for u in units if u[1]]
    if not units:                                    # nothing placed: nothing to do
        return [34], [[0, 0, hi - 1, wi - 1]]

    # ---- where to start, and which way to sweep: at the biggest clump's corner
    big = max(units, key=lambda u: (len(u[0]), -min(u[0])[0], -min(u[0])[1]))
    br = sum(r for r, _ in big[0]) / len(big[0])
    bc = sum(c for _, c in big[0]) / len(big[0])
    from_top = br <= (hi - 1) / 2.0
    from_left = bc <= (wi - 1) / 2.0
    rows_first = hi >= wi                            # sweep along the long side

    def key(cell):
        r, c = cell
        rr = r if from_top else (hi - 1 - r)
        cc = c if from_left else (wi - 1 - c)
        return (rr, cc) if rows_first else (cc, rr)

    def unit_key(u):
        return min(key(p) for p in u[0])

    def obj_key(o):
        return min(key(p) for p in o[1])

    for u in units:
        u[1].sort(key=obj_key)
    units.sort(key=unit_key)

    # ---- tangled clumps: colour the clump through, then the odd shape on top?
    # a habit worth adopting once there is more than one clump to work that way
    base_first = len([u for u in units if len(u[1]) > 1]) >= 2

    # ---- one kind well ahead of the other: get that kind out of the way first
    kinds = Counter(kind for kind, _ in objs)
    n_block, n_line = kinds["block"], kinds["line"]
    lead = None
    if 2 * n_block >= 3 * n_line and n_block > n_line:
        lead = "block"
    elif 2 * n_line >= 3 * n_block and n_line > n_block:
        lead = "line"
    if lead is not None:
        def bulk(u):
            c = Counter()
            for kind, cells in u[1]:
                c[kind] += len(cells)
            return c.most_common(1)[0][0]
        units.sort(key=lambda u: (0 if bulk(u) == lead else 1, unit_key(u)))

    # ---- lone blocks all look alike: colour one and stamp it over the others?
    lone_block = [u for u in units if len(u[1]) == 1 and u[1][0][0] == "block"]
    lone_line = [u for u in units if len(u[1]) == 1 and u[1][0][0] == "line"]
    stamp = (len(lone_block) >= 3 and len(lone_block) >= len(lone_line)
             and colour_of["block"] != 0)   # a 0-coloured block would not travel

    ops, sels = [], []
    stamped = False
    for cells_all, members in units:
        if len(members) == 1:
            kind, cells = members[0]
            if kind == "block" and stamp:
                if not stamped:
                    ops.append(colour_of[kind])
                    sels.append(sel_of(cells))        # this block's own four cells
                    ops.append(29)
                    sels.append(sel_of(cells))        # take a copy of it to stamp with
                    stamped = True
                else:
                    ops.append(30)
                    sels.append(sel_of([min(cells)]))  # stamp it down, top-left first
            else:
                ops.append(colour_of[kind])
                sels.append(sel_of(cells))            # the shape's own cells
            continue

        if base_first:
            weight = Counter()
            for kind, cells in members:
                weight[kind] += len(cells)
            top = min(weight,
                      key=lambda k: (-weight[k],
                                     min(obj_key(m) for m in members if m[0] == k)))
            ops.append(colour_of[top])
            sels.append(sel_of(cells_all))            # the whole clump, laid down flat
            for kind, cells in members:
                if kind != top:
                    ops.append(colour_of[kind])
                    sels.append(sel_of(cells))        # the odd shape back on top
        else:
            for kind, cells in members:
                ops.append(colour_of[kind])
                sels.append(sel_of(cells))            # one shape at a time

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])               # full canvas: submit as is
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
                        f"num_examples+1 ({num_examples + 1}) for task 150deff5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 150deff5"
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
                                f"for task 150deff5"
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
                    f"Failed to build a complete episode for task 150deff5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"150deff5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
