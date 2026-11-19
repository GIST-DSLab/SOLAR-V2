"""
ARC Task: 746b3537 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

# The one structural branch of this task: the stripes either run across the grid
# (the answer is a column) or down it (the answer is a row).
VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    # The rule is purely structural — "collapse repeated stripes" — so no colour
    # role has to be pinned.  One palette per episode keeps the instances looking
    # like a single puzzle.  0 stays in the palette on purpose: here it is an
    # ordinary stripe colour, not a background.
    palette = random.sample(range(10), random.randint(4, 10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None, transposed=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if transposed is None:
        transposed = random.choice([True, False])
    if not palette or len(palette) < 2:
        palette = list(range(10))

    # before the optional transpose the grid is (stripes + duplicates) x thickness
    len_cap = max(3, min(30, max_w if transposed else max_h))
    thick_cap = max(1, min(30, max_h if transposed else max_w))

    h = unifint(diff_lb, diff_ub, (2, max(2, min(15, len_cap - 1))))
    h = max(2, min(h, len_cap - 1))
    w = unifint(diff_lb, diff_ub, (1, thick_cap))
    w = max(1, min(w, thick_cap))

    def draw(pool):
        out, lastc = [], -1
        for _ in range(h):
            c = random.choice([x for x in pool if x != lastc])
            out.append(c)
            lastc = c
        return out

    nonblack = [c for c in palette if c != 0]
    for _ in range(20):
        cols = draw(palette)
        # A picture striped in black and one other colour alone is degenerate: the
        # answer is then fixed by which stripes are black, so any route reaches it.
        if 0 not in cols or len(set(cols)) > 2:
            break
    else:
        cols = draw(nonblack if len(nonblack) >= 2 else palette)

    go = [[c] for c in cols]
    gi = [[c] * w for c in cols]

    numinserts = unifint(diff_lb, diff_ub, (1, max(1, len_cap - h)))
    numinserts = max(1, min(numinserts, len_cap - h))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + [list(gi[loc])] + gi[loc + 1:]

    if transposed:                      # dmirror of a stack of solid lines == transpose
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {"input": gi, "output": go}


def derive_operations(I, O):
    """The picture is a stack of solid stripes in which some stripes are repeated.
    The answer keeps one line per stripe, so the repeats have to be squeezed out and
    a single line's worth of the stack kept.

    Everything below is read off I alone: the orientation, the runs of repeated
    lines, and the route taken through them.  The route (slide the rest of the stack
    over a repeat, or carry each stripe to its place one at a time; work from the
    top/left end or from the bottom/right end; deal with the repeats front-first,
    back-first, or biggest-first) is picked from the picture itself — its two end
    colours, its number of stripes, its number of repeated lines — so one picture
    always gets one route, and different pictures get different ones.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # ── read the stripes ────────────────────────────────────────────────────
    # Either every row is solid (stripes lie across, answer is a column) or every
    # column is; consecutive stripes always differ, so exactly one holds.
    row_mode = all(len(set(I[r].tolist())) == 1 for r in range(hi))
    seq = [int(v) for v in (I[:, 0] if row_mode else I[0, :])]
    thick = wi if row_mode else hi
    m = len(seq)

    runs = []                                   # [colour, repeat count] per stripe
    for v in seq:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    vals = [v for v, _ in runs]
    lens0 = [L for _, L in runs]
    n = len(runs)
    start0, acc = [], 0
    for L in lens0:
        start0.append(acc)
        acc += L
    dup = m - n                                 # how many lines are repeats

    def cell(i, t):
        return (i, t) if row_mode else (t, i)

    def band(i0, i1):
        """every cell of lines i0..i1 of the stack (a solid slab, full thickness)"""
        return [cell(i, t) for i in range(i0, i1 + 1) for t in range(thick)]

    MOVE_LOW = 20 if row_mode else 23           # towards the first stripe
    MOVE_HIGH = 21 if row_mode else 22          # towards the last stripe

    expected = np.array([[v] for v in vals]) if row_mode else np.array([vals])

    # ── the two ways of squeezing a repeat out ──────────────────────────────
    # A Move carries only the non-zero cells it grabs, so a stripe of colour 0
    # never travels: black stripes are left where they lie by both plans and are
    # put down with one Color0 on the line the answer keeps.
    def plan_blocks(to_low, order):
        """Slide the whole rest of the stack over the repeated lines, repeat by
        repeat.  `order` says which repeat is dealt with first."""
        todo = [i for i in range(n) if lens0[i] > 1]
        if order == "back":
            todo = todo[::-1]
        elif order == "big":
            todo = sorted(todo, key=lambda i: (-lens0[i], i))
        moves = []
        if to_low:
            pos, live = list(start0), m
            for i in todo:
                ex = lens0[i] - 1
                s = pos[i] + lens0[i]           # first line under the repeat
                if s > live - 1:                # nothing under it: the copies just go
                    live = pos[i] + 1
                elif all(vals[j] == 0 for j in range(i + 1, n)):
                    pass                        # a slab of black cannot be dragged
                else:
                    moves.append((MOVE_LOW, band(s, live - 1), ex))
                    for j in range(i + 1, n):
                        pos[j] -= ex
                    live -= ex
            keep = list(range(n))
        else:
            end, top = [start0[i] + lens0[i] - 1 for i in range(n)], 0
            for i in todo:
                ex = lens0[i] - 1
                e = end[i] - lens0[i]           # last line above the repeat
                if e < top:                     # nothing above it
                    top = end[i]
                elif all(vals[j] == 0 for j in range(i)):
                    pass                        # a slab of black cannot be dragged
                else:
                    moves.append((MOVE_HIGH, band(top, e), ex))
                    for j in range(i):
                        end[j] += ex
                    top += ex
            keep = [m - n + i for i in range(n)]
        return moves, keep

    def plan_stripes(to_low):
        """Carry each stripe on its own to the line it must end on."""
        moves = []
        if to_low:
            keep = list(range(n))
            for k in range(n):                  # front stripe first, or it would
                src = start0[k]                 # be overwritten before its turn
                if src > k and vals[k] != 0:
                    moves.append((MOVE_LOW, band(src, src), src - k))
        else:
            keep = [m - n + k for k in range(n)]
            for k in range(n - 1, -1, -1):
                src = start0[k] + lens0[k] - 1
                if m - n + k > src and vals[k] != 0:
                    moves.append((MOVE_HIGH, band(src, src), m - n + k - src))
        return moves, keep

    ROUTES = [("stripes", True, ""), ("stripes", False, ""),
              ("blocks", True, "front"), ("blocks", True, "back"),
              ("blocks", True, "big"), ("blocks", False, "front"),
              ("blocks", False, "back"), ("blocks", False, "big")]

    def make_plan(route):
        kind, to_low, order = route
        return plan_stripes(to_low) if kind == "stripes" else plan_blocks(to_low, order)

    # Route chosen from the picture: how many repeats there are, how many stripes,
    # and the colours of the two end stripes.
    want_stripes = (dup + n) % 2 == 1
    to_low = (vals[0] + vals[-1]) % 2 == 0
    order = ["front", "back", "big"][(vals[0] + n) % 3]
    if want_stripes and sum(st for _, _, st in plan_stripes(to_low)[0]) > 30:
        want_stripes = False                    # nobody nudges a stack 30 times over
    first = ("stripes", to_low, "") if want_stripes else ("blocks", to_low, order)
    cut = ROUTES.index(first)                   # the rest, should that one degenerate
    order_of_try = ROUTES[cut:] + ROUTES[:cut]

    # ── simulate exactly what ARCLE does, to keep the route honest ──────────
    def simulate(ops, sels):
        g = I.copy()
        dim = [hi, wi]
        obj = bg = None
        pos = [0, 0]
        active = False
        states = [(dim[0], dim[1], g.tobytes())]
        for op, sel in zip(ops, sels):
            cells = [(int(r), int(c)) for r, c in sel["cells"]]
            if op < 10:
                for r, c in cells:
                    g[r, c] = op
                active = False
            elif 20 <= op <= 23:
                if cells:
                    rs = [r for r, _ in cells]
                    cs = [c for _, c in cells]
                    x0, y0 = min(rs), min(cs)
                    obj = np.zeros((max(rs) - x0 + 1, max(cs) - y0 + 1), dtype=int)
                    for r, c in cells:
                        obj[r - x0, c - y0] = g[r, c]
                    bg = g.copy()
                    for r, c in cells:
                        bg[r, c] = 0
                    pos, active = [x0, y0], True
                if not active:
                    states.append((dim[0], dim[1], g.tobytes()))
                    continue
                dr, dc = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[op]
                pos = [pos[0] + dr, pos[1] + dc]
                g[:, :] = bg
                for i in range(obj.shape[0]):
                    for j in range(obj.shape[1]):
                        r, c = pos[0] + i, pos[1] + j
                        if 0 <= r < dim[0] and 0 <= c < dim[1] and obj[i, j] > 0:
                            g[r, c] = obj[i, j]
            elif op == 33:
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                x0, y0 = min(rs), min(cs)
                hh, ww = max(rs) - x0 + 1, max(cs) - y0 + 1
                want = set(cells)
                patch = np.zeros((hh, ww), dtype=int)
                for i in range(hh):
                    for j in range(ww):
                        if (x0 + i, y0 + j) in want and g[x0 + i, y0 + j] > 0:
                            patch[i, j] = g[x0 + i, y0 + j]
                g[:, :] = 0
                g[:hh, :ww] = patch
                dim = [hh, ww]
                active = False
            states.append((dim[0], dim[1], g.tobytes()))
        return g[:dim[0], :dim[1]].copy(), states

    def build(moves, keep):
        ops, sels = [], []
        for op, cells, steps in moves:
            ops.append(op)
            sels.append(sel_of(cells))            # this grabs the slab...
            for _ in range(steps - 1):
                ops.append(op)
                sels.append(sel_of([]))           # ...and it keeps gliding
        ops.append(33)
        sels.append(sel_of([cell(i, 0) for i in keep]))
        return ops, sels

    def finish(moves, keep):
        """Full candidate route + the Color0 that puts the black stripes down."""
        ops, sels = build(moves, keep)
        grid, states = simulate(ops, sels)
        if grid.shape != expected.shape:
            return None
        back = [(r, c) for r in range(expected.shape[0]) for c in range(expected.shape[1])
                if expected[r, c] == 0 and grid[r, c] != 0]
        for r in range(expected.shape[0]):
            for c in range(expected.shape[1]):
                if expected[r, c] != 0 and grid[r, c] != expected[r, c]:
                    return None                    # this route cannot reach the answer
        if back:
            ops.append(0)                          # the black stripes, where they land
            sels.append(sel_of(back))
            grid, states = simulate(ops, sels)
            if not np.array_equal(grid, expected):
                return None
        return ops, sels, states

    def every_op_shows(states):
        return all(states[i] != states[i + 1] for i in range(len(states) - 1))

    def nothing_droppable(ops, sels):
        body = len(ops)
        if body * body > 4200:                     # long route: test the rare shapes
            probe = [i for i, op in enumerate(ops)
                     if op < 10 or op == 33 or i == 0 or ops[i - 1] != op]
        else:
            probe = range(body)
        for i in probe:
            grid, _ = simulate(ops[:i] + ops[i + 1:], sels[:i] + sels[i + 1:])
            if grid.shape == expected.shape and np.array_equal(grid, expected):
                return False
        return True

    best = None
    for route in order_of_try:
        moves, keep = make_plan(route)
        if route[0] == "stripes" and sum(st for _, _, st in moves) > 30:
            continue                               # nobody nudges a stack 30 times
        got = finish(moves, keep)
        if got is None:
            continue
        ops, sels, states = got
        if best is None:
            best = (ops, sels)
        if every_op_shows(states) and nothing_droppable(ops, sels):
            best = (ops, sels)                     # every op of this one earns its place
            break

    ops, sels = best
    ops.append(34)
    sels.append(sel_of([cell(i, 0) for i in range(n)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 746b3537"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 746b3537"
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
                                f"for task 746b3537"
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
                    f"Failed to build a complete episode for task 746b3537 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"746b3537-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
