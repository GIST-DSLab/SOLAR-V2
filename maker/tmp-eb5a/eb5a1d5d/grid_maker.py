"""
ARC Task: eb5a1d5d (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = int(round(diff_lb * (b - a) + a))
    hi = int(round(diff_ub * (b - a) + a))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # The rule (collapse every run of identical rows, then of identical
    # columns) is colour-agnostic; ring colours are sampled per instance,
    # exactly as in the original generator.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **color_kwargs) -> dict:
    lim = min(max_h, max_w)
    dmax = max(2, min(10, (lim + 1) // 2))
    d = _unifint(diff_lb, diff_ub, (2, dmax))
    n = 2 * d - 1
    colss = random.sample(range(10), d)
    go = [[0] * n for _ in range(n)]
    for j, cc in enumerate(colss):
        for r in range(j, n - j):
            for c in range(j, n - j):
                go[r][c] = cc
    nvenl = _unifint(diff_lb, diff_ub, (0, max(0, min(30, max_h) - d)))
    nhenl = _unifint(diff_lb, diff_ub, (0, max(0, min(30, max_w) - d)))
    gi = [row[:] for row in go]
    enl = [nvenl, nhenl]
    while (enl[0] > 0 or enl[1] > 0) and len(gi) < max_h and len(gi[0]) < max_w:
        opts = []
        if enl[0] > 0:
            opts.append(0)
        if enl[1] > 0:
            opts.append(1)
        ch = random.choice(opts)
        if ch == 0:
            idx = random.randint(0, len(gi) - 1)
            gi = gi[:idx + 1] + [gi[idx][:]] + gi[idx + 1:]
        else:
            idx = random.randint(0, len(gi[0]) - 1)
            gi = [row[:idx + 1] + [row[idx]] + row[idx + 1:] for row in gi]
        enl[ch] -= 1
    derive_operations(gi)   # raises ValueError if no clean trajectory exists
    return {"input": [list(r) for r in gi], "output": go}


def _simulate(I, ops, sels):
    """Faithful replay of the ops used here (Color, Move with object grab,
    CropGrid, Submit) under ARCLE semantics. Returns (final grid, states)."""
    g = np.asarray(I, dtype=int).copy()
    active = False
    obj, bg, pos = None, None, (0, 0)
    states = []
    for op, sel in zip(ops, sels):
        cells = ([tuple(p) for p in sel["cells"]] if isinstance(sel, dict)
                 else [(r, c) for r in range(sel[0], sel[0] + sel[2] + 1)
                       for c in range(sel[1], sel[1] + sel[3] + 1)])
        h, w = g.shape
        if op <= 9:
            active = False
            for r, c in cells:
                g[r, c] = op
        elif 20 <= op <= 23:
            if cells:
                obj = {(r, c): int(g[r, c]) for r, c in cells}
                bg = g.copy()
                for r, c in cells:
                    bg[r, c] = 0
                pos = (0, 0)
                active = True
            if active:
                dr, dc = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[op]
                pos = (pos[0] + dr, pos[1] + dc)
                g = bg.copy()
                for (r, c), v in obj.items():
                    rr, cc = r + pos[0], c + pos[1]
                    if v and 0 <= rr < h and 0 <= cc < w:
                        g[rr, cc] = v
        elif op == 33:
            active = False
            r0, c0 = min(p[0] for p in cells), min(p[1] for p in cells)
            r1, c1 = max(p[0] for p in cells), max(p[1] for p in cells)
            g = g[r0:r1 + 1, c0:c1 + 1].copy()
        states.append((g.shape, g.tobytes()))
    return g, states


def derive_operations(I, O=None, examples=None):
    # Reading: the figure is the compact nested-ring square with some rows and
    # columns stretched into runs of identical copies. Collapse every run of
    # identical ROWS to one row, then every run of identical COLUMNS to one
    # column (dedupe rows, transpose, dedupe again).
    # Realization: each run is one unit. For a run of k identical lines, the
    # part of the figure beyond it slides back k-1 cells (one grab, then
    # empty-selection Moves) and swallows the k-1 surplus copies. A trailing
    # run has nothing beyond it; the final crop leaves its copies off.
    # Everything is computed from I alone (O is never read).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    G = I.copy()        # what ARCLE shows (colour 0 is transparent to Move)
    T = I.copy()        # the figure as the rule means it
    win = [0, hi - 1, 0, wi - 1]          # top, bot, left, right
    units = []                            # (ops, sels) per meaningful action

    def sim(cells, dr, dc, steps):
        bg = G.copy()
        T2 = T.copy()
        for r, c in cells:
            bg[r, c] = 0
            T2[r, c] = 0
        states = []
        for s in range(1, steps + 1):
            g = bg.copy()
            for r, c in cells:
                rr, cc = r + dr * s, c + dc * s
                if 0 <= rr < hi and 0 <= cc < wi and G[r, c]:
                    g[rr, cc] = G[r, c]
            states.append(g)
        for r, c in cells:
            rr, cc = r + dr * steps, c + dc * steps
            if 0 <= rr < hi and 0 <= cc < wi:
                T2[rr, cc] = T[r, c]
        return states, T2

    def visible(states):
        prev = G
        for g in states:
            if np.array_equal(g, prev):
                return False
            prev = g
        return True

    def slide(cells, dr, dc, steps, states, T2):
        nonlocal G, T
        mop = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        u_ops = [mop] * steps
        u_sels = [sel_of(cells)] + [sel_of([]) for _ in range(steps - 1)]
        units.append((u_ops, u_sels))
        G, T = states[-1], T2

    def runs(axis):
        t, b, l, r = win
        lines = ([tuple(T[i, l:r + 1]) for i in range(t, b + 1)] if axis == 0
                 else [tuple(T[t:b + 1, j]) for j in range(l, r + 1)])
        base = t if axis == 0 else l
        out = []
        for i, ln in enumerate(lines):
            if out and lines[out[-1][0] - base] == ln:
                out[-1][1] = base + i
            else:
                out.append([base + i, base + i])
        return out

    for axis in (0, 1):                    # rows first, then columns
        while True:
            rs = runs(axis)
            t, b, l, r = win
            end = b if axis == 0 else r
            # bottom-most / right-most run with surplus copies, not trailing
            cand = [ru for ru in rs if ru[1] > ru[0] and ru[1] < end]
            if not cand:
                win[1 if axis == 0 else 3] = rs[-1][0]   # crop drops the tail
                break
            a, e = cand[-1]
            k = e - a + 1
            if axis == 0:
                after = [(i, j) for i in range(e + 1, b + 1) for j in range(l, r + 1)]
                before = [(i, j) for i in range(t, a) for j in range(l, r + 1)]
                back, fwd = (-1, 0), (1, 0)
            else:
                after = [(i, j) for i in range(t, b + 1) for j in range(e + 1, r + 1)]
                before = [(i, j) for i in range(t, b + 1) for j in range(l, a)]
                back, fwd = (0, -1), (0, 1)
            # Pull everything beyond the run back over its surplus copies; if
            # ARCLE would not show that slide (colour-0 content is transparent
            # to Move), push the part before the run forward instead.
            st_a, T_a = sim(after, *back, k - 1)
            st_b, T_b = sim(before, *fwd, k - 1) if before else ([], None)
            if visible(st_a):
                slide(after, *back, k - 1, st_a, T_a)
                win[1 if axis == 0 else 3] -= k - 1
            elif before and visible(st_b):
                slide(before, *fwd, k - 1, st_b, T_b)
                win[0 if axis == 0 else 2] += k - 1
            elif all(T[p] == 0 for p in after):
                # the content beyond is all colour 0: the slide has nothing
                # ARCLE can show; the colour-0 repaint below realizes it
                T = T_a
                win[1 if axis == 0 else 3] -= k - 1
            else:
                raise ValueError("no visible slide for this run")

    t, b, l, r = win
    n_h, n_w = b - t + 1, r - l + 1
    target = T[t:b + 1, l:r + 1]
    # Colour 0 is transparent to Move: where the collapsed figure's 0-coloured
    # ring slid over another ring, the old colour still shows. Repaint the
    # whole 0-coloured ring of the collapsed figure.
    ring0 = [(i, j) for i in range(t, b + 1) for j in range(l, r + 1) if T[i, j] == 0]
    tail = []
    if ring0:
        tail.append(([0], [sel_of(ring0)]))
    if (t, b, l, r) != (0, hi - 1, 0, wi - 1):
        tail.append(([33], [[t, l, n_h - 1, n_w - 1]]))   # crop to the figure

    def assemble(us):
        o, s = [], []
        for uo, us_ in us:
            o += uo
            s += us_
        return o + [34], s + [[0, 0, n_h - 1, n_w - 1]]

    def reaches(us):
        o, s = assemble(us)
        g, _ = _simulate(I, o, s)
        return np.array_equal(g, target)

    allu = units + tail
    # drop any whole unit whose effect never survives to the result
    changed = True
    while changed:
        changed = False
        for i in range(len(allu)):
            if reaches(allu[:i] + allu[i + 1:]):
                allu = allu[:i] + allu[i + 1:]
                changed = True
                break
    ops, sels = assemble(allu)
    g, states = _simulate(I, ops, sels)
    if not np.array_equal(g, target):
        raise ValueError("trajectory does not realize the collapse")
    prev = [((hi, wi), I.tobytes())]
    if len(set(prev + states[:-1])) != len(states):
        raise ValueError("trajectory revisits a grid state")
    for k in range(len(ops) - 1):
        g2, _ = _simulate(I, ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:])
        if np.array_equal(g2, target):
            raise ValueError("trajectory contains a removable op")
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
                        f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
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
                                f"for task eb5a1d5d"
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
                    f"Failed to build a complete episode for task eb5a1d5d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"eb5a1d5d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
