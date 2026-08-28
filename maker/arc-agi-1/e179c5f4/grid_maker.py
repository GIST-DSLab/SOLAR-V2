"""
ARC Task: e179c5f4 (RE-ARC) — LLM-generated grid_maker
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
import math
import random

import numpy as np

from maker.sel_helpers import sel_of

# Discrete structural variants: grid orientation (tall = ray steps row-by-row,
# wide = ray steps column-by-column) x number of emitting corners (1..4).
VARIANTS = [
    {"transpose": False, "numlins": 1},
    {"transpose": True,  "numlins": 2},
    {"transpose": False, "numlins": 4},
    {"transpose": True,  "numlins": 3},
]


def sample_colors(num_examples=None) -> dict:
    # 8 is reserved as the output background marker; the ray colour must be
    # non-zero so it survives ARCLE's object ops (0 counts as "nothing there").
    linc = random.choice([c for c in range(1, 10) if c != 8])
    bgc = random.choice([c for c in range(10) if c != 8 and c != linc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
        if n_ex >= 2 and len({e["transpose"] for e in examples}) == 1:
            alt = [v for v in VARIANTS if v["transpose"] != examples[0]["transpose"]]
            examples[-1] = dict(random.choice(alt))
    plan = examples + [dict(random.choice(examples))]   # test case is one of the shown cases
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = int(math.ceil(a + (b - a) * diff_lb))
    hi = int(math.ceil(a + (b - a) * diff_ub))
    lo = max(a, min(lo, b))
    hi = max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _canonical_path(L, S):
    """Bouncing ray from bottom-left (L-1, 0) going up-right, reflecting off side walls."""
    r, c = L - 1, 0
    dc = 1
    path = [(r, c)]
    while True:
        r -= 1
        c += dc
        if not (0 <= r < L and 0 <= c < S):
            break
        path.append((r, c))
        if c == 0 or c == S - 1:
            dc = -dc
    return path


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, transpose=None, numlins=None) -> dict:
    if transpose is None or numlins is None:
        v = random.choice(VARIANTS)
        if transpose is None:
            transpose = v["transpose"]
        if numlins is None:
            numlins = v["numlins"]

    # L = long axis (the axis the ray steps along), S = short axis (the axis it bounces on)
    maxL = min(30, max_w if transpose else max_h)
    maxS = min(30, max_h if transpose else max_w)
    s_ub = max(2, min(10, maxS, maxL - 1))
    S = _unifint(diff_lb, diff_ub, (2, s_ub))
    S = max(2, min(S, maxS, maxL - 1))
    L = _unifint(diff_lb, diff_ub, (S + 1, max(S + 1, maxL)))

    gi = np.full((L, S), bgc, dtype=int)
    go = gi.copy()
    path = _canonical_path(L, S)
    gi[L - 1, 0] = linc
    for (r, c) in path:
        go[r, c] = linc

    if transpose:
        gi = np.ascontiguousarray(gi.T)
        go = np.ascontiguousarray(go.T)
    if random.random() < 0.5:
        gi = np.ascontiguousarray(np.fliplr(gi))
        go = np.ascontiguousarray(np.fliplr(go))
    if random.random() < 0.5:
        gi = np.ascontiguousarray(np.flipud(gi))
        go = np.ascontiguousarray(np.flipud(go))

    gix, gox = gi.copy(), go.copy()
    if numlins > 1:
        gi[np.flipud(gix) == linc] = linc
        go[np.flipud(gox) == linc] = linc
    if numlins > 2:
        gi[np.fliplr(gix) == linc] = linc
        go[np.fliplr(gox) == linc] = linc
    if numlins > 3:
        gi[np.flipud(np.fliplr(gix)) == linc] = linc
        go[np.flipud(np.fliplr(gox)) == linc] = linc

    go[go == bgc] = 8
    return {"input": gi.tolist(), "output": go.tolist()}


def _trace_ray(H, W, seed):
    """Bouncing ray emitted from a corner marker. It steps along the longer grid
    axis and reflects off the two walls of the shorter axis."""
    r, c = seed
    path = [(r, c)]
    dr = 1 if r == 0 else -1
    dc = 1 if c == 0 else -1
    tall = H > W
    while True:
        r += dr
        c += dc
        if not (0 <= r < H and 0 <= c < W):
            break
        path.append((r, c))
        if tall:
            if c == 0 or c == W - 1:
                dc = -dc
        else:
            if r == 0 or r == H - 1:
                dr = -dr
    return path


def _segments(path):
    """Split a bouncing ray into its straight diagonal strokes."""
    segs, cur, prev = [], [path[0]], None
    for i in range(1, len(path)):
        d = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if prev is None or d == prev:
            cur.append(path[i])
            prev = d
        else:
            segs.append(cur)
            cur = [path[i]]
            prev = d
    segs.append(cur)
    return segs


def _flip_object(grid, cells, axis):
    """ARCLE object-mode Flip: grab `cells`, blank them, mirror them inside their
    bounding box, composite back (zeros are transparent)."""
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    obj = [((r, c), int(grid[r, c])) for (r, c) in cells if grid[r, c] != 0]
    out = grid.copy()
    for (r, c) in cells:
        out[r, c] = 0
    for (r, c), v in obj:
        nr, nc = (r, c0 + c1 - c) if axis == "H" else (r0 + r1 - r, c)
        out[nr, nc] = v
    return out


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    # background = colour of a cell that can never hold a corner marker
    bseed = (1, 0) if H >= 3 else (0, 1)
    bgc = int(I[bseed])
    fg = [int(v) for v in np.unique(I) if int(v) != bgc]
    if fg:
        linc = fg[0]
    else:
        linc = int([v for v in np.unique(O) if int(v) != 8][0])

    seeds = sorted({(r, c) for r in (0, H - 1) for c in (0, W - 1) if I[r, c] == linc})
    rays = {s: _trace_ray(H, W, s) for s in seeds}

    ops, sels = [], []
    cur = I.copy()

    # 1) the background becomes 8 — one connected bgc region, one FloodFill8 seed
    if bgc != 8:
        ops.append(18)
        sels.append(sel_of([bseed]))
        cur[cur == bgc] = 8

    # 2) shoot the ray of the first marker, one straight stroke per bounce
    primary = seeds[0]
    for seg in _segments(rays[primary]):
        cells = [p for p in seg if cur[p] != linc]
        if not cells:
            continue
        ops.append(int(linc))
        sels.append(sel_of(cells))
        for p in cells:
            cur[p] = linc
    drawn = [primary]
    todo = [s for s in seeds if s != primary]

    # 3) every other marker's ray is the mirror image of a ray already on the
    #    grid — reflect that ray onto it, then redraw the ray it came from
    while todo:
        progressed = False
        for tgt in list(todo):
            src = None
            for d in drawn:
                if (d[0] == tgt[0]) != (d[1] == tgt[1]):   # differs on exactly one axis
                    src = d
                    break
            if src is None:
                continue
            todo.remove(tgt)
            drawn.append(tgt)
            progressed = True
            srccells = rays[src]
            tgtcells = rays[tgt]
            if set(tgtcells) == set(srccells):
                continue                                   # mirror-symmetric ray: already drawn
            axis = "V" if src[0] != tgt[0] else "H"        # rows differ -> flip up/down
            nxt = _flip_object(cur, srccells, axis)
            want = cur.copy()
            for p in srccells:
                want[p] = 0
            for p in tgtcells:
                want[p] = linc
            if np.array_equal(nxt, want):
                ops.append(27 if axis == "V" else 26)      # FlipV / FlipH of that ray
                sels.append(sel_of(srccells))
                cur = nxt
                hole = [p for p in srccells if cur[p] != linc]   # footprint the ray left behind
                if hole:
                    ops.append(int(linc))
                    sels.append(sel_of(hole))
                    for p in hole:
                        cur[p] = linc
            else:
                cells = [p for p in tgtcells if cur[p] != linc]
                if cells:
                    ops.append(int(linc))
                    sels.append(sel_of(cells))
                    for p in cells:
                        cur[p] = linc
        if not progressed:
            for tgt in list(todo):
                todo.remove(tgt)
                drawn.append(tgt)
                cells = [p for p in rays[tgt] if cur[p] != linc]
                if cells:
                    ops.append(int(linc))
                    sels.append(sel_of(cells))
                    for p in cells:
                        cur[p] = linc

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])                      # full-grid bbox for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
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
                                f"for task e179c5f4"
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
                    f"Failed to build a complete episode for task e179c5f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e179c5f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
