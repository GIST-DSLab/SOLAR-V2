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
import random
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def _tf(g, name):
    a = np.asarray(g)
    return {"identity": a, "dmirror": a.T, "cmirror": np.rot90(a, 2).T,
            "vmirror": a[:, ::-1], "hmirror": a[::-1, :], "rot90": np.rot90(a, 3),
            "rot180": np.rot90(a, 2), "rot270": np.rot90(a, 1)}[name]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    def unif(lo, hi):
        if hi < lo:
            hi = lo
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + round(d * (hi - lo))))
    if bgc is None or linc is None:
        bgc, linc = random.sample([c for c in range(10) if c != 8], 2)
    m = min(max_h, max_w, 30)          # transforms may transpose, so bound both sides
    w = unif(2, min(10, m - 1))
    h = unif(w + 1, m)
    gi = np.full((h, w), bgc, dtype=int)
    gi[h - 1, 0] = linc
    go = gi.copy()
    r, c, d = h - 1, 0, 1
    while True:
        r, c = r - 1, c + d
        if c == w - 1 or c == 0:
            d *= -1
        if r < 0:
            break
        go[r, c] = linc
    names = ["identity", "dmirror", "cmirror", "vmirror", "hmirror", "rot90", "rot180", "rot270"]
    for fn in random.sample(names, random.choice((1, 2))):
        gi = _tf(gi, fn).copy()
        go = _tf(go, fn).copy()
    gix, gox = gi.copy(), go.copy()
    n = unif(1, 4)
    variants = [lambda a: a[::-1, :], lambda a: a[:, ::-1], lambda a: a[::-1, ::-1]]
    for k in range(n - 1):
        f = variants[k]
        gi[f(gix) == linc] = linc
        go[f(gox) == linc] = linc
    go[go == bgc] = 8
    return {"input": tuple(tuple(int(v) for v in row) for row in gi),
            "output": tuple(tuple(int(v) for v in row) for row in go)}


def _trace_ray(H, W, s):
    """Bouncing diagonal from corner s, one cell per step along the long axis."""
    r0, c0 = s
    if H > W:
        L, S, a0, b0 = H, W, r0, c0
    else:
        L, S, a0, b0 = W, H, c0, r0
    P = 2 * (S - 1)
    da = -1 if a0 == L - 1 else 1
    cells = []
    for k in range(L):
        p = k % P
        if p > S - 1:
            p = P - p
        b = p if b0 == 0 else S - 1 - p
        a = a0 + da * k
        cells.append((a, b) if H > W else (b, a))
    return cells


def _segments(ray, H, W):
    """Split the ray into straight strokes; a bounce cell closes one stroke and opens the next."""
    short = 1 if H > W else 0
    S = W if H > W else H
    segs, cur = [], []
    for i, p in enumerate(ray):
        cur.append(p)
        if i > 0 and p[short] in (0, S - 1) and i < len(ray) - 1:
            segs.append(cur)
            cur = [p]
    if cur:
        segs.append(cur)
    return segs


def _flip_object(cur, cells, axis):
    rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    out = cur.copy()
    vals = {p: int(cur[p]) for p in cells}
    for p in cells:
        out[p] = 0
    for (r, c), v in vals.items():
        if v == 0:
            continue
        if axis == "V":
            out[r0 + r1 - r, c] = v
        else:
            out[r, c0 + c1 - c] = v
    return out


def derive_operations(I, O, examples=None):
    # Reading shared with the golf entry: the whole answer is ONE bouncing diagonal
    # line launched from a corner marker, one cell per step along the long axis,
    # reflecting off the short sides; background becomes 8. The golf entry only
    # handles a single marker; extra markers here are mirror images of that line,
    # so the unit is still "one ray" -- the two readings agree, nothing to recarve.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    # a non-corner cell can never hold a marker -> it is background
    bseed = (1, 0) if H >= 3 else (0, 1)
    bgc = int(I[bseed])
    fg = [int(v) for v in np.unique(I) if int(v) != bgc]
    linc = fg[0]

    seeds = sorted({(r, c) for r in (0, H - 1) for c in (0, W - 1) if I[r, c] == linc})
    rays = {s: _trace_ray(H, W, s) for s in seeds}

    ops, sels = [], []
    cur = I.copy()

    # 1) background -> 8 (one connected region, one FloodFill8)
    if bgc != 8:
        ops.append(18); sels.append(sel_of([bseed]))
        cur[cur == bgc] = 8

    # 2) the first marker's ray, one straight stroke per bounce
    primary = seeds[0]
    for seg in _segments(rays[primary], H, W):
        cells = [p for p in seg if cur[p] != linc]
        if not cells:
            continue
        ops.append(linc); sels.append(sel_of(cells))
        for p in cells:
            cur[p] = linc
    drawn = [primary]
    todo = [s for s in seeds if s != primary]

    # 3) every other marker's ray is the mirror of an already drawn ray:
    #    flip that ray onto the new marker, then redraw the ray it came from
    while todo:
        progressed = False
        for tgt in list(todo):
            src = next((d for d in drawn if (d[0] == tgt[0]) != (d[1] == tgt[1])), None)
            if src is None:
                continue
            todo.remove(tgt); drawn.append(tgt); progressed = True
            srccells, tgtcells = rays[src], rays[tgt]
            if set(tgtcells) == set(srccells):
                continue
            axis = "V" if src[0] != tgt[0] else "H"
            want = cur.copy()
            for p in srccells:
                want[p] = 0
            for p in tgtcells:
                want[p] = linc
            nxt = _flip_object(cur, srccells, axis) if linc != 0 else None
            if nxt is not None and np.array_equal(nxt, want):
                ops.append(27 if axis == "V" else 26); sels.append(sel_of(srccells))
                cur = nxt
                hole = [p for p in srccells if cur[p] != linc]
                if hole:
                    ops.append(linc); sels.append(sel_of(hole))
                    for p in hole:
                        cur[p] = linc
            else:
                cells = [p for p in tgtcells if cur[p] != linc]
                if cells:
                    ops.append(linc); sels.append(sel_of(cells))
                    for p in cells:
                        cur[p] = linc
        if not progressed:
            for tgt in list(todo):
                todo.remove(tgt); drawn.append(tgt)
                cells = [p for p in rays[tgt] if cur[p] != linc]
                if cells:
                    ops.append(linc); sels.append(sel_of(cells))
                    for p in cells:
                        cur[p] = linc

    ops.append(34); sels.append([0, 0, H - 1, W - 1])   # full-grid bbox for Submit
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
