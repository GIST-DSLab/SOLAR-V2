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
    pieces left to right; slide piece k so that its leftmost marker sits one
    column right of the accumulated snake's trailing marker; the two touching
    markers then become ordinary body cells of their own piece.
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

    acc = {(int(r), int(c)): int(I[r, c]) for (r, c) in comps[0]}
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

        cells = [(int(r), int(c)) for (r, c) in comps[k]]
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
    return {"bgc": int(bgc), "dotc": int(dotc), "W": int(W), "steps": steps, "out": out}


# ----------------------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    rest = [c for c in range(1, 10) if c != bgc]   # 0 never used as a foreground colour
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
    W = plan["W"]
    temp = 1 if bgc != 1 else 2          # stand-in colour for literal-0 cells

    for st in steps_iter(plan["steps"]):
        src = st["cells"]
        dr, dc = st["dr"], st["dc"]

        seq = []
        if dr:
            seq += [(21 if dr > 0 else 20, (1 if dr > 0 else -1, 0))] * abs(dr)
        if dc:
            seq += [(22 if dc > 0 else 23, (0, 1 if dc > 0 else -1))] * abs(dc)

        zeros = [p for p in src if int(I[p]) == 0]
        if seq and zeros:
            # ARCLE's object grab treats 0 as "nothing"; make those cells visible
            ops.append(temp)
            sels.append(sel_of(zeros))

        cur = list(src)
        first = True
        for op, (sr, sc) in seq:
            ops.append(op)
            sels.append(sel_of(cur) if first else sel_of([]))
            first = False
            cur = [(r + sr, c + sc) for (r, c) in cur]

        if seq:
            hole = sorted(set(src) - set(cur))
            if bgc != 0 and hole:
                ops.append(bgc)
                sels.append(sel_of(hole))
            if zeros:
                ops.append(0)
                sels.append(sel_of(sorted((r + dr, c + dc) for (r, c) in zeros)))

        # the two markers that now touch become body cells of their own piece
        ops.append(st["acccol"])
        sels.append(sel_of([st["dpos"]]))
        ops.append(st["headcol"])
        sels.append(sel_of([st["headdst"]]))

    if (hi, W) != (hi, wi):
        ops.append(33)
        sels.append([0, 0, hi - 1, W - 1])

    ops.append(34)
    sels.append([0, 0, hi - 1, W - 1])
    return ops, sels


def steps_iter(steps):
    for st in steps:
        yield st


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
