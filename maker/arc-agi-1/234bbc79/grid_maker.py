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
except Exception:                                    # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# ----------------------------------------------------------------------------
# 1. colours: the rule needs a stable background and a stable "joint dot" colour
#    (the colour that marks where two snake segments must be welded together).
#    The individual segment colours are irrelevant to the rule, so they stay
#    random inside generate().
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc = random.sample(cols, 2)
    return {"bgc": bgc, "dotc": dotc}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC 234bbc79, hardcoded 30s replaced by max_h / max_w)
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc) -> dict:
    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + int((hi - lo) * diff_lb)
        b = lo + int((hi - lo) * diff_ub)
        a = max(lo, min(hi, a))
        b = max(lo, min(hi, b))
        if b < a:
            a, b = b, a
        return random.randint(a, b)

    cols = list(range(10))
    remcols = [c for c in cols if c not in (bgc, dotc)]
    hmax = max(5, min(30, int(max_h)))
    wmax = min(20, int(max_w))
    wlo = min(6, wmax)

    while True:
        h = unifint(5, hmax)
        w = unifint(wlo, wmax)
        if w < 2:
            continue

        # ---- draw the snake on a (h, w) canvas -----------------------------
        go = [[bgc] * w for _ in range(h)]
        ncols = unifint(1, 8)
        ccols = random.sample(remcols, min(ncols, len(remcols)))
        spi = random.randint(0, h - 1)
        snek = [(spi, 0)]
        go[spi][0] = dotc
        while True:
            pi, pj = snek[-1]
            if pj == w - 1 and random.choice((True, False, False)):
                break
            options = []
            if pi < h - 1 and go[pi + 1][pj] == bgc:
                options.append((pi + 1, pj))
            if pi > 0 and go[pi - 1][pj] == bgc:
                options.append((pi - 1, pj))
            if pj < w - 1:
                options.append((pi, pj + 1))
            if not options:
                break
            loc = random.choice(options)
            snek.append(loc)
            go[loc[0]][loc[1]] = dotc

        # ---- cut the snake into segments at rightward steps ----------------
        objs, cobj = [], []
        for idx, cel in enumerate(snek):
            if (len(cobj) > 2
                    and (max(j for _, j in cobj) - min(j for _, j in cobj)) > 0
                    and snek[idx - 1] == (cel[0], cel[1] - 1)):
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
        ntokeep = unifint(2, nobjs)
        for _ in range(nobjs - ntokeep):
            if len(objs) < 3:
                break
            idx = random.randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx + 1]] + objs[idx + 2:]
        if len(objs) < 2:
            continue

        # ---- colour segments; mark inner endpoints with dotc ---------------
        inobjs = []
        for idx, obj in enumerate(objs):
            col = random.choice(ccols)
            for (i, j) in obj:
                go[i][j] = col
            cells = {}
            for (i, j) in obj[1:-1]:
                cells[(i, j)] = col
            cells[obj[0]] = dotc if idx > 0 else col
            cells[obj[-1]] = dotc if idx < len(objs) - 1 else col
            inobjs.append(cells)

        # ---- scatter the segments left-to-right on a wider canvas ----------
        spacings = [1] * (len(inobjs) - 1)
        fullw = unifint(w, max(w, int(max_w)))
        for _ in range(fullw - w - len(inobjs) - 1):
            if not spacings:
                break
            spacings[random.randint(0, len(spacings) - 1)] += 1
        lspacings = [0] + spacings
        gi = [[bgc] * fullw for _ in range(h)]
        ofs = 0
        ok = True
        for i, (lsp, cells) in enumerate(zip(lspacings, inobjs)):
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            oh = max(rs) - min(rs) + 1
            ow = max(cs) - min(cs) + 1
            if i == 0:
                ulc = (min(rs), min(cs))
            else:
                if h - oh < 0:
                    ok = False
                    break
                ulc = (random.randint(0, h - oh), ofs + lsp)
            ofs += ow + lsp
            if ulc[1] + ow > fullw:
                ok = False
                break
            for (r, c), v in cells.items():
                gi[ulc[0] + r - min(rs)][ulc[1] + c - min(cs)] = v
        if not ok:
            continue

        W = max(j for _, j in snek) + 1
        gout = [row[:W] for row in go]
        return {"input": gi, "output": gout}


# ----------------------------------------------------------------------------
# 3. derive_operations
#    Rule: the grid holds the pieces of one snake, scattered left to right.
#    Each piece carries "dot" cells at the ends that must be welded.  Keep the
#    leftmost piece where it is; slide every other piece so that its left dot
#    lands one column right of the growing snake's free dot; then repaint each
#    weld with the colour of the segment it belongs to; finally crop away the
#    now empty right part of the canvas.
# ----------------------------------------------------------------------------
def _components(grid, bgc):
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] != bgc:
                            seen[nx, ny] = True
                            stack.append((nx, ny))
                comps.append(cells)
    return comps


def _build_plan(I, O, bgc, dotc, comps):
    hi, wi = I.shape
    ho, wo = O.shape
    n = len(comps)
    if n < 2 or hi != ho or wo > wi:
        return None
    dots = []
    for k, cells in enumerate(comps):
        d = sorted([(r, c) for (r, c) in cells if I[r, c] == dotc], key=lambda p: (p[1], p[0]))
        need = 1 if (k == 0 or k == n - 1) else 2
        if len(d) != need:
            return None
        dots.append(d)
    offs = [(0, 0)]
    tail = dots[0][-1]
    for k in range(1, n):
        head = dots[k][0]
        o = (tail[0] - head[0], tail[1] + 1 - head[1])
        offs.append(o)
        if k < n - 1:
            t = dots[k][-1]
            tail = (t[0] + o[0], t[1] + o[1])
    occupied = {}
    for k, cells in enumerate(comps):
        dr, dc = offs[k]
        for (r, c) in cells:
            p = (r + dr, c + dc)
            if p in occupied or not (0 <= p[0] < hi) or not (0 <= p[1] < wi):
                return None
            occupied[p] = int(I[r, c])
    if min(c for _, c in occupied) != 0:
        return None
    if max(c for _, c in occupied) + 1 != wo:
        return None
    for p, v in occupied.items():
        if v == dotc:
            if int(O[p[0], p[1]]) == bgc:
                return None
        elif int(O[p[0], p[1]]) != v:
            return None
    for r in range(ho):
        for c in range(wo):
            if (int(O[r, c]) != bgc) != ((r, c) in occupied):
                return None
    return comps, dots, offs, occupied


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    counts = Counter(I.flatten().tolist())
    present = sorted(counts)

    plan = None
    for bgc in sorted(present, key=lambda c: -counts[c]):
        comps = _components(I, bgc)
        if len(comps) < 2:
            continue
        comps.sort(key=lambda cells: (min(c for _, c in cells), min(r for r, _ in cells)))
        # the joint colour occurs once or twice in EVERY piece, and is the
        # rarest such colour overall
        cand = [c for c in present if c != bgc and
                all(sum(1 for (r, cc) in cells if I[r, cc] == c) in (1, 2) for cells in comps)]
        cand.sort(key=lambda c: counts[c])
        order = cand + [c for c in present if c != bgc and c not in cand]
        for dotc in order:
            p = _build_plan(I, O, bgc, dotc, [list(x) for x in comps])
            if p is not None:
                plan = (bgc, dotc) + p
                break
        if plan is not None:
            break

    ops, sels = [], []
    cur = I.copy()

    def color_op(col, cells):
        col = int(col)
        cells = [(int(r), int(c)) for (r, c) in cells if int(cur[r, c]) != col]
        if not cells:
            return
        cells.sort()
        ops.append(col)
        sels.append(sel_of(cells))
        for (r, c) in cells:
            cur[r, c] = col

    if plan is None:                       # defensive fallback (should not run)
        if (hi, wi) != (ho, wo):
            ops.append(33)
            sels.append([0, 0, ho - 1, wo - 1])   # full rectangle kept
            cur = cur[:ho, :wo].copy()
        groups = {}
        for r in range(ho):
            for c in range(wo):
                if int(cur[r, c]) != int(O[r, c]):
                    groups.setdefault(int(O[r, c]), []).append((r, c))
        for col in sorted(groups):
            color_op(col, groups[col])
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    bgc, dotc, comps, dots, offs, occupied = plan
    n = len(comps)

    grab = {"obj": None, "snap": None, "off": (0, 0)}

    def move_op(op, grab_cells):
        if grab_cells is not None:
            obj, snap = {}, cur.copy()
            for (r, c) in grab_cells:
                if int(cur[r, c]) != 0:
                    obj[(r, c)] = int(cur[r, c])
                snap[r, c] = 0
            grab["obj"], grab["snap"], grab["off"] = obj, snap, (0, 0)
            sels.append(sel_of(sorted(grab_cells)))
        else:
            sels.append(sel_of([]))          # keep the same object grabbed
        d = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[op]
        o = (grab["off"][0] + d[0], grab["off"][1] + d[1])
        grab["off"] = o
        new = grab["snap"].copy()
        for (r, c), v in grab["obj"].items():
            nr, nc = r + o[0], c + o[1]
            if 0 <= nr < hi and 0 <= nc < wi:
                new[nr, nc] = v
        cur[:, :] = new
        ops.append(op)

    for k in range(1, n):
        src = sorted(comps[k])
        dr, dc = offs[k]
        dest = [(r + dr, c + dc) for (r, c) in src]
        dest_set = set(dest)
        # cells still to be drawn by later segments (used only to drop an
        # entirely superfluous background repair, never to shrink a selection)
        future = set()
        for j in range(k + 1, n):
            oj = offs[j]
            for (r, c) in comps[j]:
                future.add((r + oj[0], c + oj[1]))
        # the right part of the canvas is cropped away at the end
        vacated = [p for p in src if p not in dest_set and p[1] < wo]

        has_zero = any(int(I[r, c]) == 0 for (r, c) in src)
        if has_zero:
            # ARCLE's object ops treat colour 0 as transparent, so a Move would
            # destroy this segment's 0-coloured body: redraw it at its place.
            if vacated and not set(vacated) <= future:
                color_op(bgc, vacated)
            groups = {}
            for p in dest:
                groups.setdefault(int(O[p[0], p[1]]), []).append(p)
            for col in sorted(groups, key=lambda c: -len(groups[c])):
                color_op(col, groups[col])
        else:
            first = True
            if dr:
                op = 20 if dr < 0 else 21
                for _ in range(abs(dr)):
                    move_op(op, src if first else None)
                    first = False
            if dc:
                op = 23 if dc < 0 else 22
                for _ in range(abs(dc)):
                    move_op(op, src if first else None)
                    first = False
            # only the footprint the segment no longer covers reads 0 now
            if bgc != 0 and vacated and not set(vacated) <= future:
                color_op(bgc, vacated)

        # weld the seam: the snake's free dot and this segment's left dot
        po = offs[k - 1]
        ptail = (dots[k - 1][-1][0] + po[0], dots[k - 1][-1][1] + po[1])
        hpos = (dots[k][0][0] + dr, dots[k][0][1] + dc)
        seam = {}
        for p in (ptail, hpos):
            tgt = int(O[p[0], p[1]])
            if int(cur[p[0], p[1]]) != tgt:
                seam.setdefault(tgt, []).append(p)
        for tgt, cells in seam.items():
            color_op(tgt, cells)

    if (hi, wi) != (ho, wo):
        ops.append(33)
        sels.append([0, 0, ho - 1, wo - 1])   # bbox == exactly the region kept
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
