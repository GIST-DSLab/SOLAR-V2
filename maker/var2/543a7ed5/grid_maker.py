"""
ARC Task: 543a7ed5 (RE-ARC) — LLM-generated grid_maker
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

try:  # repo helper; inline equivalent when the module isn't importable
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}


def sample_colors(num_examples=None) -> dict:
    # generator draws from cols = 0..9 minus {3, 4}: 3 (frame) and 4 (pocket) are
    # hardcoded by the rule, so they are never a sampled role. The rule reacts to
    # where the blocks are, not what colour they are, so only bgc must be pinned;
    # the palette is pinned too so every instance of an episode looks alike.
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 7)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, ccols=None) -> dict:
    cols = difference(interval(0, 10, 1), (3, 4))
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        remcols = remove(bgc, cols)
        ccols = sample(remcols, unifint(diff_lb, diff_ub, (1, 7)))
    ccols = list(ccols)

    hlo = min(10, max_h)
    wlo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = randint(4, 8)
        ow = randint(4, 8)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(ccols)
        if bd.issubset(indss):
            bdibd = backdrop(frozenset({(loci + 1, locj + 1), (loci + oh - 2, locj + ow - 2)}))
            go = fill(go, col, bdibd)
            go = fill(go, 3, box(bd))
            gi = fill(gi, col, bdibd)
            if oh > 5 and ow > 5 and randint(1, 10) != 1:
                ulci, ulcj = ulcorner(bdibd)
                lrci, lrcj = lrcorner(bdibd)
                aa = randint(ulci + 1, lrci - 1)
                aa = randint(ulci + 1, aa)
                bb = randint(ulcj + 1, lrcj - 1)
                bb = randint(ulcj + 1, bb)
                cc = randint(aa, lrci - 1)
                dd = randint(bb, lrcj - 1)
                cc = randint(cc, lrci - 1)
                dd = randint(dd, lrcj - 1)
                ins = backdrop({(aa, bb), (cc, dd)})
                go = fill(go, 4, ins)
                gi = fill(gi, bgc, ins)
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


# ── derive_operations ────────────────────────────────────────────────────────
# Rule (read off the examples, never off O): every solid block of one colour gets
# a colour-3 frame on the ring of cells one step outside its bounding box, and
# every pocket of background trapped inside that bounding box becomes colour 4.
#
# The ROUTE through that rule is picked from the INPUT alone — background colour,
# how many blocks there are, how much ink is on the grid, where the first block
# sits — the way two people each find their own way round the same puzzle: which
# block to start on, which way to sweep, whether the frame goes down as one
# outline, a pen stroke round the four sides, two corner strokes, or a patch of 3
# with the block redrawn on top, whether a trapped pocket is bucket-filled or
# painted, whether the pocket is punched before or after the frame, and whether
# an identical twin block is drawn again or stamped from the first one.
# The same input always gives the same route; O is never consulted.

def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # ── background ───────────────────────────────────────────────────────────
    # Every block is inset by at least one cell from every edge (the generator
    # leaves room for the frame), so the border is pure background; the episode's
    # example inputs sit on that same background, so they get a vote too.
    border = ([int(v) for v in I[0, :]] + [int(v) for v in I[h - 1, :]] +
              [int(v) for v in I[:, 0]] + [int(v) for v in I[:, w - 1]])
    for pair in (examples or []):
        try:
            E = np.asarray(pair[0], dtype=int)
            border += [int(v) for v in E[0, :]] + [int(v) for v in E[:, 0]]
        except Exception:
            pass
    bgc = max(set(border), key=border.count)

    # ── the blocks, their frames and their trapped pockets ───────────────────
    comps = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            cs = set(cells)
            r0 = min(p[0] for p in cells); r1 = max(p[0] for p in cells)
            c0 = min(p[1] for p in cells); c1 = max(p[1] for p in cells)
            R0, R1, C0, C1 = r0 - 1, r1 + 1, c0 - 1, c1 + 1
            ring = [(rr, cc) for cc in range(C0, C1 + 1) for rr in (R0, R1)]
            ring += [(rr, cc) for rr in range(R0 + 1, R1) for cc in (C0, C1)]
            ring = sorted({(rr, cc) for rr, cc in ring if 0 <= rr < h and 0 <= cc < w})
            hole = sorted({(rr, cc) for rr in range(r0, r1 + 1)
                           for cc in range(c0, c1 + 1)} - cs)
            comps.append({"col": col, "cells": cs, "hole": hole, "ring": ring,
                          "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                          "R0": R0, "R1": R1, "C0": C0, "C1": C1,
                          "area": len(cs),
                          "off": {(rr - r0, cc - c0) for rr, cc in hole}})

    if not comps:
        return [34], [[0, 0, h - 1, w - 1]]

    # what the rule asks for on this grid — built from I, never from O
    target = I.copy()
    for cp in comps:
        for rr, cc in cp["ring"]:
            target[rr, cc] = 3
    for cp in comps:
        for rr, cc in cp["hole"]:
            target[rr, cc] = 4

    # ── small shared tools ───────────────────────────────────────────────────
    def sel_for(cells):
        cells = sorted(set(cells))
        r0 = min(p[0] for p in cells); r1 = max(p[0] for p in cells)
        c0 = min(p[1] for p in cells); c1 = max(p[1] for p in cells)
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
            return [r0, c0, r1 - r0, c1 - c0]      # exactly that full rectangle
        return sel_of(cells)

    def pieces(cells):
        """4-connected pieces of a set of cells."""
        left = set(cells)
        out = []
        while left:
            start = min(left)
            grp = {start}
            st = [start]
            left.discard(start)
            while st:
                rr, cc = st.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p = (rr + dr, cc + dc)
                    if p in left:
                        left.discard(p)
                        grp.add(p)
                        st.append(p)
            out.append(sorted(grp))
        return out

    def strips(cells, by_row, reverse):
        """Maximal straight runs of `cells`, row by row or column by column."""
        cells = set(cells)
        keys = sorted({p[0] if by_row else p[1] for p in cells}, reverse=reverse)
        out = []
        for k in keys:
            line = sorted(p[1] if by_row else p[0] for p in cells
                          if (p[0] if by_row else p[1]) == k)
            run = [line[0]]
            for v in line[1:]:
                if v == run[-1] + 1:
                    run.append(v)
                else:
                    out.append([(k, x) if by_row else (x, k) for x in run])
                    run = [v]
            out.append([(k, x) if by_row else (x, k) for x in run])
        return out

    # ── which route this particular input gets ───────────────────────────────
    n = len(comps)
    ink = sum(cp["area"] for cp in comps)
    head = min(comps, key=lambda cp: (cp["r0"], cp["c0"]))
    fr, fc = head["r0"], head["c0"]

    order_mode = (bgc + n + fr + fc) % 5
    frame_mode = (3 * bgc + n + w) % 4
    cw = ((bgc + fr) % 2 == 0)
    bucket = ((h + ink) % 2 == 0)
    stamp_twins = ((fc + n) % 2 == 0)
    by_row = ((ink + w) % 2 == 0)
    backwards = ((ink + fr) % 2 == 1)
    pocket_first = ((bgc + n + w) % 2 == 1)

    if order_mode == 0:      # reading order, top-left block first
        comps.sort(key=lambda cp: (cp["r0"], cp["c0"]))
    elif order_mode == 1:    # column by column, left to right
        comps.sort(key=lambda cp: (cp["c0"], cp["r0"]))
    elif order_mode == 2:    # biggest block first
        comps.sort(key=lambda cp: (-cp["area"], cp["r0"], cp["c0"]))
    elif order_mode == 3:    # backwards, bottom-right block first
        comps.sort(key=lambda cp: (-cp["r1"], -cp["c1"]))
    else:                    # one colour at a time
        comps.sort(key=lambda cp: (cp["col"], cp["r0"], cp["c0"]))

    ops, sels = [], []
    cur = I.copy()
    clip_rect = None

    def paint(colour, cells):
        cells = [(r, c) for r, c in cells if 0 <= r < h and 0 <= c < w]
        if not cells or all(cur[r, c] == colour for r, c in cells):
            return                      # nothing there to change: emit nothing
        ops.append(int(colour))
        sels.append(sel_for(cells))
        for r, c in cells:
            cur[r, c] = colour

    def flood_of(start):
        col = cur[start]
        reg = {start}
        st = [start]
        while st:
            rr, cc = st.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (rr + dr, cc + dc)
                if 0 <= p[0] < h and 0 <= p[1] < w and p not in reg and cur[p] == col:
                    reg.add(p)
                    st.append(p)
        return reg

    def fill_pocket(cp):
        hole = cp["hole"]
        if not hole:
            return
        if bucket and cur[hole[0]] != 4 and flood_of(hole[0]) == set(hole):
            ops.append(14)              # walled in by the block: one bucket-fill
            sels.append([hole[0][0], hole[0][1], 0, 0])
            for r, c in hole:
                cur[r, c] = 4
            return
        for grp in pieces(hole):
            r0 = min(p[0] for p in grp); r1 = max(p[0] for p in grp)
            c0 = min(p[1] for p in grp); c1 = max(p[1] for p in grp)
            if len(grp) == (r1 - r0 + 1) * (c1 - c0 + 1):
                paint(4, grp)
            else:
                for run in strips(grp, by_row, backwards):
                    paint(4, run)

    def draw_frame(cp):
        R0, R1, C0, C1 = cp["R0"], cp["R1"], cp["C0"], cp["C1"]
        if frame_mode == 0:                      # the whole outline in one stroke
            paint(3, cp["ring"])
        elif frame_mode == 1:                    # round the four sides
            top = [(R0, cc) for cc in range(C0, C1 + 1)]
            if cw:
                segs = [top,
                        [(rr, C1) for rr in range(R0 + 1, R1 + 1)],
                        [(R1, cc) for cc in range(C0, C1)],
                        [(rr, C0) for rr in range(R0 + 1, R1)]]
            else:
                segs = [top,
                        [(rr, C0) for rr in range(R0 + 1, R1 + 1)],
                        [(R1, cc) for cc in range(C0 + 1, C1 + 1)],
                        [(rr, C1) for rr in range(R0 + 1, R1)]]
            for seg in segs:
                paint(3, seg)
        elif frame_mode == 2:                    # two corner strokes
            if cw:
                l1 = [(R0, cc) for cc in range(C0, C1 + 1)] + \
                     [(rr, C1) for rr in range(R0 + 1, R1 + 1)]
                l2 = [(R1, cc) for cc in range(C0, C1)] + \
                     [(rr, C0) for rr in range(R0 + 1, R1)]
            else:
                l1 = [(R0, cc) for cc in range(C0, C1 + 1)] + \
                     [(rr, C0) for rr in range(R0 + 1, R1 + 1)]
                l2 = [(R1, cc) for cc in range(C0 + 1, C1 + 1)] + \
                     [(rr, C1) for rr in range(R0 + 1, R1)]
            paint(3, l1)
            paint(3, l2)
        else:
            # lay a patch of 3 over the whole block-and-frame, then draw the block
            # back on top of it (an explicit Color op, so a block of colour 0 lands
            # too); the pocket is punched out of it straight after.
            paint(3, [(rr, cc) for rr in range(R0, R1 + 1) for cc in range(C0, C1 + 1)])
            paint(cp["col"], [(rr, cc) for rr in range(cp["r0"], cp["r1"] + 1)
                              for cc in range(cp["c0"], cp["c1"] + 1)])

    def try_stamp(cp, done):
        """Stamp a finished identical twin instead of drawing this one again."""
        nonlocal clip_rect
        if not stamp_twins:
            return False
        if cp["R0"] < 0 or cp["C0"] < 0 or cp["R1"] >= h or cp["C1"] >= w:
            return False
        src = None
        for other in done:
            if (other["col"] == cp["col"] and other["off"] == cp["off"]
                    and other["r1"] - other["r0"] == cp["r1"] - cp["r0"]
                    and other["c1"] - other["c0"] == cp["c1"] - cp["c0"]
                    and other["R0"] >= 0 and other["C0"] >= 0
                    and other["R1"] < h and other["C1"] < w):
                src = other
                break
        if src is None:
            return False

        # Dry run first. Paste carries only non-zero clipboard cells, so any part of
        # the stamp that is colour 0 stays behind and has to be put down by hand.
        nxt = cur.copy()
        for dr in range(cp["R1"] - cp["R0"] + 1):
            for dc in range(cp["C1"] - cp["C0"] + 1):
                v = cur[src["R0"] + dr, src["C0"] + dc]
                if v != 0:
                    nxt[cp["R0"] + dr, cp["C0"] + dc] = v
        zeros = [(cp["R0"] + dr, cp["C0"] + dc)
                 for dr in range(cp["R1"] - cp["R0"] + 1)
                 for dc in range(cp["C1"] - cp["C0"] + 1)
                 if target[cp["R0"] + dr, cp["C0"] + dc] == 0
                 and nxt[cp["R0"] + dr, cp["C0"] + dc] != 0]
        for r, c in zeros:
            nxt[r, c] = 0
        box = (slice(cp["R0"], cp["R1"] + 1), slice(cp["C0"], cp["C1"] + 1))
        if not np.array_equal(nxt[box], target[box]):
            return False                 # not a true twin here: draw it by hand
        if np.array_equal(nxt, cur):
            return False                 # the stamp would change nothing

        rect = (src["R0"], src["C0"], src["R1"], src["C1"])
        if clip_rect != rect:            # full rectangle: the finished twin
            ops.append(29)
            sels.append([src["R0"], src["C0"],
                         src["R1"] - src["R0"], src["C1"] - src["C0"]])
            clip_rect = rect
        ops.append(30)
        sels.append([cp["R0"], cp["C0"], 0, 0])
        cur[:, :] = nxt
        if zeros:                        # the colour-0 cells Paste left behind
            paint(0, zeros)
        return True

    done = []
    for cp in comps:
        if not try_stamp(cp, done):
            if pocket_first and frame_mode != 3:
                fill_pocket(cp)          # punch the pocket out, then frame it
                draw_frame(cp)
            else:
                draw_frame(cp)           # frame it, then punch the pocket out
                fill_pocket(cp)
        done.append(cp)

    if not np.array_equal(cur, target):
        # anything unexpected about this grid: fall back to the plain route —
        # every frame first, then every pocket
        ops, sels = [], []
        cur = I.copy()
        for cp in comps:
            paint(3, cp["ring"])
        for cp in comps:
            paint(4, cp["hole"])

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 543a7ed5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 543a7ed5"
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
                                f"for task 543a7ed5"
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
                    f"Failed to build a complete episode for task 543a7ed5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"543a7ed5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
