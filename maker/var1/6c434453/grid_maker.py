"""
ARC Task: 6c434453 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    # the generator never uses colour 2 for anything but the drawn crosses
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ncols = random.randint(1, 8)
    ccols = random.sample(rem, ncols)
    # the rule depends only on the SHAPE of an object (hollow rectangle or not),
    # never on its colour, so only bgc + the shared palette need fixing.
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    import random as _r

    def unif(lb, ub, bounds):
        a, b = bounds
        lo = a + int((b - a) * lb)
        up = a + int((b - a) * ub)
        if up < lo:
            lo, up = up, lo
        lo = max(a, min(b, lo))
        up = max(a, min(b, up))
        return _r.randint(lo, up)

    cols = [c for c in range(10) if c != 2]
    if bgc is None:
        bgc = _r.choice(cols)
    bgc = int(bgc)
    if not ccols:
        rem = [c for c in cols if c != bgc]
        ccols = _r.sample(rem, unif(diff_lb, diff_ub, (1, 8)))
    ccols = [int(c) for c in ccols if int(c) != bgc and int(c) != 2]
    if not ccols:
        ccols = [c for c in cols if c != bgc]

    hb = max(10, min(30, int(max_h)))
    wb = max(10, min(30, int(max_w)))
    h = unif(diff_lb, diff_ub, (10, hb))
    w = unif(diff_lb, diff_ub, (10, wb))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = set((r, c) for r in range(h) for c in range(w))

    nobjs = unif(diff_lb, diff_ub, (2, max(2, (h * w) // 16)))
    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        # force the very first placed object to be a frame, so every instance
        # actually exercises the rule at least once
        if succ == 0 or _r.choice((True, False)):
            oh = _r.choice((3, 5))
            ow = _r.choice((3, 5))
            obj = set()
            for y in range(oh):
                for x in range(ow):
                    if y in (0, oh - 1) or x in (0, ow - 1):
                        obj.add((y, x))
        else:
            oh = _r.randint(1, 5)
            ow = _r.randint(1, 5)
            bounds = [(y, x) for y in range(oh) for x in range(ow)]
            ncells = _r.randint(1, oh * ow)
            obj = {_r.choice(bounds)}
            for _ in range(ncells - 1):
                cands = [p for p in bounds if p not in obj and any(
                    (p[0] + dy, p[1] + dx) in obj for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))]
                if not cands:
                    break
                obj.add(_r.choice(cands))
            mr = min(p[0] for p in obj)
            mc = min(p[1] for p in obj)
            obj = {(y - mr, x - mc) for (y, x) in obj}

        oh = max(p[0] for p in obj) + 1
        ow = max(p[1] for p in obj) + 1
        ring = {(y, x) for y in range(oh) for x in range(ow)
                if y in (0, oh - 1) or x in (0, ow - 1)}
        # exactly the verifier's predicate: a hollow rectangle, both sides >= 3
        flag = (obj == ring) and oh >= 3 and ow >= 3
        if flag:
            objo = {(y, ow // 2) for y in range(oh)} | {(oh // 2, x) for x in range(ow)}
            tocover = {(y, x) for y in range(oh) for x in range(ow)}
        else:
            objo = set(obj)
            tocover = set(obj)

        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loc = _r.choice(sorted(cands))
        plcdi = {(y + loc[0], x + loc[1]) for (y, x) in obj}
        if plcdi <= inds:
            succ += 1
            plcdo = {(y + loc[0], x + loc[1]) for (y, x) in objo}
            cov = {(y + loc[0], x + loc[1]) for (y, x) in tocover}
            blocked = set(cov)
            for (y, x) in cov:
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    blocked.add((y + dy, x + dx))
            inds -= blocked
            col = _r.choice(ccols)
            for (y, x) in plcdi:
                gi[y][x] = col
            for (y, x) in plcdo:
                go[y][x] = 2 if flag else col

    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule: every hollow rectangle (an object whose cells are exactly its bbox
    outline, both sides >= 3) is wiped away and replaced, inside its own
    footprint, by a cross: the full centre row and the full centre column of
    that footprint, drawn in the marker colour (2). Everything else is left
    alone.

    The ROUTE through that rule (which frame first, which stroke first, wipe
    the whole footprint or just the ring, redraw or stamp a copy of an
    identical cross) is chosen from a signature of the INPUT alone, so the same
    instance always yields the same route while different instances differ.
    """
    import numpy as np
    from collections import deque, Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in cells]}

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour the canvas was painted with before objects landed
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # marker colour: read from the demonstrations (the colour that appears in an
    # output where its input held something else, other than the background)
    mark = 2
    if examples:
        tally = Counter()
        for ex in examples:
            try:
                a = np.asarray(ex[0], dtype=int)
                b = np.asarray(ex[1], dtype=int)
            except Exception:
                continue
            if a.shape != b.shape:
                continue
            diff = a != b
            for v in np.unique(b[diff]).tolist():
                tally[int(v)] += int(np.sum((b == v) & diff))
        cand = [v for v, _ in tally.most_common() if v != bgc]
        if cand:
            mark = cand[0]

    # ---- find the frames (raster/detection order is the canonical order) ----
    visited = np.zeros((hi, wi), dtype=bool)
    frames = []          # (r, c, oh, ow, ring_cells)
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or I[r, c] == bgc:
                continue
            color = I[r, c]
            queue = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while queue:
                y, x = queue.popleft()
                comp.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] == color:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            r0 = min(p[0] for p in comp); r1 = max(p[0] for p in comp)
            c0 = min(p[1] for p in comp); c1 = max(p[1] for p in comp)
            oh, ow = r1 - r0 + 1, c1 - c0 + 1
            if oh < 3 or ow < 3:
                continue
            ring = {(y, c0) for y in range(r0, r1 + 1)} | {(y, c1) for y in range(r0, r1 + 1)} \
                 | {(r0, x) for x in range(c0, c1 + 1)} | {(r1, x) for x in range(c0, c1 + 1)}
            if set(comp) == ring:
                frames.append((r0, c0, oh, ow, sorted(ring)))

    ops, sels = [], []

    if frames:
        # ---- route signature: a number read off the input grid only ----
        sig = bgc * 7 + hi * 13 + wi * 11 + len(frames) * 5
        for (r, c, oh, ow, _ring) in frames:
            sig += (r + 1) * 3 + (c + 1) * 2 + oh * 5 + ow * 7
        order_mode = sig % 4
        erase_ring = ((sig // 4) % 2) == 1      # wipe just the ring, or the whole footprint
        stroke_mode = (sig // 8) % 4
        reuse = ((sig // 32) % 2) == 1          # stamp repeated crosses instead of redrawing

        # ---- which frame to deal with first ----
        if order_mode == 0:                       # top-to-bottom, left-to-right
            ordered = sorted(frames, key=lambda f: (f[0], f[1]))
        elif order_mode == 1:                     # bottom-to-top, right-to-left
            ordered = sorted(frames, key=lambda f: (-f[0], -f[1]))
        elif order_mode == 2:                     # column by column
            ordered = sorted(frames, key=lambda f: (f[1], f[0]))
        else:                                     # biggest frame first
            ordered = sorted(frames, key=lambda f: (-(f[2] * f[3]), f[0], f[1]))

        clip_shape = None
        first_of_shape = {}

        for (r, c, oh, ow, ring) in ordered:
            cr, cc = r + oh // 2, c + ow // 2
            hline = [(cr, x) for x in range(c, c + ow)]
            vline = [(y, cc) for y in range(r, r + oh)]

            src = first_of_shape.get((oh, ow)) if reuse else None

            if src is not None:
                # an identical frame has already become a cross: stamp that one.
                if bgc == 0:
                    # Paste is transparent to 0, so the old frame must go first
                    if erase_ring:
                        ops.append(bgc); sels.append(sel_of(ring))
                    else:
                        # intended cells are exactly this full rectangle (the
                        # frame's whole footprint, background included)
                        ops.append(bgc); sels.append([r, c, oh - 1, ow - 1])
                if clip_shape != (oh, ow):
                    # full rectangle of the finished cross
                    ops.append(29); sels.append([src[0], src[1], oh - 1, ow - 1])
                    clip_shape = (oh, ow)
                ops.append(30); sels.append([r, c, 0, 0])
            else:
                # rub the frame out ...
                if erase_ring:
                    ops.append(bgc); sels.append(sel_of(ring))
                else:
                    # intended cells are exactly this full rectangle
                    ops.append(bgc); sels.append([r, c, oh - 1, ow - 1])
                # ... and draw the cross in its place
                if stroke_mode == 3:
                    ops.append(mark); sels.append(sel_of(sorted(set(hline) | set(vline))))
                else:
                    if stroke_mode == 0:
                        h_first = True
                    elif stroke_mode == 1:
                        h_first = False
                    else:
                        h_first = ow >= oh          # the longer stroke first
                    if h_first:
                        ops.append(mark); sels.append(sel_of(hline))
                        ops.append(mark); sels.append(sel_of(vline))
                    else:
                        ops.append(mark); sels.append(sel_of(vline))
                        ops.append(mark); sels.append(sel_of(hline))
                first_of_shape.setdefault((oh, ow), (r, c))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6c434453"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6c434453"
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
                                f"for task 6c434453"
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
                    f"Failed to build a complete episode for task 6c434453 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6c434453-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
