"""
ARC Task: 6b9890af (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Background and frame colours are fixed for the episode.
    # Pattern colours (ccols) are not fixed: the rule copies them whatever they are.
    bgc, sqc = random.sample(range(10), 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **_) -> dict:
    def unifint(lb, ub):
        if ub < lb:
            raise ValueError
        d = random.uniform(diff_lb, diff_ub)
        return max(lb, min(ub, lb + round(d * (ub - lb))))

    cols = list(range(10))
    if bgc is None or sqc is None:
        bgc, sqc = random.sample(cols, 2)
    oh = unifint(2, 5)
    ow = unifint(2, 5)
    h = unifint(2 * oh + 2, max_h)
    w = unifint(2 * ow + 2, max_w)
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    obj = {random.choice(sorted(bounds))}

    def shp(o):
        rs = [a for a, _ in o]
        cs = [b for _, b in o]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(obj) != (oh, ow):
        nb = set()
        for (a, b) in obj:
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da or db:
                        nb.add((a + da, b + db))
        obj.add(random.choice(sorted((bounds - obj) & nb)))
    maxfac = 1
    while oh * maxfac + 2 <= h - oh and ow * maxfac + 2 <= w - ow:
        maxfac += 1
    maxfac -= 1
    fac = unifint(1, maxfac)
    remcols = [c for c in cols if c not in (bgc, sqc)]
    numc = unifint(1, 8)
    ccols = random.sample(remcols, numc)
    obj = {(random.choice(ccols), ij) for ij in obj}

    gi = np.full((h, w), bgc, dtype=int)
    SH, SW = oh * fac + 2, ow * fac + 2
    loci = random.randint(0, h - SH - oh)
    locj = random.randint(0, w - SW)
    gi[loci:loci + SH, locj:locj + SW] = sqc
    gi[loci + 1:loci + SH - 1, locj + 1:locj + SW - 1] = bgc
    pi = random.randint(loci + SH, h - oh)
    pj = random.randint(0, w - ow)
    for c, (a, b) in obj:
        gi[pi + a, pj + b] = c

    go = np.full((SH, SW), sqc, dtype=int)
    go[1:-1, 1:-1] = bgc
    for c, (a, b) in obj:
        go[1 + a * fac:1 + (a + 1) * fac, 1 + b * fac:1 + (b + 1) * fac] = c

    # A quarter turn swaps the height and width, so allow it only when the grid still fits
    rots = [0, 2] if (w > max_h or h > max_w) else [0, 1, 2, 3]
    k = random.choice(rots)
    gi = np.rot90(gi, -k)
    go = np.rot90(go, -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading: the frame becomes the canvas. The pattern is enlarged one row at a time.
    # Each pattern row is first stretched sideways (each pixel becomes a fac-wide run) on the
    # top line of its band. That stretched line is then copied down fac-1 times.
    # Every value comes from I; O is never consulted.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # find 8-connected components of one colour
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(cells)

    # the frame is the largest component whose cells are exactly the outline of its bbox
    best = None
    for cells in comps:
        rs = [y for y, _ in cells]
        cs = [x for _, x in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh < 3 or bw < 3:
            continue
        border = {(y, x) for y in range(r0, r1 + 1) for x in range(c0, c1 + 1)
                  if y in (r0, r1) or x in (c0, c1)}
        if set(cells) == border and (best is None or bh * bw > best[0]):
            best = (bh * bw, r0, c0, bh, bw, border)
    _, br, bc, bh, bw, bcells = best

    # the pattern is every other non-background cell
    pts = [(r, c) for r in range(hi) for c in range(wi)
           if I[r, c] != bgc and (r, c) not in bcells]
    pr = min(r for r, _ in pts)
    pc = min(c for _, c in pts)
    ph = max(r for r, _ in pts) - pr + 1
    pw = max(c for _, c in pts) - pc + 1
    fac = (bh - 2) // ph
    iw = bw - 2

    ops, sels = [], []
    # 1. Crop to the frame, which becomes the canvas. Its inside is plain background.
    #    This bbox is the frame's whole rectangle.
    ops.append(33); sels.append([br, bc, bh - 1, bw - 1])

    # 2. Draw the pattern row by row
    for i in range(ph):
        top = 1 + i * fac
        row = [int(I[pr + i, pc + j]) for j in range(pw)]
        # stretch the row sideways on the top line of its band
        for j, v in enumerate(row):
            if v == bgc:
                continue
            ops.append(v)
            sels.append(sel_of([(top, 1 + j * fac + k) for k in range(fac)]))
        if fac == 1:
            continue
        # copy the stretched line down the rest of the band
        if any(v != bgc and v != 0 for v in row):
            ops.append(29); sels.append([top, 1, 0, iw - 1])      # CopyO the full inside line
            for t in range(1, fac):
                ops.append(30); sels.append([top + t, 1, 0, 0])   # Paste one line lower
        # Copy/Paste skip colour 0, so extend 0-coloured runs down the band directly
        zcells = [(top + t, 1 + j * fac + k)
                  for j, v in enumerate(row) if v == 0 and v != bgc
                  for t in range(1, fac) for k in range(fac)]
        if zcells:
            ops.append(0); sels.append(sel_of(zcells))

    ops.append(34); sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6b9890af"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6b9890af"
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
                                f"for task 6b9890af"
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
                    f"Failed to build a complete episode for task 6b9890af "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6b9890af-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
