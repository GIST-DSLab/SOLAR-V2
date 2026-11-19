"""
ARC Task: f25fbde4 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
import random
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # the rule (crop to the shape, then double every cell) is colour-blind,
    # but the whole episode must agree on which colour is the canvas.
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    h = unifint(diff_lb, diff_ub, (2, max_h))
    w = unifint(diff_lb, diff_ub, (2, max_w))
    # the blob lives in a box small enough that its doubled version still fits
    bh = min(15, h - 1, max(1, max_h // 2))
    bw = min(15, w - 1, max(1, max_w // 2))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, (bh * bw) // 2)))
    nc = min(max(1, ncd), (h - 1) * (w - 1) - 1)
    c = canvas(bgc, (h, w))
    bounds = asindices(canvas(-1, (bh, bw)))
    ch = random.choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc):
        cands = totuple((bounds - shp) & mapply(neighbors, shp))
        if len(cands) == 0:
            break
        shp.add(random.choice(cands))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    plcd = shift(shp, (loci, locj))
    gi = fill(c, fgc, plcd)
    go = upscale(compress(gi), 2)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # ---- the rule, read off the input alone -------------------------------
    # background = the colour the canvas was painted with (strict majority here)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    mask = I != bgc
    rr = [r for r in range(hi) if bool(mask[r].any())]
    cc = [c for c in range(wi) if bool(mask[:, c].any())]
    r0, r1, c0, c1 = rr[0], rr[-1], cc[0], cc[-1]
    h, w = r1 - r0 + 1, c1 - c0 + 1
    core = I[r0:r1 + 1, c0:c1 + 1]

    ops, sels = [], []

    # (1) crop the canvas down to the shape's bounding box.
    #     bbox selection == exactly the rectangle we keep, background included.
    ops.append(33); sels.append([r0, c0, h - 1, w - 1])

    # simulation of the working grid (a 2h x 2w buffer; canvas is h x w for now)
    G = np.zeros((2 * h, 2 * w), dtype=int)
    G[:h, :w] = core

    # what the shape's box is mostly made of, and the other colour in it
    cnt = Counter(core.flatten().tolist())
    base = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    others = [v for v in sorted(cnt) if v != base]
    mark = others[0] if others else None

    # the minority colour's blobs inside the box (4-connected)
    comps = []
    if mark is not None:
        seen = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if core[r, c] == mark and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    cells = []
                    while stack:
                        a, b = stack.pop()
                        cells.append((a, b))
                        for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            na, nb = a + da, b + db
                            if 0 <= na < h and 0 <= nb < w and not seen[na, nb] \
                                    and core[na, nb] == mark:
                                seen[na, nb] = True
                                stack.append((na, nb))
                    comps.append(sorted(cells))

    if len(comps) <= 6:
        # ---------------------------------------------------------------
        # ROUTE 1 (few blobs to draw): open the doubled canvas, lay the
        # dominant colour over it, then draw each blob at double size.
        # ---------------------------------------------------------------
        ops.append(33); sels.append([0, 0, 2 * h - 1, 2 * w - 1])   # full rectangle

        lay = [(r, c) for r in range(2 * h) for c in range(2 * w)
               if int(G[r, c]) != base]          # cells not already the base colour
        if lay:
            ops.append(int(base)); sels.append(sel_of(lay))
            for (a, b) in lay:
                G[a, b] = base

        if base == bgc:
            # drawing the shape onto its background: big pieces first
            order = sorted(comps, key=lambda cs: (-len(cs), cs[0]))
        else:
            # poking the background holes out of a solid shape: go through them in order
            order = list(comps)

        for cells in order:
            blk = [(2 * a + dr, 2 * b + dc) for (a, b) in cells
                   for dr in (0, 1) for dc in (0, 1)]
            ops.append(int(mark)); sels.append(sel_of(blk))
            for (a, b) in blk:
                G[a, b] = mark
    else:
        # ---------------------------------------------------------------
        # ROUTE 2 (too tangled to draw blob by blob): stretch the picture,
        # copying each line of it onto the two lines it becomes.
        # ---------------------------------------------------------------
        state = {"clip": None, "th": h, "tw": w}

        def resize(nh, nw):
            ops.append(33); sels.append([0, 0, nh - 1, nw - 1])      # full rectangle

        def double_rows():
            th, tw = state["th"], state["tw"]
            for i in range(th - 1, -1, -1):     # from the bottom up: a row's copies
                src = [int(v) for v in G[i, :tw]]   # land below it, never on a row
                for t in (2 * i, 2 * i + 1):        # still waiting to be read
                    if all(int(G[t, c]) == src[c] for c in range(tw)):
                        continue
                    clear = [(t, c) for c in range(tw)
                             if src[c] == 0 and int(G[t, c]) != 0]
                    if clear:
                        # Paste never writes colour 0 -- clear those cells explicitly
                        ops.append(0); sels.append(sel_of(clear))
                        for (a, b) in clear:
                            G[a, b] = 0
                    if any(src[c] != 0 and int(G[t, c]) != src[c] for c in range(tw)):
                        key = ("row", tuple(src))
                        if state["clip"] != key:
                            ops.append(29); sels.append([i, 0, 0, tw - 1])  # whole row
                            state["clip"] = key
                        ops.append(30); sels.append([t, 0, 0, 0])
                        for c in range(tw):
                            if src[c] != 0:
                                G[t, c] = src[c]
            state["th"] = 2 * th

        def double_cols():
            th, tw = state["th"], state["tw"]
            for j in range(tw - 1, -1, -1):     # right to left, same reason
                src = [int(v) for v in G[:th, j]]
                for t in (2 * j, 2 * j + 1):
                    if all(int(G[r, t]) == src[r] for r in range(th)):
                        continue
                    clear = [(r, t) for r in range(th)
                             if src[r] == 0 and int(G[r, t]) != 0]
                    if clear:
                        ops.append(0); sels.append(sel_of(clear))
                        for (a, b) in clear:
                            G[a, b] = 0
                    if any(src[r] != 0 and int(G[r, t]) != src[r] for r in range(th)):
                        key = ("col", tuple(src))
                        if state["clip"] != key:
                            ops.append(29); sels.append([0, j, th - 1, 0])  # whole column
                            state["clip"] = key
                        ops.append(30); sels.append([0, t, 0, 0])
                        for r in range(th):
                            if src[r] != 0:
                                G[r, t] = src[r]
            state["tw"] = 2 * tw

        # where the shape sits decides whether the canvas is opened in one go
        # or one axis at a time; its proportions decide which axis is stretched first
        stagewise = ((r0 + c0) % 2 == 1)
        rows_first = (h > w)

        if stagewise:
            if rows_first:
                resize(2 * h, w); double_rows()
                resize(2 * h, 2 * w); double_cols()
            else:
                resize(h, 2 * w); double_cols()
                resize(2 * h, 2 * w); double_rows()
        else:
            resize(2 * h, 2 * w)
            if rows_first:
                double_rows(); double_cols()
            else:
                double_cols(); double_rows()

    ops.append(34); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f25fbde4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f25fbde4"
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
                                f"for task f25fbde4"
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
                    f"Failed to build a complete episode for task f25fbde4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f25fbde4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
