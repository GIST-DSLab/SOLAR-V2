"""
ARC Task: 22233c11 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Background is the only structural color role: 8 is hardcoded as the marker
    # color and the object colors are irrelevant to the rule (the rule depends on
    # the diagonal pattern's orientation and scale, never on its color).
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, **kwargs) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    nlim = max(2, (h * w) // 10)
    nobjs = unifint(diff_lb, diff_ub, (2, nlim))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    while succ < nobjs and tr < maxtr:
        if len(inds) == 0:
            break
        tr += 1
        od = randint(1, 3)
        g = canvas(bgc, (4, 4))
        g = fill(g, 8, {(0, 3), (3, 0)})
        col = choice(ccols)
        g = fill(g, col, {(1, 1), (2, 2)})
        # alternate the two discrete orientations ("\" and "/") so both appear
        # in every instance -> the test case is always learnable from examples
        if succ % 2 == 1:
            g = hmirror(g)
        g = upscale(g, od)
        inobj = recolor(col, ofcolor(g, col))
        outobj = inobj | recolor(8, ofcolor(g, 8))
        loc = choice(totuple(inds))
        outobj = shift(outobj, loc)
        inobj = shift(inobj, loc)
        outobji = toindices(outobj)
        # the whole 4d x 4d frame the markers get mirrored inside stays private to
        # this object, so no object's frame ever holds another object's cells
        frame = frozenset(
            (loc[0] + a, loc[1] + b) for a in range(4 * od) for b in range(4 * od)
        )
        framei = frame & fullinds
        if toindices(inobj).issubset(inds) and framei.issubset(inds):
            succ += 1
            inds = (inds - framei) - mapply(neighbors, outobji)
            gi = paint(gi, inobj)
            go = paint(go, outobj)
    if succ == 0:
        raise ValueError('no object placed')
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Every object is two od x od squares touching at a corner: blocks (1,1),(2,2)
    (a "\\" diagonal) or (1,2),(2,1) (a "/" diagonal) of a 4x4 frame of od-sized
    blocks.  The rule mirrors that diagonal inside the frame and marks its two
    outer corners with 8.  So per object: draw the marker pair as the plain
    continuation of the object's OWN diagonal (frame corners it points at), then
    Flip that pair inside the frame — the reflection is performed, not assumed.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Background: an object's frame origin is itself a grid cell and its squares
    # start one block inside that frame, so row 0 / col 0 are always pure canvas.
    bgc = int(I[0, 0])

    # objects: the two squares touch at a corner -> one 8-connected component
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                    and I[nr, nc] != bgc:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                comps.append(cells)

    def block(br, bc, od):
        return [(rr, cc)
                for rr in range(br, br + od)
                for cc in range(bc, bc + od)
                if 0 <= rr < hi and 0 <= cc < wi]

    cur = I.copy()                      # grid as the trajectory leaves it
    comps.sort(key=lambda cs: (min(r for r, _ in cs), min(c for _, c in cs)))

    for cells in comps:
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells)
        od = max(1, (r1 - r0 + 1) // 2)         # object bbox is 2od x 2od
        R0, C0 = r0 - od, c0 - od               # frame starts one block up/left
        Rend, Cend = R0 + 4 * od - 1, C0 + 4 * od - 1
        main_diag = I[r0, c0] != bgc            # top-left square filled -> "\"

        if main_diag:
            # mirrored diagonal -> markers on frame corners (0,3) and (3,0)
            tgt = block(R0, C0 + 3 * od, od) + block(R0 + 3 * od, C0, od)
        else:
            tgt = block(R0, C0, od) + block(R0 + 3 * od, C0 + 3 * od, od)
        if not tgt:
            continue

        # where that pair sits BEFORE the reflection: on the object's own diagonal
        # (FlipH mirrors the frame's columns, FlipV its rows — for this corner pair
        # both land on the same two blocks, so take whichever stays on the grid)
        flip_op, src = None, None
        for op, pre in ((26, [(r, C0 + Cend - c) for (r, c) in tgt]),
                        (27, [(R0 + Rend - r, c) for (r, c) in tgt])):
            if any(not (0 <= r < hi and 0 <= c < wi) for r, c in pre):
                continue
            if any(cur[r, c] != bgc for r, c in pre):
                continue
            rs = [r for r, _ in pre]
            cs = [c for _, c in pre]
            if op == 26:
                got = {(r, min(cs) + max(cs) - c) for r, c in pre}
            else:
                got = {(min(rs) + max(rs) - r, c) for r, c in pre}
            if got == set(tgt):                 # flip spans the whole frame here
                flip_op, src = op, sorted(set(pre))
                break

        if flip_op is None:
            # frame runs off the grid so hard that the pre-image of the surviving
            # marker is off-grid: there is nothing left to reflect, the marker is
            # simply drawn where the rule lands it
            ops.append(8)
            sels.append(sel_of(tgt))
            for (r, c) in tgt:
                cur[r, c] = 8
            continue

        ops.append(8)                           # extend the object's diagonal
        sels.append(sel_of(src))
        for (r, c) in src:
            cur[r, c] = 8
        ops.append(flip_op)                     # reflect the pair inside the frame
        sels.append(sel_of(src))
        for (r, c) in src:
            cur[r, c] = 0
        for (r, c) in tgt:
            cur[r, c] = 8
        if bgc != 0:
            # the flip leaves the vacated cells at 0; restore the canvas there
            ops.append(bgc)
            sels.append(sel_of(src))
            for (r, c) in src:
                cur[r, c] = bgc

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])         # full-grid bbox for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 22233c11"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22233c11"
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
                                f"for task 22233c11"
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
                    f"Failed to build a complete episode for task 22233c11 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22233c11-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
