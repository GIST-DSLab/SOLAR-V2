"""
ARC Task: 93b581b8 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(2, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        remcols = remove(bgc, cols)
        ccols = sample(remcols, unifint(diff_lb, diff_ub, (1, 9)))
    ccols = list(ccols)

    hlo = min(10, max_h)
    wlo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))

    for _attempt in range(25):
        numocc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 50)))
        succ = 0
        tr = 0
        maxtr = 10 * numocc
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        inds = asindices(gi)
        fullinds = asindices(gi)
        while tr < maxtr and succ < numocc:
            tr += 1
            cands = sfilter(inds, lambda ij: ij[0] <= h - 2 and ij[1] <= w - 2)
            if len(cands) == 0:
                break
            loc = choice(totuple(cands))
            c1, c2, c3, c4 = [choice(ccols) for k in range(4)]
            q = {(0, 0), (0, 1), (1, 0), (1, 1)}
            inobj = {(c1, (0, 0)), (c2, (0, 1)), (c3, (1, 0)), (c4, (1, 1))}
            outobj = (
                inobj
                | recolor(c4, shift(q, (-2, -2)))
                | recolor(c3, shift(q, (-2, 2)))
                | recolor(c2, shift(q, (2, -2)))
                | recolor(c1, shift(q, (2, 2)))
            )
            inobjplcd = shift(inobj, loc)
            outobjplcd = shift(outobj, loc)
            outobjplcd = sfilter(outobjplcd, lambda cij: cij[1] in fullinds)
            outobjplcdi = toindices(outobjplcd)
            if outobjplcdi.issubset(inds):
                succ += 1
                inds = (inds - outobjplcdi) - mapply(dneighbors, toindices(inobjplcd))
                gi = paint(gi, inobjplcd)
                go = paint(go, outobjplcd)
        if succ > 0:
            break

    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (read off the examples): every 2x2 block of four coloured cells throws a
    2x2 echo into each of the four diagonal directions, two cells away.  The echo
    carries the colour of the block corner DIAGONALLY OPPOSITE the direction it
    flies in: the up-left echo is the block's lower-right colour, the up-right echo
    is its lower-left colour, and so on.  Echoes are clipped at the grid edge.

    The route (which block first, which echo first, whether echoes of one colour
    are laid in one stroke) is chosen from features of the input alone, so the same
    input always yields the same trajectory while different inputs are solved in
    visibly different orders.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- find the source blocks (connected components of non-background cells)
    seen = set()
    blocks = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and (r, c) not in seen:
                cells = []
                dq = deque([(r, c)])
                seen.add((r, c))
                while dq:
                    rr, cc = dq.popleft()
                    cells.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < hi and 0 <= nc < wi
                                and (nr, nc) not in seen and I[nr, nc] != bgc):
                            seen.add((nr, nc))
                            dq.append((nr, nc))
                r0 = min(x[0] for x in cells)
                c0 = min(x[1] for x in cells)
                r0 = min(r0, hi - 2)
                c0 = min(c0, wi - 2)
                blocks.append((r0, c0))

    ops, sels = [], []
    if not blocks:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # ---- route choices, all read off the input
    ar, ac = blocks[0]                      # the first block in reading order
    a_ul = int(I[ar, ac])                   # its upper-left colour
    a_lr = int(I[ar + 1, ac + 1])           # its lower-right colour

    mode_sweep = (ar + ac + hi) % 5         # which block to deal with first
    mode_corner = (a_ul + a_lr) % 4         # which echo of a block to throw first
    mode_group = (len(blocks) + wi) % 3     # one echo at a time, or by colour

    if mode_sweep == 0:                     # top-left downwards, reading order
        order = sorted(blocks, key=lambda b: (b[0], b[1]))
    elif mode_sweep == 1:                   # bottom-right upwards
        order = sorted(blocks, key=lambda b: (-b[0], -b[1]))
    elif mode_sweep == 2:                   # column by column, left to right
        order = sorted(blocks, key=lambda b: (b[1], b[0]))
    elif mode_sweep == 3:                   # column by column, right to left
        order = sorted(blocks, key=lambda b: (-b[1], -b[0]))
    else:                                   # diagonal sweep away from the corner
        order = sorted(blocks, key=lambda b: (b[0] + b[1], b[0]))

    UL, UR, LL, LR = (-2, -2), (-2, 2), (2, -2), (2, 2)
    if mode_corner == 0:                    # reading order of the four echoes
        corner_order = [UL, UR, LL, LR]
    elif mode_corner == 1:                  # clockwise from the top-left
        corner_order = [UL, UR, LR, LL]
    elif mode_corner == 2:                  # counter-clockwise from the top-left
        corner_order = [UL, LL, LR, UR]
    else:                                   # reverse reading order
        corner_order = [LR, LL, UR, UL]

    def echo_color(r, c, d):
        # colour of the corner diagonally opposite the flight direction
        if d == UL:
            return int(I[r + 1, c + 1])
        if d == UR:
            return int(I[r + 1, c])
        if d == LL:
            return int(I[r, c + 1])
        return int(I[r, c])

    def echo_cells(r, c, d):
        dr, dc = d
        out = []
        for qr in range(2):
            for qc in range(2):
                rr, cc = r + dr + qr, c + dc + qc
                if 0 <= rr < hi and 0 <= cc < wi:
                    out.append((rr, cc))
        return out

    # the echoes, block by block, in the chosen order
    plan = []                               # list of (color, cells) per block
    for (r, c) in order:
        per_block = []
        for d in corner_order:
            cells = echo_cells(r, c, d)
            if cells:
                per_block.append((echo_color(r, c, d), cells))
        plan.append(per_block)

    if mode_group == 0:
        # one stroke per echo, block by block
        for per_block in plan:
            for col, cells in per_block:
                ops.append(col)
                sels.append(sel_of(cells))
    elif mode_group == 1:
        # finish one block at a time, but echoes sharing a colour go in one stroke
        for per_block in plan:
            merged = {}
            seq = []
            for col, cells in per_block:
                if col not in merged:
                    merged[col] = []
                    seq.append(col)
                merged[col].extend(cells)
            for col in seq:
                ops.append(col)
                sels.append(sel_of(merged[col]))
    else:
        # all echoes of one colour at once, colours in order of first appearance
        merged = {}
        seq = []
        for per_block in plan:
            for col, cells in per_block:
                if col not in merged:
                    merged[col] = []
                    seq.append(col)
                merged[col].extend(cells)
        for col in seq:
            ops.append(col)
            sels.append(sel_of(merged[col]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 93b581b8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 93b581b8"
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
                                f"for task 93b581b8"
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
                    f"Failed to build a complete episode for task 93b581b8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"93b581b8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
