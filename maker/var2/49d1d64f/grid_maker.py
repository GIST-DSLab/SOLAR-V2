"""
ARC Task: 49d1d64f (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------
# 1. colours
# ----------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """The rule is purely geometric (duplicate the outer ring of the picture),
    so no colour carries a role.  The only thing that has to be shared across
    the episode is the palette the picture is drawn from.  NOTE: 0 is an
    ordinary picture colour here (the canvas the generator paints on is -1,
    not 0), so the palette deliberately contains 0 fairly often - the route in
    derive_operations has to cope with that."""
    cols = list(range(10))
    ncols = random.randint(2, 10)
    ccols = random.sample(cols, ncols)
    if 0 not in ccols and random.random() < 0.5:
        ccols[random.randrange(len(ccols))] = 0
    return {"ccols": ccols}


# ----------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, ccols=None, **kwargs) -> dict:
    if not ccols:
        ccols = random.sample(list(range(10)), random.randint(2, 10))
    ccols = list(ccols)

    hmax = max(2, min(28, max_h - 2))
    wmax = max(2, min(28, max_w - 2))
    try:
        h = unifint(diff_lb, diff_ub, (2, hmax))          # noqa: F821
        w = unifint(diff_lb, diff_ub, (2, wmax))          # noqa: F821
    except NameError:
        h = random.randint(2, hmax)
        w = random.randint(2, wmax)

    gi = [[random.choice(ccols) for _ in range(w)] for _ in range(h)]

    go = [[0] * (w + 2) for _ in range(h + 2)]
    for r in range(h):                                    # picture at (1, 1)
        for c in range(w):
            go[r + 1][c + 1] = gi[r][c]
    for c in range(w):                                    # top / bottom edge
        go[0][c + 1] = gi[0][c]
        go[h + 1][c + 1] = gi[h - 1][c]
    for r in range(h):                                    # left / right edge
        go[r + 1][0] = gi[r][0]
        go[r + 1][w + 1] = gi[r][w - 1]

    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


# ----------------------------------------------------------------------
# 3. operations
# ----------------------------------------------------------------------
def derive_operations(I, O):
    """Rule: give the picture one ring of extra room, slide it into the middle
    of that room, then copy each of its four edge lines outwards into the ring
    (the four corners of the ring stay empty).

    The route taken is read off I alone - shape and content decide which way
    the picture is slid, in which order the four edges are drawn, and whether
    an edge is duplicated from the original input, from the picture as it now
    lies on the canvas, or drawn colour by colour.  O is never consulted."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi + 2, wi + 2

    ops, sels = [], []

    # ---- route choices, all derived from the input picture ---------------
    flat = I.flatten().tolist()
    checksum = sum((i + 1) * (v + 1) for i, v in enumerate(flat))
    ncolors = len(set(flat))

    ORDERS = [("T", "R", "B", "L"),   # clockwise from the top
              ("T", "L", "B", "R"),   # anticlockwise from the top
              ("T", "B", "L", "R"),   # the two horizontal edges, then vertical
              ("L", "R", "T", "B"),   # the two vertical edges, then horizontal
              ("L", "T", "R", "B"),
              ("B", "L", "T", "R")]
    order = ORDERS[checksum % len(ORDERS)]

    MODES = ("copyI", "copyO", "paint")
    mode = MODES[(ncolors + int(I[0, 0]) + int(I[hi - 1, wi - 1])) % 3]

    # a tall picture gets pushed down first, a wide one rightwards first
    down_first = hi > wi or (hi == wi and checksum % 2 == 0)

    # simulated working grid - built from I and the ops emitted so far
    cur = np.zeros((ho, wo), dtype=int)

    # ---- 1. one extra ring of room around the picture --------------------
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])   # whole new canvas rect
    cur[:hi, :wi] = I

    ink = bool(np.any(I != 0))

    # ---- 2. slide the whole picture one cell down and one cell right -----
    if ink:
        first, second = (21, 22) if down_first else (22, 21)
        # selection is exactly the picture's full rectangle (0-coloured cells
        # included); the canvas it slides over is empty, so the cells it leaves
        # behind and the cells of colour 0 both correctly read 0 afterwards.
        ops.append(first);  sels.append([0, 0, hi - 1, wi - 1])   # grab it
        ops.append(second); sels.append(sel_of([]))               # keep it grabbed
        cur = np.zeros((ho, wo), dtype=int)
        cur[1:hi + 1, 1:wi + 1] = I

    # ---- 3. the four edge lines, duplicated outwards ---------------------
    def edge(name):
        if name == "T":
            vals = [int(v) for v in I[0, :]]
            dst = [(0, c + 1) for c in range(wi)]
            src_in, src_out, at = [0, 0, 0, wi - 1], [1, 1, 0, wi - 1], [0, 1, 0, 0]
        elif name == "B":
            vals = [int(v) for v in I[hi - 1, :]]
            dst = [(ho - 1, c + 1) for c in range(wi)]
            src_in, src_out, at = [hi - 1, 0, 0, wi - 1], [hi, 1, 0, wi - 1], [ho - 1, 1, 0, 0]
        elif name == "L":
            vals = [int(v) for v in I[:, 0]]
            dst = [(r + 1, 0) for r in range(hi)]
            src_in, src_out, at = [0, 0, hi - 1, 0], [1, 1, hi - 1, 0], [1, 0, 0, 0]
        else:                                     # "R"
            vals = [int(v) for v in I[:, wi - 1]]
            dst = [(r + 1, wo - 1) for r in range(hi)]
            src_in, src_out, at = [0, wi - 1, hi - 1, 0], [1, wi, hi - 1, 0], [1, wo - 1, 0, 0]
        return vals, dst, src_in, src_out, at

    for name in order:
        vals, dst, src_in, src_out, at = edge(name)

        todo = [(cell, v) for cell, v in zip(dst, vals) if v != 0 and cur[cell] != v]
        if todo:
            if mode == "paint":
                # draw the line colour by colour, in the order the colours are
                # first met walking along it
                seen = []
                for _, v in todo:
                    if v not in seen:
                        seen.append(v)
                for v in seen:
                    cells = [cell for cell, cv in todo if cv == v]
                    ops.append(v); sels.append(sel_of(cells))
                    for cell in cells:
                        cur[cell] = v
            else:
                # duplicate the line: from the original input, or from the
                # picture where it now lies on the canvas
                ops.append(28 if mode == "copyI" else 29)
                sels.append(src_in if mode == "copyI" else src_out)   # a full 1-cell-wide rect
                ops.append(30); sels.append(at)                       # paste origin
                for cell, v in zip(dst, vals):
                    if v != 0:
                        cur[cell] = v

        # Cells of this edge whose colour is 0: ARCLE treats 0 as "nothing
        # there", so neither Move nor Copy/Paste carries them.  Put them down
        # explicitly wherever the canvas does not already read 0 - and emit
        # nothing when there are none.
        holes = [cell for cell, v in zip(dst, vals) if v == 0 and cur[cell] != 0]
        if holes:
            ops.append(0); sels.append(sel_of(holes))
            for cell in holes:
                cur[cell] = 0

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 49d1d64f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 49d1d64f"
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
                                f"for task 49d1d64f"
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
                    f"Failed to build a complete episode for task 49d1d64f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"49d1d64f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
