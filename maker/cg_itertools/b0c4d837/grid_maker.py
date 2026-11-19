"""
ARC Task: b0c4d837 (RE-ARC) — LLM-generated grid_maker
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
    bgc, boxc, fillc = random.sample(range(10), 3)
    return {"bgc": bgc, "boxc": boxc, "fillc": fillc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, fillc) -> dict:
    def unifint(lb, ub):
        lo, hi = lb, ub
        if hi <= lo:
            return lo
        a = lo + int(round((hi - lo) * diff_lb))
        b = lo + int(round((hi - lo) * diff_ub))
        a, b = max(lo, min(a, hi)), max(lo, min(b, hi))
        if b < a:
            a, b = b, a
        return random.randint(a, b)

    h = unifint(5, max(5, max_h))
    w = unifint(5, max(5, max_w))
    oh = unifint(3, h - 1)
    ow = unifint(3, w - 1)
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    sub = np.full((oh, ow), boxc, dtype=int)
    sub2 = np.full((oh - 1, ow - 2), fillc, dtype=int)
    n = unifint(1, min(9, oh - 2))
    sub2[:n, :] = bgc
    sub[0:oh - 1, 1:ow - 1] = sub2
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + oh, locj:locj + ow] = sub
    go = [fillc] * n + [bgc] * (9 - n)
    go = [go[:3], go[3:6][::-1], go[6:]]
    gi = np.rot90(gi, random.randint(0, 3))
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    """
    Reading (shared by the maker and the golf entry): the box is a tally board.
    Each interior line between the two walls that is entirely background is ONE
    tally.  The answer is a 3x3 board on which the fill colour is laid down once
    per tally along a snake path (row0 L->R, row1 R->L, row2 L->R); the remaining
    cells are background.  The golf reading counts the empty lines directly
    (rows whose wall cells alone sum to 10); here n is likewise the number of
    all-background interior lines, rather than a box-minus-fill dimension
    difference.  The operation route otherwise agrees with the maker's:
    take the tally colour from the input, reframe to 3x3, lay the base colour,
    paste one tally per counted line along the snake.
    """
    I = np.asarray(I, dtype=int)

    # colour roles by bounding-box area: background > frame > fill
    info = []
    for col in np.unique(I):
        cells = np.argwhere(I == col)
        a = cells.min(0)
        b = cells.max(0)
        info.append((int((b[0] - a[0] + 1) * (b[1] - a[1] + 1)), int(col),
                     (int(a[0]), int(a[1]), int(b[0]), int(b[1]))))
    info.sort(key=lambda t: -t[0])
    bgc = info[0][1]
    boxc, (r0, c0, r1, c1) = info[1][1], info[1][2]
    fillc, fbox = info[2][1], info[2][2]

    # interior lines between the two parallel walls
    left_full = all(I[r, c0] == boxc for r in range(r0, r1 + 1))
    right_full = all(I[r, c1] == boxc for r in range(r0, r1 + 1))
    if left_full and right_full:
        lines = [I[r, c0 + 1:c1] for r in range(r0, r1 + 1)]
    else:
        lines = [I[r0 + 1:r1, c] for c in range(c0, c1 + 1)]
    # each fully-empty interior line is one tally
    n = min(9, sum(1 for ln in lines if np.all(ln == bgc)))

    path = []
    for k in range(9):
        r, m = divmod(k, 3)
        path.append((r, m if r != 1 else 2 - m))

    # Paste can't carry colour 0: whichever role is 0 becomes the painted base
    if fillc != 0:
        base_col = bgc
        rep_cells = path[:n]
        src = (fbox[0], fbox[1])                     # a cell of the fill block
    else:
        base_col = fillc
        rep_cells = path[n:]
        bc = np.argwhere(I == bgc)
        src = (int(bc[0][0]), int(bc[0][1]))         # a background cell

    ops, sels = [], []
    out_cells = [(r, c) for r in range(3) for c in range(3)]

    # 1. pick up the tally colour from its object in the input
    if rep_cells:
        ops.append(28); sels.append(sel_of([src]))
    # 2. reframe the canvas to the 3x3 board (full rectangle)
    ops.append(33); sels.append(sel_of(out_cells))
    # 3. lay the base colour over the whole board unless already so
    if not np.all(I[0:3, 0:3] == base_col):
        ops.append(int(base_col)); sels.append(sel_of(out_cells))
    # 4. one paste per tally, walking the snake path
    for rc in rep_cells:
        ops.append(30); sels.append(sel_of([rc]))

    ops.append(34); sels.append(sel_of(out_cells))
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
                        f"num_examples+1 ({num_examples + 1}) for task b0c4d837"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b0c4d837"
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
                                f"for task b0c4d837"
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
                    f"Failed to build a complete episode for task b0c4d837 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b0c4d837-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
