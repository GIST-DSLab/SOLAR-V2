"""
ARC Task: 7447852a (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 4]
    bgc, linc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, linc)]
    ccols = random.sample(remcols, random.randint(1, 7))
    return {"bgc": bgc, "linc": linc, "ccols": ccols}

def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, ccols=None) -> dict:
    from dsl import (canvas, fill, add, rot90, ofcolor, totuple, paint)
    from utils import unifint
    cols = [c for c in range(10) if c != 4]
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, linc)]
    if not ccols:
        ccols = random.sample(remcols, unifint(diff_lb, diff_ub, (1, 7)))
    # pre-rotation (h, w) -> final grid is (w rows, h cols)
    w = unifint(diff_lb, diff_ub, (2, max(2, min(8, max_h))))
    h = unifint(diff_lb, diff_ub, (w + 1, max(w + 1, max_w)))
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        gi2 = fill(gi, linc, {sp})
        if gi2 == gi:
            break
        gi = gi2
    gi = rot90(gi)
    inds = ofcolor(gi, bgc)
    ncells = unifint(diff_lb, diff_ub, (0, len(inds)))
    locs = random.sample(totuple(inds), ncells)
    obj = {(random.choice(ccols), ij) for ij in locs}
    gi = paint(gi, obj)
    # output: strips between zigzag segments, ordered left->right, every third -> 4
    G = np.array(gi)
    H, W = G.shape
    go = G.copy()
    for r in range(H):
        k = 0
        c0 = 0
        for cc in range(W + 1):
            if cc == W or G[r, cc] == linc:
                if cc > c0 and k % 3 == 0:
                    go[r, c0:cc] = 4
                if cc < W:
                    k += 2 if (r in (0, H - 1) and 0 < cc < W - 1) else 1
                c0 = cc + 1
    return {"input": [list(map(int, x)) for x in G], "output": [list(map(int, x)) for x in go]}

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # The zigzag line starts at the top-left corner.
    linc = int(I[0, 0])
    ops, sels = [], []
    # Read the grid row by row, the way a horizontal scan crosses the strips:
    # every line cell is a boundary between consecutive strips. A line cell at a
    # turning point (top or bottom row, not at a grid edge) is where two diagonals
    # meet, so it closes one strip and the empty one wedged in the apex: 2 boundaries.
    # Strip index = boundaries passed so far; strips 0, 3, 6, ... are painted 4.
    for r in range(h):
        k = 0
        c0 = 0
        for c in range(w + 1):
            if c == w or I[r, c] == linc:
                if c > c0 and k % 3 == 0:
                    ops.append(4)
                    sels.append(sel_of([(r, x) for x in range(c0, c)]))  # one row run between line cells
                if c < w:
                    k += 2 if (r in (0, h - 1) and 0 < c < w - 1) else 1
                c0 = c + 1
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
                        f"num_examples+1 ({num_examples + 1}) for task 7447852a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7447852a"
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
                                f"for task 7447852a"
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
                    f"Failed to build a complete episode for task 7447852a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7447852a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
