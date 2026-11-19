"""
ARC Task: 5614dbcf (RE-ARC) — LLM-generated grid_maker
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
    # Background 0 and noise colour 5 are hardcoded in the generator.
    # Each instance draws its own palette of block colours, and the rule does not
    # depend on which colours are used, so there is nothing to fix per episode.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    colopts = [c for c in range(1, 10) if c != 5]
    cand = [(a, b) for a in range(3) for b in range(3)]
    dmax = max(2, min(10, max_h // 3, max_w // 3))
    d = max(2, min(dmax, unifint(diff_lb, diff_ub, (2, 10))))
    inds = [(i, j) for i in range(d) for j in range(d)]
    numocc = max(1, min(d * d, unifint(diff_lb, diff_ub, (1, d * d))))
    numcol = max(1, min(8, unifint(diff_lb, diff_ub, (1, 8))))
    occs = set(random.sample(inds, numocc))
    colset = random.sample(colopts, numcol)
    go = np.zeros((d, d), dtype=int)
    gi = np.zeros((3 * d, 3 * d), dtype=int)
    for (i, j) in inds:
        numnoise = max(0, min(8, unifint(diff_lb, diff_ub, (0, 8))))
        noise = random.sample(cand, numnoise)
        if (i, j) in occs:
            col = random.choice(colset)
            go[i, j] = col
            gi[3 * i:3 * i + 3, 3 * j:3 * j + 3] = col
        for (a, b) in noise:
            gi[3 * i + a, 3 * j + b] = 5
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading: I is a d x d grid upscaled 3x. Every 3x3 block holds exactly one
    # real colour (0 if the block is empty), and 5-noise covers some of its cells.
    # O is the downscale: block (i, j) becomes cell (i, j).
    # The code-golf entry reads the task the same way (block -> its one non-5
    # colour -> one cell). It only looks at the top row of each block, which is a
    # shortcut: RE-ARC noise can cover a whole row (up to 8 of the 9 cells).
    # Since the readings agree, the ops stay the same: read each block's colour
    # from I, write it into the top-left d x d corner, then crop to that corner.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    dh, dw = hi // 3, wi // 3

    block_col = [[0] * dw for _ in range(dh)]
    for i in range(dh):
        for j in range(dw):
            vals = [v for v in I[3 * i:3 * i + 3, 3 * j:3 * j + 3].flatten().tolist() if v != 5]
            block_col[i][j] = Counter(vals).most_common(1)[0][0] if vals else 0

    ops, sels = [], []
    # Paint the downscaled cells one row at a time. Neighbouring cells in a row
    # that need the same colour form one horizontal segment and get one Color op.
    # A cell that already holds its colour is skipped (this check uses I only).
    # Painting cell (i, j) only touches block (i//3, j//3), and that block was
    # already read, because block_col is computed from I up front.
    for i in range(dh):
        j = 0
        while j < dw:
            c = block_col[i][j]
            if int(I[i, j]) == c:
                j += 1
                continue
            k = j
            while k < dw and block_col[i][k] == c and int(I[i, k]) != c:
                k += 1
            ops.append(int(c))
            sels.append(sel_of([(i, x) for x in range(j, k)]))
            j = k

    # Crop to the top-left dh x dw corner, which now holds the downscaled grid.
    # The selection is the full rectangle on purpose.
    ops.append(33)
    sels.append(sel_of([(r, c) for r in range(dh) for c in range(dw)]))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(dh) for c in range(dw)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 5614dbcf"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5614dbcf"
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
                                f"for task 5614dbcf"
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
                    f"Failed to build a complete episode for task 5614dbcf "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5614dbcf-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
