"""
ARC Task: 539a4f51 (RE-ARC) — LLM-generated grid_maker
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

# The four orientations the generator can use. Every one is planned into the episode.
ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    # bgc is always 0 in the generator. Each instance samples its own band colours,
    # and the rule doesn't depend on which colours they are. The only structural case
    # is which corner holds the apex (the rotation), so plan it per instance.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, rot=None, **color_kwargs) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b <= a:
            return a
        x = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * x)))

    cols = list(range(1, 10))
    dmax = max(2, min(15, max_h // 2, max_w // 2))   # the output is 2d x 2d
    d = unifint(diff_lb, diff_ub, (2, dmax))
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(cols, numc)
    numocc = unifint(diff_lb, diff_ub, (1, d))
    arr = [random.choice(ccols) for _ in range(numocc)]
    while len(set(arr)) == 1:
        arr = [random.choice(ccols) for _ in range(d)]
    gi = np.zeros((d, d), dtype=int)
    for j, col in enumerate(arr):
        gi[j, :j + 1] = col
        gi[:j + 1, j] = col
    go = np.zeros((2 * d, 2 * d), dtype=int)
    for j in range(2 * d):
        col = arr[j % len(arr)]
        go[j, :j + 1] = col
        go[:j + 1, j] = col
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (taken from the golf entry, colour = seq[max(i, j) % n]):
    #   - The colour sequence and its period n are read from I alone. Both the maker and
    #     the golf entry do this the same way: an L-band at Chebyshev distance j from the
    #     apex gets arr[j % n]. Nothing changes there.
    #   - Where the golf reading carves the task differently: the unit of colour is the
    #     RESIDUE CLASS j mod n, not each band on its own. All bands that share a residue
    #     are one periodic object and get painted together.
    #   - The seed block from I (the n nested L-bands) survives unchanged at the apex
    #     corner of the doubled canvas, so it is MOVED there rather than repainted.
    I = np.asarray(I, dtype=int)
    d = I.shape[0]

    nz = np.argwhere(I != 0)
    r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
    r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
    n = max(r1 - r0 + 1, c1 - c0 + 1)   # period = side of the seed square

    def ring_cells(R, C, sr, sc, j, lim):
        cells = []
        for k in range(j + 1):
            cells.append((R + sr * j, C + sc * k))
            if k != j:
                cells.append((R + sr * k, C + sc * j))
        return [(r, c) for (r, c) in cells if 0 <= r < lim and 0 <= c < lim]

    # The apex is the grid corner from which I's nonzero cells form constant-colour rings.
    apex = None
    for (R, C) in [(0, 0), (0, d - 1), (d - 1, 0), (d - 1, d - 1)]:
        sr = 1 if R == 0 else -1
        sc = 1 if C == 0 else -1
        ok = True
        mask = np.zeros((d, d), dtype=bool)
        for j in range(n):
            cells = ring_cells(R, C, sr, sc, j, d)
            vals = {int(I[r, c]) for (r, c) in cells}
            if len(vals) != 1 or 0 in vals:
                ok = False
                break
            for rc in cells:
                mask[rc] = True
        if ok and np.all(I[~mask] == 0):
            apex = (R, C, sr, sc)
            break
    if apex is None:
        R = 0 if r0 == 0 else d - 1
        C = 0 if c0 == 0 else d - 1
        apex = (R, C, 1 if R == 0 else -1, 1 if C == 0 else -1)
    Ri, Ci, sr, sc = apex

    # The colour sequence, read outward from the apex.
    arr = [int(I[Ri + sr * j, Ci + sc * j]) for j in range(n)]

    D = 2 * d
    ops, sels = [], []

    # 1. Double the canvas. The selection is the whole rectangle, so a bbox is exact.
    #    I lands at the top-left.
    ops.append(33)
    sels.append([0, 0, D - 1, D - 1])

    # 2. Carry the seed block (every nonzero cell of I) to the apex corner of the
    #    doubled canvas. Grab once per direction leg, then use empty selections.
    #    bgc is 0, so the vacated footprint is already background.
    block = [(int(r), int(c)) for r, c in nz]
    dr = (D - d) if Ri != 0 else 0
    dc = (D - d) if Ci != 0 else 0
    cur = list(block)
    for op, steps, (a, b) in ((21, dr, (1, 0)), (22, dc, (0, 1))):
        for s in range(steps):
            ops.append(op)
            sels.append(sel_of(cur) if s == 0 else sel_of([]))
            cur = [(r + a, c + b) for r, c in cur]

    # 3. Extend the period outward. For each residue k, paint all bands at distance
    #    j >= n with j % n == k in colour arr[k], as one selection. These cells are
    #    still empty: they lie outside the seed block.
    Ro = 0 if Ri == 0 else D - 1
    Co = 0 if Ci == 0 else D - 1
    for k in range(n):
        cells = []
        for j in range(n + k, D, n):
            cells += ring_cells(Ro, Co, sr, sc, j, D)
        if cells:
            ops.append(arr[k])
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, D - 1, D - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 539a4f51"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 539a4f51"
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
                                f"for task 539a4f51"
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
                    f"Failed to build a complete episode for task 539a4f51 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"539a4f51-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
