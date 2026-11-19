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

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    # Background is fixed at 0 and the rule ignores which colours appear, so no
    # colours are fixed here. Plan the apex corner (rotation) so the examples cover each one.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append({"rot": random.choice(ex)})
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, rot=None, **kwargs) -> dict:
    from dsl import canvas, fill, connect, interval, identity, rot90, rot180, rot270
    from random import sample, choice
    from utils import unifint
    cols = interval(1, 10, 1)
    dmax = max(2, min(15, max_h // 2, max_w // 2))
    d = unifint(diff_lb, diff_ub, (2, dmax))
    h, w = d, d
    gi = canvas(0, (h, w))
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(cols, numc)
    numocc = unifint(diff_lb, diff_ub, (1, d))
    arr = [choice(ccols) for k in range(numocc)]
    while len(set(arr)) == 1:
        arr = [choice(ccols) for k in range(d)]
    for j, col in enumerate(arr):
        gi = fill(gi, col, connect((j, 0), (j, j)) | connect((0, j), (j, j)))
    go = canvas(0, (2 * d, 2 * d))
    for j in range(2 * d):
        col = arr[j % len(arr)]
        go = fill(go, col, connect((j, 0), (j, j)) | connect((0, j), (j, j)))
    if rot is None:
        rot = random.choice(ROTS)
    rotf = (identity, rot90, rot180, rot270)[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    d = I.shape[0]

    # side n of the nested-L band square drawn in I
    nz = np.argwhere(I != 0)
    n = max(int(nz[:, 0].max() - nz[:, 0].min()) + 1,
            int(nz[:, 1].max() - nz[:, 1].min()) + 1)

    def ring(R, C, j, lim):
        # L-band j = Chebyshev ring j around corner (R, C), clipped to a lim x lim grid
        return [(r, c) for r in range(lim) for c in range(lim)
                if max(abs(r - R), abs(c - C)) == j]

    # apex = the corner from which I's nonzero cells form single-colour L-bands
    apex = None
    for (R, C) in [(0, 0), (0, d - 1), (d - 1, 0), (d - 1, d - 1)]:
        ok = True
        mask = np.zeros((d, d), dtype=bool)
        for j in range(n):
            cells = ring(R, C, j, d)
            vals = {int(I[r, c]) for r, c in cells}
            if len(vals) != 1 or 0 in vals:
                ok = False
                break
            for r, c in cells:
                mask[r, c] = True
        if ok and np.all(I[~mask] == 0):
            apex = (R, C)
            break
    if apex is None:
        r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
        apex = (0 if r0 == 0 else d - 1, 0 if c0 == 0 else d - 1)
    Ri, Ci = apex

    # the colour period, read outward from the apex along the diagonal
    sr = 1 if Ri == 0 else -1
    sc = 1 if Ci == 0 else -1
    arr = [int(I[Ri + sr * j, Ci + sc * j]) for j in range(n)]

    # output: side doubles, apex stays at the same corner
    D = 2 * d
    Ro = 0 if Ri == 0 else D - 1
    Co = 0 if Ci == 0 else D - 1

    ops, sels = [], []
    # 1. grow the canvas to 2d x 2d (whole rectangle -> bbox is exact)
    ops.append(33); sels.append([0, 0, D - 1, D - 1])

    # 2. if the apex is not the top-left corner, carry I's n x n band square
    #    (a full rectangle with no zero cells) to the matching corner of the canvas
    qr = Ri if Ri == 0 else d - n
    qc = Ci if Ci == 0 else d - n
    dr = Ro if Ro == 0 else D - n
    dc = Co if Co == 0 else D - n
    if (qr, qc) != (dr, dc):
        ops.append(28); sels.append([qr, qc, n - 1, n - 1])
        ops.append(30); sels.append([dr, dc, 0, 0])

    # 3. continue the period outward: each band j >= n from the apex takes the
    #    colour of input band j mod n; bands that share a colour in the input
    #    period are painted in one stroke
    by_col = {}
    for j in range(n, D):
        by_col.setdefault(arr[j % n], []).extend(ring(Ro, Co, j, D))
    for col in sorted(by_col, key=lambda k: arr.index(k)):
        ops.append(col); sels.append(sel_of(by_col[col]))

    ops.append(34); sels.append([0, 0, D - 1, D - 1])
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
