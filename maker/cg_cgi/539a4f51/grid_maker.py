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


def sample_colors(num_examples=None) -> dict:
    # Background is hardcoded 0; band colours are sampled per instance and the
    # rule (repeat the band sequence outward) does not depend on which colours.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    dmax = max(2, min(15, max_h // 2, max_w // 2))
    d = random.randint(2, dmax)
    cols = list(range(1, 10))
    numc = random.randint(2, 9)
    ccols = random.sample(cols, numc)
    numocc = random.randint(1, d)
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
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    d = I.shape[0]

    # side of the nested-L square in I (= period of the band sequence)
    nz = np.argwhere(I != 0)
    r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
    r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
    n = max(r1 - r0 + 1, c1 - c0 + 1)

    # apex corner: cells at Chebyshev distance j < n from it share one nonzero
    # colour per j, and everything farther is empty
    apex = None
    for (R, C) in [(0, 0), (0, d - 1), (d - 1, 0), (d - 1, d - 1)]:
        ok = True
        rings = {}
        for r in range(d):
            for c in range(d):
                j = max(abs(r - R), abs(c - C))
                if j < n:
                    rings.setdefault(j, set()).add(int(I[r, c]))
                elif I[r, c] != 0:
                    ok = False
        if ok:
            for j in range(n):
                vals = rings.get(j, set())
                if len(vals) != 1 or 0 in vals:
                    ok = False
                    break
        if ok:
            apex = (R, C)
            break
    if apex is None:
        apex = (0 if r0 == 0 else d - 1, 0 if c0 == 0 else d - 1)
    Ri, Ci = apex
    sr = 1 if Ri == 0 else -1
    sc = 1 if Ci == 0 else -1

    # band sequence read from I along the diagonal away from the apex
    arr = [int(I[Ri + sr * j, Ci + sc * j]) for j in range(n)]

    D = 2 * d
    Ro = 0 if Ri == 0 else D - 1
    Co = 0 if Ci == 0 else D - 1

    ops, sels = [], []
    # 1. expand canvas to 2d x 2d (whole rectangle -> bbox is exact)
    ops.append(33)
    sels.append([0, 0, D - 1, D - 1])
    G = np.zeros((D, D), dtype=int)
    G[:d, :d] = I

    def paint(cells, col):
        if all(G[r, c] == col for (r, c) in cells):
            return  # no visible change -> skip
        ops.append(col)
        sels.append(sel_of(cells))
        for (r, c) in cells:
            G[r, c] = col

    # 2. base: every line parallel to the apex column carries the periodic
    #    band sequence (each row of the output = the sequence)
    for c in range(D):
        paint([(r, c) for r in range(D)], arr[abs(c - Co) % n])

    # 3. overlay: row at distance j from the apex gets its prefix (cells within
    #    distance j of the apex column) covered by band colour j
    for r in range(D):
        j = abs(r - Ro)
        paint([(r, c) for c in range(D) if abs(c - Co) <= j], arr[j % n])

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
