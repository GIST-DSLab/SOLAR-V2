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

ROTS = [0, 1, 2, 3]   # identity, rot90, rot180, rot270 -> which grid corner is the apex


def sample_colors(num_examples=None) -> dict:
    # Background is hardcoded 0 and the band colours are read from each input,
    # so no colour needs fixing. The one structural variant is the apex corner
    # (the rotation), so plan it per instance and make sure each corner shows up.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append({"rot": random.choice(ex)})
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, rot=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo = max(a, min(b, lo))
        hi = max(lo, min(b, hi))
        return random.randint(lo, hi)

    if rot is None:
        rot = random.choice(ROTS)
    dmax = max(2, min(15, max_h // 2, max_w // 2))     # the output is 2d x 2d
    d = unifint(diff_lb, diff_ub, (2, dmax))
    cols = list(range(1, 10))
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
    D = 2 * d
    go = np.zeros((D, D), dtype=int)
    for j in range(D):
        col = arr[j % len(arr)]
        go[j, :j + 1] = col
        go[:j + 1, j] = col
    gi = np.rot90(gi, k=-rot)   # clockwise rot90 applied `rot` times
    go = np.rot90(go, k=-rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    d = I.shape[0]
    nz = np.argwhere(I != 0)
    r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
    r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
    n = max(r1 - r0 + 1, c1 - c0 + 1)      # side of the nested-L square = colour period

    def band(R, C, sr, sc, j):
        cells = set()
        for k in range(j + 1):
            cells.add((R + sr * j, C + sc * k))
            cells.add((R + sr * k, C + sc * j))
        return cells

    # apex = the grid corner from which I's nonzero cells form single-colour L-bands
    apex = None
    for (R, C) in [(0, 0), (0, d - 1), (d - 1, 0), (d - 1, d - 1)]:
        sr = 1 if R == 0 else -1
        sc = 1 if C == 0 else -1
        ok, mask = True, np.zeros((d, d), dtype=bool)
        for j in range(n):
            cells = band(R, C, sr, sc, j)
            if any(not (0 <= r < d and 0 <= c < d) for r, c in cells):
                ok = False
                break
            vals = {int(I[r, c]) for r, c in cells}
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
    # band colours read outward from the apex along the diagonal
    arr = [int(I[Ri + sr * j, Ci + sc * j]) for j in range(n)]

    D = 2 * d
    Ro = 0 if Ri == 0 else D - 1
    Co = 0 if Ci == 0 else D - 1

    ops, sels = [], []
    # 1. grow the canvas to 2d x 2d (the whole rectangle, so a bbox is exact)
    ops.append(33)
    sels.append([0, 0, D - 1, D - 1])
    G = np.zeros((D, D), dtype=int)
    G[:d, :d] = I

    # 2. Column layer: paint each full column, counted outward from the apex
    #    edge, in the band colour for its distance (period n). This also covers
    #    the old input content left at the top-left after the resize.
    for v in range(D):
        col = arr[v % n]
        c = Co + sc * v
        cells = [(r, c) for r in range(D)]
        ops.append(col)
        sels.append(sel_of(cells))
        for rc in cells:
            G[rc] = col

    # 3. Row layer: below the diagonal the row's band colour wins. In each row,
    #    paint the stretch from the apex edge up to the diagonal, skipping cells
    #    that already hold that colour in the current grid.
    for u in range(1, D):
        col = arr[u % n]
        r = Ro + sr * u
        seg = [(r, Co + sc * v) for v in range(u)]
        cells = [rc for rc in seg if G[rc] != col]
        if not cells:
            continue
        ops.append(col)
        sels.append(sel_of(cells))
        for rc in cells:
            G[rc] = col

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
