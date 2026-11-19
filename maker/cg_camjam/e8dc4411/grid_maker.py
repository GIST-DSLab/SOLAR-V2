"""
ARC Task: e8dc4411 (RE-ARC) — LLM-generated grid_maker
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
    bgc, objc, remc = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "remc": remc}


def _unif(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    t = random.uniform(lb, ub)
    return max(a, min(b, a + int(round(t * (b - a)))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, remc) -> dict:
    h = _unif(diff_lb, diff_ub, (9, max(9, max_h)))
    w = _unif(diff_lb, diff_ub, (9, max(9, max_w)))
    d = _unif(diff_lb, diff_ub, (3, max(3, min(h, w) // 2 - 1)))

    # 4-fold symmetric shape on a d x d canvas
    inds = [(i, j) for i in range(d) for j in range(d) if i >= d // 2 and j >= d // 2]
    ncd = _unif(diff_lb, diff_ub, (1, max(1, len(inds) // 2)))
    nc = random.choice((ncd, len(inds) - ncd))
    nc = min(max(2, nc), len(inds) - 1)
    cells = set(random.sample(inds, nc))
    cells |= {random.choice(((d // 2, d // 2), (d // 2, d // 2 - 1)))}
    cells |= {(j, i) for i, j in cells}
    arr = np.zeros((d, d), dtype=int)
    for _ in range(4):
        for (i, j) in cells:
            arr[i, j] = 1
        arr = np.rot90(arr)
    while arr.shape[0] > 3:
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        if border.any():
            break
        arr = arr[1:-1, 1:-1]
    obj = {(int(i), int(j)) for i, j in zip(*np.nonzero(arr))}
    ri = [i for i, _ in obj]
    od = max(ri) - min(ri) + 1

    loci = random.randint(1, h - 2 * od)
    locj = random.randint(1, w - 2 * od)
    obj = {(i + loci, j + locj) for i, j in obj}
    rmin = min(i for i, _ in obj); rmax = max(i for i, _ in obj)
    cmin = min(j for _, j in obj); cmax = max(j for _, j in obj)
    p = 0
    while any(rmin <= i + p <= rmax and cmin <= j + p <= cmax for i, j in obj):
        p += 1
    obj2 = {(i + p, j + p) for i, j in obj}

    def nb(s):
        return {(i + a, j + b) for i, j in s for a in (-1, 0, 1) for b in (-1, 0, 1)
                if (a, b) != (0, 0)}
    nbhs = nb(obj)
    while not (obj2 & nbhs):
        nbhs = nb(nbhs)
    indic = obj2 & nbhs

    gi = np.full((h, w), bgc, dtype=int)
    for i, j in obj:
        gi[i, j] = objc
    for i, j in indic:
        if 0 <= i < h and 0 <= j < w:
            gi[i, j] = remc
    go = gi.copy()
    for k in range(30):
        new = go.copy()
        for i, j in obj:
            a, b = i + p * (k + 1), j + p * (k + 1)
            if 0 <= a < h and 0 <= b < w:
                new[a, b] = remc
        if np.array_equal(new, go):
            break
        go = new
    rk = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, rk)
    go = np.rot90(go, rk)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """The code-golf entry reads the task the same way the maker does. The step vector
    is the diagonal offset that puts the first copy of the shape on the marker. Each
    unit is one copy of the shape, stamped in the marker colour. Copies are placed
    k*step away for k = 1, 2, ... until they leave the grid. The golf entry works out
    the step from where the marker sits against the shape's edge. The maker works it out
    as the smallest diagonal shift that clears the shape's bounding box, with the marker
    choosing the direction. Both give the same vector, so the carving does not change:
    one Color op per stamped copy, and every selection is the cells that copy lands on.
    O is never read."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    full = [(r, c) for r in range(h) for c in range(w)]

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    classes = {}
    for r in range(h):
        for c in range(w):
            v = int(I[r, c])
            if v != bgc:
                classes.setdefault(v, []).append((r, c))

    ops, sels = [], []
    if len(classes) < 2:
        ops.append(34); sels.append(sel_of(full))
        return ops, sels

    objc = max(classes, key=lambda k: (len(classes[k]), -k))
    remc = min([k for k in classes if k != objc], key=lambda k: (len(classes[k]), k))
    obj = set(classes[objc])
    mark = set(classes[remc])

    rmin = min(r for r, _ in obj); rmax = max(r for r, _ in obj)
    cmin = min(c for _, c in obj); cmax = max(c for _, c in obj)

    # step size: first diagonal shift that clears the shape's own bounding box
    span = max(rmax - rmin, cmax - cmin) + 2
    p = 1
    while p <= span:
        if not any(rmin <= r + p <= rmax and cmin <= c + p <= cmax for r, c in obj):
            break
        p += 1

    # direction: the diagonal whose first copy covers the marker
    best_dir, best_hit = (1, 1), -1
    for dr in (-1, 1):
        for dc in (-1, 1):
            hit = len({(r + p * dr, c + p * dc) for r, c in obj} & mark)
            if hit > best_hit:
                best_hit, best_dir = hit, (dr, dc)
    dr, dc = best_dir

    # stamp each copy of the shape, in the marker colour, walking off the grid
    cur = I.copy()
    for k in range(1, max(h, w) + 1):
        cells = [(r + k * p * dr, c + k * p * dc) for (r, c) in sorted(obj)]
        cells = [(r, c) for (r, c) in cells if 0 <= r < h and 0 <= c < w]
        if not cells:
            break
        if all(cur[r, c] == remc for (r, c) in cells):
            continue
        ops.append(int(remc))
        sels.append(sel_of(cells))          # exact cells this copy lands on
        for (r, c) in cells:
            cur[r, c] = remc

    ops.append(34)
    sels.append(sel_of(full))               # Submit: whole grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task e8dc4411"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e8dc4411"
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
                                f"for task e8dc4411"
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
                    f"Failed to build a complete episode for task e8dc4411 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e8dc4411-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
