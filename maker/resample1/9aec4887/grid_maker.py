"""
ARC Task: 9aec4887 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    pc = random.choice([c for c in cols if c != 0])  # pc must be nonzero so Move can carry it
    rest = [c for c in cols if c != pc]
    bgc, c1, c2, c3, c4 = random.sample(rest, 5)
    return {"bgc": bgc, "pc": pc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, pc, c1, c2, c3, c4) -> dict:
    h = _unifint(diff_lb, diff_ub, (12, max(12, max_h)))
    w = _unifint(diff_lb, diff_ub, (12, max(12, max_w)))
    oh = _unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = _unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((oh, ow), bgc, dtype=int)
    lines = [
        (c1, [(i, 0) for i in range(1, oh - 1)]),
        (c2, [(i, ow - 1) for i in range(1, oh - 1)]),
        (c3, [(0, j) for j in range(1, ow - 1)]),
        (c4, [(oh - 1, j) for j in range(1, ow - 1)]),
    ]
    for col, cells in lines:
        for (i, j) in cells:
            go[i, j] = col
    bounds = {(i, j) for i in range(oh - 2) for j in range(ow - 2)}

    def nbrs(cell):
        i, j = cell
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    objA = {random.choice(sorted(bounds))}
    ncells = _unifint(diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))

    def grow():
        cand = set()
        for c in objA:
            cand |= nbrs(c)
        cand = sorted((bounds - objA) & cand)
        objA.add(random.choice(cand))

    for _ in range(ncells - 1):
        grow()

    def shp():
        rs = [i for i, _ in objA]; cs = [j for _, j in objA]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp() != (oh - 2, ow - 2):
        grow()
    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    gi[loci:loci + oh, locj:locj + ow] = go
    plcdi = {(loci + i, locj + j) for i in range(oh) for j in range(ow)}
    rems = [(i, j) for i in range(h) for j in range(w)
            if (i, j) not in plcdi and loci + oh <= i <= h - oh + 2 and j <= w - ow + 2]
    li, lj = random.choice(rems)
    for (i, j) in objA:
        gi[i + li, j + lj] = pc
    for (i, j) in objA:
        ii, jj = i + 1, j + 1
        ds = []
        for col, cells in lines:
            ds.append((min(abs(ii - a) + abs(jj - b) for a, b in cells), col))
        m = min(d for d, _ in ds)
        winners = [col for d, col in ds if d == m]
        go[ii, jj] = winners[0] if len(winners) == 1 else pc
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, -k)
    go = np.rot90(go, -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # classify colours: the four frame sides are straight lines, the shape is not
    lines, pc, shape = [], None, None
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        cells = [tuple(int(v) for v in x) for x in np.argwhere(I == col)]
        rs = {r for r, _ in cells}; cs = {c for _, c in cells}
        if len(rs) == 1 or len(cs) == 1:
            lines.append((col, cells))
        else:
            pc, shape = col, cells
    allc = [p for _, cells in lines for p in cells]
    fr0 = min(r for r, _ in allc); fr1 = max(r for r, _ in allc)
    fc0 = min(c for _, c in allc); fc1 = max(c for _, c in allc)
    sr0 = min(r for r, _ in shape); sc0 = min(c for _, c in shape)
    dr = (fr0 + 1) - sr0
    dc = (fc0 + 1) - sc0

    ops, sels = [], []
    # slide the shape into the frame interior: one grab, then empty-selection moves
    cur = list(shape)
    steps = [(20 if dr < 0 else 21, (-1 if dr < 0 else 1, 0))] * abs(dr) + \
            [(23 if dc < 0 else 22, (0, -1 if dc < 0 else 1))] * abs(dc)
    for k, (op, (a, b)) in enumerate(steps):
        ops.append(op); sels.append(sel_of(cur) if k == 0 else sel_of([]))
        cur = [(r + a, c + b) for r, c in cur]
    # the source footprint always lies outside the frame bbox, so the crop discards it

    # recolour each shape cell by its uniquely nearest frame side (ties keep pc)
    groups = {}
    for (r, c) in cur:
        ds = [(min(abs(r - a) + abs(c - b) for a, b in cells), col) for col, cells in lines]
        m = min(d for d, _ in ds)
        win = [col for d, col in ds if d == m]
        col = win[0] if len(win) == 1 else pc
        if col != pc:
            groups.setdefault(col, []).append((r, c))
    for col, cells in lines:
        if col in groups:
            ops.append(col); sels.append(sel_of(sorted(groups[col])))

    # crop to the frame's full bounding rectangle (whole rectangle intended)
    ops.append(33); sels.append([fr0, fc0, fr1 - fr0, fc1 - fc0])
    ho, wo = fr1 - fr0 + 1, fc1 - fc0 + 1
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
                        f"num_examples+1 ({num_examples + 1}) for task 9aec4887"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9aec4887"
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
                                f"for task 9aec4887"
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
                    f"Failed to build a complete episode for task 9aec4887 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9aec4887-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
