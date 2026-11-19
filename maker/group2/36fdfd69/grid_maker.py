"""
ARC Task: 36fdfd69 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 4]
    bgc, fgc, objc = random.sample(cols, 3)
    return {"bgc": bgc, "fgc": fgc, "objc": objc}


def _unif(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    d = random.uniform(diff_lb, diff_ub)
    return max(lo, min(hi, int(round(lo + d * (hi - lo)))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, objc=None) -> dict:
    if bgc is None or fgc is None or objc is None:
        cols = [c for c in range(10) if c != 4]
        bgc, fgc, objc = random.sample(cols, 3)
    h = _unif(diff_lb, diff_ub, 10, max(10, min(30, max_h)))
    w = _unif(diff_lb, diff_ub, 10, max(10, min(30, max_w)))
    nobjs = _unif(diff_lb, diff_ub, 1, max(1, (h * w) // 30))
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    namt = random.randint(int(0.35 * h * w), int(0.65 * h * w))
    for (i, j) in random.sample(sorted(inds), namt):
        gi[i][j] = fgc
    go = [row[:] for row in gi]

    def nbrs8(cell):
        i, j = cell
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    def count(g, c):
        return sum(row.count(c) for row in g)

    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(2, 7)
        ow = random.randint(2, 7)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loci, locj = random.choice(sorted(cands))
        bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        if not bd.issubset(inds):
            continue
        ncells = random.randint(2, oh * ow - 1)

        def grow():
            obj = {random.choice(sorted(bd))}
            for _ in range(ncells - 1):
                fr = set()
                for c in obj:
                    fr |= nbrs8(c)
                fr = (fr & bd) - obj
                if not fr:
                    break
                obj.add(random.choice(sorted(fr)))
            return obj

        def is_full(obj):
            rs = [p[0] for p in obj]; cs = [p[1] for p in obj]
            return len(obj) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

        obj = grow()
        guard = 0
        while is_full(obj) and guard < 50:
            obj = grow(); guard += 1
        if is_full(obj):
            continue
        rs = [p[0] for p in obj]; cs = [p[1] for p in obj]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        bd2 = {(i, j) for i in range(r0, r1 + 1) for j in range(c0, c1 + 1)}
        gi2 = [row[:] for row in gi]
        for (i, j) in bd2:
            gi2[i][j] = fgc
        for (i, j) in obj:
            gi2[i][j] = objc
        if count(gi2, objc) < min(count(gi2, fgc), count(gi2, bgc)):
            succ += 1
            ring = {(i, j) for i in range(r0 - 2, r1 + 3) for j in range(c0 - 2, c1 + 3)}
            inds = inds - ring
            gi = gi2
            for (i, j) in bd2:
                go[i][j] = 4
            for (i, j) in obj:
                go[i][j] = objc
    if succ == 0:
        raise ValueError("no object placed")
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    """Same rule as the maker, carved differently: work one colour layer at a time
    over the whole grid.  (1) Paint the union of every object's backdrop rectangle
    with 4, covering the object cells too (laying the base).  (2) Repaint all cells
    of the object colour, taken from I, with that colour so the objects show on top
    of the base.  Nothing here reads O."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    least = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    obj_cells = sorted((r, c) for r in range(hi) for c in range(wi) if I[r, c] == least)

    def closure(cells):
        out = set()
        cl = list(cells)
        for (r0, c0) in cl:
            near = [(r, c) for (r, c) in cl if max(abs(r - r0), abs(c - c0)) < 3]
            rs = [p[0] for p in near]; cs = [p[1] for p in near]
            for r in range(min(rs), max(rs) + 1):
                for c in range(min(cs), max(cs) + 1):
                    out.add((r, c))
        return out

    if obj_cells:
        region = closure(closure(obj_cells))
        # Layer 1: the whole backdrop region (all objects at once) becomes 4.
        ops.append(4); sels.append(sel_of(sorted(region)))
        # Layer 2: every object-colour cell of I is repainted with its colour.
        ops.append(int(least)); sels.append(sel_of(obj_cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 36fdfd69"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 36fdfd69"
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
                                f"for task 36fdfd69"
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
                    f"Failed to build a complete episode for task 36fdfd69 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"36fdfd69-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
