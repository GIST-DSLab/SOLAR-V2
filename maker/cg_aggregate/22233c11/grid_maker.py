"""
ARC Task: 22233c11 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 8)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols) -> dict:
    h = random.randint(5, max(5, max_h))
    w = random.randint(5, max(5, max_w))
    nobjs = random.randint(1, max(1, (h * w) // 10))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}
    full = set(inds)
    succ, tr, maxtr = 0, 0, 10 * nobjs
    while succ < nobjs and tr < maxtr and inds:
        tr += 1
        od = random.randint(1, 3)
        col = random.choice(ccols)
        mk = {(0, 3), (3, 0)}
        ob = {(1, 1), (2, 2)}
        if random.choice((True, False)):
            mk = {(3 - r, c) for r, c in mk}
            ob = {(3 - r, c) for r, c in ob}
        lr, lc = random.choice(sorted(inds))
        up = lambda S: {(lr + r * od + i, lc + c * od + j) for r, c in S
                        for i in range(od) for j in range(od)}
        inobj, mko = up(ob), up(mk)
        outi = inobj | mko
        if inobj <= inds and (outi & full) <= inds:
            succ += 1
            nb = {(r + a, c + b) for r, c in outi for a in (-1, 0, 1) for b in (-1, 0, 1)}
            inds = inds - outi - nb
            for r, c in inobj:
                gi[r][c] = col
                go[r][c] = col
            for r, c in mko & full:
                go[r][c] = 8
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Each object is two od-squares touching at a corner inside a 2od x 2od box;
    the other two squares of that box are empty.  Rule: mark each empty square
    with 8 and push it diagonally OUTWARD (away from the box centre) by od cells.
    Per empty square: Color8 it (named from the input alone: the box cells the
    object does not cover), slide it diagonally with unit Moves (grab once, then
    empty selections), then restore the canvas at its starting footprint."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = int(I[0, 0])        # row 0 is never part of an object (squares start od inside the frame)
    ops, sels = [], []

    seen = np.zeros((hi, wi), bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                st = [(r, c)]; seen[r, c] = True; cells = []
                while st:
                    a, b = st.pop(); cells.append((a, b))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            na, nb = a + dr, b + dc
                            if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] \
                                    and I[na, nb] != bgc:
                                seen[na, nb] = True; st.append((na, nb))
                comps.append(cells)
    comps.sort(key=lambda cs: (min(x for x, _ in cs), min(y for _, y in cs)))

    def inside(S):
        return [(a, b) for a, b in S if 0 <= a < hi and 0 <= b < wi]

    for cells in comps:
        r0 = min(x for x, _ in cells); c0 = min(y for _, y in cells)
        od = max(1, (max(x for x, _ in cells) - r0 + 1) // 2)   # bbox is 2od x 2od
        cellset = set(cells)
        for bi in (0, 1):
            for bj in (0, 1):
                blk = [(r0 + bi * od + i, c0 + bj * od + j)
                       for i in range(od) for j in range(od)]
                if any(p in cellset for p in blk):
                    continue          # this square is part of the object
                vr = -1 if bi == 0 else 1     # outward direction from box centre
                vc = -1 if bj == 0 else 1
                dest = [(a + vr * od, b + vc * od) for a, b in blk]
                if not inside(dest):
                    continue          # marker would land entirely off the grid
                ops.append(8); sels.append(sel_of(blk))
                cur = list(blk)
                first = True
                steps = []
                for _ in range(od):
                    steps += [(20 if vr < 0 else 21, vr, 0),
                              (23 if vc < 0 else 22, 0, vc)]
                for op, dr, dc in steps:
                    if not inside(cur):
                        break             # fully clipped: further moves change nothing
                    ops.append(op)
                    sels.append(sel_of(cur) if first else sel_of([]))
                    first = False
                    cur = inside([(a + dr, b + dc) for a, b in cur])
                hole = sorted(set(blk) - set(cur))
                if bgc != 0 and hole:
                    ops.append(bgc); sels.append(sel_of(hole))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])   # full-grid bbox for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 22233c11"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22233c11"
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
                                f"for task 22233c11"
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
                    f"Failed to build a complete episode for task 22233c11 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22233c11-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
