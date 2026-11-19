"""
ARC Task: 48d8fb45 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    def nbrs8(cell):
        r, c = cell
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 15)))
    gi = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}
    tr, maxtr, succ = 0, 4 * nobjs, 0
    done = False
    go = None
    target_placed = False
    while tr < maxtr and succ < nobjs:
        oh = random.randint(2, 6)
        ow = random.randint(2, 6)
        bx = {(r, c) for r in range(oh) for c in range(ow)}
        nc = random.randint(3, oh * ow)
        sp = random.choice(sorted(bx))
        bx.discard(sp)
        obj = {sp}
        for _ in range(nc - 1):
            cand = set()
            for cell in obj:
                cand |= nbrs8(cell)
            cand = (bx - obj) & cand
            obj.add(random.choice(sorted(cand)))
        is_target = False
        if not done:
            done = True
            is_target = True
            idx = random.choice(sorted(obj))
            coll = random.choice(remcols)
            mainc = random.choice([c for c in remcols if c != coll])
            rest = obj - {idx}
            pobj = {(coll, idx)} | {(mainc, cell) for cell in rest}
            r0 = min(r for r, c in rest); c0 = min(c for r, c in rest)
            r1 = max(r for r, c in rest); c1 = max(c for r, c in rest)
            go = [[bgc] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
            for (r, c) in rest:
                go[r - r0][c - c0] = mainc
        else:
            colr = random.choice(remcols)
            pobj = {(colr, cell) for cell in obj}
        tr += 1
        locopts = [ij for ij in sorted(inds) if ij[0] <= h - oh and ij[1] <= w - ow]
        if not locopts:
            continue
        lr, lc = random.choice(locopts)
        plcd = {(v, (r + lr, c + lc)) for v, (r, c) in pobj}
        plcdi = {cell for _, cell in plcd}
        if plcdi.issubset(inds):
            for v, (r, c) in plcd:
                gi[r][c] = v
            succ += 1
            if is_target:
                target_placed = True
            rem = set(plcdi)
            for cell in plcdi:
                rem |= nbrs8(cell)
            inds = inds - rem
    if not target_placed:
        raise ValueError("target not placed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background: the canvas colour the generator paints before placing sparse objects
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected non-background components (multicolour allowed)
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            na, nb = a + da, b + db
                            if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and I[na, nb] != bgc:
                                seen[na, nb] = True
                                stack.append((na, nb))
                comps.append(cells)

    # the pattern object that the indicator pixel touches: the only two-colour component
    target = max(comps, key=lambda cs: len(set(int(I[a, b]) for a, b in cs)))
    cnt = Counter(int(I[a, b]) for a, b in target)
    main = cnt.most_common(1)[0][0]
    pattern = [(a, b) for a, b in target if int(I[a, b]) == main]
    indicator = [(a, b) for a, b in target if int(I[a, b]) != main]
    pset = set(pattern)

    r0 = min(a for a, b in pattern); r1 = max(a for a, b in pattern)
    c0 = min(b for a, b in pattern); c1 = max(b for a, b in pattern)

    def in_win(a, b):
        return r0 <= a <= r1 and c0 <= b <= c1

    ops, sels = [], []

    # remove the indicator pixel if it lies in the pattern's crop window
    ind_in = sorted((a, b) for a, b in indicator if in_win(a, b))
    if ind_in:
        ops.append(int(bgc)); sels.append(sel_of(ind_in))

    # remove pieces of any other object that intrude into the crop window
    for cs in comps:
        if cs is target:
            continue
        intr = sorted((a, b) for a, b in cs if in_win(a, b) and (a, b) not in pset)
        if intr:
            ops.append(int(bgc)); sels.append(sel_of(intr))

    # crop to the pattern's bbox: intended cells are exactly that full rectangle
    ops.append(33)
    sels.append([r0, c0, r1 - r0, c1 - c0])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 48d8fb45"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 48d8fb45"
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
                                f"for task 48d8fb45"
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
                    f"Failed to build a complete episode for task 48d8fb45 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"48d8fb45-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
