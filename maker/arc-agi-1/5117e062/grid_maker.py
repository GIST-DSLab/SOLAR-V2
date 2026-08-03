"""
ARC Task: 5117e062 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    nobjs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 15)))
    tr = 0
    maxtr = 4 * nobjs
    done = False
    succ = 0
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    go = canvas(bgc, (1, 1))
    while tr < maxtr and succ < nobjs:
        oh = randint(2, 6)
        ow = randint(2, 6)
        bx = asindices(canvas(-1, (oh, ow)))
        nc = randint(3, oh * ow)
        sp = choice(totuple(bx))
        bx = remove(sp, bx)
        obj = {sp}
        for k in range(nc - 1):
            cand = totuple((bx - obj) & mapply(neighbors, obj))
            if len(cand) == 0:
                break
            obj.add(choice(cand))
        if not done:
            done = True
            idx = choice(totuple(obj))
            coll = choice(remcols)
            obj2 = {(coll, idx)}
            coll2 = choice(remove(coll, remcols))
            obj3 = recolor(coll2, remove(idx, obj))
            obj = obj2 | obj3
            go = fill(canvas(bgc, shape(obj)), coll2, normalize(obj))
        else:
            obj = recolor(choice(remcols), obj)
        locopts = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        tr += 1
        if len(locopts) == 0:
            continue
        loc = choice(totuple(locopts))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            gi = paint(gi, plcd)
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = most common color of I
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected components of non-bg cells (colors merged; generator keeps objects apart)
    visited = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] != bgc:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(cells)

    # the unique multi-color object
    target = None
    for cells in comps:
        if len({I[y, x] for (y, x) in cells}) > 1:
            target = cells
            break
    if target is None:
        target = max(comps, key=len)

    ys = [y for y, x in target]
    xs = [x for y, x in target]
    r0, r1 = min(ys), max(ys)
    c0, c1 = min(xs), max(xs)
    oh = r1 - r0 + 1
    ow = c1 - c0 + 1

    maj = Counter(I[y, x] for (y, x) in target).most_common(1)[0][0]
    objset = set(target)

    ops = []
    sels = []

    # 1) clear any non-object cells inside the bbox to bgc (skip cells already bgc)
    clear_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                   if (r, c) not in objset and I[r, c] != bgc]
    if clear_cells:
        ops.append(int(bgc))
        sels.append(sel_of(clear_cells))

    # 2) unify object color: recolor minority cells to the majority color
    minority = [(y, x) for (y, x) in target if I[y, x] != maj]
    if minority:
        ops.append(int(maj))
        sels.append(sel_of(minority))

    # 3) crop to the object's bounding box (full rectangle incl. bgc corners)
    ops.append(33)
    sels.append([r0, c0, oh - 1, ow - 1])

    ops.append(34)
    sels.append([0, 0, oh - 1, ow - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 5117e062"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5117e062"
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
                                f"for task 5117e062"
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
                    f"Failed to build a complete episode for task 5117e062 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5117e062-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
