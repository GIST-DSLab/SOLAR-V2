"""
ARC Task: 6e82a1ae (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import deque, Counter

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Generator samples bgc and fgc from {0,4,5,6,7,8,9} (1,2,3 are reserved
    # for the output colors). Both are fixed once per episode.
    cols = [0, 4, 5, 6, 7, 8, 9]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, fgc: int) -> dict:
    b = frozenset({frozenset({ORIGIN, RIGHT}), frozenset({ORIGIN, DOWN})})
    c = frozenset({
        frozenset({ORIGIN, DOWN, UNITY}),
        frozenset({ORIGIN, DOWN, RIGHT}),
        frozenset({UNITY, DOWN, RIGHT}),
        frozenset({UNITY, ORIGIN, RIGHT}),
        shift(frozenset({ORIGIN, UP, DOWN}), DOWN),
        shift(frozenset({ORIGIN, LEFT, RIGHT}), RIGHT)
    })
    d = set()
    for k in range(100):
        shp = {(0, 0)}
        for jj in range(3):
            shp.add(choice(totuple(mapply(dneighbors, shp) - shp)))
        shp = frozenset(normalize(shp))
        d.add(shp)
    d = frozenset(d)
    d, b, c = totuple(d), totuple(b), totuple(c)
    prs = [(b, 3), (c, 2), (d, 1)]

    hub = max(5, min(30, int(max_h)))
    wub = max(5, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (5, hub))
    w = unifint(diff_lb, diff_ub, (5, wub))

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    reminds = asindices(gi)

    nub = max(3, ((h * w) // 2) // 3)
    nobjs = unifint(diff_lb, diff_ub, (3, nub))

    # every instance shows all three size classes -> the size->color map is
    # always readable from the examples
    plan = [prs[0], prs[1], prs[2]] + [choice(prs) for _ in range(nobjs - 3)]
    random.shuffle(plan)

    maxtr = 50
    for objs, col in plan:
        obj = choice(objs)
        ntr = 0
        while ntr < maxtr:
            loc = choice(totuple(reminds))
            olcd = shift(obj, loc)
            if olcd.issubset(reminds):
                gi = fill(gi, fgc, olcd)
                go = fill(go, col, olcd)
                reminds = (reminds - olcd) - mapply(dneighbors, olcd)
                break
            ntr += 1

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    def components(color):
        seen = np.zeros((hi, wi), dtype=bool)
        comps = []
        for r in range(hi):
            for c in range(wi):
                if I[r, c] != color or seen[r, c]:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                and I[nr, nc] == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                comps.append(cells)
        return comps

    # Two colors only: background is the one whose cells form a big region.
    present = sorted(set(I.flatten().tolist()))
    comps_by_color = {col: components(col) for col in present}
    bgc = None
    best = -1
    for col in present:
        mx = max((len(cm) for cm in comps_by_color[col]), default=0)
        if mx > best:
            best = mx
            bgc = col
    if best <= 4:  # fallback: no color has an oversized region
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    objs = []
    for col in present:
        if col == bgc:
            continue
        objs.extend(comps_by_color[col])

    # Rule measured from I: an object's cell count decides its new color.
    # 2 -> 3, 3 -> 2, 4 -> 1.
    def size_class(cells):
        n = len(cells)
        return 2 if n <= 2 else (3 if n == 3 else 4)

    CLASS_COLOR = {2: 3, 3: 2, 4: 1}

    def sweep(group, direction):
        # 0: top->bottom / left->right, 1: bottom->top / right->left,
        # 2: column by column, left->right, 3: column by column, right->left
        if direction == 0:
            return sorted(group, key=lambda ob: min(ob))
        if direction == 1:
            return sorted(group, key=lambda ob: min(ob), reverse=True)
        if direction == 2:
            return sorted(group, key=lambda ob: min((c, r) for r, c in ob))
        return sorted(group, key=lambda ob: min((c, r) for r, c in ob),
                      reverse=True)

    counts = {s: 0 for s in (2, 3, 4)}
    for ob in objs:
        counts[size_class(ob)] += 1

    ASC = (2, 3, 4)                      # smallest objects first
    DESC = (4, 3, 2)                     # biggest objects first
    COMMON = tuple(sorted((2, 3, 4), key=lambda s: (-counts[s], s)))  # busiest class first

    # Which route to take is read off the puzzle itself (its shape and how
    # crowded it is), never off the answer: same input -> same route.
    nfg = int(np.count_nonzero(I != bgc))
    route = (hi * 7 + wi * 13 + len(objs) * 29 + nfg * 3) % 9

    # (kind, class order, sweep direction)
    #  "size"    : work through one size class at a time, object by object
    #  "spatial" : walk the grid once, recoloring each object as it is met
    #  "batch"   : one stroke per size class, all its objects together
    ROUTES = [
        ("size", ASC, 0),
        ("size", DESC, 1),
        ("size", COMMON, 2),
        ("size", ASC, 2),
        ("spatial", ASC, 0),
        ("spatial", ASC, 1),
        ("spatial", ASC, 2),
        ("batch", ASC, 0),
        ("batch", COMMON, 0),
    ]
    kind, class_order, direction = ROUTES[route]

    ops, sels = [], []

    if kind == "spatial":
        for cells in sweep(objs, direction):
            ops.append(CLASS_COLOR[size_class(cells)])
            sels.append(sel_of(cells))
    elif kind == "batch":
        for s in class_order:
            group = [ob for ob in objs if size_class(ob) == s]
            if not group:
                continue
            cells = [rc for ob in group for rc in ob]
            ops.append(CLASS_COLOR[s])
            sels.append(sel_of(cells))
    else:
        for s in class_order:
            group = [ob for ob in objs if size_class(ob) == s]
            if not group:
                continue
            for cells in sweep(group, direction):
                ops.append(CLASS_COLOR[s])
                sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6e82a1ae"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6e82a1ae"
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
                                f"for task 6e82a1ae"
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
                    f"Failed to build a complete episode for task 6e82a1ae "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6e82a1ae-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
