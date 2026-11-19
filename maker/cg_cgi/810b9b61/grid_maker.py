"""
ARC Task: 810b9b61 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque, Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 6)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    def box_of(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        return frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                         if r in (r0, r1) or c in (c0, c1))

    def shape_of(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    def normalize(cells):
        r0 = min(p[0] for p in cells)
        c0 = min(p[1] for p in cells)
        return frozenset((r - r0, c - c0) for r, c in cells)

    def dmirror(cells):
        return frozenset((c, r) for r, c in cells)

    def cmirror(cells):
        h, w = shape_of(cells)
        return frozenset((w - 1 - c, h - 1 - r) for r, c in normalize(cells))

    def vmirror(cells):
        h, w = shape_of(cells)
        return frozenset((r, w - 1 - c) for r, c in normalize(cells))

    def hmirror(cells):
        h, w = shape_of(cells)
        return frozenset((h - 1 - r, c) for r, c in normalize(cells))

    def identity(cells):
        return cells

    cols = [c for c in range(10) if c != 3]
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, unifint(diff_lb, diff_ub, (1, 6)))

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (3, (h * w) // 10))
    succ, tr, maxtr = 0, 0, 5 * nobjs
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = set((i, j) for i in range(h) for j in range(w))
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(3, 5)
        ow = random.randint(3, 5)
        obj = box_of([(0, 0), (oh - 1, ow - 1)])
        mfs = [identity, dmirror, cmirror, vmirror, hmirror]
        for fn in random.sample(mfs, random.choice((1, 2))):
            obj = normalize(fn(obj))
        oh, ow = shape_of(obj)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        li, lj = random.choice(cands)
        plcd = frozenset((r + li, c + lj) for r, c in obj)
        if random.choice((True, False)):
            k = unifint(diff_lb, diff_ub, (1, len(plcd) - 1))
            inobj = frozenset(random.sample(sorted(plcd), k))
        else:
            inobj = plcd
        if inobj.issubset(inds):
            succ += 1
            nb = set()
            for r, c in inobj:
                nb |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
            inds = (inds - inobj) - nb
            col = random.choice(ccols)
            isframe = box_of(inobj) == inobj and min(shape_of(inobj)) > 2
            for r, c in inobj:
                gi[r][c] = col
                go[r][c] = 3 if isframe else col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): a frame is a closed loop -- an
    # outline that seals off a pocket of background from the rest of the
    # canvas. So: (1) find background pockets unreachable from the border,
    # (2) the shapes walling each pocket are the loop candidates, (3) a wall
    # that is exactly the outline of its own bounding box (min side > 2) is a
    # closed frame and gets recoloured 3 as one unit (one FloodFill3).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi, wi

    # background: the canvas colour the generator paints first; objects are sparse
    if examples:
        cnt = Counter()
        for ex_i, _ in examples:
            cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
        cnt.update(I.flatten().tolist())
        cand = Counter(I.flatten().tolist()).most_common(1)[0][0]
        bgc = cand
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # 1. background reachable from the border
    outside = np.zeros((hi, wi), dtype=bool)
    dq = deque()
    for r in range(hi):
        for c in range(wi):
            if (r in (0, hi - 1) or c in (0, wi - 1)) and I[r, c] == bgc:
                outside[r, c] = True
                dq.append((r, c))
    while dq:
        y, x = dq.popleft()
        for dy, dx in N4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < hi and 0 <= nx < wi and not outside[ny, nx] and I[ny, nx] == bgc:
                outside[ny, nx] = True
                dq.append((ny, nx))

    def component(r, c):
        color = I[r, c]
        seen = {(r, c)}
        q = deque([(r, c)])
        while q:
            y, x = q.popleft()
            for dy, dx in N4:
                ny, nx = y + dy, x + dx
                if 0 <= ny < hi and 0 <= nx < wi and (ny, nx) not in seen and I[ny, nx] == color:
                    seen.add((ny, nx))
                    q.append((ny, nx))
        return seen

    ops, sels = [], []
    pocket_seen = np.zeros((hi, wi), dtype=bool)
    done = set()
    # 2. walk each enclosed pocket, collect the shapes that wall it
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc or outside[r, c] or pocket_seen[r, c]:
                continue
            pocket_seen[r, c] = True
            q = deque([(r, c)])
            walls = []
            while q:
                y, x = q.popleft()
                for dy, dx in N4:
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < hi and 0 <= nx < wi):
                        continue
                    if I[ny, nx] == bgc:
                        if not pocket_seen[ny, nx]:
                            pocket_seen[ny, nx] = True
                            q.append((ny, nx))
                    else:
                        walls.append((ny, nx))
            for (wy, wx) in walls:
                if (wy, wx) in done:
                    continue
                comp = component(wy, wx)
                done |= comp
                rs = [p[0] for p in comp]
                cs = [p[1] for p in comp]
                rmin, rmax, cmin, cmax = min(rs), max(rs), min(cs), max(cs)
                if min(rmax - rmin + 1, cmax - cmin + 1) <= 2:
                    continue
                perim = {(y, x) for y in range(rmin, rmax + 1) for x in range(cmin, cmax + 1)
                         if y in (rmin, rmax) or x in (cmin, cmax)}
                if comp == perim:
                    # 3. closed frame -> recolour the whole loop: FloodFill3 from its corner
                    ops.append(13)
                    sels.append([rmin, cmin, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 810b9b61"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 810b9b61"
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
                                f"for task 810b9b61"
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
                    f"Failed to build a complete episode for task 810b9b61 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"810b9b61-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
