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
import numpy as np
from collections import deque, Counter


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
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        return random.randint(min(lo, hi), max(lo, hi))

    def box_of(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        return frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                         if r in (r0, r1) or c in (c0, c1))

    def normalize(cells):
        r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
        return frozenset((r - r0, c - c0) for r, c in cells)

    def shape(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    cols = [c for c in range(10) if c != 3]
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, random.randint(1, 6))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    h = min(h, max_h); w = min(w, max_w)
    nobjs = unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 10)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    mfs = [lambda o: o,
           lambda o: frozenset((c, r) for r, c in o),
           lambda o: frozenset((-c, -r) for r, c in o),
           lambda o: frozenset((r, -c) for r, c in o),
           lambda o: frozenset((-r, c) for r, c in o)]
    succ = tr = 0
    maxtr = 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(3, 5); ow = random.randint(3, 5)
        if oh > h or ow > w:
            continue
        obj = box_of([(0, 0), (oh - 1, ow - 1)])
        for fn in random.sample(mfs, random.choice((1, 2))):
            obj = normalize(fn(obj))
        oh, ow = shape(obj)
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
        if inobj <= inds:
            succ += 1
            nb = {(r + dr, c + dc) for r, c in inobj
                  for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
            inds = (inds - inobj) - nb
            col = random.choice(ccols)
            frame = box_of(inobj) == inobj and min(shape(inobj)) > 2
            for r, c in inobj:
                gi[r][c] = col
                go[r][c] = 3 if frame else col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): a frame is an object that SEALS OFF
    # a pocket of background. Flood the background in from the grid border;
    # whatever background is left unreached is an enclosed hole. Each hole
    # points at the ring around it, and that ring is recoloured to 3 as one
    # piece (FloodFill3). The generator can also make a box missing a corner,
    # which still encloses a hole but is not a full outline, so the verifier's
    # check is kept: the ring must equal its whole bounding-box outline.
    # Apart from that, this matches the maker: one FloodFill3 per frame. The only
    # change is that candidate frames are found from their holes, in hole order.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # background: the generator paints a solid canvas first; confirm with examples
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_i, _ in examples:
            cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # 1. flood background inward from the border
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
        col = I[r, c]
        seen = {(r, c)}
        q = deque([(r, c)])
        while q:
            y, x = q.popleft()
            for dy, dx in N4:
                ny, nx = y + dy, x + dx
                if 0 <= ny < hi and 0 <= nx < wi and (ny, nx) not in seen and I[ny, nx] == col:
                    seen.add((ny, nx)); q.append((ny, nx))
        return seen

    ops, sels = [], []
    done_holes = np.zeros((hi, wi), dtype=bool)
    done_objs = set()
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc or outside[r, c] or done_holes[r, c]:
                continue
            # 2. one enclosed background pocket
            hole = {(r, c)}
            done_holes[r, c] = True
            q = deque([(r, c)])
            while q:
                y, x = q.popleft()
                for dy, dx in N4:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not done_holes[ny, nx] \
                            and I[ny, nx] == bgc and not outside[ny, nx]:
                        done_holes[ny, nx] = True
                        hole.add((ny, nx)); q.append((ny, nx))
            # 3. the rings bordering it
            walls = set()
            for y, x in hole:
                for dy, dx in N4:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and I[ny, nx] != bgc:
                        walls.add((ny, nx))
            for (wy, wx) in sorted(walls):
                if (wy, wx) in done_objs:
                    continue
                comp = component(wy, wx)
                done_objs |= comp
                rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
                r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
                if min(r1 - r0, c1 - c0) < 2:
                    continue
                if not all(r0 < y < r1 and c0 < x < c1 for y, x in hole):
                    continue  # not the ring around this pocket
                outline = {(y, x) for y in range(r0, r1 + 1) for x in range(c0, c1 + 1)
                           if y in (r0, r1) or x in (c0, c1)}
                if comp == outline:
                    ops.append(13)               # FloodFill3 the whole ring
                    sels.append([r0, c0, 0, 0])  # seed at its corner (a frame cell)
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
