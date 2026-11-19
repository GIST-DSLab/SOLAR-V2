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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background matters for the rule. The marker colour and the object
    # colours are read from each input.
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

    def nbrs(loc):
        i, j = loc
        return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if not (di == 0 and dj == 0)}

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    hub = max(10, max_h)
    wub = max(10, max_w)
    h = unifint(diff_lb, diff_ub, (10, hub))
    w = unifint(diff_lb, diff_ub, (10, wub))
    h = max(10, min(h, max_h)) if max_h >= 10 else max_h
    w = max(10, min(w, max_w)) if max_w >= 10 else max_w
    nobjs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 15)))
    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    tr, maxtr, succ, done = 0, 4 * nobjs, 0, False
    go = None
    while tr < maxtr and succ < nobjs:
        oh = random.randint(2, min(6, h))
        ow = random.randint(2, min(6, w))
        bx = {(i, j) for i in range(oh) for j in range(ow)}
        nc = random.randint(3, oh * ow)
        sp = random.choice(sorted(bx))
        bx.discard(sp)
        obj = {sp}
        for _ in range(nc - 1):
            cand = set()
            for c in obj:
                cand |= nbrs(c)
            cand = sorted((bx - obj) & cand)
            obj.add(random.choice(cand))
        tr += 1
        if not done:
            idx = random.choice(sorted(obj))
            coll = random.choice(remcols)
            coll2 = random.choice([c for c in remcols if c != coll])
            cobj = {(coll, idx)} | {(coll2, p) for p in obj if p != idx}
            mi = min(p[0] for p in obj); mj = min(p[1] for p in obj)
            Mi = max(p[0] for p in obj); Mj = max(p[1] for p in obj)
            g = [[bgc] * (Mj - mj + 1) for _ in range(Mi - mi + 1)]
            for (a, b) in obj:
                g[a - mi][b - mj] = coll2
            cand_go = g
        else:
            col = random.choice(remcols)
            cobj = {(col, p) for p in obj}
        locopts = sorted(ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow)
        if not locopts:
            continue
        loc = random.choice(locopts)
        plcd = {(c, (p[0] + loc[0], p[1] + loc[1])) for c, p in cobj}
        plcdi = {p for _, p in plcd}
        if plcdi.issubset(inds):
            for c, (a, b) in plcd:
                gi[a][b] = c
            succ += 1
            rem = set(plcdi)
            for p in plcdi:
                rem |= nbrs(p)
            inds = inds - rem
            if not done:
                done = True
                go = cand_go
    if go is None:
        raise ValueError("marked object not placed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the canvas colour, which is shared by every demonstration input.
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_i, _ in examples:
            cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # 8-connected non-background components (the generator keeps objects 8-separated)
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
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] \
                                and I[ny, nx] != bgc:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(cells)

    # The marked object is the one component that contains two colours.
    target = None
    for cells in comps:
        if len({int(I[y, x]) for (y, x) in cells}) > 1:
            target = cells
            break
    if target is None:
        target = max(comps, key=len)

    objset = set(target)
    body = Counter(int(I[y, x]) for (y, x) in target).most_common(1)[0][0]
    marker = [(y, x) for (y, x) in target if I[y, x] != body]

    r0 = min(y for y, _ in target); r1 = max(y for y, _ in target)
    c0 = min(x for _, x in target); c1 = max(x for _, x in target)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []

    # 1) Crop the canvas to the marked object's bounding box. This is the full
    #    rectangle, background corners included.
    ops.append(33)
    sels.append([r0, c0, oh - 1, ow - 1])

    # 2) Any other object that reaches into the box does not belong to the window:
    #    paint those cells back to background. These are I's non-bg cells in the box
    #    outside the marked component, so they can be named from I.
    intr = [(r - r0, c - c0) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if (r, c) not in objset and I[r, c] != bgc]
    if intr:
        ops.append(int(bgc))
        sels.append(sel_of(intr))

    # 3) The marker cell takes the object's own colour.
    if marker:
        ops.append(int(body))
        sels.append(sel_of([(y - r0, x - c0) for (y, x) in marker]))

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
