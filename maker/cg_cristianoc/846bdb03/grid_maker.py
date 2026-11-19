"""
ARC Task: 846bdb03 (RE-ARC) — LLM-generated grid_maker
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

# Per-instance structural plan: rotation of the whole scene.
# Rotations 0 and 2 have the box's two coloured lines on its left and right sides.
# Rotations 1 and 3 have them on its top and bottom.
ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc, dotc, c1, c2 = random.sample(range(10), 4)
    n_ex = num_examples if num_examples else 3
    if n_ex >= 2:
        # make sure both line orientations (left/right and top/bottom) are shown
        examples = [random.choice([0, 2]), random.choice([1, 3])]
        examples += [random.choice(ROTS) for _ in range(n_ex - 2)]
        random.shuffle(examples)
    else:
        examples = [random.choice(ROTS) for _ in range(n_ex)]
    plan = [{"rot": r} for r in examples]
    plan.append({"rot": random.choice(examples)})
    return {"bgc": bgc, "dotc": dotc, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, c1=None, c2=None,
             rot=None, **kw) -> dict:
    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + diff_lb * (hi - lo)
        b = lo + diff_ub * (hi - lo)
        return int(round(random.uniform(a, b)))

    if bgc is None:
        bgc, dotc, c1, c2 = random.sample(range(10), 4)
    if rot is None:
        rot = random.choice(ROTS)
    max_h = min(30, max_h)
    max_w = min(30, max_w)
    h = unifint(12, max(12, max_h))
    w = unifint(12, max(12, max_w))
    oh = unifint(4, h // 2 - 2)
    ow = unifint(4, w // 2 - 2)

    go = np.full((oh, ow), bgc, dtype=int)
    go[1:oh - 1, 0] = c1
    go[1:oh - 1, ow - 1] = c2
    for (r, c) in [(0, 0), (0, ow - 1), (oh - 1, 0), (oh - 1, ow - 1)]:
        go[r, c] = dotc

    bh, bw = oh - 2, ow - 2
    bounds = {(i, j) for i in range(bh) for j in range(bw)}

    def nbrs(cells):
        out = set()
        for (i, j) in cells:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        out.add((i + di, j + dj))
        return out

    objA = {random.choice(sorted(bounds))}
    ncells = unifint(1, max(1, (bh * bw) // 2))
    for _ in range(ncells - 1):
        cand = sorted((bounds - objA) & nbrs(objA))
        if not cand:
            break
        objA.add(random.choice(cand))

    def shp(o):
        rs = [i for i, _ in o]; cs = [j for _, j in o]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(objA) != (bh, bw):
        cand = sorted((bounds - objA) & nbrs(objA))
        objA.add(random.choice(cand))

    gi = np.full((h, w), bgc, dtype=int)
    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    plcdi = {(loci + i, locj + j) for i in range(oh) for j in range(ow)}
    rems = [(i, j) for i in range(h) for j in range(w)
            if (i, j) not in plcdi and loci + oh <= i <= h - oh + 2 and j <= w - ow + 2]
    loc = random.choice(rems)
    plcdA = {(i + loc[0], j + loc[1]) for (i, j) in objA}
    lm = min(j for _, j in plcdA)
    mp = lm + bw // 2
    colA = {(i, j): (c1 if j < mp else c2) for (i, j) in plcdA}

    gi[loci:loci + oh, locj:locj + ow] = go
    ism = random.choice((True, False))
    rm = max(j for _, j in plcdA)
    for (i, j), v in colA.items():
        jj = lm + rm - j if ism else j
        gi[i, jj] = v
    ui = min(i for i, _ in plcdA)
    for (i, j), v in colA.items():
        go[i - ui + 1, j - lm + 1] = v

    # rot k = number of clockwise quarter turns
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule: a box is marked by 4 corner dots and two coloured lines on opposite sides.
    A loose two-coloured shape elsewhere on the grid has a bounding box exactly the
    size of the box's interior. The shape slides into the box as one object, and it
    is mirrored first if its two colours sit on the wrong sides relative to the box's
    lines. Then the grid is cropped to the box.
    (The golf reading moves the same shape cell by cell, mirroring by reversing column
    order. It treats the same thing as one unit and makes the same mirror decision,
    so the operations are the same.)
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # ---- locate the box: a colour whose 4 cells are exactly 4 bbox corners ----
    frame = None
    for col, n in cnt.items():
        if col == bgc or n != 4:
            continue
        cells = [(int(r), int(c)) for r, c in np.argwhere(I == col).tolist()]
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        if set(cells) != {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        sub = I[r0:r1 + 1, c0:c1 + 1]
        left = sub[1:-1, 0].tolist(); right = sub[1:-1, -1].tolist()
        top = sub[0, 1:-1].tolist(); bot = sub[-1, 1:-1].tolist()
        if (len(set(left)) == 1 and len(set(right)) == 1 and left[0] != bgc
                and right[0] != bgc and all(v == bgc for v in top + bot)):
            frame = (r0, c0, r1, c1, 'v'); break
        if (len(set(top)) == 1 and len(set(bot)) == 1 and top[0] != bgc
                and bot[0] != bgc and all(v == bgc for v in left + right)):
            frame = (r0, c0, r1, c1, 'h'); break
    r0, c0, r1, c1, orient = frame
    ho, wo = r1 - r0 + 1, c1 - c0 + 1

    # ---- the loose shape: every non-background cell outside the box ----
    blob = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v == bgc or (r0 <= r <= r1 and c0 <= c <= c1):
                continue
            blob[(r, c)] = v
    br0 = min(r for r, _ in blob); br1 = max(r for r, _ in blob)
    bc0 = min(c for _, c in blob); bc1 = max(c for _, c in blob)

    # ---- mirror needed? compare the shape's leading colour with the box's leading line ----
    first = {}
    if orient == 'v':
        frame_side = int(I[r0 + 1, c0])
        for (r, c), v in blob.items():
            if v not in first or c < first[v]:
                first[v] = c
    else:
        frame_side = int(I[r0, c0 + 1])
        for (r, c), v in blob.items():
            if v not in first or r < first[v]:
                first[v] = r
    blob_side = min(first.items(), key=lambda kv: kv[1])[0]
    need_mirror = frame_side != blob_side

    def mirrored(p):
        r, c = p
        if not need_mirror:
            return (r, c)
        return (r, bc0 + bc1 - c) if orient == 'v' else (br0 + br1 - r, c)

    dr = (r0 + 1) - br0
    dc = (c0 + 1) - bc0

    ops, sels = [], []
    cells = sorted(blob.keys())

    grabbed = False
    if need_mirror:
        ops.append(26 if orient == 'v' else 27)
        sels.append(sel_of(cells))           # the shape's exact cells
        grabbed = True

    steps = []
    steps += [20] * (-dr) if dr < 0 else [21] * dr
    steps += [23] * (-dc) if dc < 0 else [22] * dc
    for i, mop in enumerate(steps):
        ops.append(mop)
        if i == 0 and not grabbed:
            sels.append(sel_of(cells))       # first Move grabs the shape
        else:
            sels.append(sel_of([]))          # keep the same object grabbed

    # colour-0 parts of the shape cannot be carried by object ops; draw them
    zeros = sorted([(r + dr, c + dc) for p, v in blob.items()
                    for (r, c) in [mirrored(p)] if v == 0])
    if zeros:
        ops.append(0)
        sels.append(sel_of(zeros))

    # crop to the box (full rectangle: border plus interior)
    ops.append(33)
    sels.append([int(r0), int(c0), int(ho - 1), int(wo - 1)])

    ops.append(34)
    sels.append([0, 0, int(ho - 1), int(wo - 1)])
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
                        f"num_examples+1 ({num_examples + 1}) for task 846bdb03"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 846bdb03"
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
                                f"for task 846bdb03"
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
                    f"Failed to build a complete episode for task 846bdb03 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"846bdb03-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
