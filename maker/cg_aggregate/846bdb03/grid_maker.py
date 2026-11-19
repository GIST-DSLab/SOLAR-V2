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


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return max(a, min(b, a + int(round((b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    bgc, dotc, c1, c2 = random.sample(range(10), 4)
    n_ex = num_examples if num_examples else 3
    isms = [True, False] * ((n_ex + 1) // 2)
    isms = isms[:n_ex]
    random.shuffle(isms)
    examples = [{"ism": m, "rot": random.randint(0, 3)} for m in isms]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotc": dotc, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, c1, c2, ism=None, rot=None) -> dict:
    if ism is None:
        ism = random.choice((True, False))
    if rot is None:
        rot = random.randint(0, 3)
    h = _unifint(diff_lb, diff_ub, (12, max(12, max_h)))
    w = _unifint(diff_lb, diff_ub, (12, max(12, max_w)))
    oh = _unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = _unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((oh, ow), bgc, dtype=int)
    go[1:oh - 1, 0] = c1
    go[1:oh - 1, ow - 1] = c2
    for (r, c) in [(0, 0), (0, ow - 1), (oh - 1, 0), (oh - 1, ow - 1)]:
        go[r, c] = dotc
    bh, bw = oh - 2, ow - 2
    bounds = {(i, j) for i in range(bh) for j in range(bw)}

    def nbrs(p):
        i, j = p
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    objA = {random.choice(sorted(bounds))}
    ncells = _unifint(diff_lb, diff_ub, (1, (bh * bw) // 2))

    def grow():
        cand = set()
        for p in objA:
            cand |= nbrs(p)
        cand = sorted((cand & bounds) - objA)
        objA.add(random.choice(cand))

    for _ in range(ncells - 1):
        grow()
    while (max(i for i, _ in objA) - min(i for i, _ in objA) + 1,
           max(j for _, j in objA) - min(j for _, j in objA) + 1) != (bh, bw):
        grow()
    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    rems = [(i, j) for i in range(h) for j in range(w)
            if not (loci <= i < loci + oh and locj <= j < locj + ow)
            and loci + oh <= i <= h - oh + 2 and j <= w - ow + 2]
    li, lj = random.choice(rems)
    mp = bw // 2
    shape = {(i, j): (c1 if j < mp else c2) for (i, j) in objA}
    gi[loci:loci + oh, locj:locj + ow] = go
    for (i, j), v in shape.items():
        jj = (bw - 1 - j) if ism else j
        gi[li + i, lj + jj] = v
    for (i, j), v in shape.items():
        go[1 + i, 1 + j] = v
    gi = np.rot90(gi, -rot) if rot else gi   # rot90 (CW) applied rot times
    go = np.rot90(go, -rot) if rot else go
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Rule (maker and golf reading agree): the dotted-corner box with its two coloured
    side-lines is the output canvas; the loose two-coloured shape outside it is exactly
    the box's interior.  The shape is mirrored when its leading-side colour differs from
    the box's line on that side, slid into the interior, and the grid is cropped to the
    box.  The golf entry writes the same thing row by row (frame colour, shape row maybe
    reversed, frame colour): same units (box, whole shape), same single mirror decision,
    so the trajectory stays: mirror shape -> slide shape -> crop to box.
    Everything is measured from I only.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # ---- the box: a colour on exactly the 4 corners of a rectangle whose interior is
    #      empty and whose two opposite sides are distinct solid lines
    frame = None
    for col, n in cnt.items():
        if col == bgc or n != 4:
            continue
        cells = [(int(r), int(c)) for r, c in np.argwhere(I == col)]
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        if set(cells) != {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        sub = I[r0:r1 + 1, c0:c1 + 1]
        if not np.all(sub[1:-1, 1:-1] == bgc):
            continue
        left = sub[1:-1, 0].tolist(); right = sub[1:-1, -1].tolist()
        top = sub[0, 1:-1].tolist(); bot = sub[-1, 1:-1].tolist()
        if (len(set(left)) == 1 and len(set(right)) == 1 and left[0] != bgc
                and right[0] != bgc and left[0] != right[0]
                and set(top) == {bgc} and set(bot) == {bgc}):
            frame = (r0, c0, r1, c1, 'v'); break
        if (len(set(top)) == 1 and len(set(bot)) == 1 and top[0] != bgc
                and bot[0] != bgc and top[0] != bot[0]
                and set(left) == {bgc} and set(right) == {bgc}):
            frame = (r0, c0, r1, c1, 'h'); break
    r0, c0, r1, c1, orient = frame
    ho, wo = r1 - r0 + 1, c1 - c0 + 1

    # ---- the loose shape: all non-background cells outside the box
    blob = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc and not (r0 <= r <= r1 and c0 <= c <= c1):
                blob[(r, c)] = v
    br0 = min(r for r, _ in blob); br1 = max(r for r, _ in blob)
    bc0 = min(c for _, c in blob); bc1 = max(c for _, c in blob)

    # ---- mirror decision: the shape's leading-side colour vs the box's line on that side
    if orient == 'v':
        side_col = int(I[r0 + 1, c0])
        lead = {v for (r, c), v in blob.items() if c == bc0}
    else:
        side_col = int(I[r0, c0 + 1])
        lead = {v for (r, c), v in blob.items() if r == br0}
    need_mirror = side_col not in lead

    def mirrored(p):
        r, c = p
        if not need_mirror:
            return (r, c)
        return (r, bc0 + bc1 - c) if orient == 'v' else (br0 + br1 - r, c)

    dr = (r0 + 1) - br0
    dc = (c0 + 1) - bc0
    cells = sorted(blob.keys())
    ops, sels = [], []

    grabbed = False
    if need_mirror:
        ops.append(26 if orient == 'v' else 27)
        sels.append(sel_of(cells))           # the shape's own cells
        grabbed = True

    # ---- choose a slide route whose every unit step visibly changes the grid
    #      (a step is invisible when the shape glides along a same-coloured frame
    #      line); simulated from I alone.
    vs = [20] * (-dr) if dr < 0 else [21] * dr
    hs = [23] * (-dc) if dc < 0 else [22] * dc
    routes = [vs + hs, hs + vs]
    a, b = list(vs), list(hs)
    stair = []
    while a or b:
        if a: stair.append(a.pop(0))
        if b: stair.append(b.pop(0))
    routes.append(stair)
    a, b = list(vs), list(hs)
    stair = []
    while a or b:
        if b: stair.append(b.pop(0))
        if a: stair.append(a.pop(0))
    routes.append(stair)
    D = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}
    base = I.copy()
    for p in blob:
        base[p] = 0
    obj0 = {mirrored(p): v for p, v in blob.items() if v != 0}

    def render(obj):
        g = base.copy()
        for (r, c), v in obj.items():
            if 0 <= r < hi and 0 <= c < wi:
                g[r, c] = v
        return g

    def visible(route):
        prev = render(obj0)
        if need_mirror and np.array_equal(prev, I):
            return False
        oy = ox = 0
        for mop in route:
            oy += D[mop][0]; ox += D[mop][1]
            cur = render({(r + oy, c + ox): v for (r, c), v in obj0.items()})
            if np.array_equal(cur, prev):
                return False
            prev = cur
        return True

    steps = next((rt for rt in routes if visible(rt)), routes[0])
    for i, mop in enumerate(steps):
        ops.append(mop)
        sels.append(sel_of(cells) if (i == 0 and not grabbed) else sel_of([]))

    # colour-0 parts of the shape are not carried by object ops: draw them where they land
    zeros = sorted((r + dr, c + dc) for p, v in blob.items()
                   for (r, c) in [mirrored(p)] if v == 0)
    if zeros:
        ops.append(0); sels.append(sel_of(zeros))

    # crop to the box: full rectangle (border + interior) is exactly what is kept
    ops.append(33); sels.append([int(r0), int(c0), int(ho - 1), int(wo - 1)])
    ops.append(34); sels.append([0, 0, int(ho - 1), int(wo - 1)])
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
