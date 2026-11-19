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

try:
    from dsl import *
    from utils import unifint
except Exception:  # pragma: no cover
    from re_arc.dsl import *
    from re_arc.utils import unifint

# Two discrete things decide what each instance looks like:
#   ism  -> the loose shape is drawn mirrored relative to the box's line colours
#   rot  -> how the whole scene is rotated (quarter turns)
VARIANTS = [
    {"ism": False, "rot": 0},
    {"ism": True,  "rot": 1},
    {"ism": False, "rot": 2},
    {"ism": True,  "rot": 3},
    {"ism": True,  "rot": 0},
    {"ism": False, "rot": 1},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc, c1, c2 = random.sample(cols, 4)
    n_ex = num_examples if num_examples else 3
    # Make sure both the mirrored and the unmirrored case show up, and
    # both box orientations (an even and an odd number of quarter turns).
    base = [{"ism": False, "rot": random.choice([0, 2])},
            {"ism": True, "rot": random.choice([1, 3])}]
    if n_ex >= 3:
        base.append({"ism": True, "rot": random.choice([0, 2])})
    if n_ex >= 4:
        base.append({"ism": False, "rot": random.choice([1, 3])})
    examples = [dict(v) for v in base[:n_ex]]
    while len(examples) < n_ex:
        examples.append(dict(random.choice(VARIANTS)))
    random.shuffle(examples)
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotc": dotc, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, c1=None, c2=None,
             ism=None, rot=None) -> dict:
    if bgc is None or dotc is None or c1 is None or c2 is None:
        bgc, dotc, c1, c2 = random.sample(range(10), 4)
    if ism is None:
        ism = random.choice((True, False))
    if rot is None:
        rot = random.randint(0, 3)
    h = unifint(diff_lb, diff_ub, (12, max(12, max_h)))
    w = unifint(diff_lb, diff_ub, (12, max(12, max_w)))
    oh = unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (oh, ow))
    ln1 = connect((1, 0), (oh - 2, 0))
    ln2 = connect((1, ow - 1), (oh - 2, ow - 1))
    go = fill(go, c1, ln1)
    go = fill(go, c2, ln2)
    go = fill(go, dotc, corners(asindices(go)))
    objB = asobject(go)
    bounds = asindices(canvas(-1, (oh - 2, ow - 2)))
    objA = {choice(totuple(bounds))}
    ncells = unifint(diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))
    for k in range(ncells - 1):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    while shape(objA) != (oh - 2, ow - 2):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    fullinds = asindices(gi)
    loci = randint(0, h - 2 * oh + 2)
    locj = randint(0, w - ow)
    plcdB = shift(objB, (loci, locj))
    plcdi = toindices(plcdB)
    rems = sfilter(fullinds - plcdi,
                   lambda ij: loci + oh <= ij[0] <= h - oh + 2 and ij[1] <= w - ow + 2)
    loc = choice(totuple(rems))
    plcdA = shift(objA, loc)
    mp = center(plcdA)[1]
    plcdAL = sfilter(plcdA, lambda ij: ij[1] < mp)
    plcdAR = plcdA - plcdAL
    plcdA = recolor(c1, plcdAL) | recolor(c2, plcdAR)
    gi = paint(gi, plcdB)
    gi = paint(gi, vmirror(plcdA) if ism else plcdA)
    objA = shift(normalize(plcdA), (1, 1))
    go = paint(go, objA)
    rotf = (identity, rot90, rot180, rot270)[rot % 4]
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    How this compares with the code-golf reading: the two agree. The golf entry
    pulls out the loose shape, reverses it when its two colours are on the wrong
    sides compared with the box's lines, and wraps it in the box. That is the
    same rule as "mirror the shape if needed, slide it into the box's interior,
    crop to the box". Both treat the two-coloured shape as one unit, both decide
    the mirror the same way (which colour comes first along the axis across the
    lines), and both keep the box frame fixed. So there is nothing to change:
    grab the shape once (flip it if needed), move it one cell at a time into the
    box, then crop to the box.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # ---- find the box: a colour whose only 4 cells are a rectangle's corners ----
    frame = None
    loose = None
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
        inner = sub[1:-1, 1:-1]
        left = sub[1:-1, 0].tolist(); right = sub[1:-1, -1].tolist()
        top = sub[0, 1:-1].tolist(); bot = sub[-1, 1:-1].tolist()
        if np.all(inner == bgc):
            if (len(set(left)) == 1 and len(set(right)) == 1 and left[0] != bgc
                    and right[0] != bgc and left[0] != right[0]
                    and all(v == bgc for v in top) and all(v == bgc for v in bot)):
                frame = (r0, c0, r1, c1, 'v'); break
            if (len(set(top)) == 1 and len(set(bot)) == 1 and top[0] != bgc
                    and bot[0] != bgc and top[0] != bot[0]
                    and all(v == bgc for v in left) and all(v == bgc for v in right)):
                frame = (r0, c0, r1, c1, 'h'); break
        if loose is None:
            loose = (r0, c0, r1, c1, 'v' if left and left[0] != bgc else 'h')
    if frame is None:
        frame = loose
    r0, c0, r1, c1, orient = frame
    bh, bw = r1 - r0 + 1, c1 - c0 + 1

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

    # ---- is a mirror needed so the colours line up with the box's lines? ----
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
    need_mirror = (frame_side != blob_side)

    def mirrored(p):
        r, c = p
        if not need_mirror:
            return (r, c)
        return (r, bc0 + bc1 - c) if orient == 'v' else (br0 + br1 - r, c)

    dr = (r0 + 1) - br0
    dc = (c0 + 1) - bc0

    ops, sels = [], []
    cells = sorted(blob.keys())

    # 1. mirror the shape in place (this also grabs it)
    grabbed = False
    if need_mirror:
        ops.append(26 if orient == 'v' else 27)
        sels.append(sel_of(cells))
        grabbed = True

    # 2. slide the shape into the box one cell at a time:
    #    the first Move grabs it, every later Move uses an empty selection
    steps = []
    steps += [20] * (-dr) if dr < 0 else [21] * dr
    steps += [23] * (-dc) if dc < 0 else [22] * dc
    for i, mop in enumerate(steps):
        ops.append(mop)
        if i == 0 and not grabbed:
            sels.append(sel_of(cells))
        else:
            sels.append(sel_of([]))

    # 3. cells of colour 0 don't travel with object ops, so paint them in place
    zeros = sorted([(r + dr, c + dc) for p, v in blob.items()
                    for (r, c) in [mirrored(p)] if v == 0])
    if zeros:
        ops.append(0)
        sels.append(sel_of(zeros))

    # 4. crop to the box (the whole rectangle: frame plus interior)
    ops.append(33)
    sels.append([int(r0), int(c0), int(bh - 1), int(bw - 1)])

    ops.append(34)
    sels.append([0, 0, int(bh - 1), int(bw - 1)])
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
