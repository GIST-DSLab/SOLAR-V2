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


VARIANTS = [
    {"ism": False, "rot": 0},
    {"ism": True,  "rot": 1},
    {"ism": True,  "rot": 0},
    {"ism": False, "rot": 2},
    {"ism": False, "rot": 3},
    {"ism": True,  "rot": 2},
    {"ism": False, "rot": 1},
    {"ism": True,  "rot": 3},
]


def sample_colors(num_examples=None) -> dict:
    bgc, dotc, c1, c2 = random.sample(list(range(10)), 4)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        # always show both the mirrored and the un-mirrored case
        pool_m = [v for v in VARIANTS if v["ism"]]
        pool_n = [v for v in VARIANTS if not v["ism"]]
        examples = [dict(random.choice(pool_m))]
        if n_ex >= 2:
            examples.append(dict(random.choice(pool_n)))
        rest = [v for v in VARIANTS if v not in examples]
        examples += [dict(v) for v in random.sample(rest, max(0, n_ex - len(examples)))]
        random.shuffle(examples)
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotc": dotc, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, c1, c2, ism=None, rot=None) -> dict:
    if ism is None:
        ism = choice((True, False))
    if rot is None:
        rot = choice((0, 1, 2, 3))
    h = unifint(diff_lb, diff_ub, (min(12, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(12, max_w), max_w))
    oh = unifint(diff_lb, diff_ub, (4, max(4, h // 2 - 2)))
    ow = unifint(diff_lb, diff_ub, (4, max(4, w // 2 - 2)))
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
    rems = sfilter(fullinds - plcdi, lambda ij: loci + oh <= ij[0] <= h - oh + 2 and ij[1] <= w - ow + 2)
    loc = choice(totuple(rems))
    plcdA = shift(objA, loc)
    mp = center(plcdA)[1]
    plcdAL = sfilter(plcdA, lambda ij: ij[1] < mp)
    plcdAR = plcdA - plcdAL
    plcdA = recolor(c1, plcdAL) | recolor(c2, plcdAR)
    gi = paint(gi, plcdB)
    gi = paint(gi, vmirror(plcdA) if ism else plcdA)
    objAn = shift(normalize(plcdA), (1, 1))
    go = paint(go, objAn)
    rotf = (identity, rot90, rot180, rot270)[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Reading (block transplant): the box (4 dot corners + two coloured side lines)
    has an empty interior; the loose two-coloured shape's BOUNDING RECTANGLE has
    exactly the interior's size.  That rectangle is copied as one block into the
    interior, the interior block is mirrored when its colours face the wrong lines,
    and the canvas is cropped to the box.  The original shape is not moved -- it is
    simply left outside and cut away by the crop.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]   # shape fills <= half of a small box: bg dominates

    # ---- the box: a colour occupying exactly the 4 corners of a rectangle ----
    frame = None
    for col, n in cnt.items():
        if col == bgc or n != 4:
            continue
        cells = [tuple(p) for p in np.argwhere(I == col).tolist()]
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        if set(cells) != {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        sub = I[r0:r1 + 1, c0:c1 + 1]
        left, right = sub[1:-1, 0], sub[1:-1, -1]
        top, bot = sub[0, 1:-1], sub[-1, 1:-1]
        if not np.all(sub[1:-1, 1:-1] == bgc):
            continue
        if (len(set(left.tolist())) == 1 and len(set(right.tolist())) == 1
                and left[0] != bgc and right[0] != bgc and np.all(top == bgc) and np.all(bot == bgc)):
            frame = (r0, c0, r1, c1, 'v'); break
        if (len(set(top.tolist())) == 1 and len(set(bot.tolist())) == 1
                and top[0] != bgc and bot[0] != bgc and np.all(left == bgc) and np.all(right == bgc)):
            frame = (r0, c0, r1, c1, 'h'); break
    r0, c0, r1, c1, orient = frame
    ir, ic = r0 + 1, c0 + 1                     # interior top-left
    ih, iw = r1 - r0 - 1, c1 - c0 - 1           # interior size

    # ---- the loose shape: non-bg cells outside the box; its bounding rectangle ----
    blob = {(r, c): int(I[r, c]) for r in range(hi) for c in range(wi)
            if I[r, c] != bgc and not (r0 <= r <= r1 and c0 <= c <= c1)}
    br0 = min(r for r, _ in blob); bc0 = min(c for _, c in blob)
    br1 = max(r for r, _ in blob); bc1 = max(c for _, c in blob)
    assert (br1 - br0 + 1, bc1 - bc0 + 1) == (ih, iw)

    # ---- which colour leads (left for 'v', top for 'h') in the shape vs the box ----
    if orient == 'v':
        lead_box = int(I[ir, c0])
        key = lambda p: p[1]
    else:
        lead_box = int(I[r0, ic])
        key = lambda p: p[0]
    first = {}
    for p, v in blob.items():
        first[v] = min(first.get(v, 10 ** 6), key(p))
    lead_shape = min(first.items(), key=lambda kv: kv[1])[0]
    need_mirror = lead_box != lead_shape

    ops, sels = [], []
    # 1. copy the shape's bounding rectangle (whole block, background included)
    ops.append(28); sels.append([br0, bc0, ih - 1, iw - 1])
    # 2. drop it into the box interior (top-left anchor)
    ops.append(30); sels.append([ir, ic, 0, 0])
    # 3. mirror the interior block in place when colours face the wrong lines
    #    (full interior rectangle, background included -- intended)
    if need_mirror:
        ops.append(26 if orient == 'v' else 27)
        sels.append([ir, ic, ih - 1, iw - 1])

    # 4. shape cells of colour 0 are transparent to Copy/Paste: draw them in
    def dest(p):
        r, c = p[0] - br0, p[1] - bc0
        if need_mirror:
            if orient == 'v':
                c = iw - 1 - c
            else:
                r = ih - 1 - r
        return (ir + r, ic + c)
    zeros = sorted(dest(p) for p, v in blob.items() if v == 0)
    if zeros:
        ops.append(0); sels.append(sel_of(zeros))

    # 5. crop to the box (full rectangle: border + interior)
    ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])
    ops.append(34); sels.append([0, 0, r1 - r0, c1 - c0])
    return ops, sels


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
