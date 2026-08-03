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
    while True:
        bgc, dotc, c1, c2 = random.sample(list(range(10)), 4)
        # c1/c2 are the copied shape colours; 0 is "transparent" for CopyI/Paste
        if c1 != 0 and c2 != 0:
            break
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotc": dotc, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, c1, c2, ism=None, rot=None) -> dict:
    if ism is None:
        ism = choice((True, False))
    if rot is None:
        rot = choice((0, 1, 2, 3))

    hlo = min(12, max_h)
    wlo = min(12, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))
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


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- locate the frame from I: 4 dots at the corners of a rectangle whose
    #      two opposite sides are solid non-background lines ----
    cells_by_color = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                cells_by_color.setdefault(v, []).append((r, c))

    frame = None
    for col, cells in cells_by_color.items():
        if len(cells) != 4:
            continue
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1c = min(rs), max(rs), min(cs), max(cs)
        if r1 - r0 < 3 or c1c - c0 < 3:
            continue
        if set(cells) != {(r0, c0), (r0, c1c), (r1, c0), (r1, c1c)}:
            continue
        lv = {int(I[r, c0]) for r in range(r0 + 1, r1)}
        rv = {int(I[r, c1c]) for r in range(r0 + 1, r1)}
        tv = {int(I[r0, c]) for c in range(c0 + 1, c1c)}
        bv = {int(I[r1, c]) for c in range(c0 + 1, c1c)}
        if len(lv) == 1 and len(rv) == 1 and lv != {bgc} and rv != {bgc} and tv == {bgc} and bv == {bgc}:
            frame = (r0, c0, r1, c1c, True)   # lines are vertical
            break
        if len(tv) == 1 and len(bv) == 1 and tv != {bgc} and bv != {bgc} and lv == {bgc} and rv == {bgc}:
            frame = (r0, c0, r1, c1c, False)  # lines are horizontal
            break

    r0, c0, r1, c1c, vertical = frame
    fh, fw = r1 - r0 + 1, c1c - c0 + 1
    ir, ic = r0 + 1, c0 + 1          # frame interior top-left  (shift by UNITY)
    ih, iw = fh - 2, fw - 2

    # ---- the shape object: every non-background cell outside the frame ----
    shape_cells = []
    for col, cells in cells_by_color.items():
        for (r, c) in cells:
            if not (r0 <= r <= r1 and c0 <= c <= c1c):
                shape_cells.append((r, c))
    shape_cells.sort()

    # ---- mirror decision, measured from I: does the shape's colour order match
    #      the frame's line order along the axis across the two lines? ----
    if vertical:
        L = int(I[r0 + 1, c0])       # colour of the left line
        R = int(I[r0 + 1, c1c])      # colour of the right line
        posL = min(c for (r, c) in shape_cells if int(I[r, c]) == L)
        posR = min(c for (r, c) in shape_cells if int(I[r, c]) == R)
        flip_op = 26                 # FlipH (left<->right)
    else:
        L = int(I[r0, c0 + 1])       # colour of the top line
        R = int(I[r1, c0 + 1])       # colour of the bottom line
        posL = min(r for (r, c) in shape_cells if int(I[r, c]) == L)
        posR = min(r for (r, c) in shape_cells if int(I[r, c]) == R)
        flip_op = 27                 # FlipV (up<->down)
    mirror = not (posL < posR)

    # ---- stamp the shape object into the frame interior ----
    ops.append(28); sels.append(sel_of(shape_cells))       # CopyI: the shape's exact cells
    ops.append(30); sels.append([ir, ic, 0, 0])            # Paste at interior top-left
    if mirror:
        # interior rectangle now holds exactly the stamped shape on background
        ops.append(flip_op); sels.append([ir, ic, ih - 1, iw - 1])

    # ---- keep only the frame ----
    ops.append(33); sels.append([r0, c0, fh - 1, fw - 1])  # CropGrid to the frame bbox
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
