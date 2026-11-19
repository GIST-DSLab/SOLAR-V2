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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc, c1, c2 = random.sample(cols, 4)

    n_ex = int(num_examples) if num_examples else 3
    n_ex = max(1, n_ex)

    rots = list(range(4))
    random.shuffle(rots)

    if n_ex >= 2:
        plan = [{"ism": False, "rotk": rots[0]},
                {"ism": True, "rotk": rots[1]}]
        for i in range(n_ex - 2):
            plan.append({"ism": random.choice([True, False]),
                         "rotk": rots[(2 + i) % 4]})
        random.shuffle(plan)
    else:
        plan = [{"ism": random.choice([True, False]), "rotk": rots[0]}]

    plan.append(dict(random.choice(plan)))          # test case was demonstrated
    return {"bgc": bgc, "dotc": dotc, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, dotc=None, c1=None, c2=None,
             ism=None, rotk=None) -> dict:
    if ism is None:
        ism = choice((True, False))
    if rotk is None:
        rotk = choice((0, 1, 2, 3))

    # a quarter turn transposes the canvas, so bound h/w accordingly
    if rotk % 2 == 1:
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    hb = max(12, min(30, int(hb)))
    wb = max(12, min(30, int(wb)))

    h = unifint(diff_lb, diff_ub, (12, hb))
    w = unifint(diff_lb, diff_ub, (12, wb))
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

    rotf = (identity, rot90, rot180, rot270)[rotk]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Copy the loose shape from the input straight into the box interior,
    mirror it there (whole interior rectangle) when its colours sit on the
    wrong sides, draw any colour-0 parts, then crop to the box."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # ---- box: a colour occupying exactly the 4 corners of a rectangle -------
    frame = None
    for col, n in cnt.items():
        if col == bgc or n != 4:
            continue
        cells = [tuple(map(int, p)) for p in np.argwhere(I == col)]
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        if set(cells) != {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        sub = I[r0:r1 + 1, c0:c1 + 1]
        if not np.all(sub[1:-1, 1:-1] == bgc):
            continue
        left = sub[1:-1, 0]; right = sub[1:-1, -1]
        top = sub[0, 1:-1]; bot = sub[-1, 1:-1]
        if (len(set(left.tolist())) == 1 and len(set(right.tolist())) == 1
                and left[0] != bgc and right[0] != bgc and np.all(top == bgc)
                and np.all(bot == bgc)):
            frame = (r0, c0, r1, c1, 'v'); break
        if (len(set(top.tolist())) == 1 and len(set(bot.tolist())) == 1
                and top[0] != bgc and bot[0] != bgc and np.all(left == bgc)
                and np.all(right == bgc)):
            frame = (r0, c0, r1, c1, 'h'); break
    r0, c0, r1, c1, orient = frame

    # ---- the loose shape: non-background cells outside the box --------------
    blob = {(r, c): int(I[r, c]) for r in range(hi) for c in range(wi)
            if I[r, c] != bgc and not (r0 <= r <= r1 and c0 <= c <= c1)}
    br0 = min(r for r, _ in blob); br1 = max(r for r, _ in blob)
    bc0 = min(c for _, c in blob); bc1 = max(c for _, c in blob)
    bh, bw = br1 - br0 + 1, bc1 - bc0 + 1

    # ---- mirror needed? compare the colour nearest the first line ----------
    first = {}
    for (r, c), v in blob.items():
        k = c if orient == 'v' else r
        if v not in first or k < first[v]:
            first[v] = k
    blob_side = min(first.items(), key=lambda kv: kv[1])[0]
    frame_side = int(I[r0 + 1, c0]) if orient == 'v' else int(I[r0, c0 + 1])
    need_mirror = frame_side != blob_side

    ir, ic = r0 + 1, c0 + 1          # interior top-left = paste origin
    ops, sels = [], []

    # 1. copy the shape's bounding rectangle from the input (full rectangle:
    #    shape + background only; it never overlaps the box)
    ops.append(28); sels.append([br0, bc0, bh - 1, bw - 1])
    # 2. paste it into the box interior
    ops.append(30); sels.append([ir, ic, 0, 0])
    # 3. mirror the whole interior rectangle in place (it holds only the copy)
    if need_mirror:
        ops.append(26 if orient == 'v' else 27)
        sels.append([ir, ic, bh - 1, bw - 1])

    # 4. colour-0 parts of the shape are not carried by Copy/Paste: draw them
    #    at the cells the copy landed on
    zeros = []
    for (r, c), v in blob.items():
        if v != 0:
            continue
        rr, cc = r - br0, c - bc0
        if need_mirror:
            if orient == 'v':
                cc = bw - 1 - cc
            else:
                rr = bh - 1 - rr
        zeros.append((ir + rr, ic + cc))
    if zeros:
        ops.append(0); sels.append(sel_of(sorted(zeros)))

    # 5. crop to the box (full rectangle: border + interior)
    ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])
    ops.append(34); sels.append([0, 0, r1 - r0, c1 - c0])
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
