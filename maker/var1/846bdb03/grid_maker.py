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


# ----------------------------------------------------------------------------- #
# 1. colours + per-instance structural plan
# ----------------------------------------------------------------------------- #
def sample_colors(num_examples=None) -> dict:
    """bgc / dotc / c1 / c2 are the four colours the generator samples.

    `ism` (is the loose shape already on the right side, or mirrored?) and
    `rotk` (which quarter turn the whole scene gets) are the two discrete
    structural cases, so they are planned per instance and shared by index.
    """
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


# ----------------------------------------------------------------------------- #
# 2. generator
# ----------------------------------------------------------------------------- #
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


# ----------------------------------------------------------------------------- #
# 3. trajectory
# ----------------------------------------------------------------------------- #
def derive_operations(I, O):
    """
    Rule: a rectangular 'box' is marked by 4 dot-coloured corners and two coloured
    lines on opposite sides.  A loose two-coloured shape lies elsewhere; its
    bounding box is exactly the box's interior.  The shape is slid into the box
    (mirrored when its two colours sit on the wrong sides w.r.t. the box's lines),
    and the grid is cropped down to the box.

    The rule is fixed; the ROUTE through it is read off the input, so two
    different inputs get two different (but always legal) routes:
      * path_style  (how many cells the shape has)  -> down-then-across /
                                                       across-then-down / diagonal
      * mirror_when (where the frame sits)          -> mirror the shape before the
                                                       slide, or mirror the frame's
                                                       interior after the slide, or
                                                       after the crop
      * grouped     (the corner-dot colour)         -> the shape's colour-0 cells
                                                       painted in one go or region
                                                       by region
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # ---- locate the box: colour whose cells are exactly 4 bbox corners -------
    strict = None
    fallback = None
    for col, n in cnt.items():
        if col == bgc or n != 4:
            continue
        pts = [(int(r), int(c)) for r, c in np.argwhere(I == col).tolist()]
        a0 = min(p[0] for p in pts); a1 = max(p[0] for p in pts)
        b0 = min(p[1] for p in pts); b1 = max(p[1] for p in pts)
        if a1 - a0 < 3 or b1 - b0 < 3:
            continue
        if set(pts) != {(a0, b0), (a0, b1), (a1, b0), (a1, b1)}:
            continue
        sub = I[a0:a1 + 1, b0:b1 + 1]
        inner = sub[1:-1, 1:-1]
        left = sub[1:-1, 0].tolist(); right = sub[1:-1, -1].tolist()
        top = sub[0, 1:-1].tolist(); bot = sub[-1, 1:-1].tolist()
        if np.all(inner == bgc):
            if (len(set(left)) == 1 and len(set(right)) == 1 and left[0] != bgc
                    and right[0] != bgc and left[0] != right[0]
                    and all(v == bgc for v in top) and all(v == bgc for v in bot)):
                strict = (a0, b0, a1, b1, 'v'); break
            if (len(set(top)) == 1 and len(set(bot)) == 1 and top[0] != bgc
                    and bot[0] != bgc and top[0] != bot[0]
                    and all(v == bgc for v in left) and all(v == bgc for v in right)):
                strict = (a0, b0, a1, b1, 'h'); break
        if fallback is None:
            fallback = (a0, b0, a1, b1, 'v' if (left and left[0] != bgc) else 'h')

    frame = strict if strict is not None else fallback
    r0, c0, r1, c1, orient = frame
    dotc = int(I[r0, c0])
    ho, wo = r1 - r0 + 1, c1 - c0 + 1

    # ---- the loose shape: every non-background cell outside the box ---------
    blob = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v == bgc:
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            blob[(r, c)] = v
    br0 = min(r for r, _ in blob); br1 = max(r for r, _ in blob)
    bc0 = min(c for _, c in blob); bc1 = max(c for _, c in blob)

    # ---- does the shape need mirroring to match the box's line sides? -------
    first = {}
    if orient == 'v':
        frame_side = int(I[r0 + 1, c0])                 # colour of the left line
        for (r, c), v in blob.items():
            if v not in first or c < first[v]:
                first[v] = c
    else:
        frame_side = int(I[r0, c0 + 1])                 # colour of the top line
        for (r, c), v in blob.items():
            if v not in first or r < first[v]:
                first[v] = r
    blob_side = min(first.items(), key=lambda kv: kv[1])[0]
    need_mirror = (frame_side != blob_side)
    flip_op = 26 if orient == 'v' else 27               # v -> left/right, h -> up/down

    def mirrored(p):
        r, c = p
        if not need_mirror:
            return (r, c)
        return (r, bc0 + bc1 - c) if orient == 'v' else (br0 + br1 - r, c)

    cells = sorted(blob.keys())
    dr = (r0 + 1) - br0
    dc = (c0 + 1) - bc0

    # ---- route choices, every one of them read off the input ----------------
    path_style = len(cells) % 3                         # a count
    mirror_when = (r0 + c0) % 3                         # a position
    grouped = (dotc % 2 == 1)                           # a colour

    nv, nh = abs(dr), abs(dc)
    vop = 21 if dr > 0 else 20
    hop = 22 if dc > 0 else 23
    if path_style == 0:                                 # straight down/up, then across
        seq = [vop] * nv + [hop] * nh
    elif path_style == 1:                               # across first, then down/up
        seq = [hop] * nh + [vop] * nv
    else:                                               # walk it diagonally
        seq = []
        i = j = 0
        turn_v = nv >= nh
        while i < nv or j < nh:
            if turn_v and i < nv:
                seq.append(vop); i += 1
            elif (not turn_v) and j < nh:
                seq.append(hop); j += 1
            elif i < nv:
                seq.append(vop); i += 1
            else:
                seq.append(hop); j += 1
            turn_v = not turn_v

    # colour-0 parts of the shape cannot be carried by object ops; they are drawn.
    # If the mirror is still to come they are drawn un-mirrored and the flip
    # carries them; otherwise they are drawn where they finally belong.
    late_mirror = need_mirror and mirror_when in (1, 2)
    zsrc = [p for p in cells if blob[p] == 0]
    if late_mirror:
        zpos = [(r + dr, c + dc) for (r, c) in zsrc]
    else:
        zpos = [(mirrored(p)[0] + dr, mirrored(p)[1] + dc) for p in zsrc]

    ops, sels = [], []

    def regions(pts):
        pool = set(pts)
        out = []
        while pool:
            seed = min(pool)
            pool.discard(seed)
            stack, comp = [seed], set()
            while stack:
                r, c = stack.pop()
                comp.add((r, c))
                for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if nb in pool:
                        pool.discard(nb); stack.append(nb)
            out.append(sorted(comp))
        out.sort()
        return out

    def emit_zeros(pts, off=(0, 0)):
        if not pts:
            return
        moved = [(r + off[0], c + off[1]) for (r, c) in pts]
        if grouped:
            for g in regions(moved):
                ops.append(0); sels.append(sel_of(g))
        else:
            ops.append(0); sels.append(sel_of(sorted(moved)))

    # 1. mirror the shape where it lies (this also grabs it as the working object)
    grabbed = False
    if need_mirror and mirror_when == 0:
        ops.append(flip_op)
        sels.append(sel_of(cells))          # the shape's exact cells, not its bbox
        grabbed = True

    # 2. slide the shape into the box, one cell at a time
    for k, mop in enumerate(seq):
        ops.append(mop)
        if not grabbed:
            sels.append(sel_of(cells))      # first Move grabs the shape
            grabbed = True
        else:
            sels.append(sel_of([]))         # empty -> keep the same object grabbed

    # 3. finish: zeros, mirror (if it was left for later), crop to the box
    crop_sel = [int(r0), int(c0), int(r1 - r0), int(c1 - c0)]   # full box rectangle
    if need_mirror and mirror_when == 2:
        ops.append(33); sels.append(crop_sel)
        emit_zeros(zpos, (-r0, -c0))
        # full rectangle: the frame's whole interior, background included
        ops.append(flip_op); sels.append([1, 1, int(ho - 3), int(wo - 3)])
    elif need_mirror and mirror_when == 1:
        emit_zeros(zpos)
        # full rectangle: the frame's whole interior, background included
        ops.append(flip_op)
        sels.append([int(r0 + 1), int(c0 + 1), int(ho - 3), int(wo - 3)])
        ops.append(33); sels.append(crop_sel)
    elif mirror_when == 2:
        ops.append(33); sels.append(crop_sel)
        emit_zeros(zpos, (-r0, -c0))
    else:
        emit_zeros(zpos)
        ops.append(33); sels.append(crop_sel)

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
