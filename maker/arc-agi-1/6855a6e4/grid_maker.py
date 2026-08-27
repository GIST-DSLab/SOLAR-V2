"""
ARC Task: 6855a6e4 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# The task: a "bracket box" made of two parallel full-length lines with small
# end-caps.  Outside each bracket sits a scattered pattern.  Each pattern is
# mirrored (across the axis parallel to the bracket) and slid THROUGH the
# bracket line so that it comes to rest 2 cells inside the box (i.e. right
# behind the end-cap row/column).  The whole configuration may be transposed
# (brackets horizontal -> vertical), which is the only structural variant.
# ----------------------------------------------------------------------------

VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    boxc = random.choice([c for c in cols if c != bgc])
    # objc != 0 so the moving pattern is a real (non-transparent) ARCLE object
    objc = random.choice([c for c in cols if c not in (bgc, boxc) and c != 0])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, boxc,
             transposed=None, **kwargs) -> dict:
    if transposed is None:
        transposed = random.choice(VARIANTS)["transposed"]

    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + int((hi - lo) * diff_lb)
        b = lo + int((hi - lo) * diff_ub)
        a = max(lo, min(hi, a))
        b = max(lo, min(hi, b))
        if b < a:
            a, b = b, a
        return random.randint(a, b)

    # canonical frame = brackets horizontal; transpose at the very end
    if transposed:
        hcap, wcap = min(30, max_w), min(30, max_h)
    else:
        hcap, wcap = min(30, max_h), min(30, max_w)
    hcap = max(10, hcap)
    wcap = max(4, wcap)

    h = unifint(10, hcap)
    w = unifint(4, wcap)
    fullh = unifint(10, h)
    fullw = unifint(3, w)
    objh = (fullh // 2 - 3) // 2
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)

    gi = [[bgc] * w for _ in range(h)]
    tl = objh + 1                 # top bracket line row (relative)
    bl = fullh - objh - 2         # bottom bracket line row (relative)
    for j in range(fullw):
        gi[loci + tl][locj + j] = boxc
        gi[loci + bl][locj + j] = boxc
    for (r, c) in ((tl + 1, 0), (tl + 1, fullw - 1), (bl - 1, 0), (bl - 1, fullw - 1)):
        gi[loci + r][locj + c] = boxc
    go = [row[:] for row in gi]

    ntot = objh * (fullw - 2)
    for side in (0, 1):
        cands = [(r, c) for r in range(objh) for c in range(1, fullw - 1)]
        if side == 1:
            cands = [(fullh - 1 - r, c) for (r, c) in cands]
        d = unifint(0, ntot // 2)
        n = random.choice((d, ntot - d))
        n = min(max(1, n), ntot)
        cells = random.sample(cands, n)
        rmin = min(r for r, _ in cells)
        rmax = max(r for r, _ in cells)
        # mirror inside own bbox, then park 2 cells inside the bracket
        off = (tl + 2) - rmin if side == 0 else (bl - 2) - rmax
        for (r, c) in cells:
            gi[loci + r][locj + c] = objc
            go[loci + (rmin + rmax - r) + off][locj + c] = objc

    if transposed:
        gi = [list(x) for x in zip(*gi)]
        go = [list(x) for x in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])
    others = [int(c) for c in sorted(cnt) if int(c) != bgc]

    def cells_of(g, col):
        return [(int(r), int(c)) for r, c in np.argwhere(g == col)]

    def bbox(cs):
        rs = [r for r, _ in cs]
        cl = [c for _, c in cs]
        return min(rs), max(rs), min(cl), max(cl)

    def box_score(cs):
        # the bracket colour: every cell on its bbox border, size == 2*side + 4
        if not cs:
            return -1
        r0, r1, c0, c1 = bbox(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if not all(r in (r0, r1) or c in (c0, c1) for r, c in cs):
            return -1
        if len(cs) not in (2 * bw + 4, 2 * bh + 4):
            return -1
        S = set(cs)
        s = 0
        if all((r0, c) in S for c in range(c0, c1 + 1)) and \
           all((r1, c) in S for c in range(c0, c1 + 1)):
            s += 1
        if all((r, c0) in S for r in range(r0, r1 + 1)) and \
           all((r, c1) in S for r in range(r0, r1 + 1)):
            s += 1
        return s

    if len(others) < 2:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    boxc = sorted(others, key=lambda c: -box_score(cells_of(I, c)))[0]
    objc = [c for c in others if c != boxc][0]

    box = cells_of(I, boxc)
    r0, r1, c0, c1 = bbox(box)
    center_c = c0 + (c1 - c0 + 1) // 2
    horizontal_lines = any(c == center_c for _, c in box)
    axis = 0 if horizontal_lines else 1          # axis along which things travel

    LO, HI = (r0, r1) if axis == 0 else (c0, c1)  # the two bracket lines
    flip_op = 27 if axis == 0 else 26             # FlipV / FlipH
    pos_op = 21 if axis == 0 else 22              # MoveD / MoveR
    neg_op = 20 if axis == 0 else 23              # MoveU / MoveL

    def sh(p, s):
        return (p[0] + s, p[1]) if axis == 0 else (p[0], p[1] + s)

    obj = cells_of(I, objc)
    groups = [
        ([p for p in obj if p[axis] < LO], LO + 2, 'lo'),
        ([p for p in obj if p[axis] > HI], HI - 2, 'hi'),
    ]

    for grp, target, side in groups:
        if not grp:
            continue
        co = [p[axis] for p in grp]
        gmin, gmax = min(co), max(co)

        def mir(p):
            return (gmin + gmax - p[0], p[1]) if axis == 0 else (p[0], gmin + gmax - p[1])

        mirrored = [mir(p) for p in grp]
        shift = (target - gmin) if side == 'lo' else (target - gmax)

        def fits(cellset, s):
            for p in cellset:
                q = sh(p, s)
                if not (0 <= q[0] < ho and 0 <= q[1] < wo):
                    return False
                if O[q[0], q[1]] != objc:
                    return False
            return True

        carried = mirrored
        use_shift = shift
        do_flip = set(mirrored) != set(grp)
        if not fits(mirrored, shift):
            picked = None
            for cand, flipped in ((mirrored, True), (grp, False)):
                best = None
                for s in range(-(ho + wo), ho + wo + 1):
                    if s == 0:
                        continue
                    if fits(cand, s) and (best is None or abs(s - shift) < abs(best - shift)):
                        best = s
                if best is not None:
                    picked = (cand, best, flipped and set(cand) != set(grp))
                    break
            if picked is not None:
                carried, use_shift, do_flip = picked
        if use_shift == 0:
            continue

        dest = [sh(p, use_shift) for p in carried]

        if objc == 0:
            # colour 0 is transparent to every ARCLE object op, so the pattern
            # cannot be grabbed and slid: draw it at its destination and clear
            # the vacated pattern instead.
            ops.append(0); sels.append(sel_of(sorted(set(dest))))
            ops.append(bgc); sels.append(sel_of(sorted(set(grp))))
            continue

        # 1) mirror the pattern in place (about its own bounding box)
        if do_flip:
            ops.append(flip_op); sels.append(sel_of(grp))
        # 2) slide it through the bracket line until it rests inside the box
        steps = abs(use_shift)
        mop = pos_op if use_shift > 0 else neg_op
        for k in range(steps):
            ops.append(mop)
            sels.append(sel_of(carried) if k == 0 else sel_of([]))
        # 3) the grabbed footprint (original + mirrored positions) is left at 0
        holes = (set(grp) | set(carried)) - set(dest)
        if bgc != 0 and holes:
            ops.append(bgc); sels.append(sel_of(sorted(holes)))

    # whole grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
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
                                f"for task 6855a6e4"
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
                    f"Failed to build a complete episode for task 6855a6e4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6855a6e4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
