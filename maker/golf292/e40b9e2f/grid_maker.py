"""
ARC Task: e40b9e2f (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    d = unifint(diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = randint(0, h - d)
    locj = randint(0, w - d)
    loc = (loci, locj)
    if bgc is None:
        bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    subg = canvas(bgc, (d, d))
    inds = asindices(subg)
    if d % 2 == 0:
        q = sfilter(inds, lambda ij: ij[0] < d//2 and ij[1] < d//2)
        cp = {(d//2-1, d//2-1), (d//2, d//2-1), (d//2-1, d//2), (d//2, d//2)}
    else:
        q = sfilter(inds, lambda ij: ij[0] < d//2 and ij[1] <= d//2)
        cp = {(d//2, d//2)} | ineighbors((d//2, d//2))
    nrings = unifint(diff_lb, diff_ub, (1, max(1, (d-2)//2)))
    rings = set()
    for k in range(nrings):
        ring = box({(k, k), (d-k-1, d-k-1)})
        rings = rings | ring
    qin = q - rings
    qout = rings & q
    ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
    tailobjcells = sample(totuple(q), ntailobjcells)
    tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
    tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    while hmirror(tailobj) == tailobj and vmirror(tailobj) == tailobj:
        ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
        tailobjcells = sample(totuple(q), ntailobjcells)
        tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
        tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    for k in range(4):
        subg = paint(subg, tailobj)
        subg = rot90(subg)
    fxobj = recolor(choice(ccols), cp)
    subg = paint(subg, fxobj)
    subgi = subg
    subgo = tuple(e for e in subgi)
    subgi = fill(subgi, bgc, rings)
    nsplits = unifint(diff_lb, diff_ub, (1, 4))
    splits = [set() for k in range(nsplits)]
    for idx, cel in enumerate(tailobj):
        splits[idx % nsplits].add(cel)
    for jj in range(4):
        if jj < len(splits):
            subgi = paint(subgi, splits[jj])
        subgi = rot90(subgi)
    subgi = paint(subgi, fxobj)
    rotf = choice((identity, rot90, rot180, rot270))
    subgi = rotf(subgi)
    subgo = rotf(subgo)
    gi = paint(canvas(bgc, (h, w)), shift(asobject(subgi), loc))
    go = paint(canvas(bgc, (h, w)), shift(asobject(subgo), loc))
    return {'input': gi, 'output': go}


def _find_bgc(I, examples):
    hi, wi = I.shape
    for r in range(hi):
        if np.all(I[r] == I[r, 0]):
            return int(I[r, 0])
    for c in range(wi):
        if np.all(I[:, c] == I[0, c]):
            return int(I[0, c])
    return int(Counter(I.flatten().tolist()).most_common(1)[0][0])


def _find_pivot(I, bgc):
    """Pivot (doubled coords) of the pinwheel, measured from I alone."""
    hi, wi = I.shape
    fg = [(r, c, int(I[r, c])) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    if not fg:
        return None
    r0 = min(p[0] for p in fg); r1 = max(p[0] for p in fg)
    c0 = min(p[1] for p in fg); c1 = max(p[1] for p in fg)

    def rot_images(r, c, cr2, cc2):
        s = (cr2 + cc2) // 2
        t = (cc2 - cr2) // 2
        return [(s - c, t + r), (cr2 - r, cc2 - c), (c - t, s - r)]

    best = None
    for cr2 in range(2 * (hi - 1) + 1):
        for cc2 in range(2 * (wi - 1) + 1):
            if (cr2 - cc2) % 2:
                continue
            ext = max(cr2 - 2 * r0, 2 * r1 - cr2, cc2 - 2 * c0, 2 * c1 - cc2)
            if ext < 0:
                continue
            if cr2 - ext < 0 or cr2 + ext > 2 * (hi - 1):
                continue
            if cc2 - ext < 0 or cc2 + ext > 2 * (wi - 1):
                continue
            if cr2 % 2:
                mark = [(cr2 // 2, cc2 // 2), (cr2 // 2, cc2 // 2 + 1),
                        (cr2 // 2 + 1, cc2 // 2), (cr2 // 2 + 1, cc2 // 2 + 1)]
            else:
                pr, pc = cr2 // 2, cc2 // 2
                mark = [(pr, pc), (pr-1, pc-1), (pr-1, pc+1), (pr+1, pc-1), (pr+1, pc+1)]
            if any(not (0 <= p[0] < hi and 0 <= p[1] < wi) for p in mark):
                continue
            mv = {int(I[p]) for p in mark}
            if len(mv) != 1 or mv.pop() == bgc:
                continue
            agree = 0
            ok = True
            for (r, c, v) in fg:
                for (nr, nc) in rot_images(r, c, cr2, cc2):
                    if not (0 <= nr < hi and 0 <= nc < wi):
                        ok = False; break
                    u = int(I[nr, nc])
                    if u == v:
                        agree += 1
                    elif u != bgc:
                        ok = False; break
                if not ok:
                    break
            if not ok:
                continue
            side_hi = 7 if cr2 % 2 == 0 else 6
            core = 0
            for side in range(side_hi, 0, -1):
                if side % 2 != (0 if cr2 % 2 else 1):
                    continue
                tr = (cr2 - (side - 1)) // 2
                tc = (cc2 - (side - 1)) // 2
                if tr < 0 or tc < 0 or tr + side > hi or tc + side > wi:
                    continue
                P = I[tr:tr + side, tc:tc + side]
                if np.array_equal(P, np.rot90(P)) and int((P != bgc).sum()) >= 2:
                    core = side; break
            key = (-core, abs(cr2 - (r0 + r1)) + abs(cc2 - (c0 + c1)), -agree, ext)
            if best is None or key < best[0]:
                best = (key, (cr2, cc2, ext))
    return None if best is None else best[1]


def derive_operations(I, O, examples=None):
    """Complete the pinwheel by stamping the figure's quarter turns onto itself.

    The figure's square block (centred on the solid centre mark, just big enough
    to hold every foreground cell) is copied, rotated a quarter turn in place,
    and the copy pasted back over it: figure + its 90-degree image.  That union
    is copied, turned half a turn (FlipH+FlipV) and pasted back: all four
    quarter turns.  Paste only treats 0 as transparent, so when the background
    is not 0 the block's background is first turned to 0 and restored at the
    end; foreground 0s (invisible to Copy/Paste) then get their four quarter
    turns drawn in one Color0 stroke.
    Everything is measured from I; O is never inspected.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    full = [0, 0, hi - 1, wi - 1]
    bgc = _find_bgc(I, examples)
    piv = _find_pivot(I, bgc)
    if piv is None:
        return [34], [full]
    cr2, cc2, ext = piv
    tr, tc, S = (cr2 - ext) // 2, (cc2 - ext) // 2, ext + 1
    sq = [tr, tc, S - 1, S - 1]          # the figure's square block (full rectangle)
    sq_cells = [(r, c) for r in range(tr, tr + S) for c in range(tc, tc + S)]

    cur = I.copy()
    ops, sels = [], []

    def emit(op, sel, newgrid):
        nonlocal cur
        if op < 28 and np.array_equal(newgrid, cur):
            return False
        ops.append(op); sels.append(sel); cur = newgrid
        return True

    def paint(color, cells):
        g = cur.copy()
        for p in cells:
            g[p] = color
        return g

    swap = bgc != 0
    if swap:
        # make the block's background transparent for Paste: bgc -> 0
        bgs = [p for p in sq_cells if cur[p] == bgc]
        emit(0, sel_of(bgs), paint(0, bgs))

    def block():
        return cur[tr:tr + S, tc:tc + S].copy()

    def paste(clip):
        g = cur.copy()
        sub = g[tr:tr + S, tc:tc + S]
        m = clip != 0
        sub[m] = clip[m]
        return g

    def put(B):
        g = cur.copy(); g[tr:tr + S, tc:tc + S] = B; return g

    # figure + its quarter turn
    clip = block()
    ops.append(28 if not swap else 29); sels.append(sq)
    emit(24, sq, put(np.rot90(block(), 1)))
    emit(30, [tr, tc, 0, 0], paste(clip))
    # that pair + its half turn
    clip = block()
    ops.append(29); sels.append(sq)
    emit(26, sq, put(np.fliplr(block())))
    emit(27, sq, put(np.flipud(block())))
    emit(30, [tr, tc, 0, 0], paste(clip))

    if swap:
        # restore the background: every 0 left in the block -> bgc
        zeros = [p for p in sq_cells if cur[p] == 0]
        emit(bgc, sel_of(zeros), paint(bgc, zeros))
        # foreground 0s were invisible to Copy/Paste: draw their four quarter
        # turns (the cells their copies land on) in one stroke
        s2, t2 = (cr2 + cc2) // 2, (cc2 - cr2) // 2
        fz = set()
        for (r, c) in zip(*np.nonzero(I == 0)):
            r, c = int(r), int(c)
            fz |= {(r, c), (s2 - c, t2 + r), (cr2 - r, cc2 - c), (c - t2, s2 - r)}
        fz = sorted(fz)
        if fz:
            emit(0, sel_of(fz), paint(0, fz))

    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
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
                                f"for task e40b9e2f"
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
                    f"Failed to build a complete episode for task e40b9e2f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e40b9e2f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
