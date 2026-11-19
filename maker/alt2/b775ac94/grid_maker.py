"""
ARC Task: b775ac94 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors
# Only the canvas background is a real "role" colour here: the rule is purely
# positional (a marker cell sitting next to a corner of a shape says on which
# side the shape is repeated and about which axis it is mirrored).  The body /
# marker colours are arbitrary per object, so they stay free.
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


# ---------------------------------------------------------------- generator
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    gi = canvas(0, (1, 1))
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(2, 5)
        ow = randint(2, 5)
        canv = canvas(bgc, (oh, ow))
        c1, c2, c3, c4 = sample(remcols, 4)
        obj = {(0, 0)}
        ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        bounds = asindices(canv)
        for k in range(ncells):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        gLR = fill(canv, c1, obj)
        gLL = replace(vmirror(gLR), c1, c2)
        gUR = replace(hmirror(gLR), c1, c3)
        gUL = replace(vmirror(hmirror(gLR)), c1, c4)
        gU = hconcat(gUL, gUR)
        gL = hconcat(gLL, gLR)
        g = vconcat(gU, gL)
        g2 = canvas(bgc, (oh * 2, ow * 2))
        g2 = fill(g2, c1, shift(obj, (oh, ow)))
        nkeepcols = unifint(diff_lb, diff_ub, (1, 3))
        keepcols = sample((c2, c3, c4), nkeepcols)
        for cc in (c2, c3, c4):
            if cc not in keepcols:
                g = replace(g, cc, bgc)
            else:
                ofsi = -1 if cc in (c3, c4) else 0
                ofsj = -1 if cc in (c2, c4) else 0
                g2 = fill(g2, cc, {(oh + ofsi, ow + ofsj)})
        rotf = choice((identity, rot90, rot180, rot270))
        g = rotf(g)
        g2 = rotf(g2)
        obji = asobject(g2)
        objo = asobject(g)
        objo = sfilter(objo, lambda cij: cij[0] != bgc)
        obji = sfilter(obji, lambda cij: cij[0] != bgc)
        tonorm = invert(ulcorner(objo))
        obji = shift(obji, tonorm)
        objo = shift(objo, tonorm)
        oh, ow = shape(objo)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcdi = shift(obji, loc)
        plcdo = shift(objo, loc)
        plcdoi = toindices(plcdo)
        if plcdoi.issubset(inds):
            succ += 1
            inds = (inds - plcdoi) - mapply(neighbors, plcdoi)
            gi = paint(gi, plcdi)
            go = paint(go, plcdo)
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- trajectory
def derive_operations(I, O, examples=None):
    """
    Every shape carries one or more single marker cells touching a corner of its
    bounding box.  A marker one block to the left/right -> the shape is repeated
    on that side mirrored left<->right; above/below -> mirrored up<->down;
    diagonally -> both (rot180).  The copy takes the marker's colour.

    Trajectory per marker: lay the shape down again on the indicated side with a
    single Color op, then express the mirror with FlipV / FlipH on that copy.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = the canvas colour (shared by every demonstration input)
    cnt = Counter(I.flatten().tolist())
    if examples:
        for pair in examples:
            try:
                ex_in = np.asarray(pair[0], dtype=int)
            except Exception:
                continue
            cnt.update(ex_in.flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])

    # diagonally-connected non-background components: body + its markers
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                comp = []
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                    and I[nr, nc] != bgc:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                comps.append(sorted(comp))

    ops, sels = [], []

    for comp in sorted(comps, key=lambda cm: cm[0]):
        colcnt = Counter(int(I[r, c]) for (r, c) in comp)
        body_col = colcnt.most_common(1)[0][0]          # the shape (markers are single cells)
        body = sorted(p for p in comp if int(I[p[0], p[1]]) == body_col)
        markers = sorted(p for p in comp if int(I[p[0], p[1]]) != body_col)
        if len(body) < 2 or not markers:
            continue
        r0 = min(r for r, _ in body)
        r1 = max(r for r, _ in body)
        c0 = min(c for _, c in body)
        c1 = max(c for _, c in body)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1

        for (mr, mc) in markers:
            mcol = int(I[mr, mc])
            di = -1 if mr < r0 else (1 if mr > r1 else 0)
            dj = -1 if mc < c0 else (1 if mc > c1 else 0)
            if di == 0 and dj == 0:
                continue
            R0, C0 = r0 + di * bh, c0 + dj * bw          # destination block origin
            if R0 < 0 or C0 < 0 or R0 + bh > hi or C0 + bw > wi:
                continue

            laid = [(r + di * bh, c + dj * bw) for (r, c) in body]

            if mcol == 0:
                # ARCLE's object ops only grab NON-zero cells, so a 0-coloured
                # copy cannot be flipped in place; place the mirrored shape with
                # one Color0 op instead.
                fin = list(laid)
                if di != 0:
                    fin = [(2 * R0 + bh - 1 - r, c) for (r, c) in fin]
                if dj != 0:
                    fin = [(r, 2 * C0 + bw - 1 - c) for (r, c) in fin]
                ops.append(0)
                sels.append(sel_of(sorted(set(fin))))
                continue

            cur = list(laid)
            touched = set(cur)
            # lay the shape down again on the side the marker points to
            ops.append(mcol)
            sels.append(sel_of(sorted(set(cur))))

            if di != 0:                                   # mirror up<->down
                nxt = [(2 * R0 + bh - 1 - r, c) for (r, c) in cur]
                if set(nxt) != set(cur):                  # skip when symmetric (no-op)
                    ops.append(27)
                    sels.append(sel_of(sorted(set(cur))))
                    touched |= set(nxt)
                cur = nxt
            if dj != 0:                                   # mirror left<->right
                nxt = [(r, 2 * C0 + bw - 1 - c) for (r, c) in cur]
                if set(nxt) != set(cur):
                    ops.append(26)
                    sels.append(sel_of(sorted(set(cur))))
                    touched |= set(nxt)
                cur = nxt

            # the flip zeroes the cells the copy left behind; restore them
            hole = sorted(touched - set(cur))
            if bgc != 0 and hole:
                ops.append(bgc)
                sels.append(sel_of(hole))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])                   # whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task b775ac94"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b775ac94"
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
                                f"for task b775ac94"
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
                    f"Failed to build a complete episode for task b775ac94 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b775ac94-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
