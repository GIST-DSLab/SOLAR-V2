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


# ----------------------------------------------------------------------------
# 1. colours
# ----------------------------------------------------------------------------
# The rule is purely positional: a single marker cell touching a shape says on
# which side the shape is repeated and about which axis it is mirrored.  The
# marker's colour only tells which colour the copy is drawn in, and that colour
# is read straight off the marker, so the per-object foreground colours may stay
# random.  Only the canvas background has to be fixed for the whole episode.
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC b775ac94, colours injected, 30 -> max_h / max_w)
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    remcols = remove(bgc, cols)
    cap = max(1, (h * w) // 25)
    nobjs = unifint(diff_lb, diff_ub, (min(2, cap), cap))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    ohub = max(2, min(5, h // 2))
    owub = max(2, min(5, w // 2))
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(2, ohub)
        ow = randint(2, owub)
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


# ----------------------------------------------------------------------------
# 3. ARCLE trajectory
# ----------------------------------------------------------------------------
# Rule (read off the examples): every shape carries 1-3 single marker cells
# sitting just outside one corner / side of its bounding box.  A marker at
# block-offset (dr, dc) means "repeat the shape in that neighbouring block,
# mirrored up/down when dr != 0 and left/right when dc != 0, drawn in the
# marker's own colour".  The marker cell itself is always covered by that copy.
#
# Route per marker: dress the shape in the marker's colour, copy its block,
# paste it on the marked side, then mirror the pasted block with FlipV / FlipH.
# Afterwards the shape's own colour is put back.  Colour 0 never travels through
# Copy/Paste (ARCLE reads it as "nothing there"), so a copy whose colour is 0 is
# laid down with Color0 on the mirrored footprint instead.
def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = the colour every grid of the episode is drawn on
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            try:
                ei = np.asarray(ex[0], dtype=int)
            except Exception:
                continue
            cnt.update(ei.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # ---------------- tiny ARCLE simulator (Color / CopyO / Paste / Flip) ----
    def mask_of(sel):
        m = np.zeros((hi, wi), dtype=bool)
        if isinstance(sel, dict):
            for (r, c) in sel.get("cells", []):
                if 0 <= r < hi and 0 <= c < wi:
                    m[r, c] = True
        else:
            r, c, dh, dw = sel
            m[max(0, r):r + dh + 1, max(0, c):c + dw + 1] = True
        return m

    def apply_op(g, cl, op, sel):
        g = g.copy()
        m = mask_of(sel)
        if 0 <= op <= 9:
            g[m] = op
        elif op == 29:                                   # CopyO
            if m.any():
                rs, cs = np.where(m)
                r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
                sub_m = m[r0:r1 + 1, c0:c1 + 1]
                cl = np.where(sub_m, g[r0:r1 + 1, c0:c1 + 1], 0).copy()
        elif op == 30:                                   # Paste (transparent)
            if m.any() and cl.size:
                rs, cs = np.where(m)
                r0, c0 = rs.min(), cs.min()
                ch, cw = cl.shape
                rr = min(ch, hi - r0)
                cc = min(cw, wi - c0)
                if rr > 0 and cc > 0:
                    patch = cl[:rr, :cc]
                    blk = g[r0:r0 + rr, c0:c0 + cc]
                    blk[patch != 0] = patch[patch != 0]
        elif op in (26, 27):                             # FlipH / FlipV
            if m.any():
                rs, cs = np.where(m)
                r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
                sub_m = m[r0:r1 + 1, c0:c1 + 1]
                obj = np.where(sub_m, g[r0:r1 + 1, c0:c1 + 1], 0)
                blk = g[r0:r1 + 1, c0:c1 + 1]
                blk[sub_m] = 0
                f = np.fliplr(obj) if op == 26 else np.flipud(obj)
                nz = f != 0
                blk[nz] = f[nz]
        return g, cl

    grid = I.copy()
    clip = np.zeros((0, 0), dtype=int)
    ops, sels = [], []

    def emit(op, sel):
        nonlocal grid, clip
        ng, ncl = apply_op(grid, clip, op, sel)
        if op != 29 and np.array_equal(ng, grid):
            return False                                  # no visible change
        grid, clip = ng, ncl
        ops.append(op)
        sels.append(sel)
        return True

    # ---------------- objects: shape + its marker cells ---------------------
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                    and I[nr, nc] != bgc:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                comps.append(cells)

    for cells in comps:
        vals = Counter(int(I[r, c]) for (r, c) in cells)
        main_col = max(vals.items(), key=lambda kv: kv[1])[0]
        S = sorted([(r, c) for (r, c) in cells if I[r, c] == main_col])
        markers = sorted([((r, c), int(I[r, c])) for (r, c) in cells
                          if I[r, c] != main_col])
        if not S or not markers:
            continue
        r0 = min(p[0] for p in S)
        r1 = max(p[0] for p in S)
        c0 = min(p[1] for p in S)
        c1 = max(p[1] for p in S)
        oh, ow = r1 - r0 + 1, c1 - c0 + 1
        Sset = set(S)
        src_extra = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                     if (r, c) not in Sset]
        recoloured = False

        for (mpos, mcol) in markers:
            mr, mc = mpos
            dr = -1 if mr < r0 else (1 if mr > r1 else 0)
            dc = -1 if mc < c0 else (1 if mc > c1 else 0)
            if dr == 0 and dc == 0:
                continue
            tr = r0 + dr * oh
            tc = c0 + dc * ow
            if tr < 0 or tc < 0 or tr + oh > hi or tc + ow > wi:
                continue

            # the mirrored copy's own cells
            T = []
            for (r, c) in S:
                a, b = r - r0, c - c0
                if dr:
                    a = oh - 1 - a
                if dc:
                    b = ow - 1 - b
                T.append((tr + a, tc + b))
            Tset = set(T)
            rect = [(r, c) for r in range(tr, tr + oh) for c in range(tc, tc + ow)]
            pasted = {(tr + (r - r0), tc + (c - c0)) for (r, c) in S}

            # the block route overwrites/carries whole blocks: only take it when
            # no neighbouring object's cell sits inside either block
            clean = all(grid[p] == bgc for p in rect if p not in Tset) and \
                    all(grid[p] == bgc for p in src_extra)

            if mcol == 0 or not clean:
                # colour 0 never travels through Copy/Paste -> lay the mirrored
                # copy down directly (same for the rare crowded-block case)
                emit(mcol, sel_of(T))
                continue

            if bgc == 0 and mpos not in pasted:
                # transparent paste would leave this marker cell behind and the
                # mirror would carry it to a wrong place
                emit(bgc, sel_of([mpos]))
            emit(mcol, sel_of(S))                       # dress shape in marker colour
            emit(29, [r0, c0, oh - 1, ow - 1])          # copy the shape's block
            emit(30, [tr, tc, 0, 0])                    # lay it on the marked side
            # whole-block mirrors: the intended cells are exactly this rectangle,
            # background included, so the bbox selection is the right one
            if dr:
                emit(27, [tr, tc, oh - 1, ow - 1])      # FlipV (up<->down)
            if dc:
                emit(26, [tr, tc, oh - 1, ow - 1])      # FlipH (left<->right)
            recoloured = True

            miss = [p for p in T if grid[p] != mcol]
            if miss:
                emit(mcol, sel_of(miss))                # cells the paste could not carry
            stray = [p for p in rect if p not in Tset and grid[p] != bgc]
            if stray:
                emit(bgc, sel_of(stray))

        if recoloured:
            emit(main_col, sel_of(S))                   # put the shape's colour back

    # ---------------- drop any single op that is not load bearing -----------
    def simulate(o_list, s_list):
        g = I.copy()
        cl = np.zeros((0, 0), dtype=int)
        for op, sel in zip(o_list, s_list):
            g, cl = apply_op(g, cl, op, sel)
        return g

    final = simulate(ops, sels)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ops):
            t_ops = ops[:i] + ops[i + 1:]
            t_sels = sels[:i] + sels[i + 1:]
            if np.array_equal(simulate(t_ops, t_sels), final):
                ops, sels = t_ops, t_sels
                changed = True
            else:
                i += 1

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
