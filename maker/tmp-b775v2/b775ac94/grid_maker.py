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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of




def sample_colors(num_examples=None) -> dict:
    # Background is the only episode-level colour role.  The rule ("mirror the
    # body into the block its marker pixel points at, in that marker's colour")
    # reads positions, never particular colours, so c1..c4 stay per-object
    # random exactly as in the original generator.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    if bgc is None:
        bgc = choice(cols)
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


# ── trajectory ────────────────────────────────────────────────────────────────
#
# Every placed instance is one 8-connected blob: a body (the colour that repeats)
# plus 1..3 single marker pixels hanging off one corner of the body's bounding
# box.  A marker asks for the body MIRRORED into the neighbouring bbox-sized
# block it points at, drawn in the marker's own colour: across the shared
# horizontal edge when the block is above/below, across the shared vertical edge
# when it is left/right, across both when the block is diagonal.  The marker
# pixel itself is the mirror image of the body's corner cell, so the finished
# copy always lands back on it.
#
# Per marker: clear the block it points into (that erases the marker pixel, so
# nothing foreign is inside when the block is mirrored), stamp a plain copy of
# the body there in the marker's colour, then FLIP that block - FlipV (27) for an
# up/down mirror, FlipH (26) for a left/right one, both for a diagonal block.


def _components(I, bgc):
    """8-connected blobs of non-background cells, in raster order."""
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w and not seen[nr, nc]
                                and I[nr, nc] != bgc):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(sorted(cells))
    return comps


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # the colour the generator fills the canvas with before placing instances
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    grid = I.copy()          # working grid, kept in step with the ops emitted

    for comp in _components(I, bgc):
        cnt = Counter(int(I[r, c]) for r, c in comp)
        body_col = max(cnt.items(), key=lambda kv: kv[1])[0]
        body = sorted(p for p in comp if int(I[p]) == body_col)
        markers = sorted(p for p in comp if int(I[p]) != body_col)
        if len(body) < 2 or not markers:
            continue

        r0 = min(r for r, _ in body)
        c0 = min(c for _, c in body)
        bh = max(r for r, _ in body) - r0 + 1
        bw = max(c for _, c in body) - c0 + 1

        for m in markers:
            mc = int(I[m])
            # which neighbouring block does this marker hang off?
            vd = -1 if m[0] < r0 else (1 if m[0] > r0 + bh - 1 else 0)
            hd = -1 if m[1] < c0 else (1 if m[1] > c0 + bw - 1 else 0)
            if vd == 0 and hd == 0:
                continue
            tr0, tc0 = r0 + vd * bh, c0 + hd * bw
            if tr0 < 0 or tc0 < 0 or tr0 + bh > hi or tc0 + bw > wi:
                continue
            rect = [(r, c) for r in range(tr0, tr0 + bh)
                    for c in range(tc0, tc0 + bw)]

            # the body copied straight across, and where the mirror puts it
            trans = [(r + vd * bh, c + hd * bw) for r, c in body]
            mirr = list(trans)
            if vd:
                mirr = [(2 * tr0 + bh - 1 - r, c) for r, c in mirr]
            if hd:
                mirr = [(r, 2 * tc0 + bw - 1 - c) for r, c in mirr]

            # anything in that block other than this marker belongs to another
            # instance; mirroring the block would carry it along, so that copy
            # is drawn straight into its mirrored place instead.
            foreign = [p for p in rect if grid[p] != bgc and p != tuple(m)]
            if foreign:
                cells = [p for p in mirr if grid[p] != mc]
                if cells:
                    ops.append(mc)
                    sels.append(sel_of(cells))
                    for p in cells:
                        grid[p] = mc
                continue

            # which mirror does this block need?  A body symmetric about the
            # shared edge already sits where the mirror puts it - no flip then.
            flips = []
            if set(mirr) != set(trans):
                if vd and hd:
                    if {(r, 2 * tc0 + bw - 1 - c) for r, c in trans} == set(mirr):
                        flips = [26]
                    elif {(2 * tr0 + bh - 1 - r, c) for r, c in trans} == set(mirr):
                        flips = [27]
                    else:
                        flips = [27, 26]
                elif vd:
                    flips = [27]
                else:
                    flips = [26]

            # 1. clear what the copy will not cover out of the block it points
            #    into (its own marker pixel), so the flip carries nothing else.
            if flips:
                clear = [p for p in rect
                         if grid[p] != bgc and p not in set(trans)]
                if clear:
                    ops.append(bgc)
                    sels.append(sel_of(clear))
                    for p in clear:
                        grid[p] = bgc

            # 2. stamp a copy of the body into that block, in the marker colour
            stamp = [p for p in trans if grid[p] != mc]
            if stamp:
                ops.append(mc)
                sels.append(sel_of(stamp))
                for p in stamp:
                    grid[p] = mc

            # 3. mirror the block.  The selection is that whole bbox-sized
            #    block, background included - exactly the region the reflection
            #    acts on:  FlipV (27) across the shared horizontal edge, FlipH
            #    (26) across the shared vertical one, both when it is diagonal.
            for fop in flips:
                ops.append(fop)
                sels.append([tr0, tc0, bh - 1, bw - 1])
                block = grid[tr0:tr0 + bh, tc0:tc0 + bw]
                grid[tr0:tr0 + bh, tc0:tc0 + bw] = (
                    np.flipud(block) if fop == 27 else np.fliplr(block))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
