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
    # Only the background is a per-episode colour role.  The rule ("mirror the
    # body across the bbox edge/corner the marker hangs off, in the marker's
    # colour") depends on marker *position*, never on which foreground colours
    # were drawn, so c1..c4 stay per-object exactly as in the original generator.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **color_kwargs) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    if bgc is None:
        bgc = choice(cols)
    # Foreground colours are never 0: colour 0 is "transparent" to every ARCLE
    # object operation, so a 0-coloured body could not be reflected with a Flip.
    # When bgc == 0 this is exactly the original remove(bgc, cols).
    remcols = tuple(c for c in range(1, 10) if c != bgc)
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


def _components(I, bgc):
    """8-connected non-background blobs of I, in raster order of their first cell.

    One placed instance is exactly one blob: the body plus its 1..3 marker
    pixels (each marker is 8-adjacent to the body's outer corner cell), and the
    generator keeps different instances >=2 cells apart, so blobs never merge.
    """
    h, w = I.shape
    seen = set()
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if I[r0, c0] == bgc or (r0, c0) in seen:
                continue
            stack = [(r0, c0)]
            seen.add((r0, c0))
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and (nr, nc) not in seen and I[nr, nc] != bgc):
                            seen.add((nr, nc))
                            stack.append((nr, nc))
            comps.append(comp)
    return comps


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # Background = the canvas colour the generator paints before placing objects.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    grid = I.copy()

    for comp in _components(I, bgc):
        # Body = the colour that occurs more than once; markers are single pixels.
        cnt = Counter(int(I[r, c]) for r, c in comp)
        bc = max(cnt.items(), key=lambda kv: kv[1])[0]
        body = sorted((r, c) for r, c in comp if int(I[r, c]) == bc)
        markers = sorted((r, c) for r, c in comp if int(I[r, c]) != bc)
        if not markers or len(body) < 2:
            continue

        r0 = min(r for r, _ in body)
        c0 = min(c for _, c in body)
        bh = max(r for r, _ in body) - r0 + 1
        bw = max(c for _, c in body) - c0 + 1

        for m in markers:
            mc = int(I[m])
            # Which neighbouring bbox-sized block does this marker sit in?
            dr = -1 if m[0] < r0 else (1 if m[0] >= r0 + bh else 0)
            dc = -1 if m[1] < c0 else (1 if m[1] >= c0 + bw else 0)
            if dr == 0 and dc == 0:
                continue
            tr0, tc0 = r0 + dr * bh, c0 + dc * bw          # copy's bbox origin
            trans = sorted((r + dr * bh, c + dc * bw) for r, c in body)
            hmir = lambda p: (2 * tr0 + bh - 1 - p[0], p[1])
            vmir = lambda p: (p[0], 2 * tc0 + bw - 1 - p[1])
            want = set(trans)
            if dr:
                want = {hmir(p) for p in want}
            if dc:
                want = {vmir(p) for p in want}
            if any(not (0 <= r < h and 0 <= c < w) for r, c in list(trans) + list(want)):
                continue

            S = set(trans)
            # Two cases the copy-then-flip route cannot take, both drawn straight
            # into place instead:
            #   * colour 0 is transparent to ARCLE object ops, so a 0-coloured
            #     copy cannot be grabbed and flipped at all;
            #   * a cell of some *other* instance sitting where the un-mirrored
            #     copy would land would be destroyed by laying the copy there.
            if mc == 0 or any(int(I[p]) != bgc for p in S - want):
                cells = sorted(p for p in want if grid[p] != mc)
                if cells:
                    ops.append(mc)
                    sels.append(sel_of(cells))
                    for p in cells:
                        grid[p] = mc
                continue

            # 1. lay a copy of the body, in the marker's colour, in the block the
            #    marker points at (the marker pixel is already that colour).
            paint_cells = [p for p in trans if grid[p] != mc]
            if paint_cells:
                ops.append(mc)
                sels.append(sel_of(paint_cells))
                for p in paint_cells:
                    grid[p] = mc

            # 2. reflect that copy across the edge it shares with the body:
            #    across the horizontal edge when the block is above/below,
            #    across the vertical edge when it is left/right, both when
            #    it is diagonal.  Symmetric copies need no flip at all.
            flips = []
            if S != want:
                if dr and dc:
                    if {hmir(p) for p in S} == want:
                        flips = [27]
                    elif {vmir(p) for p in S} == want:
                        flips = [26]
                    else:
                        flips = [27, 26]
                elif dr:
                    flips = [27]
                else:
                    flips = [26]
            cur, vacated = S, set()
            for fop in flips:
                ops.append(fop)
                sels.append(sel_of(cur))          # grab the copy where it is now
                vacated |= cur
                cur = {(hmir(p) if fop == 27 else vmir(p)) for p in cur}
            if flips:
                for p in vacated - cur:
                    grid[p] = 0
                for p in cur:
                    grid[p] = mc
                # 3. flipping left the cells the copy no longer covers at 0;
                #    give them the background back (nothing to do when bgc == 0).
                holes = sorted(vacated - cur)
                if bgc != 0 and holes:
                    ops.append(bgc)
                    sels.append(sel_of(holes))
                    for p in holes:
                        grid[p] = bgc

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
