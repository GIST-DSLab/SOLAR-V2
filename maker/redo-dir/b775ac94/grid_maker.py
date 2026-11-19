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
    # Background is the only episode-level colour role.  The rule ("mirror the body
    # into the block its marker pixel points at, painted in that marker's colour")
    # reads positions only, never particular colours, so c1..c4 stay per-object
    # random exactly as in the original generator.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    hub = max(10, min(30, int(max_h)))
    wub = max(10, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (min(10, hub), hub))
    w = unifint(diff_lb, diff_ub, (min(10, wub), wub))
    if bgc is None:
        bgc = choice(cols)
    # colour 0 is left out of the palette: ARCLE's clipboard reads 0 as "nothing",
    # and the trajectory below copies and pastes the bodies around.
    remcols = remove(bgc, remove(0, cols))
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
        # the whole bounding box is claimed (not just the painted cells): the
        # trajectory copies, pastes and flips whole rectangles of an instance,
        # so no other instance may reach inside one
        plcdoi = backdrop(toindices(plcdo))
        if plcdoi.issubset(inds):
            succ += 1
            inds = (inds - plcdoi) - mapply(neighbors, plcdoi)
            gi = paint(gi, plcdi)
            go = paint(go, plcdo)
    return {'input': gi, 'output': go}


# ── trajectory ────────────────────────────────────────────────────────────────
#
# Each placed instance is one 8-connected blob: a BODY (the colour that occupies
# two or more cells) plus one to three single MARKER pixels hanging off one
# corner of the body's bounding box.  A marker asks for the body REFLECTED into
# the adjacent bbox-sized block it points at, drawn in the marker's own colour:
# across the shared vertical edge when the block is left/right, across the shared
# horizontal edge when it is above/below, across both when it is diagonal.  The
# marker pixel is itself the mirror image of the body's corner cell, so the
# finished copy always lands back on top of it.
#
# So, per marker: paint the body in that marker's colour, CopyO the body's
# rectangle, Paste it into the block the marker points at, and FlipH / FlipV that
# block - that flip is the reflection the task is about.  A flip whose block is
# already symmetric about that axis is left out; it would change nothing.  Once
# an instance's copies are made the body is painted back in its own colour.

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


def _sel_cells(sel):
    """(cells, bbox) of a selection in either form."""
    if isinstance(sel, dict):
        cells = [(int(r), int(c)) for r, c in sel["cells"]]
    else:
        r, c, h, w = sel
        cells = [(i, j) for i in range(r, r + h + 1) for j in range(c, c + w + 1)]
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return cells, (min(rs), max(rs), min(cs), max(cs))


def _replay(I, ops, sels):
    """ARCLE's behaviour for the handful of ops used here, for self-checking."""
    grid = np.asarray(I, dtype=int).copy()
    h, w = grid.shape
    clip = None
    for op, sel in zip(ops, sels):
        cells, (rmin, rmax, cmin, cmax) = _sel_cells(sel)
        if op <= 9:
            for (r, c) in cells:
                grid[r, c] = op
        elif op == 28:
            clip = np.asarray(I, dtype=int)[rmin:rmax + 1, cmin:cmax + 1].copy()
        elif op == 29:
            clip = grid[rmin:rmax + 1, cmin:cmax + 1].copy()
        elif op == 30:
            if clip is None:
                continue
            ch, cw = clip.shape
            for i in range(min(ch, h - rmin)):
                for j in range(min(cw, w - cmin)):
                    if clip[i, j] != 0:
                        grid[rmin + i, cmin + j] = clip[i, j]
        elif op in (26, 27):
            sub = grid[rmin:rmax + 1, cmin:cmax + 1]
            grid[rmin:rmax + 1, cmin:cmax + 1] = (np.fliplr(sub) if op == 26
                                                  else np.flipud(sub))
    return grid


def _drop_useless(I, O, ops, sels):
    """Remove any operation the finished grid does not depend on."""
    changed = True
    while changed:
        changed = False
        for k in range(len(ops) - 1):          # never the Submit
            if np.array_equal(_replay(I, ops[:k] + ops[k + 1:],
                                      sels[:k] + sels[k + 1:]), O):
                ops = ops[:k] + ops[k + 1:]
                sels = sels[:k] + sels[k + 1:]
                changed = True
                break
    return ops, sels


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    # the colour the generator fills the canvas with before placing instances
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    grid = I.copy()          # working grid, kept in step with the ops emitted
    clip = None              # clipboard contents, likewise

    for comp in _components(I, bgc):
        cnt = Counter(int(I[r, c]) for r, c in comp)
        body_col = max(cnt.items(), key=lambda kv: kv[1])[0]   # markers are single cells
        body = sorted(p for p in comp if int(I[p]) == body_col)
        markers = sorted(p for p in comp if int(I[p]) != body_col)
        if len(body) < 2 or not markers:
            continue

        r0 = min(r for r, _ in body); r1 = max(r for r, _ in body)
        c0 = min(c for _, c in body); c1 = max(c for _, c in body)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1

        # the body's shape inside its own rectangle: its symmetries tell which
        # flips below are real moves and which would change nothing
        mask = np.zeros((bh, bw), dtype=bool)
        for (r, c) in body:
            mask[r - r0, c - c0] = True
        lr_same = np.array_equal(mask, np.fliplr(mask))
        ud_same = np.array_equal(mask, np.flipud(mask))
        r180_same = np.array_equal(mask, np.rot90(mask, 2))

        compset = set(comp)
        cur_col = body_col
        for (mr, mc) in markers:
            mcol = int(I[mr, mc])
            vd = -1 if mr < r0 else (1 if mr > r1 else 0)   # block above / below
            hd = -1 if mc < c0 else (1 if mc > c1 else 0)   # block left / right
            if vd == 0 and hd == 0:
                continue
            tr0, tc0 = r0 + vd * bh, c0 + hd * bw           # block the marker points at
            block = [tr0, tc0, bh - 1, bw - 1]              # exactly that rectangle
            # where the reflection sends each body cell
            mirrored = []
            for (r, c) in body:
                nr = 2 * r0 - 1 - r if vd < 0 else (2 * r1 + 1 - r if vd > 0 else r)
                nc = 2 * c0 - 1 - c if hd < 0 else (2 * c1 + 1 - c if hd > 0 else c)
                mirrored.append((nr, nc))

            # A neighbouring instance reaching into this block would be dragged
            # along by the copy/flip, so such a block is painted cell by cell.
            if (tr0 < 0 or tc0 < 0 or tr0 + bh > grid.shape[0]
                    or tc0 + bw > grid.shape[1]
                    or any(I[r, c] != bgc and (r, c) not in compset
                           for r in range(tr0, tr0 + bh)
                           for c in range(tc0, tc0 + bw))):
                ops.append(mcol); sels.append(sel_of(mirrored))
                for p in mirrored:
                    grid[p] = mcol
                continue

            # ARCLE's clipboard reads colour 0 as "nothing", so a copy that has to
            # come out black is carried in a spare colour and blackened at the end.
            pc = mcol if mcol != 0 else min(c for c in range(1, 10)
                                            if c not in cnt and c != bgc)

            # paint the body in the colour this copy has to wear, and pick it up
            # (the clipboard survives between instances, so an identical copy
            # that is still on it is simply reused)
            patch = np.where(mask, pc, bgc)
            if clip is None or clip.shape != patch.shape or not np.array_equal(clip, patch):
                if cur_col != pc:
                    ops.append(pc); sels.append(sel_of(body))
                    for p in body:
                        grid[p] = pc
                    cur_col = pc
                ops.append(29); sels.append([r0, c0, bh - 1, bw - 1])
                clip = patch.copy()

            # Paste writes nothing where the copy is colour 0, so a cell of the
            # block it leaves untouched - the marker pixel, typically - would
            # survive into the reflected copy: clear the block first.
            if any(patch[i, j] == 0 and grid[tr0 + i, tc0 + j] != bgc
                   for i in range(bh) for j in range(bw)):
                ops.append(bgc); sels.append(block)
                grid[tr0:tr0 + bh, tc0:tc0 + bw] = bgc
            # stamp the body into the block the marker points at ...
            ops.append(30); sels.append([tr0, tc0, 0, 0])
            for i in range(bh):
                for j in range(bw):
                    if patch[i, j] != 0:
                        grid[tr0 + i, tc0 + j] = patch[i, j]

            # ... and reflect it there.  Flipping a block whose shape is already
            # symmetric about that axis would change nothing, so it is left out.
            def _flip(op):
                ops.append(op); sels.append(block)
                sub = grid[tr0:tr0 + bh, tc0:tc0 + bw]
                grid[tr0:tr0 + bh, tc0:tc0 + bw] = (np.fliplr(sub) if op == 26
                                                    else np.flipud(sub))
            if vd and hd:
                if not r180_same:                    # rot180 = FlipH then FlipV
                    if not lr_same:
                        _flip(26)
                    if not ud_same:
                        _flip(27)
            elif hd and not lr_same:
                _flip(26)
            elif vd and not ud_same:
                _flip(27)

            if pc != mcol:
                ops.append(mcol); sels.append(sel_of(mirrored))
                for p in mirrored:
                    grid[p] = mcol

        # give the body its own colour back
        if cur_col != body_col:
            ops.append(body_col); sels.append(sel_of(body))
            for p in body:
                grid[p] = body_col

    # nothing here should be droppable, but never hand over an operation the
    # finished grid does not actually depend on
    ops, sels = _drop_useless(I, O, ops, sels)

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
