"""
ARC Task: 3345333e (RE-ARC) — LLM-generated grid_maker
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


# The object the generator builds is mirror-symmetric about a VERTICAL axis and the
# occluding rectangle always sits on its RIGHT half; the random dihedral transform
# applied at the end turns that into one of four discrete situations.  Plan them.
VARIANTS = [
    {"axis": "vertical",   "side": "right"},
    {"axis": "vertical",   "side": "left"},
    {"axis": "horizontal", "side": "bottom"},
    {"axis": "horizontal", "side": "top"},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc, occcol = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "occcol": occcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, occcol,
             axis=None, side=None) -> dict:
    if axis is None or side is None:
        v = choice(VARIANTS)
        axis, side = v["axis"], v["side"]

    # the class transform: base construction is (vertical axis, occluder on the right)
    classfn = {("vertical", "right"): identity,
               ("vertical", "left"): vmirror,
               ("horizontal", "bottom"): dmirror,
               ("horizontal", "top"): cmirror}[(axis, side)]
    fns = [classfn]
    # a further mirror along the symmetry axis keeps both axis and side intact
    if choice((True, False)):
        fns.append(hmirror if axis == "vertical" else vmirror)
    swapped = axis == "horizontal"      # dmirror / cmirror transpose the canvas

    hub = max(10, max_w if swapped else max_h)
    wub = max(10, max_h if swapped else max_w)

    h = unifint(diff_lb, diff_ub, (10, hub))
    w = unifint(diff_lb, diff_ub, (10, wub))
    oh = unifint(diff_lb, diff_ub, (4, h - 2))
    ow = unifint(diff_lb, diff_ub, (4, (w - 2) // 2))
    nc = unifint(diff_lb, diff_ub, (min(oh, ow), (oh * ow) // 3 * 2))
    shp = {(0, 0)}
    bounds = asindices(canvas(-1, (oh, ow)))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    while height(shp) < 3 or width(shp) < 3:
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    vmshp = vmirror(shp)
    if choice((True, False)):
        vmshp = sfilter(vmshp, lambda ij: ij[1] != width(shp) - 1)
    shp = normalize(combine(shp, shift(vmshp, (0, -width(vmshp)))))
    oh, ow = shape(shp)
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    loc = (loci, locj)
    shp = shift(shp, loc)
    c = canvas(bgc, (h, w))
    go = fill(c, objc, shp)
    boxh = unifint(diff_lb, diff_ub, (2, oh - 1))
    boxw = unifint(diff_lb, diff_ub, (2, ow // 2))
    ulci = randint(loci - 1, loci + oh - boxh + 1)
    ulcj = randint(locj + ow // 2 + 1, locj + ow - boxw + 1)
    bx = backdrop(frozenset({(ulci, ulcj), (ulci + boxh - 1, ulcj + boxw - 1)}))
    gi = fill(go, occcol, bx)
    for fn in fns:
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """A mirror-symmetric object is partly hidden under a solid rectangle of a third
    colour.  Everything below is read off I: the background is the grid border, the
    occluder is the colour that forms a solid rectangle, and the mirror line is the
    only line the *visible* grid is symmetric about (the occluder counting as a
    wild card).  The hidden part is then the reflection of the strip on the other
    side of that line: copy that strip out of the input, paste it over the
    rectangle and flip it.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # ── background: the object keeps a >=1 cell margin from every edge ──────────
    border = (list(I[0, :]) + list(I[h - 1, :]) + list(I[:, 0]) + list(I[:, w - 1]))
    bgc = Counter(int(v) for v in border).most_common(1)[0][0]

    # ── occluder: the non-background colour whose cells are a solid rectangle ───
    boxes = []
    for col in sorted({int(v) for v in np.unique(I)} - {bgc}):
        cells = np.argwhere(I == col)
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0)
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if len(cells) == area:                       # solid filled rectangle
            boxes.append((area, col, (int(r0), int(c0), int(r1), int(c1))))
    boxes.sort()

    # ── mirror line: reflect the whole grid, occluder cells are wild cards ──────
    def mirrored(axis, S):
        if axis == 1:
            idx = S - np.arange(w)
            ok = (idx >= 0) & (idx < w)
            return np.where(ok[None, :], I[:, np.clip(idx, 0, w - 1)], -1), ok
        idx = S - np.arange(h)
        ok = (idx >= 0) & (idx < h)
        return np.where(ok[:, None], I[np.clip(idx, 0, h - 1), :], -1), ok

    def consistent(occ, axis, S):
        B, _ = mirrored(axis, S)
        out = B < 0
        if np.any(out & (I != bgc)):
            return False                    # something real reflects off the canvas
        if np.any((I == occ) & (B == occ)):
            return False                    # hidden on both sides -> unrecoverable
        wild = (I == occ) | (B == occ)
        return not np.any(~out & ~wild & (I != B))

    found = None
    for _area, occ, box in boxes:
        for axis, lim in ((1, w), (0, h)):
            for S in range(2 * lim - 1):
                if consistent(occ, axis, S):
                    found = (occ, box, axis, S)
                    break
            if found:
                break
        if found:
            break

    if found is None:                       # no mirror line: nothing can be restored
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    occ, (r0, c0, r1, c1), axis, S = found
    box_sel = [r0, c0, r1 - r0, c1 - c0]     # the occluder IS exactly this rectangle
    if axis == 1:                            # vertical mirror line: reflect columns
        sr0, sr1, sc0, sc1 = r0, r1, S - c1, S - c0
        flip_op = 26                         # FlipH (left<->right)
    else:                                    # horizontal mirror line: reflect rows
        sr0, sr1, sc0, sc1 = S - r1, S - r0, c0, c1
        flip_op = 27                         # FlipV (up<->down)
    src = I[sr0:sr1 + 1, sc0:sc1 + 1]        # the strip the hidden part mirrors

    # Paste is transparent: a 0 in the strip writes nothing.  If those cells only
    # have to become background, wipe the rectangle to background first and let the
    # paste draw the object on top of it.
    blind = src == 0
    if not np.any(blind) or bgc == 0:
        if np.any(blind):                    # bgc == 0: lay the background base
            ops.append(int(bgc))
            sels.append(box_sel)
        ops.append(28)                                        # CopyI the strip
        sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0])         # whole rectangle
        ops.append(30)                                        # Paste onto the box
        sels.append([r0, c0, 0, 0])
        ops.append(flip_op)                                   # mirror it in place
        sels.append(box_sel)                                  # whole rectangle
    else:
        # object colour is 0, which Copy/Paste cannot carry: clear the rectangle to
        # background and paint the reflected object cells directly.
        ops.append(int(bgc))
        sels.append(box_sel)
        cells = [(r0 + (r1 - r0 - i if axis == 0 else i),
                  c0 + (c1 - c0 - j if axis == 1 else j))
                 for i in range(sr1 - sr0 + 1) for j in range(sc1 - sc0 + 1)
                 if src[i, j] == 0]
        if cells:
            ops.append(0)
            sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3345333e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3345333e"
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
                                f"for task 3345333e"
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
                    f"Failed to build a complete episode for task 3345333e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3345333e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
