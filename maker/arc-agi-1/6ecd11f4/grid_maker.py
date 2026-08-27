"""
ARC Task: 6ecd11f4 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- 1. colors
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fgc = random.sample(cols, 2)
    remcols = [c for c in cols if c != bgc]          # fgc may be reused inside the patch
    ncols = random.randint(2, 9)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "fgc": fgc, "ccols": ccols}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, ccols) -> dict:
    ccols = list(ccols)

    # minimal canvas needed later is 3*h+1 rows / 3*w+1 cols (hscf>=2 plus margin)
    h_ub = max(2, min(7, (max_h - 1) // 3))
    w_ub = max(2, min(7, (max_w - 1) // 3))
    h = unifint(diff_lb, diff_ub, (2, h_ub))
    w = unifint(diff_lb, diff_ub, (2, w_ub))

    inds = asindices(canvas(bgc, (h, w)))
    nlocsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nlocs = choice((nlocsd, h * w - nlocsd))
    nlocs = min(max(3, nlocs), h * w - 1)
    sp = choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(nlocs):
        ij = choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    h, w = shape(shp)
    canv = canvas(bgc, (h, w))
    objbase = fill(canv, fgc, shp)

    maxhscf = max(2, min((2 * h + h + 1) // h, (max_h - h - 1) // h))
    maxwscf = max(2, min((2 * w + w + 1) // w, (max_w - w - 1) // w))
    hscf = unifint(diff_lb, diff_ub, (2, maxhscf))
    wscf = unifint(diff_lb, diff_ub, (2, maxwscf))
    obj = asobject(hupscale(vupscale(objbase, hscf), wscf))
    oh, ow = shape(obj)

    inds = asindices(canv)
    objx = {(choice(ccols), ij) for ij in inds}
    if len(palette(objx)) == 1:
        objxodo = first(objx)
        objx = insert((choice(remove(objxodo[0], ccols)), objxodo[1]), remove(objxodo, objx))

    fullh = unifint(diff_lb, diff_ub, (min(hscf * h + h + 1, max_h), max_h))
    fullw = unifint(diff_lb, diff_ub, (min(wscf * w + w + 1, max_w), max_w))
    gi = canvas(bgc, (fullh, fullw))
    fullinds = asindices(gi)
    while True:
        loci = randint(0, fullh - oh)
        locj = randint(0, fullw - ow)
        loc = (loci, locj)
        gix = paint(gi, shift(obj, loc))
        ofc = ofcolor(gix, fgc)
        delt = (fullinds - ofc)
        delt2 = delt - mapply(neighbors, ofc)
        scands = sfilter(
            delt2,
            lambda ij: ij[0] <= fullh - oh and ij[1] <= fullw - ow
        )
        if len(scands) == 0:
            continue
        locc = choice(totuple(scands))
        shftd = shift(objx, locc)
        if toindices(shftd).issubset(delt2):
            gi = paint(gix, shftd)
            break

    go = paint(canv, objx)
    go = fill(go, bgc, ofcolor(objbase, bgc))
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- 3. derive
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background: the colour the generator paints the whole canvas with (always dominant here)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- find the two objects: 8-connected components of non-background cells
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                seen[r, c] = True
                stack = [(r, c)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != bgc:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                comps.append(cells)

    def ncolors(cells):
        return len({int(I[y, x]) for y, x in cells})

    def bbox(cells):
        rs = [y for y, _ in cells]
        cs = [x for _, x in cells]
        return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    if len(comps) < 2:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # multi-coloured patch = most distinct colours ; monochrome upscaled shape = fewest
    patch = max(comps, key=ncolors)
    shape_cells = min([c for c in comps if c is not patch], key=ncolors)

    pr, pc, ph, pw = bbox(patch)
    sr, sc, sh, sw = bbox(shape_cells)
    fgc = int(I[shape_cells[0][0], shape_cells[0][1]])

    hscf = max(1, sh // ph)
    wscf = max(1, pw and sw // pw)

    # --- downscaled stencil of the big shape: which patch cells survive
    keep = np.zeros((ph, pw), dtype=bool)
    for i in range(ph):
        for j in range(pw):
            y = sr + i * hscf
            x = sc + j * wscf
            if 0 <= y < hi and 0 <= x < wi and I[y, x] == fgc:
                keep[i, j] = True

    # --- erase the patch cells the stencil does not cover (region by region)
    off = {(pr + i, pc + j) for i in range(ph) for j in range(pw) if not keep[i, j]}
    rem = set(off)
    while rem:
        seed = min(rem)
        rem.discard(seed)
        stack = [seed]
        group = []
        while stack:
            y, x = stack.pop()
            group.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (ny, nx) in rem:
                    rem.discard((ny, nx))
                    stack.append((ny, nx))
        ops.append(int(bgc))
        sels.append(sel_of(sorted(group)))

    # --- crop the canvas down to the patch (bbox format is exact here: a full rectangle)
    ops.append(33); sels.append([pr, pc, ph - 1, pw - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 6ecd11f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6ecd11f4"
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
                                f"for task 6ecd11f4"
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
                    f"Failed to build a complete episode for task 6ecd11f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6ecd11f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
