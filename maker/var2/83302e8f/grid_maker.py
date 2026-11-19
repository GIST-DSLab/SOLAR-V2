"""
ARC Task: 83302e8f (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    # the task paints its answer in 3 (intact cell) and 4 (everything else),
    # so the generator's palette never contains 3 or 4.  0 IS an ordinary
    # colour here: it may be the canvas colour or the lattice-line colour.
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, linc=None, **kwargs) -> dict:
    # lattice of cells of size h x w, separated by 1-thick lines of `linc`;
    # a few line cells are punched back to `bgc`, merging neighbouring cells.
    hub = 5
    while hub > 2 and max_h // (hub + 1) < 3:
        hub -= 1
    wub = 5
    while wub > 2 and max_w // (wub + 1) < 3:
        wub -= 1
    h = unifint(diff_lb, diff_ub, (2, max(2, hub)))
    w = unifint(diff_lb, diff_ub, (2, max(2, wub)))
    nh_ub = max(2, max_h // (h + 1))
    nw_ub = max(2, max_w // (w + 1))
    nh = unifint(diff_lb, diff_ub, (min(3, nh_ub), nh_ub))
    nw = unifint(diff_lb, diff_ub, (min(3, nw_ub), nw_ub))
    while nh > 2 and (h + 1) * nh - 1 > max_h:
        nh -= 1
    while h > 2 and (h + 1) * nh - 1 > max_h:
        h -= 1
    while nw > 2 and (w + 1) * nw - 1 > max_w:
        nw -= 1
    while w > 2 and (w + 1) * nw - 1 > max_w:
        w -= 1

    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    for iloc in range(h, fullh, h + 1):
        gi = fill(gi, linc, hfrontier((iloc, 0)))
    for jloc in range(w, fullw, w + 1):
        gi = fill(gi, linc, vfrontier((0, jloc)))
    ofc = ofcolor(gi, linc)
    dots = sfilter(ofc, lambda ij: dneighbors(ij).issubset(ofc))
    tmp = fill(gi, bgc, dots)
    lns = apply(toindices, colorfilter(objects(tmp, True, False, False), linc))
    dts = apply(initset, dots)
    cands = lns | dts
    nbreaks = unifint(diff_lb, diff_ub, (0, len(cands) // 2))
    breaklocs = set()
    breakobjs = sample(totuple(cands), nbreaks)
    for breakobj in breakobjs:
        loc = choice(totuple(breakobj))
        breaklocs.add(loc)
    gi = fill(gi, bgc, breaklocs)
    objs = objects(gi, True, False, False)
    objs = colorfilter(objs, bgc)
    objs = sfilter(objs, lambda o: len(o) == h * w)
    res = toindices(merge(objs)) if len(objs) > 0 else frozenset()
    go = fill(gi, 3, res)
    go = replace(go, bgc, 4)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # (0,0) is always inside a lattice cell, never on a line -> canvas colour.
    bgc = int(I[0, 0])

    # ---- the pockets of the lattice: 4-connected regions of canvas colour ----
    seen = [[False] * wi for _ in range(hi)]
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r][c] or int(I[r, c]) != bgc:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny][nx] \
                            and int(I[ny, nx]) == bgc:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            comps.append(cells)

    # an INTACT cell is a solid rectangle of more than one square; anything
    # else (two cells merged through a broken line, or a lone hole at a line
    # crossing) is a spoiled pocket.  Read off I's geometry alone.
    full, broken = [], []
    for cells in comps:
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, c0 = min(rs), min(cs)
        bh, bw = max(rs) - r0 + 1, max(cs) - c0 + 1
        if len(cells) >= 2 and len(cells) == bh * bw:
            full.append((cells, (r0, c0, bh, bw)))
        else:
            broken.append(cells)

    all_bg = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) == bgc]
    nf, nb = len(full), len(broken)

    # ---- how this particular board is worked, read off the board itself ----
    # start the sweep on the side where the first spoiled pocket sits, and run
    # along the grid's long axis, the way you would scan the picture by eye.
    if broken:
        ar, ac = min(min(cells) for cells in broken)
    else:
        ar, ac = 0, 0
    row_dir = 1 if 2 * ar <= hi else -1
    col_dir = 1 if 2 * ac <= wi else -1
    major_row = wi >= hi

    def cell_key(p):
        kr = p[0] if row_dir > 0 else -p[0]
        kc = p[1] if col_dir > 0 else -p[1]
        return (kr, kc) if major_row else (kc, kr)

    def comp_key(cells):
        return min(cell_key(p) for p in cells)

    ops, sels = [], []

    # everything the route lays down is colour 3 or colour 4, never 0, so no
    # Paste/Copy in this route can silently drop a cell; the canvas colour
    # itself (which may be 0) is only ever read, never carried.
    def stamp_copy(box):
        r0, c0, bh, bw = box
        ops.append(29)
        sels.append(sel_of([(r, c) for r in range(r0, r0 + bh)
                            for c in range(c0, c0 + bw)]))

    if nb == 0:
        # nothing is spoiled: the whole field is intact cells, one stroke of 3.
        ops.append(3)
        sels.append(sel_of(all_bg))
    elif nf == 0:
        # nothing survived intact: the whole field is 4.
        ops.append(4)
        sels.append(sel_of(all_bg))
    else:
        dims = set(box[2:] for _, box in full)
        # repeated identical cell -> paint the first, then stamp copies of it
        stamp = nf >= 3 and len(dims) == 1 and nb % 2 == 0
        # a pocket that spans more than one square is naturally flooded;
        # single punched holes are just dabbed.
        flood = max(len(cells) for cells in broken) > 1

        if nb * 2 >= nf:
            route = 'base4'          # spoiled pockets everywhere: lay 4 down first
        elif (nf + nb) % 2 == 0:
            route = 'base3'          # mostly intact: lay 3 down, flood the spoiled ones back
        else:
            route = 'regions'        # pocket by pocket along the sweep

        def paint_full(cells, box, state):
            r0, c0, bh, bw = box
            if stamp and state[0]:
                ops.append(30)
                sels.append(sel_of([(r0, c0)]))   # paste the stamped cell here
            else:
                ops.append(3)
                sels.append(sel_of(cells))
                if stamp and not state[0]:
                    stamp_copy(box)
                state[0] = True

        def clear_broken(cells):
            if flood:
                ops.append(14)                         # FloodFill4 from one square
                sels.append(sel_of([min(cells, key=cell_key)]))
            else:
                ops.append(4)
                sels.append(sel_of(cells))

        if route == 'base4':
            ops.append(4)                              # the whole field as 4 ...
            sels.append(sel_of(all_bg))
            state = [False]
            for cells, box in sorted(full, key=lambda t: comp_key(t[0])):
                paint_full(cells, box, state)          # ... then the intact cells on top
        elif route == 'base3':
            ops.append(3)                              # the whole field as 3 ...
            sels.append(sel_of(all_bg))
            for cells in sorted(broken, key=comp_key):
                clear_broken(cells)                    # ... then the spoiled pockets back to 4
        else:
            items = [(cells, box) for cells, box in full] + [(cells, None) for cells in broken]
            items.sort(key=lambda t: comp_key(t[0]))
            state = [False]
            for cells, box in items:
                if box is None:
                    clear_broken(cells)
                else:
                    paint_full(cells, box, state)

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
                        f"num_examples+1 ({num_examples + 1}) for task 83302e8f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 83302e8f"
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
                                f"for task 83302e8f"
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
                    f"Failed to build a complete episode for task 83302e8f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"83302e8f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
