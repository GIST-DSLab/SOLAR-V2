"""
ARC Task: 63613498 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    """Episode-level colours.  The rule is structural (recolour the twin of the
    shape locked inside the L-shaped separator, using the separator's colour), so
    only the background and the separator colour need to be pinned down; the
    object palette may vary freely per instance.

    The generator also applies one of four whole-grid rotations, which decides
    WHICH CORNER the separator sits in.  That is a discrete structural variant,
    so it is planned per instance up front."""
    cols = list(range(10))
    bgc = random.choice(cols)
    sepc = random.choice([c for c in cols if c != bgc])

    ROTS = ["identity", "rot90", "rot180", "rot270"]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sepc": sepc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, sepc=None, rot=None) -> dict:
    cols = interval(0, 10, 1)

    if bgc is None:
        bgc = choice(cols)
    if sepc is None:
        sepc = choice(remove(bgc, cols))
    if rot is None:
        rot = choice(("identity", "rot90", "rot180", "rot270"))

    # a 90/270 rotation swaps the axes, so both sides must fit both caps there
    if rot in ("rot90", "rot270"):
        hub = wub = min(max_h, max_w)
    else:
        hub, wub = max_h, max_w
    hub = max(10, min(30, hub))
    wub = max(10, min(30, wub))

    h = unifint(diff_lb, diff_ub, (10, hub))
    w = unifint(diff_lb, diff_ub, (10, wub))

    remcols = remove(bgc, remove(sepc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    objh = unifint(diff_lb, diff_ub, (1, h // 3))
    objw = unifint(diff_lb, diff_ub, (1, w // 3))
    bounds = asindices(canvas(-1, (objh, objw)))
    sp = choice(totuple(bounds))
    obj = {sp}
    ncells = unifint(diff_lb, diff_ub, (1, (objh * objw)))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    gi = canvas(bgc, (h, w))
    objc = choice(ccols)
    gi = fill(gi, objc, obj)
    sep = connect((objh + 1, 0), (objh + 1, objw + 1)) | connect((0, objw + 1), (objh + 1, objw + 1))
    gi = fill(gi, sepc, sep)
    inds = asindices(gi)
    inds -= backdrop(sep)
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    baseobj = normalize(obj)
    obj = normalize(obj)
    go = tuple(e for e in gi)
    while (succ < nobjs and tr < maxtr) or succ == 0:
        tr += 1
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            col = choice(ccols)
            gi = fill(gi, col, plcd)
            go = fill(go, sepc if succ == 0 else col, plcd)
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
        objh = randint(1, h // 3)
        objw = randint(2 if objh == 1 else 1, w // 3)
        if choice((True, False)):
            objh, objw = objw, objh
        bounds = asindices(canvas(-1, (objh, objw)))
        sp = choice(totuple(bounds))
        obj = {sp}
        ncells = unifint(diff_lb, diff_ub, (1, (objh * objw)))
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        obj = set(obj)
        if obj == baseobj:
            if len(obj) < objh * objw:
                obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
            else:
                obj = remove(choice(totuple(corners(obj))), obj)
        obj = normalize(obj)

    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (unchanged): one L-shaped separator line touches the border and pens a
    single shape into its corner.  Somewhere else on the grid the very same shape
    sits in another colour.  That twin gets repainted in the separator's colour.
    Nothing else on the grid changes.

    Only one region is ever repainted, so there is no "which region first" choice
    to make.  The genuine choice a solver still faces is HOW to lay that colour
    down, and that is read off the input:
      * two or three cells, or a single row/column of them -> one dab / one stroke;
      * a solid block -> stroke it row by row (or column by column along its long
        axis), starting on the side that faces the separator and sweeping away;
      * a ragged blob -> one flood fill, seeded on the cell nearest the separator.
    Every emitted op paints cells that still hold the old colour, so none of them
    is removable and none repaints anything twice."""
    import numpy as np
    from collections import Counter, deque
    from maker.sel_helpers import sel_of

    A = np.asarray(I, dtype=int)
    B = np.asarray(O, dtype=int)
    h, w = A.shape
    ho, wo = B.shape

    bgc = Counter(A.flatten().tolist()).most_common(1)[0][0]

    # --- connected same-colour components of non-background cells (4-connectivity) ---
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if A[r, c] == bgc or seen[r, c]:
                continue
            col = int(A[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and A[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append((col, frozenset(cells)))

    def norm(cells):
        r0 = min(p[0] for p in cells)
        c0 = min(p[1] for p in cells)
        return frozenset((p[0] - r0, p[1] - c0) for p in cells)

    def bbox(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return min(rs), min(cs), max(rs), max(cs)

    # --- separator candidates: L/line shaped (size == bboxH + bboxW - 1),
    #     touching the grid border, and enclosing exactly one foreground colour
    #     plus background inside its bbox (delta has 2 colours) ---
    cands = []
    for col, cells in comps:
        r0, c0, r1, c1 = bbox(cells)
        ch, cw = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != ch + cw - 1:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        delta = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                 if (r, c) not in cells]
        if not delta:
            continue
        if len({int(A[r, c]) for r, c in delta}) != 2:
            continue
        cands.append((col, cells, delta))

    cands.sort(key=lambda t: -len(t[1]))

    ops, sels = [], []

    target = None
    sep_cells = None
    newc = None
    for scol, scells, delta in cands:
        # template = the non-background shape penned inside the separator's corner
        tmpl = frozenset((r, c) for r, c in delta if A[r, c] != bgc)
        if not tmpl:
            continue
        key = norm(tmpl)
        # the twin: the object elsewhere on the grid with the same normalized shape
        twins = [cs for cl, cs in comps if cs != tmpl and norm(cs) == key]
        if not twins:
            continue
        twins.sort(key=lambda cs: sorted(cs)[0])
        target = sorted((int(p[0]), int(p[1])) for p in twins[0])
        sep_cells = sorted((int(p[0]), int(p[1])) for p in scells)
        newc = int(scol)
        break

    if target is not None and int(A[target[0][0], target[0][1]]) != newc:
        r0, c0, r1, c1 = bbox(target)
        th, tw = r1 - r0 + 1, c1 - c0 + 1
        n = len(target)

        # where the separator sits relative to the twin: the side you come from
        scr = sum(p[0] for p in sep_cells) / float(len(sep_cells))
        scc = sum(p[1] for p in sep_cells) / float(len(sep_cells))
        tcr, tcc = (r0 + r1) / 2.0, (c0 + c1) / 2.0

        solid = (n == th * tw)

        if n <= 3 or th == 1 or tw == 1:
            # a couple of cells, or one straight run: a single stroke covers it
            ops.append(newc)
            sels.append(sel_of(target))
        elif solid:
            # a filled block: stroke it along its long axis, starting on the side
            # facing the separator and sweeping away from it
            if tw >= th:
                lines = [[(r, c) for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
                far_side_first = scr > tcr           # separator below -> start at the bottom
            else:
                lines = [[(r, c) for r in range(r0, r1 + 1)] for c in range(c0, c1 + 1)]
                far_side_first = scc > tcc           # separator to the right -> start at the right
            if far_side_first:
                lines = lines[::-1]
            for line in lines:
                # every cell in this stroke still holds the old colour -> visible change
                ops.append(newc)
                sels.append(sel_of(line))
        else:
            # a ragged blob reads as one region: flood it in one gesture, seeded on
            # the cell of the blob nearest the separator
            seed = min(target, key=lambda p: (abs(p[0] - scr) + abs(p[1] - scc), p[0], p[1]))
            ops.append(10 + newc)
            sels.append([seed[0], seed[1], 0, 0])   # FloodFill needs exactly 1 seed cell

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
                        f"num_examples+1 ({num_examples + 1}) for task 63613498"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 63613498"
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
                                f"for task 63613498"
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
                    f"Failed to build a complete episode for task 63613498 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"63613498-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
