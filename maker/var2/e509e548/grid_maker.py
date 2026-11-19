"""
ARC Task: e509e548 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover — fallback if the helper module is absent
    def sel_of(cells):
        return {"cells": [[int(r), int(c)]
                          for r, c in sorted({(int(r), int(c)) for r, c in cells})]}


# 1 / 2 / 6 are the answer's alphabet: the generator never draws with them, so a
# shape's own colour can be anything else at all — including 0, which here is an
# ordinary colour of the picture and not "nothing".
ANSWER_COLS = (1, 2, 6)


# ----------------------------------------------------------------------------
# 1. episode-level colours
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """Only the canvas colour has to be held still.

    The rule reads shape and nothing else — a bare corner becomes 1, a shape that
    lies wholly on its bounding box becomes 6, a shape with a cell inside that box
    becomes 2 — so letting the shape palette change from instance to instance is
    exactly what shows a solver that colour is not what is being read.  The route
    below never *carries* cells (no Copy, Paste or Move), it only repaints them,
    so a shape drawn in colour 0 is handled like any other: a fill seeded on its
    own value, or a Color op over its own cells.
    """
    cols = [c for c in range(10) if c not in ANSWER_COLS]
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ----------------------------------------------------------------------------
# 2. instance generator
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 6))
    getL = lambda h, w: connect((0, 0), (h - 1, 0)) | connect((0, 0), (0, w - 1))
    getU = lambda h, w: connect((0, 0), (0, w - 1)) | connect((0, 0), (randint(1, h - 1), 0)) | connect((0, w - 1), (randint(1, h - 1), w - 1))
    getH = lambda h, w: connect((0, 0), (0, w - 1)) | shift(connect((0, 0), (h - 1, 0)) | connect((h - 1, 0), (h - 1, randint(1, w - 1))), (0, randint(1, w - 2)))
    minshp_getter_pairs = ((2, 2, getL), (2, 3, getU), (3, 3, getH))
    colmapper = {getL: 1, getU: 6, getH: 2}

    hub = max(10, min(30, max_h))
    wub = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (min(10, hub), hub))
    w = unifint(diff_lb, diff_ub, (min(10, wub), wub))

    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 6))
    ccols = sample(remcols, ncols)
    nobjs = unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 10)))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)

    # Learnability: the first three shapes that land are one of each family, so
    # every instance — the examples and the test alike — shows all three answer
    # colours, and no family can turn up for the first time in the test.
    pending = list(sample(minshp_getter_pairs, 3))

    while succ < nobjs and tr < maxtr:
        tr += 1
        minh, minw, getter = pending[0] if pending else choice(minshp_getter_pairs)
        oh = randint(minh, 6)
        ow = randint(minw, 6)
        obj = getter(oh, ow)
        mfs = (identity, dmirror, cmirror, vmirror, hmirror)
        nmfs = choice((1, 2))
        for fn in sample(mfs, nmfs):
            obj = fn(obj)
            obj = normalize(obj)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            if pending and getter is pending[0][2]:
                pending.pop(0)
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            col = choice(ccols)
            gi = fill(gi, col, plcd)
            go = fill(go, colmapper[getter], plcd)

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. operations
#
# The rule — read off the worked examples, then applied to the input alone:
#   a bare corner  (its cell count is height + width - 1)        -> 1
#   a shape lying wholly on its bounding box's border            -> 6
#   a shape with at least one cell inside that border            -> 2
#
# The route that carries the rule out is settled by the picture, never by the
# answer grid:
#   * whether the 2-family is the rarest of the three decides whether every
#     shape is washed in 2 first and the other two drawn on top of that wash;
#   * meeting two shapes of one family one after the other is what starts you
#     sorting the rest into families instead of taking them as they come;
#   * how many colours the shapes are drawn in decides between dropping a fill
#     into a shape and selecting the shape's own cells;
#   * where the biggest shape sits decides which corner the sweep starts at, and
#     the grid's proportions whether it runs along rows or along columns.
# One picture therefore always yields one route, and two pictures rarely the
# same one; handed a different answer grid, this returns the same operations.
# ----------------------------------------------------------------------------
def _background(I):
    """The canvas colour the shapes are scattered on — the commonest colour.

    Checked against the generator: the shapes are at most 6x6 outlines and never
    touch each other, so they can never out-number the canvas.
    """
    vals, cnts = np.unique(I, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _shapes(I, bgc):
    """Every 4-connected block of one colour, tagged with the family it is in."""
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    out = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or int(I[r, c]) == bgc:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and int(I[ny, nx]) == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            oh, ow = r1 - r0 + 1, c1 - c0 + 1
            if len(cells) == oh + ow - 1:
                fam = "L"                                   # a single bare corner
            elif all(p[0] in (r0, r1) or p[1] in (c0, c1) for p in cells):
                fam = "U"                                   # only on the border
            else:
                fam = "H"                                   # a cell inside the box
            out.append({"cells": sorted(cells), "col": col, "fam": fam,
                        "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                        "size": len(cells)})
    return out


def _answer_colours(examples):
    """Which colour each family turns into, read where a solver reads it: off the
    worked examples.  Falls back on the three colours those examples always show
    when they are not to hand (the caller may pass I and O only)."""
    fallback = {"L": 1, "U": 6, "H": 2}
    if not examples:
        return fallback
    votes = {"L": Counter(), "U": Counter(), "H": Counter()}
    try:
        for pair in examples:
            ei = np.asarray(pair[0], dtype=int)
            eo = np.asarray(pair[1], dtype=int)
            if ei.shape != eo.shape:
                return fallback
            for o in _shapes(ei, _background(ei)):
                painted = {int(eo[r, c]) for r, c in o["cells"]}
                if len(painted) == 1:
                    votes[o["fam"]][painted.pop()] += 1
    except Exception:
        return fallback
    learned = {}
    for fam in ("L", "U", "H"):
        if not votes[fam]:
            return fallback
        learned[fam] = int(votes[fam].most_common(1)[0][0])
    if len(set(learned.values())) != 3:
        return fallback
    return learned


def _sweep_key(r, c, corner, major):
    """Reading order from `corner` (0 TL, 1 TR, 2 BL, 3 BR), by rows or columns."""
    down = r if corner in (0, 1) else -r
    across = c if corner in (0, 2) else -c
    return (down, across) if major == "row" else (across, down)


def _touching(shapes):
    """Do two different shapes sit edge to edge?  In this task they never do —
    but a fill would run out of one into the other if they did, and a 2-wash
    would weld them into one region, so check rather than trust, and fall back
    on selecting cells when the check fails."""
    owner = {}
    for i, o in enumerate(shapes):
        for p in o["cells"]:
            owner[p] = i
    for i, o in enumerate(shapes):
        for (y, x) in o["cells"]:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = owner.get((y + dy, x + dx))
                if j is not None and j != i:
                    return True
    return False


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = _background(I)
    shapes = _shapes(I, bgc)
    if not shapes:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    answer = _answer_colours(examples)
    counts = Counter(o["fam"] for o in shapes)

    # ── the route this particular picture asks for ───────────────────────────
    big = max(shapes, key=lambda o: (o["size"], -o["r0"], -o["c0"]))
    corner = (0 if (big["r0"] + big["r1"]) <= hi - 1 else 2) + \
             (1 if (big["c0"] + big["c1"]) > wi - 1 else 0)
    major = "row" if wi >= hi else "col"
    order = sorted(shapes,
                   key=lambda o: min(_sweep_key(r, c, corner, major) for r, c in o["cells"]))
    # a 2-wash is worth laying down unless the 2-family is the rarest of the three
    wash = counts["H"] > 0 and counts["H"] >= min(counts["L"], counts["U"])
    # two of a kind met one after the other is what starts you sorting by family
    by_family = len(order) > 1 and order[0]["fam"] == order[1]["fam"]
    # in four colours or more the shapes read as separate islands and each gets a
    # fill dropped into it; in three or fewer they are picked out by their cells
    use_fill = len({o["col"] for o in shapes}) >= 4 and not _touching(shapes)

    def paint(group, colour):
        """Give one family its colour — a fill dropped into each shape, or the
        whole family in one selection, or one shape's own cells at a time."""
        if not group:
            return
        if use_fill:
            for o in group:
                seed = min(o["cells"], key=lambda p: _sweep_key(p[0], p[1], corner, major))
                ops.append(10 + colour)                     # FloodFill<colour>
                sels.append(sel_of([seed]))
        elif by_family:
            ops.append(colour)                              # every shape of the family
            sels.append(sel_of([p for o in group for p in o["cells"]]))
        else:
            for o in group:
                ops.append(colour)                          # this shape's own cells
                sels.append(sel_of(o["cells"]))

    if wash:
        # most of the ink is going to be 2 anyway: lay it over every shape first
        # and draw the two other families on top of the wash
        ops.append(answer["H"])
        sels.append(sel_of([p for o in order for p in o["cells"]]))
        todo = [o for o in order if o["fam"] != "H"]
    else:
        todo = list(order)

    if by_family:
        met = [o["fam"] for o in todo]
        # carry on with the family already begun, then the rest, commonest first
        for fam in sorted(set(met), key=lambda f: (f != met[0], -counts[f], met.index(f))):
            paint([o for o in todo if o["fam"] == fam], answer[fam])
    else:
        for o in todo:
            paint([o], answer[o["fam"]])

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
                        f"num_examples+1 ({num_examples + 1}) for task e509e548"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e509e548"
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
                                f"for task e509e548"
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
                    f"Failed to build a complete episode for task e509e548 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e509e548-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
