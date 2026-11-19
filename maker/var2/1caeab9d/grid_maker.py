"""
ARC Task: 1caeab9d (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------- 1
def sample_colors(num_examples=None) -> dict:
    """
    Colour 1 is hardcoded by the generator (it marks the anchor object), so it is
    NOT sampled here.  The only randomly sampled colour role that every instance
    of an episode must share is the canvas background.  The object colours are
    irrelevant to the rule (any non-background colour behaves identically), so
    they stay free -- including the colour 0, which the generator may hand to an
    ordinary object whenever bgc != 0.
    """
    bgc = random.choice([c for c in range(10) if c != 1])
    return {"bgc": bgc}


# ----------------------------------------------------------------------------- 2
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = difference(interval(0, 10, 1), (1,))
    if bgc is None:
        bgc = choice(cols)

    hub = max(3, min(30, int(max_h)))
    wub = max(6, min(30, int(max_w)))

    h = unifint(diff_lb, diff_ub, (3, hub))
    w = unifint(diff_lb, diff_ub, (6, wub))
    oh = unifint(diff_lb, diff_ub, (1, h // 2))
    ow = unifint(diff_lb, diff_ub, (1, w // 3))
    bb = asindices(canvas(-1, (oh, ow)))
    sp = choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for k in range(ncells):
        obj.add(choice(totuple((bb - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = randint(0, h - oh)
    numo = unifint(diff_lb, diff_ub, (2, min(8, w // ow))) - 1
    itv = interval(0, w, 1)
    locj = randint(0, w - ow)
    objp = shift(obj, (loci, locj))
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    gi = fill(c, 1, objp)
    go = fill(c, 1, objp)
    itv = difference(itv, interval(locj, locj + ow, 1))
    for k in range(numo):
        cands = sfilter(itv, lambda j: set(interval(j, j + ow, 1)).issubset(set(itv)))
        if len(cands) == 0:
            break
        locj = choice(cands)
        col = choice(remcols)
        remcols = remove(col, remcols)
        gi = fill(gi, col, shift(obj, (randint(0, h - oh), locj)))
        go = fill(go, col, shift(obj, (loci, locj)))
        itv = difference(itv, interval(locj, locj + ow, 1))
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------- 3
def _majority_colour(grid):
    return int(Counter(np.asarray(grid, dtype=int).flatten().tolist()).most_common(1)[0][0])


def derive_operations(I, O, examples=None):
    """
    Rule (read off the demonstrations): the colour-1 shape is the anchor and does
    not move.  Every other shape slides VERTICALLY -- its columns never change --
    until its lowermost row sits on the anchor's lowermost row.  Shapes live in
    mutually disjoint column bands, so a slide never crosses another shape.

    Nothing below looks at O: the anchor, the target row, every offset and every
    routing decision come from I alone (plus the examples, used only to confirm
    which colour is the canvas background).  Handed a different answer grid, this
    function returns exactly the same operations.

    Route variation.  A person solving two of these would not walk the same path
    twice, so three decisions a solver genuinely faces are taken from a signature
    of the input picture itself (anchor position, grid size, object count and
    palette).  The same instance therefore always yields the same route, and the
    variety comes from the instances differing:

      * which shape to deal with first -- sweep left to right, sweep right to
        left, take the longest journey first, or take the shortest first;
      * tidy up as you go, or leave every vacated footprint and repaint them all
        at the end with one stroke;
      * for a shape that must be drawn rather than dragged: draw it at its new
        place and then rub out what is left behind, or first lay the background
        back down over its whole old footprint and draw on top of that.

    Colour 0.  ARCLE's object ops grab only NON-ZERO cells, so a 0-coloured shape
    cannot be dragged at all -- a Move would be a silent no-op.  That one shape is
    therefore put down with an explicit Color0 at the place it lands (and nothing
    is emitted when the instance contains no 0-coloured shape).  Every other
    shape really is dragged, one cell at a time.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- background: the canvas colour shared by every instance of the episode
    own = _majority_colour(I)
    votes = [own]
    if examples:
        for pair in examples:
            try:
                votes.append(_majority_colour(pair[0]))
            except Exception:
                pass
    tally = Counter(votes)
    bgc = int(max(tally.items(), key=lambda kv: (kv[1], kv[0] == own))[0])

    # --- the shapes: one per non-background colour (the generator gives every
    #     shape its own colour, and each occupies its own column band)
    cells_by_col = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v == bgc:
                continue
            cells_by_col.setdefault(v, []).append((r, c))

    if 1 not in cells_by_col:
        return [34], [[0, 0, hi - 1, wi - 1]]

    anchor = cells_by_col[1]
    anchor_bottom = max(r for r, _c in anchor)
    anchor_left = min(c for _r, c in anchor)

    # --- every shape that actually has to travel, with its vertical offset
    targets = []
    for col, cells in cells_by_col.items():
        cells = sorted(cells)
        dr = anchor_bottom - max(r for r, _c in cells)
        if dr == 0:
            continue                      # the anchor itself, or already aligned
        targets.append((col, cells, dr, min(c for _r, c in cells)))

    # --- routing signature, read entirely off the input picture
    sig = (7 * anchor_bottom + 5 * anchor_left + 3 * hi + wi
           + 11 * len(cells_by_col) + sum(cells_by_col.keys()))
    order_mode = sig % 4
    batch_cleanup = (sig // 4) % 2 == 1
    base_first = (sig // 8) % 2 == 1

    if order_mode == 0:                                   # sweep left to right
        targets.sort(key=lambda t: t[3])
    elif order_mode == 1:                                 # sweep right to left
        targets.sort(key=lambda t: -t[3])
    elif order_mode == 2:                                 # longest journey first
        targets.sort(key=lambda t: (-abs(t[2]), t[3]))
    else:                                                 # shortest journey first
        targets.sort(key=lambda t: (abs(t[2]), t[3]))

    ops, sels = [], []
    pending = []                      # footprints left at 0, repainted at the end

    for col, cells, dr, _left in targets:
        dst = [(r + dr, c) for r, c in cells]

        if col == 0:
            # ungrabbable by ARCLE's object ops -> this one is drawn, not dragged
            if base_first and not batch_cleanup:
                # lay the background back over the whole old footprint, draw on top
                ops.append(bgc)
                sels.append(sel_of(cells))
                ops.append(0)
                sels.append(sel_of(dst))
            else:
                ops.append(0)
                sels.append(sel_of(dst))
                hole = sorted(set(cells) - set(dst))
                if hole:
                    if batch_cleanup:
                        pending.extend(hole)
                    else:
                        ops.append(bgc)
                        sels.append(sel_of(hole))
            continue

        # --- drag it: one grab, then unit steps with an empty selection so ARCLE
        #     keeps the same shape held and restores everything it glides over
        move_op = 21 if dr > 0 else 20        # 21 = MoveD, 20 = MoveU
        step = 1 if dr > 0 else -1
        ops.append(move_op)
        sels.append(sel_of(cells))
        cur = [(r + step, c) for r, c in cells]
        for _ in range(abs(dr) - 1):
            ops.append(move_op)
            sels.append(sel_of([]))
            cur = [(r + step, c) for r, c in cur]

        # only the original footprint the shape no longer covers reads 0
        hole = sorted(set(cells) - set(cur))
        if bgc != 0 and hole:
            if batch_cleanup:
                pending.extend(hole)
            else:
                ops.append(bgc)
                sels.append(sel_of(hole))

    if pending:
        ops.append(bgc)
        sels.append(sel_of(sorted(set(pending))))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # full-grid rectangle: submit the canvas
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
                        f"num_examples+1 ({num_examples + 1}) for task 1caeab9d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1caeab9d"
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
                                f"for task 1caeab9d"
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
                    f"Failed to build a complete episode for task 1caeab9d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1caeab9d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
