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


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
# The only randomly sampled colour whose role matters is the background.
# The anchor colour is HARDCODED to 1 by the generator (never sampled), and the
# other object colours are pure decoration: the rule ("align every object's
# bottom row with the bottom row of the 1-object") does not depend on them.
# There are no discrete structural variants -> no instance_plan needed.
def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ----------------------------------------------------------------------------
# 2. generate  (RE-ARC generator, 30 -> max_h/max_w, bgc injected)
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = difference(interval(0, 10, 1), (1,))
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    oh = unifint(diff_lb, diff_ub, (1, max(1, h // 2)))
    ow = unifint(diff_lb, diff_ub, (1, max(1, w // 3)))
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
    numo = unifint(diff_lb, diff_ub, (2, min(8, max(2, w // ow)))) - 1
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


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    """
    Rule: every coloured object slides VERTICALLY (its columns never change) until
    its lowermost row coincides with the lowermost row of the colour-1 object.
    The colour-1 object is the anchor and stays where it is.

    Objects live in mutually disjoint column bands (the generator guarantees it),
    so a vertical slide never crosses another object -> plain Move chains.

    Special case: an object whose colour is 0.  ARCLE's object ops (Move/Rotate/
    Flip) grab only NON-ZERO cells of a selection, so a 0-coloured object cannot
    be grabbed at all - a Move would be a silent no-op.  For that object only, the
    translation is expressed by painting: Color0 on the destination cells, then
    bgc on the cells the object no longer covers.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # Background: the canvas colour the generator fills before placing objects.
    # Objects cover at most half the grid, and each single object colour covers at
    # most h*w/6 cells, so the majority colour is always the background.
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # --- connected components (same colour, 8-connectivity, background excluded)
    comps = []
    seen = set()
    for r in range(hi):
        for c in range(wi):
            if int(I[r, c]) == bgc or (r, c) in seen:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and (nr, nc) not in seen \
                                and int(I[nr, nc]) == col:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
            comps.append((col, sorted(cells)))

    anchor_rows = [r for col, cells in comps if col == 1 for (r, _c) in cells]
    if not anchor_rows:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    anchor_bottom = max(anchor_rows)

    # process objects left to right (each occupies its own column band)
    for col, cells in sorted(comps, key=lambda t: min(c for _r, c in t[1])):
        bottom = max(r for r, _c in cells)
        dr = anchor_bottom - bottom
        if dr == 0:
            continue  # anchor itself (and any object already aligned)

        dst = [(r + dr, c) for r, c in cells]

        if col == 0:
            # 0-coloured object: ungrabbable by ARCLE object ops -> paint it over.
            ops.append(0)
            sels.append(sel_of(dst))
            hole = sorted(set(cells) - set(dst))
            if hole:
                ops.append(bgc)
                sels.append(sel_of(hole))
        else:
            move_op = 21 if dr > 0 else 20          # 21 = MoveD, 20 = MoveU
            step = 1 if dr > 0 else -1
            cur = list(cells)
            # first Move GRABS the object (non-empty selection = its true cells)
            ops.append(move_op)
            sels.append(sel_of(cur))
            cur = [(r + step, c) for r, c in cur]
            # every further step keeps the SAME object grabbed (empty selection),
            # so ARCLE restores every cell the object glides over
            for _ in range(abs(dr) - 1):
                ops.append(move_op)
                sels.append(sel_of([]))
                cur = [(r + step, c) for r, c in cur]
            # only the original footprint the object no longer covers reads 0
            hole = sorted(set(cells) - set(cur))
            if bgc != 0 and hole:
                ops.append(bgc)
                sels.append(sel_of(hole))

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
