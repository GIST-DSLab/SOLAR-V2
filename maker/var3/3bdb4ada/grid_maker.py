"""
ARC Task: 3bdb4ada (RE-ARC) — LLM-generated grid_maker
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
    # The rule paints holes in the CANVAS colour, so bgc must be shared across the
    # whole episode.  The rectangles' own colours are irrelevant to the rule
    # (only their shape/parity matters), so they stay freely sampled.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    hlb = min(10, max_h)
    wlb = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (hlb, max_h))
    w = unifint(diff_lb, diff_ub, (wlb, max_w))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (2, 8))
    indss = asindices(gi)
    maxtrials = 6 * num
    tr = 0
    succ = 0
    # guarantee both orientations show up in EVERY instance, so a solver never has
    # to generalise "odd offsets from the corner" from one orientation alone
    forced = (True, False)
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        if succ < len(forced):
            horiz = forced[succ]
        else:
            horiz = choice((True, False))
        if horiz:
            oh = 3
            ow = unifint(diff_lb, diff_ub, (1, max(1, w // 2 - 1))) * 2 + 1
        else:
            ow = 3
            oh = unifint(diff_lb, diff_ub, (1, max(1, h // 2 - 1))) * 2 + 1
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            go = fill(go, col, bd)
            if oh == 3:
                ln = {(loci + 1, j) for j in range(locj + 1, locj + ow, 2)}
            else:
                ln = {(j, locj + 1) for j in range(loci + 1, loci + oh, 2)}
            go = fill(go, bgc, ln)
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- background: the one colour whose cells are NOT a solid rectangle ---
    # (every placed object is a filled rectangle of a unique colour; only the
    #  canvas colour is left over as a non-rectangular remainder)
    non_rect = []
    for col in np.unique(I):
        cells = np.argwhere(I == col)
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0)
        if len(cells) != (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1):
            non_rect.append((len(cells), int(col)))
    if len(non_rect) == 1:
        bgc = non_rect[0][1]
    elif len(non_rect) > 1:
        bgc = max(non_rect)[1]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- each non-background colour is one solid rectangle in I ---
    objs = []
    for col in np.unique(I):
        col = int(col)
        if col == bgc:
            continue
        cells = np.argwhere(I == col)
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0)
        objs.append((int(r0), int(c0), int(r1), int(c1), col))
    objs.sort()                      # canonical reading order (used for the signature)

    # --- the route choices, read off the INPUT alone -------------------------
    # Two people punching the same holes still differ in: which bar they start
    # with, which way they sweep along a bar, and whether they clear the whole
    # middle stripe first and redraw the teeth on top of it, or dot the holes
    # straight onto the bar.  Those choices are fixed here by quantities that
    # are visible in I (grid size, canvas colour, how many bars, their colours,
    # where the first one sits) -- never by a random draw, never by O.
    ncols_sum = sum(o[4] for o in objs)
    first_r = objs[0][0] if objs else 0
    first_c = objs[0][1] if objs else 0
    order_mode = (int(bgc) + len(objs) + first_r) % 4
    style = (hi + wi + ncols_sum) % 3
    sweep_rev = ((int(bgc) + wi + first_c) % 2) == 1

    if order_mode == 0:                                   # top-left downwards
        order = sorted(objs)
    elif order_mode == 1:                                 # bottom-right upwards
        order = sorted(objs, reverse=True)
    elif order_mode == 2:                                 # column by column
        order = sorted(objs, key=lambda o: (o[1], o[0]))
    else:                                                 # biggest bar first
        order = sorted(objs, key=lambda o: (-((o[2] - o[0] + 1) * (o[3] - o[1] + 1)),
                                            o[0], o[1]))

    ops, sels = [], []

    # Rule: inside each rectangle, punch background-coloured holes at every cell
    # whose row-offset AND column-offset from the rectangle's top-left are odd.
    # (A 3-tall bar -> dashes along its middle row; a 3-wide bar -> dashes down
    #  its middle column.)
    for r0, c0, r1, c1, col in order:
        holes = [(r, c)
                 for r in range(r0 + 1, r1 + 1, 2)
                 for c in range(c0 + 1, c1 + 1, 2)]
        if not holes:
            continue

        if len(holes) == 1 or style == 0:
            # punch the bar's holes in one go
            ops.append(int(bgc))
            sels.append(sel_of(holes))

        elif style == 1:
            # walk the bar and dot the holes one at a time, sweeping either
            # forwards or backwards along its long axis
            seq = sorted(holes, reverse=sweep_rev)
            for cell in seq:
                ops.append(int(bgc))
                sels.append(sel_of([cell]))

        else:
            # clear the bar's whole middle stripe to the canvas colour, then
            # draw the remaining teeth back on top of it
            horiz = (r1 - r0) == 2
            if horiz:
                stripe = [(r0 + 1, c) for c in range(c0, c1 + 1)]
                teeth = [(r0 + 1, c) for c in range(c0, c1 + 1, 2)]
            else:
                stripe = [(r, c0 + 1) for r in range(r0, r1 + 1)]
                teeth = [(r, c0 + 1) for r in range(r0, r1 + 1, 2)]
            ops.append(int(bgc))
            sels.append(sel_of(stripe))
            if teeth:
                ops.append(int(col))
                sels.append(sel_of(teeth))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # full-grid rectangle: Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 3bdb4ada"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3bdb4ada"
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
                                f"for task 3bdb4ada"
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
                    f"Failed to build a complete episode for task 3bdb4ada "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3bdb4ada-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
