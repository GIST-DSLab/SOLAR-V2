"""
ARC Task: 08ed6ac7 (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """
    The rule ranks bars by LENGTH, not by colour, so the bar colours may stay
    free (they are re-sampled per instance, exactly as the original generator
    does -- and they may legitimately be 0).  Only the ground colour is an
    episode-wide constant.  The generator never uses 1-4 as background.
    """
    bgc = random.choice([c for c in range(10) if c not in (1, 2, 3, 4)])
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc=None) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    if bgc is None:
        bgc = choice(difference(colopts, (1, 2, 3, 4)))
    remcols = remove(bgc, colopts)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    barrange = (4, w)
    locopts = interval(0, w, 1)
    nbars = unifint(diff_lb, diff_ub, barrange)
    barlocs = sample(locopts, nbars)
    barhopts = interval(0, h, 1)
    barhs = sample(barhopts, 4)
    barcols = [choice(remcols) for j in range(nbars)]
    barhsfx = [choice(barhs) for j in range(nbars - 4)] + list(barhs)
    shuffle(barhsfx)
    ordered = sorted(barhs)
    colord = interval(1, 5, 1)
    for col, (loci, locj) in zip(barcols, list(zip(barhsfx, barlocs))):
        bar = connect((loci, locj), (h - 1, locj))
        gi = fill(gi, col, bar)
        go = fill(go, colord[ordered.index(loci)], bar)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (read off the demonstrations, not off O): every mark is a 1-wide
    vertical bar hanging down to the bottom row.  There are exactly four
    distinct bar lengths; the longest class becomes 1, then 2, then 3, and the
    shortest class becomes 4.  Original bar colours are irrelevant.

    Route: the rule is performed class by class.  WHICH class is taken first,
    which way the columns are swept, and how a class's repeated bars are
    realised (one sweep-wide stroke / a base coat drawn over / stamping the
    first bar onto its twins / one bar at a time) are the choices a human
    actually faces, and they are read off the picture -- the number of bars,
    the ground colour, the columns of the outermost bars.  The same input
    therefore always yields the same route; O is never consulted.

    Colour 0 safety: ARCLE carries nothing through a Copy/Paste, so the only
    content ever stamped is a bar ALREADY recoloured to its rank colour
    (1-4, never 0).  Bars that are colour 0 in the input, and a ground of
    colour 0, are only ever handled by Color ops, which carry 0 fine.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # ---- 1. read the picture: bottom-anchored 1-wide bars on a plain ground --
    def bars_for(bg):
        """Bars assuming ground colour `bg`; None if the reading is inconsistent."""
        found = []
        for c in range(w):
            col = I[:, c]
            if int(col[h - 1]) == bg:
                if np.any(col != bg):
                    return None              # a mark not hanging from the bottom
                continue
            x = int(col[h - 1])
            r = h - 1
            while r - 1 >= 0 and int(col[r - 1]) == x:
                r -= 1
            if np.any(col[:r] != bg):
                return None                  # something above the bar
            found.append((c, r, h - r))      # (column, top row, length)
        return found

    bars, bgc = None, None
    for v, _ in Counter(I.flatten().tolist()).most_common():
        if v in (1, 2, 3, 4):                # the generator never grounds on these
            continue
        cand = bars_for(v)
        if cand is not None and len(cand) >= 4 and len({t[2] for t in cand}) == 4:
            bars, bgc = cand, v
            break

    if bars is None:                         # defensive, input-only fallback
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
        bars = []
        for c in range(w):
            col = I[:, c]
            if int(col[h - 1]) == bgc:
                continue
            x = int(col[h - 1])
            r = h - 1
            while r - 1 >= 0 and int(col[r - 1]) == x:
                r -= 1
            bars.append((c, r, h - r))

    if not bars:
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # ---- 2. rank the length classes: longest -> 1 ... shortest -> 4 ----------
    lengths = sorted({L for (_, _, L) in bars}, reverse=True)
    rank = {L: i + 1 for i, L in enumerate(lengths)}

    # ---- 3. the route, chosen from the picture ------------------------------
    nbars = len(bars)
    cols_sorted = sorted(b[0] for b in bars)
    left_to_right = ((nbars + bgc) % 2 == 0)
    strategy = (nbars + bgc + cols_sorted[0]) % 4
    order_mode = (bgc + cols_sorted[-1]) % 3

    groups = {
        L: sorted([b for b in bars if b[2] == L],
                  key=lambda t: t[0], reverse=not left_to_right)
        for L in lengths
    }

    if order_mode == 0:
        class_order = list(lengths)                      # longest class first
    elif order_mode == 1:
        class_order = list(reversed(lengths))            # shortest class first
    else:                                                # in sweep order
        class_order = sorted(lengths, key=lambda L: groups[L][0][0])

    G = I.copy()                                         # working grid, to drop no-ops

    def cells_of(bar):
        c, r0, L = bar
        return [(r, c) for r in range(r0, r0 + L)]

    def class_cells(L):
        out = []
        for b in groups[L]:
            out.extend(cells_of(b))
        return out

    def paint(cells, colour):
        """One Color stroke over the cells that this stroke actually changes."""
        todo = [(r, c) for (r, c) in cells if int(G[r, c]) != colour]
        if not todo:
            return
        ops.append(colour)
        sels.append(sel_of(todo))
        for (r, c) in todo:
            G[r, c] = colour

    def stamp(L, colour):
        """Recolour the first bar of the sweep, then stamp it onto its twins.

        Bars of one class are identical segments (same rows, same width), so the
        replication is performed rather than re-painted.  What travels is always
        a 1-4 coloured bar, so nothing is lost to ARCLE's zero-transparency.
        """
        group = groups[L]
        src = cells_of(group[0])
        paint(src, colour)
        twins = [b for b in group[1:]
                 if any(int(G[r, c]) != colour for (r, c) in cells_of(b))]
        if not twins:
            return
        ops.append(29)                                   # CopyO the recoloured bar
        sels.append(sel_of(src))
        for b in twins:
            c2, r2, _ = b
            ops.append(30)                               # Paste at the twin's top cell
            sels.append(sel_of([(r2, c2)]))
            for (r, c) in cells_of(b):
                G[r, c] = colour

    longest = lengths[0]
    base_coat_pays = any(int(I[r0, c]) != 1 for (c, r0, _) in groups[longest])

    if strategy == 3 and base_coat_pays:
        # Base coat: every bar goes to 1, then the shorter classes are drawn
        # over it -- laying a ground and correcting on top.
        all_cells = []
        for L in lengths:
            all_cells.extend(class_cells(L))
        paint(all_cells, 1)
        for L in [x for x in class_order if x != longest]:
            paint(class_cells(L), rank[L])
    else:
        for L in class_order:
            colour = rank[L]
            if strategy == 1:
                stamp(L, colour)
            elif strategy == 2:
                for b in groups[L]:                      # one bar at a time
                    paint(cells_of(b), colour)
            else:
                paint(class_cells(L), colour)            # the whole class at once

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])                    # full grid == exactly this rect
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
                        f"num_examples+1 ({num_examples + 1}) for task 08ed6ac7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 08ed6ac7"
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
                                f"for task 08ed6ac7"
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
                    f"Failed to build a complete episode for task 08ed6ac7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"08ed6ac7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
