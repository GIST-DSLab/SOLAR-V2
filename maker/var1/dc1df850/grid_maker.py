"""
ARC Task: dc1df850 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # The generator samples exactly one color role at random: the background.
    # 1 (the spray color) and 2 (the seed color) are hardcoded and therefore
    # excluded here; the per-cell "noise" colors are pure decoration - the rule
    # never depends on which of them appears where - so they stay random.
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (0, (h * w) // 2 - 1))
    nreddev = unifint(diff_lb, diff_ub, (0, nc // 2))
    nred = choice((nreddev, nc - nreddev))
    nred = min(max(0, nred), nc)
    inds = totuple(asindices(c))
    occ = sample(inds, nc)
    reds = sample(occ, nred)
    others = difference(occ, reds)
    c = fill(c, 2, reds)
    obj = frozenset({(choice(remcols), ij) for ij in others})
    c = paint(c, obj)
    gi = tuple(r for r in c)
    go = underfill(c, 1, mapply(neighbors, frozenset(reds)))
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. operations
# ----------------------------------------------------------------------------
def derive_operations(I, O, examples=None):
    """Rule (read off I and the demonstrations): every cell of color 2 sprays its
    eight neighbours with color 1, but only onto cells that are still background.
    Non-background neighbours (the scattered noise pixels, other 2s) are left alone.

    The rule is fixed; the ROUTE through it is not. A person picks an order and a
    grain - clump by clump from the top-left, clump by clump from the right, the
    biggest blob first, or a straight sweep row by row - and different puzzles
    invite different habits. The route below is chosen from the input itself
    (the number of seeds plus the position of the first one plus the grid size),
    so one grid always yields one route, and the variety comes from the grids.
    Nothing about O is consulted anywhere.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Background: the generator paints the canvas bgc and then occupies strictly
    # fewer than half the cells, so bgc is the majority color of I. bgc is never 1 or 2.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 2]
    if not seeds:                      # no seeds -> nothing sprays, grid is already the answer
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    seedset = set(seeds)

    def halo_of(cells):
        """Background cells touching any of `cells` - what those seeds spray."""
        out = set()
        for (r, c) in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < hi and 0 <= cc < wi and I[rr, cc] == bgc:
                        out.add((rr, cc))
        return out

    # The seeds clump into 8-connected blobs; a blob and its halo is the natural
    # unit a person deals with in one stroke.
    seen, comps = set(), []
    for s in seeds:
        if s in seen:
            continue
        comp, dq = [], deque([s])
        seen.add(s)
        while dq:
            r, c = dq.popleft()
            comp.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in seedset and nb not in seen:
                        seen.add(nb)
                        dq.append(nb)
        comp.sort()
        comps.append(comp)

    # ---- pick the route from the input -------------------------------------
    anchor_r, anchor_c = seeds[0]               # first seed in reading order
    route = (len(seeds) + anchor_r + anchor_c + hi + wi) % 10

    groups = []
    if route in (0, 1, 2, 3, 4, 5):
        # Blob-at-a-time. Only the order of the blobs differs.
        if route == 0:                          # reading order, top-left blob first
            order = sorted(comps, key=lambda cp: cp[0])
        elif route == 1:                        # reading order reversed, bottom-right first
            order = sorted(comps, key=lambda cp: cp[0], reverse=True)
        elif route == 2:                        # column-major, sweeping left to right
            order = sorted(comps, key=lambda cp: (min(c for _, c in cp),
                                                  min(r for r, _ in cp)))
        elif route == 3:                        # column-major, sweeping right to left
            order = sorted(comps, key=lambda cp: (-max(c for _, c in cp),
                                                  min(r for r, _ in cp)))
        elif route == 4:                        # biggest clump first, then reading order
            order = sorted(comps, key=lambda cp: (-len(cp), cp[0]))
        else:                                   # from the middle of the grid outwards
            cr, cc = (hi - 1) / 2.0, (wi - 1) / 2.0
            def dist(cp):
                mr = sum(r for r, _ in cp) / float(len(cp))
                mc = sum(c for _, c in cp) / float(len(cp))
                return (abs(mr - cr) + abs(mc - cc), cp[0])
            order = sorted(comps, key=dist)
        groups = [halo_of(cp) for cp in order]
    else:
        # Straight sweep: take the sprayed cells one line at a time.
        all_halo = halo_of(seeds)
        if route in (6, 7):                     # row by row
            lines = sorted({r for r, _ in all_halo}, reverse=(route == 7))
            groups = [{(r, c) for (r, c) in all_halo if r == line} for line in lines]
        else:                                   # column by column
            lines = sorted({c for _, c in all_halo}, reverse=(route == 9))
            groups = [{(r, c) for (r, c) in all_halo if c == line} for line in lines]

    # ---- spray ---------------------------------------------------------------
    # Each stroke paints exactly the cells of this group that are STILL background
    # in the grid as it stands (cells an earlier stroke already turned to 1 are no
    # longer background, so they are not repainted). Every op changes something.
    painted = set()
    for grp in groups:
        cells = sorted(grp - painted)
        if not cells:
            continue
        ops.append(1)                           # Color1
        sels.append(sel_of(cells))
        painted |= set(cells)

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
                        f"num_examples+1 ({num_examples + 1}) for task dc1df850"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task dc1df850"
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
                                f"for task dc1df850"
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
                    f"Failed to build a complete episode for task dc1df850 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"dc1df850-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
