"""
ARC Task: 11852cab (RE-ARC) — LLM-generated grid_maker
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
    # The rule only depends on which cells are marked, not on their colours;
    # the background is the one colour role that must stay fixed.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = backdrop(frozenset(r1))
    hub = max(7, min(30, int(max_h)))
    wub = max(7, min(30, int(max_w)))
    res = None
    for _attempt in range(40):
        h = unifint(diff_lb, diff_ub, (7, hub))
        w = unifint(diff_lb, diff_ub, (7, wub))
        remcols = remove(bgc, cols)
        numc = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, numc)
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        inds = shift(asindices(trim(gi)), UNITY)
        nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 36)))
        succ = 0
        tr = 0
        maxtr = 10 * nobjs
        while succ < nobjs and tr < maxtr:
            tr += 1
            cands = sfilter(inds, lambda ij: ij[0] <= h - 5 and ij[0] <= w - 5)
            if len(cands) == 0:
                break
            loc = choice(totuple(cands))
            plcd = shift(bx, loc)
            if plcd.issubset(inds):
                inds = (inds - plcd) - outbox(plcd)
                ringcols = [choice(ccols) for k in range(4)]
                plcdrings = [shift(r, loc) for r in rings]
                gi = fill(gi, ringcols[0], plcdrings[0])
                go = fill(go, ringcols[0], plcdrings[0])
                idx = randint(1, 3)
                gi = fill(gi, ringcols[idx], plcdrings[idx])
                go = fill(go, ringcols[idx], plcdrings[idx])
                remrings = plcdrings[1:idx] + plcdrings[idx + 1:]
                remringcols = ringcols[1:idx] + ringcols[idx + 1:]
                numrs = unifint(diff_lb, diff_ub, (1, 2))
                locs = sample((0, 1), numrs)
                remrings = [rr for j, rr in enumerate(remrings) if j in locs]
                remringcols = [rr for j, rr in enumerate(remringcols) if j in locs]
                tofillgi = merge(frozenset(
                    recolor(col, frozenset(sample(totuple(remring), 4 - unifint(diff_lb, diff_ub, (0, 3)))))
                    for remring, col in zip(remrings, remringcols)
                ))
                tofillgo = merge(frozenset(
                    recolor(col, remring) for remring, col in zip(remrings, remringcols)
                ))
                if min(shape(tofillgi)) == 5:
                    succ += 1
                    gi = paint(gi, tofillgi)
                    go = paint(go, tofillgo)
        res = {'input': gi, 'output': go}
        if gi != go:
            return res
    return res


_RINGS = (((0, 0), (0, 4), (4, 0), (4, 4)),
          ((0, 2), (2, 0), (2, 4), (4, 2)),
          ((1, 1), (1, 3), (3, 1), (3, 3)))


def _find_blocks(I, bgc):
    """5x5 ring placements, located from I alone: a 7x7 window whose outer
    frame is background, whose odd-parity cells are background, whose centre
    is marked, and whose rings are each single-coloured with at least one ring
    complete. (Abandoned placements of the generator leave such patterns too;
    they are already symmetric and get no ops.)"""
    h, w = I.shape
    found = []
    for r0 in range(1, h - 5):
        for c0 in range(1, w - 5):
            win = I[r0 - 1:r0 + 6, c0 - 1:c0 + 6]
            core = win[1:6, 1:6]
            frame = np.ones((7, 7), bool); frame[1:6, 1:6] = False
            if (win[frame] != bgc).any():
                continue
            ok = True
            for i in range(5):
                for j in range(5):
                    if (i + j) % 2 == 1 and core[i, j] != bgc:
                        ok = False
            if not ok:
                continue
            if core[2, 2] == bgc:
                continue
            full = False
            for ring in _RINGS:
                vals = [int(core[i, j]) for (i, j) in ring if core[i, j] != bgc]
                if len(set(vals)) > 1:
                    ok = False                  # a ring is drawn in one colour
                if len(vals) == 4:
                    full = True                 # one ring is always complete
            if not (ok and full):
                continue
            found.append((r0, c0))

    # Every mark in the grid belongs to exactly one placement, and placements
    # never touch. A window can pass the test above while straddling two real
    # placements, so keep the set of candidates that tiles all marks exactly.
    marks = sorted(map(tuple, np.argwhere(I != bgc).tolist()))

    def cells(rc):
        return {(rc[0] + i, rc[1] + j) for i in range(5) for j in range(5)}

    def halo(rc):
        return {(rc[0] + i, rc[1] + j) for i in range(-1, 6) for j in range(-1, 6)}

    def solve(covered, chosen):
        rest = [m for m in marks if m not in covered]
        if not rest:
            return chosen
        first = rest[0]
        for rc in found:
            if first in cells(rc) and not (halo(rc) & covered):
                got = solve(covered | cells(rc), chosen + [rc])
                if got is not None:
                    return got
        return None

    best = solve(set(), [])
    return sorted(best) if best is not None else found


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    G = I.copy()
    ops, sels = [], []

    # Each 5x5 block is a pattern with four-fold rotational symmetry whose
    # rings are only partly drawn. Turn the whole block a quarter turn CW,
    # then redraw on top every mark it held just before the turn. After at
    # most three turns each mark has been carried to all four of its rotated
    # positions, so every ring is closed. Nothing here looks at O.
    for (r0, c0) in _find_blocks(I, bgc):
        rect = [r0, c0, 4, 4]          # the whole 5x5 block, background included
        for _ in range(3):
            before = G[r0:r0 + 5, c0:c0 + 5].copy()
            turned = np.rot90(before, k=3)          # op25 = CW
            if np.array_equal(turned, before):
                break                               # already symmetric: done
            ops.append(25); sels.append(rect)
            G[r0:r0 + 5, c0:c0 + 5] = turned
            bycol = {}
            for i in range(5):
                for j in range(5):
                    v = int(before[i, j])
                    if v != bgc and G[r0 + i, c0 + j] != v:
                        bycol.setdefault(v, []).append((r0 + i, c0 + j))
            for v in sorted(bycol):
                ops.append(v); sels.append(sel_of(bycol[v]))
                for (r, c) in bycol[v]:
                    G[r, c] = v

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 11852cab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 11852cab"
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
                                f"for task 11852cab"
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
                    f"Failed to build a complete episode for task 11852cab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"11852cab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
