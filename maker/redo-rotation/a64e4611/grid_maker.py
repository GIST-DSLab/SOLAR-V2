"""
ARC Task: a64e4611 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------------
# The task: a grid of two-colour noise (bgc / noisec) is crossed by a "corridor" that
# was wiped clean to bgc: a vertical TRUNK band (width dim, from row spi down to the
# bottom edge) and one or more horizontal ARM bands (height hh, shooting left and/or
# right from the trunk to the grid edge).  The output fills the INTERIOR of every band
# with 3: the band shrunk by one cell on each of its two long sides and by one cell at
# its closed end (a grid edge is not a closed end).
#
# The rule is direction-symmetric — the verifier says so literally with
# `power(compose(rot90, rule), FOUR)`: the very same "fill this band's interior"
# procedure is applied to the grid in all four orientations.  The trajectory below
# performs that: it fills the bands that run RIGHTWARD in the current frame, rotates
# the grid 90deg (op24), fills again, ... four times, so the grid ends up back in its
# original orientation with every band filled.
# ----------------------------------------------------------------------------------


VARIANTS = [
    {"mode": "right"},   # arms shoot right only
    {"mode": "left"},    # arms shoot left only
    {"mode": "both"},    # one arm group shooting both ways (same rows)
    {"mode": "extra"},   # two arm groups, opposite directions, independent rows
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test case was shown
    return {"bgc": bgc, "noisec": noisec, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, mode=None, **kwargs) -> dict:
    if mode is None:
        mode = random.choice([v["mode"] for v in VARIANTS])

    # square canvas: the rule is applied in four orientations, and ARCLE's Rotate is
    # position-correct only on square selections.
    hi = min(max_h, max_w, 30)
    lo = min(18, hi)
    n = unifint(diff_lb, diff_ub, (lo, hi))
    h = w = n

    lb = int(0.4 * h * w)
    ub = int(0.5 * h * w)
    nbgc = unifint(diff_lb, diff_ub, (lb, ub))

    gi = canvas(noisec, (h, w))
    inds = totuple(asindices(gi))
    bgcinds = random.sample(inds, nbgc)
    gi = fill(gi, bgc, bgcinds)

    sinds = asindices(canvas(-1, (3, 3)))
    bgcf = recolor(bgc, sinds)
    noisecf = recolor(noisec, sinds)
    addn = set()
    addb = set()
    for occ in occurrences(gi, bgcf):
        occi, occj = occ
        addn.add((random.randint(0, 2) + occi, random.randint(0, 2) + occj))
    for occ in occurrences(gi, noisecf):
        occi, occj = occ
        addb.add((random.randint(0, 2) + occi, random.randint(0, 2) + occj))
    gi = fill(gi, noisec, addn)
    gi = fill(gi, bgc, addb)

    go = tuple(e for e in gi)

    dim = random.randint(random.randint(3, 8), 8)
    locj = random.randint(3, w - dim - 4)
    spi = random.choice((0, random.randint(3, h // 2)))

    # trunk: wipe the band, then fill its interior with 3
    for j in range(locj, locj + dim):
        ln = connect((spi, j), (h - 1, j))
        gi = fill(gi, bgc, ln)
        go = fill(go, bgc, ln)
    for j in range(locj + 1, locj + dim - 1):
        ln = connect((spi + 1 if spi > 0 else spi, j), (h - 1, j))
        go = fill(go, 3, ln)

    def draw_arms(gi, go, sgns):
        startloc = random.choice((spi, random.randint(spi + 3, h - 6)))
        hh = random.randint(3, min(8, h - startloc - 3))
        for sgn in sgns:
            for ii in range(startloc, startloc + hh):
                ln = shoot((ii, locj), (0, sgn))
                gi = fill(gi, bgc, ln)
                go = fill(go, bgc, ln - ofcolor(go, 3))
        for sgn in sgns:
            for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1):
                ln = shoot((ii, locj + dim - 2 if sgn == -1 else locj + 1), (0, sgn))
                go = fill(go, 3, ln)
        return gi, go

    if mode == "both":
        gi, go = draw_arms(gi, go, (-1, 1))
    elif mode == "left":
        gi, go = draw_arms(gi, go, (-1,))
    elif mode == "right":
        gi, go = draw_arms(gi, go, (1,))
    else:  # "extra": one group each way, at independent rows
        first = random.choice((-1, 1))
        gi, go = draw_arms(gi, go, (first,))
        gi, go = draw_arms(gi, go, (-first,))

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = O.shape
    M = (O == 3)

    # ---- measure the corridor's bands (all measurements are of band geometry) -----
    # The trunk always runs down to the bottom edge; the arms never reach it, so the
    # bottom row identifies the trunk's interior columns.
    trunk_cols = [c for c in range(w) if M[h - 1, c]]
    c1, c2 = trunk_cols[0], trunk_cols[-1]
    # trunk's closed end: walk up while the whole trunk width is still filled
    a = h - 1
    while a - 1 >= 0 and M[a - 1, c1:c2 + 1].all():
        a -= 1
    # arms always run out to a side edge, so the edge columns identify their rows
    left_rows = [r for r in range(h) if M[r, 0]]
    right_rows = [r for r in range(h) if M[r, w - 1]]

    def row_groups(rows):
        grps = []
        for r in rows:
            if grps and r == grps[-1][-1] + 1:
                grps[-1].append(r)
            else:
                grps.append([r])
        return grps

    # A band is stored by the orientation in which it runs RIGHTWARD:
    #   k=0 rightward arms, k=1 the (downward) trunk, k=2 leftward arms, k=3 upward.
    bands = {0: [], 1: [], 2: [], 3: []}
    for grp in row_groups(right_rows):
        bands[0].append([(r, c) for r in grp for c in range(c1, w)])
    bands[1].append([(r, c) for r in range(a, h) for c in range(c1, c2 + 1)])
    for grp in row_groups(left_rows):
        bands[2].append([(r, c) for r in grp for c in range(0, c2 + 1)])

    ops, sels = [], []

    if h == w:
        n = h
        # one CCW turn of the grid maps (r, c) -> (n-1-c, r)
        def rot(cells, k):
            out = list(cells)
            for _ in range(k):
                out = [(n - 1 - c, r) for (r, c) in out]
            return out

        # Apply the same "fill this band's interior" stroke in each of the four
        # orientations, rotating the whole grid between them (op24 = Rotate90/CCW).
        # Four rotations = one full turn, so the grid is submitted in its original
        # orientation.  The selection is exactly the whole (square) grid rectangle.
        for k in range(4):
            for cells in bands[k]:
                ops.append(3)                       # Color3 on this band's interior
                sels.append(sel_of(rot(cells, k)))
            ops.append(24)
            sels.append([0, 0, n - 1, n - 1])       # bbox == exactly the whole grid
    else:
        # non-square safety net: ARCLE's Rotate is only position-correct on square
        # selections, so fill each band in place instead.
        for k in range(4):
            for cells in bands[k]:
                ops.append(3)
                sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task a64e4611"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a64e4611"
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
                                f"for task a64e4611"
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
                    f"Failed to build a complete episode for task a64e4611 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a64e4611-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
