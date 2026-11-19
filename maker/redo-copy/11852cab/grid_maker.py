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
import numpy as np
from collections import Counter

from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    """Fix the background once per episode; ring colours are pattern-level only
    (the rule never depends on which colour a ring has), but we still share the
    same foreground palette across all instances of the episode."""
    cols = list(range(10))
    bgc = random.choice(cols)
    rest = [c for c in cols if c != bgc]
    random.shuffle(rest)
    return {"bgc": bgc, "palette": rest}


# -------------------------------------------------------------- generator ----
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int = 0, palette=None) -> dict:
    """Places 5x5 'concentric ring' motifs.

    Ring layout (relative to the 5x5 patch, all cells have even (i+j)):
        r4 = center, r3 = inner diagonals, r2 = edge midpoints, r1 = corners.
    The centre ring plus one further ring are drawn complete in BOTH grids;
    1-2 more rings are drawn PARTIALLY in the input and COMPLETE in the output.

    Unlike the original generator we only commit a motif when it also satisfies
    the detection predicate the task's rule uses (7x7 border clean, centre set,
    inner ring non-empty, outer ring non-empty, non-bg content spanning the full
    5x5).  Motifs that would be left untouched by the rule are simply not drawn,
    so every motif in the episode demonstrates the same visible rule.
    """
    cols = interval(0, 10, 1)
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = backdrop(frozenset(r1))

    hub = max(7, min(30, int(max_h)))
    wub = max(7, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (7, hub))
    w = unifint(diff_lb, diff_ub, (7, wub))

    pool = [c for c in (palette if palette else cols) if c != bgc]
    if not pool:
        pool = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (1, 9))
    numc = max(1, min(numc, len(pool)))
    ccols = pool[:numc]

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = shift(asindices(trim(gi)), UNITY)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 36)))

    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - 5 and ij[1] <= w - 5)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcd = shift(bx, loc)
        if not plcd.issubset(inds):
            continue

        ringcols = [choice(ccols) for _ in range(4)]
        idx = randint(1, 3)
        rest_rings = list(rings[1:idx]) + list(rings[idx + 1:])
        rest_cols = list(ringcols[1:idx]) + list(ringcols[idx + 1:])
        numrs = unifint(diff_lb, diff_ub, (1, 2))
        picks = sample((0, 1), numrs)

        parts = []                      # (full ring, partial cells, colour)
        for j in range(len(rest_rings)):
            if j not in picks:
                continue
            k = 4 - unifint(diff_lb, diff_ub, (0, 3))   # 1..4 visible cells
            k = max(1, min(4, k))
            partial = tuple(sample(list(rest_rings[j]), k))
            parts.append((rest_rings[j], partial, rest_cols[j]))
        if not parts:
            continue

        partcells = [p for _, pr, _ in parts for p in pr]
        rs = [p[0] for p in partcells]
        cs = [p[1] for p in partcells]
        if max(rs) - min(rs) != 4 or max(cs) - min(cs) != 4:
            continue                    # partial content must span the full 5x5

        allcells = set(rings[0]) | set(rings[idx]) | set(partcells)
        on_perimeter = any(a in (0, 4) or b in (0, 4) for a, b in allcells)
        on_inbox = any(1 <= a <= 3 and 1 <= b <= 3 and (a, b) != (2, 2)
                       for a, b in allcells)
        if not (on_perimeter and on_inbox):
            continue                    # would not be picked up by the rule

        inds = (inds - plcd) - outbox(plcd)
        succ += 1

        def sh(cells):
            return frozenset((loc[0] + a, loc[1] + b) for a, b in cells)

        gi = fill(gi, ringcols[0], sh(rings[0]))
        go = fill(go, ringcols[0], sh(rings[0]))
        gi = fill(gi, ringcols[idx], sh(rings[idx]))
        go = fill(go, ringcols[idx], sh(rings[idx]))
        for fullring, partial, col in parts:
            gi = fill(gi, col, sh(partial))
            go = fill(go, col, sh(fullring))

    return {'input': gi, 'output': go}


# ------------------------------------------------------------ derivation ----
def derive_operations(I, O):
    """Everything below is measured from I only.

    Rule (read off the generator/verifier): find every 5x5 ring-motif in I -- a
    patch whose surrounding 7x7 border is pure background, whose odd-parity
    cells are all background, whose centre is set, which has content both on its
    perimeter and on its inner 3x3 ring, and whose content spans the whole 5x5.
    For each such motif, every ring (inner diagonals, edge midpoints, corners)
    that shows at least one coloured cell is COMPLETED to its full 4-fold
    symmetric set, in that ring's own colour.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # rings ordered inner -> outer (the centre ring is never incomplete)
    RINGS = [
        [(1, 1), (1, 3), (3, 1), (3, 3)],      # inner diagonals
        [(0, 2), (2, 0), (2, 4), (4, 2)],      # edge midpoints
        [(0, 0), (0, 4), (4, 0), (4, 4)],      # corners
    ]
    PERIM = [(a, b) for a in range(5) for b in range(5)
             if a in (0, 4) or b in (0, 4)]
    INBOX = [(a, b) for a in range(1, 4) for b in range(1, 4)
             if (a, b) != (2, 2)]

    ops, sels = [], []

    for r in range(1, h - 5):
        for c in range(1, w - 5):
            W = I[r - 1:r + 6, c - 1:c + 6]
            if W.shape != (7, 7):
                continue
            # the 7x7 halo around the motif must be pure background
            if not (np.all(W[0, :] == bgc) and np.all(W[6, :] == bgc)
                    and np.all(W[:, 0] == bgc) and np.all(W[:, 6] == bgc)):
                continue
            P = I[r:r + 5, c:c + 5]
            # motif cells always sit on even parity positions
            if any(P[a, b] != bgc for a in range(5) for b in range(5)
                   if (a + b) % 2 == 1):
                continue
            if P[2, 2] == bgc:
                continue
            if not any(P[a, b] != bgc for a, b in PERIM):
                continue
            if not any(P[a, b] != bgc for a, b in INBOX):
                continue
            occ = [(a, b) for a in range(5) for b in range(5) if P[a, b] != bgc]
            ars = [a for a, _ in occ]
            acs = [b for _, b in occ]
            if max(ars) - min(ars) != 4 or max(acs) - min(acs) != 4:
                continue

            # complete each partially drawn ring, innermost first
            for ring in RINGS:
                vals = [int(P[a, b]) for a, b in ring if P[a, b] != bgc]
                if not vals:
                    continue
                col = Counter(vals).most_common(1)[0][0]
                missing = [(r + a, c + b) for a, b in ring if int(P[a, b]) != col]
                if missing:
                    ops.append(col)
                    sels.append(sel_of(sorted(missing)))

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
