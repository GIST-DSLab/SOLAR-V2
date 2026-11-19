"""
ARC Task: 1b60fb0c (RE-ARC) — LLM-generated grid_maker
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
    # the generator draws bgc, objc from interval(0, 10) with 2 removed;
    # 2 is hardcoded as the colour the missing quarter is revealed in
    cols = [c for c in range(10) if c != 2]
    bgc, objc = random.sample(cols, 2)
    return {"bgc": bgc, "objc": objc}


# -------------------------------------------------------------- generate ----
def _unifint(diff_lb, diff_ub, bounds):
    lo, hi = bounds
    a = lo + int((hi - lo) * diff_lb)
    b = lo + int((hi - lo) * diff_ub)
    if a > b:
        a, b = b, a
    return random.randint(a, b)


def _rot90cw(cells, n):
    # DSL rot90 (clockwise) acting on the cell set of an n x n quadrant
    return {(c, n - 1 - r) for r, c in cells}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc) -> dict:
    for _attempt in range(200):
        h = _unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
        w = _unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
        odh = _unifint(diff_lb, diff_ub, (2, min(h, w) // 2))
        loci = random.randint(0, h - 2 * odh)
        locj = random.randint(0, w - 2 * odh)

        ncellsd = _unifint(diff_lb, diff_ub, (0, odh ** 2 // 2))
        ncells = random.choice((ncellsd, odh ** 2 - ncellsd))
        ncells = min(max(1, ncells), odh ** 2 - 1)
        allcells = [(r, c) for r in range(odh) for c in range(odh)]
        q1 = set(random.sample(allcells, ncells))
        q2 = _rot90cw(q1, odh)
        q3 = _rot90cw(q2, odh)
        q4 = _rot90cw(q3, odh)

        s = random.randint(0, odh)
        c1 = {(r, c + s) for r, c in q1}
        c2 = {(r + s, c + odh) for r, c in q2}
        c3 = {(r + odh, c + odh - s) for r, c in q3}
        c4 = {(r + odh - s, c) for r, c in q4}

        cs = [c1, c2, c3, c4]
        k = random.randrange(4)
        rempart = cs[k]
        inobj = set()
        for j in range(4):
            if j != k:
                inobj |= cs[j]
        rempart = rempart - inobj
        if not rempart:            # nothing would be revealed -> retry
            continue

        inobj = {(r + loci, c + locj) for r, c in inobj}
        rempart = {(r + loci, c + locj) for r, c in rempart}

        gi = [[bgc] * w for _ in range(h)]
        for r, c in inobj:
            gi[r][c] = objc
        go = [row[:] for row in gi]
        for r, c in rempart:
            go[r][c] = 2
        return {
            "input": tuple(tuple(row) for row in gi),
            "output": tuple(tuple(row) for row in go),
        }
    raise ValueError("generation failed")


# ------------------------------------------------------- derive_operations --
# A clockwise quarter turn about a centre (pr, pc) is  (r, c) -> (a + c, b - r)
# with a = pr - pc and b = pr + pc; the centre lies between cells, so a + b is odd.
def _find_turn(S, hi, wi):
    """The quarter turn that carries the visible pattern onto itself and
    reveals fresh ground.  Returns (a, b) or None."""
    rows = [r for r, _ in S]
    cols = [c for _, c in S]
    rmin, rmax, cmin, cmax = min(rows), max(rows), min(cols), max(cols)
    best = None
    for a in range(-cmin, hi - cmax):              # keeps rows in the grid
        for b in range(rmax, wi + rmin):           # keeps cols in the grid
            if (a + b) % 2 == 0:                   # centre lies between cells
                continue
            T = frozenset((a + c, b - r) for r, c in S)
            if not (T - S):                        # reveals nothing
                continue
            F = S | T
            if frozenset((a + cc, b - rr) for rr, cc in F) != F:
                continue                           # union must be 4-fold symmetric
            score = len(S & T)                     # turn must land on the pattern
            if best is None or score > best[0]:
                best = (score, a, b)
    return None if best is None else (best[1], best[2])


def derive_operations(I, O=None, examples=None):
    """The pattern has four-fold rotational symmetry with one quarter missing.
    Mark the visible pattern, turn it about the pattern's centre, then put the
    original pattern back in its own colour: what is still marked is exactly
    the quarter that was missing.

    The turn may be a quarter turn either way round, or a half turn (two
    mirrors) -- all three carry the three visible quarters onto the same fresh
    ground.  Which one this grid gets, which quadrant the marking starts from,
    which way it sweeps and whether it goes quadrant by quadrant or in one
    stroke are all read off the input, so different grids are solved along
    different routes and the same grid always the same way.  O is never read:
    handed a different answer this returns the same operations."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    whole = [0, 0, hi - 1, wi - 1]

    # --- background colour --------------------------------------------
    # In the demonstrations the revealed cells were background in the input,
    # which names the background outright; without them, the input holds just
    # two colours and only one of them leaves a turnable pattern behind.
    counts = Counter(I.flatten().tolist())
    cands = [c for c, _ in counts.most_common()]
    if examples:
        for ex in examples:
            ei = np.asarray(ex[0], dtype=int)
            eo = np.asarray(ex[1], dtype=int)
            hit = [(r, c) for r in range(eo.shape[0]) for c in range(eo.shape[1])
                   if eo[r, c] == 2 and ei[r, c] != 2]
            if hit:
                b = int(ei[hit[0]])
                if b in counts:
                    cands = [b] + [c for c in cands if c != b]
                break

    found = None
    for bgc in cands:
        S = frozenset((r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc)
        if not S:
            continue
        turn = _find_turn(S, hi, wi)
        if turn is not None:
            found = (bgc, S, turn)
            break
    if found is None:                              # degenerate: nothing to reveal
        ops.append(34); sels.append(whole)
        return ops, sels
    bgc, S, (a, b) = found

    objc = int(I[min(S)])
    rows = [r for r, _ in S]
    cols = [c for _, c in S]
    rmin, rmax, cmin, cmax = min(rows), max(rows), min(cols), max(cols)

    # --- the region the turn acts on: centred on the turn's centre ----
    T = frozenset((a + c, b - r) for r, c in S)
    F = S | T
    cr2, cc2 = a + b, b - a                        # twice the centre (odd, odd)
    dr2 = max(abs(2 * r - cr2) for r, _ in F)
    dc2 = max(abs(2 * c - cc2) for _, c in F)
    # mirrors only need the tight box centred on the turn's centre; a quarter
    # turn needs that box square, and the four-fold pattern always makes it one
    rect = [(cr2 - dr2) // 2, (cc2 - dc2) // 2, dr2, dc2]
    d2 = max(dr2, dc2)
    sr0, sc0 = (cr2 - d2) // 2, (cc2 - d2) // 2
    square = [sr0, sc0, d2, d2]
    square_fits = sr0 >= 0 and sc0 >= 0 and sr0 + d2 < hi and sc0 + d2 < wi

    # --- pick a route out of the input --------------------------------
    # all three turns carry the three visible quarters onto the same fresh
    # ground, so the grid itself may choose: its cell count, colour and corner
    sig = len(S) + objc + rmin + cmin
    route = sig % 3 if square_fits else 2          # 0 quarter CW, 1 quarter CCW, 2 half
    half_h_first = (len(S) + cmin) % 2 == 0        # which mirror of the half turn leads
    turn_cw = (route == 0) or (route == 2 and half_h_first)

    # --- the pattern, quadrant by quadrant around that centre ---------
    quads = [(0, 0), (0, 1), (1, 1), (1, 0)]       # TL, TR, BR, BL, clockwise
    groups = {q: [] for q in quads}
    for (r, c) in sorted(S):
        groups[(0 if 2 * r < cr2 else 1, 0 if 2 * c < cc2 else 1)].append((r, c))
    # start where the pattern is heaviest and sweep the way the turn goes
    start = max(range(4), key=lambda i: (len(groups[quads[i]]), -i))
    step = 1 if turn_cw else -1
    order = [quads[(start + step * i) % 4] for i in range(4)]
    # sweep the quadrants one by one, or take the pattern in a single stroke;
    # an empty quadrant gets no stroke
    if ((rmax - rmin) + (cmax - cmin) + wi) % 2 == 0:
        strokes = [groups[q] for q in order if groups[q]]
    else:
        strokes = [sorted(S)]

    # 1. mark the whole visible pattern with the answer colour
    for cells in strokes:
        ops.append(2); sels.append(sel_of(cells))
    # 2. turn the marked pattern about its centre.  The bbox selection is
    #    intended: the WHOLE region turns, background included.  Nothing of
    #    colour 0 has to survive the turn -- the pattern is colour 2 by now,
    #    and a 0 background lands back on the 0 the grab left behind.
    if route == 0:
        ops.append(25); sels.append(square)        # quarter turn clockwise
    elif route == 1:
        ops.append(24); sels.append(square)        # quarter turn anticlockwise
    else:
        # a half turn is two mirrors; a mirror the marks already sit
        # symmetrically about would move nothing, so it is not made
        marks = S
        pair = [(26, lambda cs: frozenset((r, cc2 - c) for r, c in cs)),
                (27, lambda cs: frozenset((cr2 - r, c) for r, c in cs))]
        if not half_h_first:
            pair.reverse()
        for op, mirror in pair:
            turned = mirror(marks)
            if turned != marks:
                ops.append(op); sels.append(rect)
                marks = turned
    # 3. put the original pattern back in its own colour; the marks the turn
    #    carried onto fresh ground are the missing quarter
    for cells in strokes:
        ops.append(objc); sels.append(sel_of(cells))

    ops.append(34); sels.append(whole)
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
                        f"num_examples+1 ({num_examples + 1}) for task 1b60fb0c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1b60fb0c"
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
                                f"for task 1b60fb0c"
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
                    f"Failed to build a complete episode for task 1b60fb0c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1b60fb0c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
