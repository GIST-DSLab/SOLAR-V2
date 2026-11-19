"""
ARC Task: 681b3aeb (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- helpers ----
def _ui(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = int(a + (b - a) * diff_lb)
    hi = int(a + (b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _norm(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _bbox_hw(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _is_rect(cells):
    h, w = _bbox_hw(cells)
    return len(cells) == h * w


def _dneighbors(S):
    out = set()
    for r, c in S:
        out.add((r - 1, c)); out.add((r + 1, c))
        out.add((r, c - 1)); out.add((r, c + 1))
    return out


def _pack(p1, p2):
    """The task's rule: the two pieces slide together until they tile a solid
    rectangle -- no overlap, no gap.  Deterministic scan, so generate() and
    derive_operations() always agree on the same assembly."""
    n = len(p1) + len(p2)
    h1, w1 = _bbox_hw(p1)
    h2, w2 = _bbox_hw(p2)
    for square_only in (True, False):
        for dr in range(-h2, h1 + 1):
            for dc in range(-w2, w1 + 1):
                s2 = {(r + dr, c + dc) for r, c in p2}
                if s2 & set(p1):
                    continue
                u = set(p1) | s2
                rs = [r for r, c in u]
                cs = [c for r, c in u]
                rmin, rmax = min(rs), max(rs)
                cmin, cmax = min(cs), max(cs)
                H, W = rmax - rmin + 1, cmax - cmin + 1
                if H * W != n:
                    continue
                if square_only and H != W:
                    continue
                return (-rmin, -cmin), (dr - rmin, dc - cmin), (H, W)
    return None


# ------------------------------------------------------------- 1. colors -----
def sample_colors(num_examples=None) -> dict:
    # Piece colours are kept non-zero: ARCLE's object ops treat 0 as "nothing",
    # so a 0-coloured piece could never be grabbed and moved.
    ca, cb = random.sample(range(1, 10), 2)
    bgc = random.choice([c for c in range(10) if c not in (ca, cb)])
    return {"bgc": bgc, "ca": ca, "cb": cb}


# ----------------------------------------------------------- 2. generate -----
def generate(diff_lb, diff_ub, max_h, max_w, bgc, ca, cb) -> dict:
    max_h = min(int(max_h), 30)
    max_w = min(int(max_w), 30)
    while True:
        hi_ub = max(2, min(8, max_h // 3, max_w))
        hi = _ui(diff_lb, diff_ub, (2, hi_ub))
        wi_ub = min(8, max(2, max_w // 3))
        wi = _ui(diff_lb, diff_ub, (2, wi_ub))
        h_lb = max(hi, min(3 * hi, max_h))
        h = _ui(diff_lb, diff_ub, (h_lb, max_h))
        w_lb = max(hi, min(3 * wi, max_w))
        w = _ui(diff_lb, diff_ub, (w_lb, max_w))

        # split the hi x hi square into two interlocking connected pieces
        cells = [(r, c) for r in range(hi) for c in range(hi)]
        A = B = None
        for _ in range(60):
            pa, pb = random.sample(cells, 2)
            rem = set(cells) - {pa, pb}
            SA, SB = {pa}, {pb}
            ok = True
            while rem:
                ac = rem & _dneighbors(SA)
                bc = rem & _dneighbors(SB)
                opts = []
                if ac:
                    opts.append(0)
                if bc:
                    opts.append(1)
                if not opts:
                    ok = False
                    break
                if random.choice(opts) == 0:
                    loc = random.choice(sorted(ac)); SA.add(loc)
                else:
                    loc = random.choice(sorted(bc)); SB.add(loc)
                rem.discard(loc)
            if not ok:
                continue
            if _is_rect(SA) and _is_rect(SB):
                continue                 # at least one piece must be non-trivial
            A, B = SA, SB
            break
        if A is None:
            continue

        pA, pB = _norm(A), _norm(B)
        # canonical (colour-ordered) pairing, shared with derive_operations
        if ca <= cb:
            p1, col1, p2, col2 = pA, ca, pB, cb
        else:
            p1, col1, p2, col2 = pB, cb, pA, ca
        res = _pack(p1, p2)
        if res is None:
            continue
        off1, off2, (H, W) = res
        if H > h or W > w:
            continue

        go = [[bgc] * W for _ in range(H)]
        for r, c in p1:
            go[r + off1[0]][c + off1[1]] = col1
        for r, c in p2:
            go[r + off2[0]][c + off2[1]] = col2

        # one piece already sits at its assembled place, the other lies apart
        if random.random() < 0.5:
            anc_p, anc_off, anc_col = p1, off1, col1
            mov_p, mov_off, mov_col = p2, off2, col2
        else:
            anc_p, anc_off, anc_col = p2, off2, col2
            mov_p, mov_off, mov_col = p1, off1, col1

        wr = random.randint(0, h - H)
        wc = random.randint(0, w - W)
        mh, mw = _bbox_hw(mov_p)
        cands = [
            (mr, mc)
            for mr in range(h - mh + 1)
            for mc in range(w - mw + 1)
            if (mr + mh <= wr or mr >= wr + H or mc + mw <= wc or mc >= wc + W)
        ]
        if not cands:
            continue
        mr, mc = random.choice(cands)

        gi = [[bgc] * w for _ in range(h)]
        for r, c in anc_p:
            gi[wr + anc_off[0] + r][wc + anc_off[1] + c] = anc_col
        for r, c in mov_p:
            gi[mr + r][mc + c] = mov_col

        return {"input": gi, "output": go}


# --------------------------------------------------- 3. derive_operations ----
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hI, wI = I.shape

    # background = the colour the canvas was painted with before the two pieces
    # were dropped on it; the pieces cover at most a third of the grid.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    pieces = {}
    for r in range(hI):
        for c in range(wI):
            v = int(I[r, c])
            if v != bgc:
                pieces.setdefault(v, []).append((r, c))
    cols = sorted(pieces)                        # canonical colour ordering
    cells1, cells2 = pieces[cols[0]], pieces[cols[1]]
    p1, p2 = _norm(cells1), _norm(cells2)

    # THE RULE, measured from I alone: the two pieces interlock into a rectangle
    packed = _pack(p1, p2)
    if packed is None:                           # malformed pair: no assembly
        rs = [r for r, c in cells1 + cells2]
        cs = [c for r, c in cells1 + cells2]
        return ([33, 34],
                [[min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)],
                 [0, 0, max(rs) - min(rs), max(cs) - min(cs)]])
    off1, off2, (H, W) = packed

    pos1 = (min(r for r, c in cells1), min(c for r, c in cells1))
    pos2 = (min(r for r, c in cells2), min(c for r, c in cells2))

    # leaving one piece where it is fixes where the assembled rectangle lands;
    # keep whichever choice puts that rectangle inside the grid.
    options = []
    for (off_a, pos_a, cells_b, off_b, pos_b, col_b) in (
        (off1, pos1, cells2, off2, pos2, cols[1]),
        (off2, pos2, cells1, off1, pos1, cols[0]),
    ):
        wr = pos_a[0] - off_a[0]
        wc = pos_a[1] - off_a[1]
        if 0 <= wr <= hI - H and 0 <= wc <= wI - W:
            options.append((col_b == 0, wr, wc, cells_b, off_b, pos_b))
    if not options:
        options.append((False, max(0, min(pos1[0] - off1[0], hI - H)),
                        max(0, min(pos1[1] - off1[1], wI - W)),
                        cells2, off2, pos2))
    options.sort(key=lambda o: o[0])             # never try to move a 0-coloured piece
    _, wr, wc, mov_cells, mov_off, mov_pos = options[0]

    ops, sels = [], []

    # --- carry the loose piece over into its slot of the assembled rectangle ---
    dr = (wr + mov_off[0]) - mov_pos[0]
    dc = (wc + mov_off[1]) - mov_pos[1]
    cur = list(mov_cells)
    grabbed = False
    for step_dr, step_dc, op, cnt in ((-1, 0, 20, max(0, -dr)),
                                      (1, 0, 21, max(0, dr)),
                                      (0, -1, 23, max(0, -dc)),
                                      (0, 1, 22, max(0, dc))):
        for _ in range(cnt):
            ops.append(op)
            if grabbed:
                sels.append(sel_of([]))          # keep the same object grabbed
            else:
                sels.append(sel_of(cur))         # grab the piece's true shape
                grabbed = True
            cur = [(r + step_dr, c + step_dc) for r, c in cur]
    # rows move first, then columns, so the piece is never pushed off the grid.
    # ARCLE leaves the vacated footprint at 0, but the two pieces were disjoint,
    # so that footprint lies entirely outside the assembled rectangle and the
    # crop below discards it -- no repair op is needed.

    # --- the answer is the assembled rectangle itself -------------------------
    # both selections are exactly full rectangles (a canvas crop, then submit)
    ops.append(33); sels.append([wr, wc, H - 1, W - 1])
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
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
                                f"for task 681b3aeb"
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
                    f"Failed to build a complete episode for task 681b3aeb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"681b3aeb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
