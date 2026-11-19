"""
ARC Task: 150deff5 (RE-ARC) — LLM-generated grid_maker
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
import sys
import random
import numpy as np
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # generator: cols = interval(0,10) minus (2, 8)  -> 2 and 8 are reserved
    # output colors, so they can never be bgc/fgc.
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(int(a + (b - a) * lb), int(a + (b - a) * ub))

    hub = max(8, min(30, int(max_h)))
    wub = max(8, min(30, int(max_w)))

    h = unifint(diff_lb, diff_ub, (8, hub))
    w = unifint(diff_lb, diff_ub, (8, wub))

    bo = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})          # 2x2 block  -> 8
    ro1 = frozenset({(0, 0), (0, 1), (0, 2)})                 # 1x3 line   -> 2
    ro2 = frozenset({(0, 0), (1, 0), (2, 0)})                 # 3x1 line   -> 2

    boforb = set()
    reforb = set()

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    noccs = unifint(diff_lb, diff_ub, (2, (h * w) // 10))

    # every instance shows all three object kinds at least once (learnability):
    # the remaining placements stay faithful to the original random choice.
    forced = [(bo, 8), (ro1, 2), (ro2, 2)]
    random.shuffle(forced)
    picks = list(forced)
    for _ in range(max(0, noccs - len(forced))):
        picks.append(random.choice([(bo, 8), (random.choice([ro1, ro2]), 2)]))

    inds = {(i, j) for i in range(h) for j in range(w)}

    for obj, col in picks:
        oh = max(i for i, j in obj) + 1
        ow = max(j for i, j in obj) + 1
        forb = boforb if col == 8 else reforb
        cands = []
        for (i, j) in inds:
            if i <= h - oh and j <= w - ow and (i, j) not in forb:
                if all((i + a, j + b) in inds for a, b in obj):
                    cands.append((i, j))
        if not cands:
            break
        loc = random.choice(sorted(cands))
        li, lj = loc
        if col == 8:
            boforb.add((li - 2, lj))
            boforb.add((li + 2, lj))
            boforb.add((li, lj + 2))
            boforb.add((li, lj - 2))
        else:
            if obj == ro1:
                reforb.add((li, lj + 3))
                reforb.add((li, lj - 3))
            else:
                reforb.add((li + 1, lj))
                reforb.add((li - 1, lj))
        plcd = {(li + a, lj + b) for a, b in obj}
        for (r, c) in plcd:
            gi[r][c] = fgc
            go[r][c] = col
        inds -= plcd

    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


# ----------------------------------------------------------------------------
# 3. operations
# ----------------------------------------------------------------------------
def _cover_squares(cells):
    """Exact cover of an 8-region by disjoint 2x2 blocks (forced greedy)."""
    rem = set(cells)
    out = []
    while rem:
        r, c = min(rem)
        blk = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
        if not all(p in rem for p in blk):
            return None
        out.append((r, c))
        for p in blk:
            rem.discard(p)
    return out


def _cover_lines(cells):
    """Exact cover of a 2-region by disjoint 3-cell lines (DFS, min-cell first)."""
    rem = set(cells)
    out = []

    def rec():
        if not rem:
            return True
        r, c = min(rem)
        for kind, pts in (("h", [(r, c), (r, c + 1), (r, c + 2)]),
                          ("v", [(r, c), (r + 1, c), (r + 2, c)])):
            if all(p in rem for p in pts):
                for p in pts:
                    rem.discard(p)
                out.append((kind, r, c))
                if rec():
                    return True
                out.pop()
                for p in pts:
                    rem.add(p)
        return False

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 4000))
    try:
        ok = rec()
    finally:
        sys.setrecursionlimit(old)
    return out if ok else None


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    ops, sels = [], []

    eight_cells = {(int(r), int(c)) for r, c in zip(*np.where(O == 8))}
    two_cells = {(int(r), int(c)) for r, c in zip(*np.where(O == 2))}

    squares = _cover_squares(eight_cells) or []
    lines = _cover_lines(two_cells) or []
    hlines = [(r, c) for k, r, c in lines if k == "h"]
    vlines = [(r, c) for k, r, c in lines if k == "v"]
    hlines.sort()
    vlines.sort()

    # ---- 2x2 blocks: paint the first one, then replicate it at every occurrence
    if squares:
        r0, c0 = squares[0]
        ops.append(8)
        sels.append(sel_of([(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]))
        if len(squares) > 1:
            # bbox == exactly the 2x2 stamp region
            ops.append(29); sels.append([r0, c0, 1, 1])
            for (r, c) in squares[1:]:
                ops.append(30); sels.append([r, c, 0, 0])

    # ---- 3-cell lines: build ONE horizontal stamp and replicate it
    have_stamp = False
    if hlines:
        r0, c0 = hlines[0]
        ops.append(2)
        sels.append(sel_of([(r0, c0), (r0, c0 + 1), (r0, c0 + 2)]))
        if len(hlines) > 1 or vlines:
            # bbox == exactly the 1x3 stamp region
            ops.append(29); sels.append([r0, c0, 0, 2])
            have_stamp = True
        for (r, c) in hlines[1:]:
            ops.append(30); sels.append([r, c, 0, 0])

    # ---- vertical lines: turn the grid a quarter turn so they read horizontally,
    #      stamp the SAME line there, then turn the grid back.
    if vlines:
        sq = max(hi, wi)
        ops.append(33); sels.append([0, 0, sq - 1, sq - 1])          # square canvas
        ops.append(25); sels.append([0, 0, sq - 1, sq - 1])          # rotate CW
        ops.append(33); sels.append([0, sq - hi, wi - 1, hi - 1])    # keep the turned grid
        # old (r, c) -> new (c, hi-1-r); a vertical line becomes a horizontal one
        rot = [(c, hi - 3 - r) for (r, c) in vlines]
        start = 0
        if not have_stamp:
            rr, cc = rot[0]
            ops.append(2)
            sels.append(sel_of([(rr, cc), (rr, cc + 1), (rr, cc + 2)]))
            if len(rot) > 1:
                ops.append(29); sels.append([rr, cc, 0, 2])
            start = 1
        for (rr, cc) in rot[start:]:
            ops.append(30); sels.append([rr, cc, 0, 0])
        ops.append(33); sels.append([0, 0, sq - 1, sq - 1])          # square canvas
        ops.append(24); sels.append([0, 0, sq - 1, sq - 1])          # rotate CCW back
        ops.append(33); sels.append([sq - hi, 0, hi - 1, wi - 1])    # original framing

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 150deff5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 150deff5"
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
                                f"for task 150deff5"
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
                    f"Failed to build a complete episode for task 150deff5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"150deff5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
