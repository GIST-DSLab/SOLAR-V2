"""
ARC Task: d687bc17 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    # bgc + the four frame colours are all sampled by the original generator.
    # The four frame colours must be NON-ZERO: the interior dots are carried to
    # their line with ARCLE Move ops, and ARCLE's object buffer keeps only
    # non-zero cells (a 0-coloured object would be grabbed as nothing).
    cols = list(range(10))
    bgc = random.choice(cols)
    pool = [c for c in cols if c != bgc and c != 0]
    c1, c2, c3, c4 = random.sample(pool, 4)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


# -------------------------------------------------------------- generator ---
def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4) -> dict:
    def _unifint(bounds):
        a, b = bounds
        lo = a + int((b - a) * diff_lb)
        hi = a + int((b - a) * diff_ub)
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    hub = max(5, min(30, int(max_h)))
    wub = max(5, min(30, int(max_w)))
    h = _unifint((5, hub))
    w = _unifint((5, wub))

    gi = [[bgc] * w for _ in range(h)]
    for j in range(w):
        gi[0][j] = c1                      # top row
    for i in range(h):
        gi[i][0] = c2                      # left column
    for i in range(h):
        gi[i][w - 1] = c3                  # right column
    for j in range(w):
        gi[h - 1][j] = c4                  # bottom row
    for (i, j) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        gi[i][j] = bgc                     # corners are background
    go = [row[:] for row in gi]

    # dots live strictly inside the ring the lines collapse onto
    cands = [(i, j) for i in range(2, h - 2) for j in range(2, w - 2)]
    ndots = _unifint((1, min(len(cands), h + w)))
    ndots = max(1, min(ndots, len(cands)))
    picked = random.sample(cands, ndots)
    dcol = {ij: random.choice((c1, c2, c3, c4)) for ij in picked}

    # c1 dots -> project onto row 1 (at most one dot per column)
    n1 = [ij for ij in picked if dcol[ij] == c1]
    cov = sorted({j for (i, j) in n1})
    if len(cov) == w - 4 and w > 5:
        cov.remove(random.choice(cov))
    for jj in cov:
        loci = random.choice([i for (i, j) in n1 if j == jj])
        gi[loci][jj] = c1
        go[1][jj] = c1

    # c2 dots -> project onto column 1 (at most one dot per row)
    n2 = [ij for ij in picked if dcol[ij] == c2]
    cov = sorted({i for (i, j) in n2})
    if len(cov) == h - 4 and h > 5:
        cov.remove(random.choice(cov))
    for ii in cov:
        locj = random.choice([j for (i, j) in n2 if i == ii])
        gi[ii][locj] = c2
        go[ii][1] = c2

    # c4 dots -> project onto row h-2 (at most one dot per column)
    n3 = [ij for ij in picked if dcol[ij] == c4]
    cov = sorted({j for (i, j) in n3})
    if len(cov) == w - 4 and w > 5:
        cov.remove(random.choice(cov))
    for jj in cov:
        loci = random.choice([i for (i, j) in n3 if j == jj])
        gi[loci][jj] = c4
        go[h - 2][jj] = c4

    # c3 dots -> project onto column w-2 (at most one dot per row)
    n4 = [ij for ij in picked if dcol[ij] == c3]
    cov = sorted({i for (i, j) in n4})
    if len(cov) == h - 4 and h > 5:
        cov.remove(random.choice(cov))
    for ii in cov:
        locj = random.choice([j for (i, j) in n4 if i == ii])
        gi[ii][locj] = c3
        go[ii][w - 2] = c3

    # scatter litter over background cells (never on a dot, never on a line)
    noisecands = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc]
    noisecols = [c for c in range(10) if c not in (bgc, c1, c2, c3, c4)]
    ub = ((h * w) - 2 * h - 2 * (w - 2)) // 2 - ndots - 1
    nnoise = _unifint((0, max(0, ub)))
    nnoise = max(0, min(nnoise, len(noisecands)))
    for (i, j) in random.sample(noisecands, nnoise):
        gi[i][j] = random.choice(noisecols)

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# ------------------------------------------------------------ derivation ----
def derive_operations(I, O):
    """Each dot slides onto the frame line that shares its colour; the litter
    is swept back to background.  Nothing below ever reads O -- the route is
    planned from I alone, and I also decides WHICH route: where to start
    walking round the frame, which way round it, and whether to sweep the
    board before the dots are carried out or after."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # ---- read the frame -------------------------------------------------
    # A line cell is never litter (litter only lands on background cells), so
    # one cell of each line names that line's colour.
    c_top    = int(I[0, 1])
    c_left   = int(I[1, 0])
    c_right  = int(I[1, w - 1])
    c_bottom = int(I[h - 1, 1])
    frame = {c_top, c_left, c_right, c_bottom}

    # the board proper: everything inside the frame, plus the four corner
    # cells (background in the frame, but litter can land there too)
    board = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    board += [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]

    # background = the colour that floods the board (dots and litter together
    # never reach half of it); the frame colours are excluded -- those are dots
    tally = {}
    for (r, c) in board:
        v = int(I[r, c])
        if v in frame:
            continue
        tally[v] = tally.get(v, 0) + 1
    if not tally:
        for (r, c) in board:
            v = int(I[r, c])
            tally[v] = tally.get(v, 0) + 1
    bgc = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    # ---- the four sides, clockwise from the top -------------------------
    # name, colour, Move op, where a dot at (r, c) is headed, walk-order key
    sides = [
        ("top",    c_top,    20, lambda r, c: (1, c),     lambda r, c:  c),
        ("right",  c_right,  22, lambda r, c: (r, w - 2), lambda r, c:  r),
        ("bottom", c_bottom, 21, lambda r, c: (h - 2, c), lambda r, c: -c),
        ("left",   c_left,   23, lambda r, c: (r, 1),     lambda r, c: -r),
    ]

    groups = {nm: [] for (nm, _, _, _, _) in sides}
    litter = []
    for (r, c) in board:
        v = int(I[r, c])
        if v == bgc:
            continue
        hit = None
        for (nm, col, _, _, _) in sides:
            if v == col:
                hit = nm
                break
        if hit is None:
            litter.append((r, c))
        else:
            groups[hit].append((r, c))

    # ---- choose the route, out of the input itself ----------------------
    # Start on the busiest line -- the pile of work you notice first -- then
    # keep walking round the frame towards whichever neighbour is busier.
    load = [len(groups[nm]) for (nm, _, _, _, _) in sides]
    start = max(range(4), key=lambda k: (load[k], -k))
    clockwise = load[(start + 1) % 4] >= load[(start - 1) % 4]
    order = [(start + (1 if clockwise else -1) * k) % 4 for k in range(4)]
    # walking the other way round hands you each line's dots in reverse too
    turn = 1 if clockwise else -1

    # A board buried under litter (a fifth of it or more) is cleared first, so
    # the dots travel over clean ground.  A board with only a few specks is
    # done the other way round: carry the dots out to their lines, then sweep
    # the whole middle back to background in one stroke.
    sweep_first = 5 * len(litter) >= len(board)

    grid = I.copy()
    placed = set()                 # ring cells a dot has come to rest on
    ops, sels = [], []

    def wipe(cells):
        cells = [(r, c) for (r, c) in cells if int(grid[r, c]) != bgc]
        if not cells:
            return
        ops.append(bgc)
        sels.append(sel_of(sorted(cells)))
        for (r, c) in cells:
            grid[r, c] = bgc

    def carry(r, c, colour, mop, dest):
        steps = abs(dest[0] - r) + abs(dest[1] - c)
        if steps <= 0:
            return
        if colour == 0:
            # an all-zero object cannot be grabbed by ARCLE; set it down at the
            # far end instead (bgc is non-zero whenever a dot colour is 0)
            ops.append(0)
            sels.append(sel_of([dest]))
        else:
            ops.append(mop)
            sels.append(sel_of([(r, c)]))       # this Move grabs the dot
            for _ in range(steps - 1):
                ops.append(mop)
                sels.append(sel_of([]))         # empty -> the same dot glides on
        grid[r, c] = 0                          # the grab hollowed out its old cell
        grid[dest[0], dest[1]] = colour
        placed.add((dest[0], dest[1]))
        if sweep_first and bgc != 0:
            ops.append(bgc)                     # tidy behind each dot as it leaves
            sels.append(sel_of([(r, c)]))
            grid[r, c] = bgc

    if sweep_first:
        wipe(litter)

    for k in order:
        nm, colour, mop, dest_of, key = sides[k]
        for (r, c) in sorted(groups[nm], key=lambda p: turn * key(p[0], p[1])):
            carry(r, c, colour, mop, dest_of(r, c))

    if not sweep_first:
        # every dot is out on the ring now, so the whole core rectangle --
        # rows 2..h-3, cols 2..w-3, holding the litter and the hollows the
        # dots left behind -- goes back to background in one stroke
        core = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
        if any(int(grid[r, c]) != bgc for (r, c) in core):
            ops.append(bgc)
            sels.append([2, 2, (h - 3) - 2, (w - 3) - 2])   # exactly that rectangle
            for (r, c) in core:
                grid[r, c] = bgc
        # then the litter out on the ring and in the corners, all but the
        # specks a dot has just come to rest on top of
        wipe([(r, c) for (r, c) in litter
              if not (2 <= r <= h - 3 and 2 <= c <= w - 3)
              and (r, c) not in placed])

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox = exactly the whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task d687bc17"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d687bc17"
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
                                f"for task d687bc17"
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
                    f"Failed to build a complete episode for task d687bc17 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d687bc17-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
