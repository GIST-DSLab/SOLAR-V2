"""
ARC Task: 8e1813be (RE-ARC) — LLM-generated grid_maker
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


# --------------------------------------------------------------------------
# Task 8e1813be
#
# Structure of every instance (read off the generator):
#   * `nbars` stripes, each a solid line (1 cell thick) of a unique colour,
#     spanning the whole grid, separated by `hmarg` background lines.
#   * one solid `nbars` x `nbars` square of colour `sqc`, surrounded by a
#     background ring (`outbox`), which erases the piece of every stripe that
#     runs through that band.
#   * with probability 1/2 BOTH the input and the output are dmirrored, so the
#     stripes are columns instead of rows.
#
# The verifier's `dmirror` (x12) only undoes that joint transposition of input
# AND output - it is orientation normalisation, not a visible I->O reflection
# (the mirror cancels between gi and go).  What is actually visible between I
# and O is: the square is an nbars x nbars canvas, each stripe TRANSLATES into
# its slot of that canvas (k-th stripe in reading order -> k-th line of the
# square), and the canvas is then cropped out.  So the route below is
# Move-based, and every parameter (which stripes, which order, which canvas,
# which axis, how far) is measured from I alone; O is never inspected.
# --------------------------------------------------------------------------


VARIANTS = [{"transposed": False}, {"transposed": True}]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    bgc, sqc = random.sample(list(range(10)), 2)
    # Stripe colours never include 0: ARCLE's object ops (Move) only pick up
    # non-zero cells, so a colour-0 stripe could not be translated at all.
    barpool = [c for c in range(1, 10) if c != bgc and c != sqc]
    random.shuffle(barpool)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "barpool": barpool, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, barpool,
             transposed=None, **kwargs) -> dict:
    if transposed is None:
        transposed = random.choice([True, False])

    # build un-transposed (stripes = rows), then transpose if asked; the
    # size limits swap accordingly
    lim_h, lim_w = (max_w, max_h) if transposed else (max_h, max_w)

    pool = list(barpool)
    nb_ub = min(8, len(pool), lim_h // 3, lim_w - 3)
    nb_ub = max(3, nb_ub)
    nbars = _unifint(diff_lb, diff_ub, (3, nb_ub))
    nbars = max(3, min(nbars, len(pool)))
    ccols = random.sample(pool, nbars)

    w = _unifint(diff_lb, diff_ub, (nbars + 3, max(nbars + 3, lim_w)))
    hmarg = _unifint(diff_lb, diff_ub, (2 * nbars, max(2 * nbars, lim_h - nbars)))

    go = [[c] * nbars for c in ccols]
    gi = [[c] * w for c in ccols]
    for _ in range(hmarg):
        idx = random.randint(0, nbars - 1)
        gi.insert(idx, [bgc] * w)

    h2 = nbars + hmarg
    oh = ow = nbars
    loci = random.randint(1, max(1, h2 - oh - 2))
    locj = random.randint(1, max(1, w - ow - 2))

    for r in range(loci, min(h2, loci + oh)):
        for c in range(locj, min(w, locj + ow)):
            gi[r][c] = sqc
    for c in range(locj - 1, locj + ow + 1):
        for r in (loci - 1, loci + oh):
            if 0 <= r < h2 and 0 <= c < w:
                gi[r][c] = bgc
    for r in range(loci - 1, loci + oh + 1):
        for c in (locj - 1, locj + ow):
            if 0 <= r < h2 and 0 <= c < w:
                gi[r][c] = bgc

    if transposed:
        gi = [list(row) for row in zip(*gi)]
        go = [list(row) for row in zip(*go)]

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def _apply_move(g, cells, dr, dc, steps):
    """Mirror of ARCLE's object Move chain: grab (source cells zeroed in the
    background snapshot), then paste the object at its final offset."""
    R, C = g.shape
    obj = [(r, c, int(g[r, c])) for (r, c) in cells if g[r, c] != 0]
    out = g.copy()
    for (r, c) in cells:
        out[r, c] = 0
    for (r, c, v) in obj:
        rr, cc = r + dr * steps, c + dc * steps
        if 0 <= rr < R and 0 <= cc < C:
            out[rr, cc] = v
    return out


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour the generator paints the margin lines with
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- measure the structure of I (colour-partition, exactly like the task)
    row_bars, col_bars, blocks = [], [], []
    for c in sorted(set(int(v) for v in I.flatten().tolist())):
        if c == bgc:
            continue
        pos = np.argwhere(I == c)
        r0, r1 = int(pos[:, 0].min()), int(pos[:, 0].max())
        c0, c1 = int(pos[:, 1].min()), int(pos[:, 1].max())
        if r0 == r1 and c1 > c0:
            row_bars.append((r0, c))          # horizontal stripe
        elif c0 == c1 and r1 > r0:
            col_bars.append((c0, c))          # vertical stripe
        else:
            blocks.append((int(pos.shape[0]), r0, c0))   # the square

    by_row = len(row_bars) > len(col_bars)    # stripes run horizontally?
    bars = sorted(row_bars) if by_row else sorted(col_bars)
    n = len(bars)
    if n == 0:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # the square is the only non-line object: it is the n x n output canvas
    blocks.sort(key=lambda b: -b[0])
    sr, sc = (blocks[0][1], blocks[0][2]) if blocks else (0, 0)
    tr = max(0, min(sr, hi - n))
    tc = max(0, min(sc, wi - n))

    pos = [b[0] for b in bars]                # current line index of stripe k
    cols = [b[1] for b in bars]               # its colour
    dest = [(tr if by_row else tc) + k for k in range(n)]
    disp = [dest[k] - pos[k] for k in range(n)]

    # process order so a stripe never lands on a stripe that has not moved yet
    order = [k for k in range(n) if disp[k] == 0]
    order += sorted([k for k in range(n) if disp[k] < 0])
    order += sorted([k for k in range(n) if disp[k] > 0], reverse=True)

    g = I.copy()
    ops, sels = [], []

    for k in order:
        c = cols[k]
        p = pos[k]
        if by_row:
            line = [(p, j) for j in range(tc, tc + n)]
            dline = [(dest[k], j) for j in range(tc, tc + n)]
        else:
            line = [(i, p) for i in range(tr, tr + n)]
            dline = [(i, dest[k]) for i in range(tr, tr + n)]

        if c == 0 and disp[k] != 0:
            # ARCLE cannot grab a colour-0 object, so this stripe is drawn at
            # the slot it belongs to; the slot comes from I's geometry.
            bad = [(r, cc) for (r, cc) in dline if g[r, cc] != 0]
            if bad:
                ops.append(0)
                sels.append(sel_of(bad))
                for (r, cc) in bad:
                    g[r, cc] = 0
            continue

        # restore the piece of the stripe the square / its ring erased, but
        # only over the span that has to reach the canvas
        bad = [(r, cc) for (r, cc) in line if g[r, cc] != c]
        if bad:
            ops.append(int(c))
            sels.append(sel_of(bad))
            for (r, cc) in bad:
                g[r, cc] = c

        if disp[k] == 0:
            continue

        cells = [(int(r), int(cc)) for r, cc in np.argwhere(g == c)]
        steps = abs(disp[k])
        if by_row:
            dr, dc = (1 if disp[k] > 0 else -1), 0
            mop = 21 if disp[k] > 0 else 20
        else:
            dr, dc = 0, (1 if disp[k] > 0 else -1)
            mop = 22 if disp[k] > 0 else 23

        ops.append(mop)
        sels.append(sel_of(cells))            # first step grabs the stripe
        for _ in range(steps - 1):
            ops.append(mop)
            sels.append(sel_of([]))           # empty -> keep gliding it
        g = _apply_move(g, cells, dr, dc, steps)

    # crop out the canvas: a full rectangle, so a bbox selection is exact
    ops.append(33)
    sels.append([tr, tc, n - 1, n - 1])
    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
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
                                f"for task 8e1813be"
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
                    f"Failed to build a complete episode for task 8e1813be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e1813be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
