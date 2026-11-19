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


# ===========================================================================
# structure of the task
#
#   The input carries a "pipe": one vertical band of background colour that
#   runs from some row down to the bottom edge, plus up to two horizontal
#   arms that run out of that band to the left / right edge.  Everything else
#   is two-colour noise that (by construction) never contains a solid 3x3
#   block.  The rule paints the INSIDE of the pipe with 3: the pipe keeps a
#   one-cell wall everywhere it borders the noise, and is filled right up to
#   the grid edges, where there is no wall.
#
#   Every measurement below is taken from the input alone.
# ===========================================================================

VARIANTS = [
    {"sgn_mode": "left",  "top_mode": "inner"},
    {"sgn_mode": "right", "top_mode": "top"},
    {"sgn_mode": "both",  "top_mode": "inner"},
    {"sgn_mode": "both",  "top_mode": "top"},
    {"sgn_mode": "left",  "top_mode": "top"},
    {"sgn_mode": "right", "top_mode": "inner"},
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "noisec": noisec, "instance_plan": plan}


# ------------------------------------------------------------------ detect --
def _pipe_colour(g):
    """The noise never holds a solid 3x3 block; the pipe is full of them."""
    h, w = g.shape
    best_cnt, best_c = -1, int(g[0, 0])
    for c in sorted(set(g.flatten().tolist())):
        m = (g == c)
        cnt = 0
        for r in range(h - 2):
            for cc in range(w - 2):
                if m[r:r + 3, cc:cc + 3].all():
                    cnt += 1
        if cnt > best_cnt:
            best_cnt, best_c = cnt, c
    return best_c


def _detect(I):
    """Read the pipe off the input and return the rectangles to fill with 3.

    rects[0] is the vertical band's interior, the rest are the arms'."""
    g = np.asarray(I, dtype=int)
    h, w = g.shape
    bgc = _pipe_colour(g)
    m = (g == bgc)

    # the band is the only group of columns solid from the bottom edge up
    # through at least half of the grid
    bot = []
    for j in range(w):
        k = 0
        for r in range(h - 1, -1, -1):
            if m[r, j]:
                k += 1
            else:
                break
        bot.append(k)
    thr = (h + 1) // 2
    best, run = [], []
    for j in range(w):
        if bot[j] >= thr:
            run.append(j)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    if len(best) < 3:
        raise ValueError("no vertical band found")
    bc0, bc1 = best[0], best[-1]
    top = max(h - bot[j] for j in best)

    # band interior: side walls kept, top wall kept unless it is the grid edge,
    # filled down to the bottom edge
    rects = [(top + 1 if top > 0 else 0, bc0 + 1, h - 1, bc1 - 1)]

    # an arm row is a row that is solid bgc from the band all the way to an edge
    for side in (-1, 1):
        if side == -1 and bc0 == 0:
            continue
        if side == 1 and bc1 == w - 1:
            continue
        flags = []
        for r in range(h):
            if side == -1:
                flags.append(bool(m[r, 0:bc1 + 1].all()))
            else:
                flags.append(bool(m[r, bc0:w].all()))
        r = 0
        while r < h:
            if not flags[r]:
                r += 1
                continue
            s = r
            while r < h and flags[r]:
                r += 1
            e = r - 1
            if e - s + 1 < 3:
                continue
            a = s + 1 if s > 0 else 0          # top wall, unless it is the edge
            b = e - 1                          # bottom wall
            if b < a:
                continue
            if side == -1:
                rects.append((a, 0, b, bc1 - 1))
            else:
                rects.append((a, bc0 + 1, b, w - 1))
    return rects


# ---------------------------------------------------------------- generate --
def _build(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sgn_mode, top_mode):
    hlo, wlo = min(18, max_h), min(18, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))
    lb, ub = int(0.4 * h * w), int(0.5 * h * w)
    nbgc = unifint(diff_lb, diff_ub, (lb, ub))
    gi = canvas(noisec, (h, w))
    inds = totuple(asindices(gi))
    gi = fill(gi, bgc, random.sample(inds, nbgc))
    sinds = asindices(canvas(-1, (3, 3)))
    addn, addb = set(), set()
    for occ in occurrences(gi, recolor(bgc, sinds)):
        addn.add((random.randint(0, 2) + occ[0], random.randint(0, 2) + occ[1]))
    for occ in occurrences(gi, recolor(noisec, sinds)):
        addb.add((random.randint(0, 2) + occ[0], random.randint(0, 2) + occ[1]))
    gi = fill(gi, noisec, addn)
    gi = fill(gi, bgc, addb)
    go = tuple(e for e in gi)

    dmax = min(8, min(h, w) - 10)
    if dmax < 3:
        dmax = 3
    dim = random.randint(random.randint(3, dmax), dmax)
    jhi = min(h, w) - dim - 4                 # keep the band inside the grid
    if jhi < 3:
        jhi = 3
    locj = random.randint(3, jhi)

    if top_mode == "top":
        spi = 0
    elif top_mode == "inner":
        spi = random.randint(3, max(3, h // 2))
    else:
        spi = random.choice((0, random.randint(3, max(3, h // 2))))

    for j in range(locj, locj + dim):
        ln = connect((spi, j), (h - 1, j))
        gi = fill(gi, bgc, ln)
        go = fill(go, bgc, ln)
    for j in range(locj + 1, locj + dim - 1):
        ln = connect((spi + 1 if spi > 0 else spi, j), (h - 1, j))
        go = fill(go, 3, ln)

    def arm(g_in, g_out, sgn):
        if spi + 3 <= h - 6:
            startloc = random.choice((spi, random.randint(spi + 3, h - 6)))
        else:
            startloc = spi
        hh = random.randint(3, max(3, min(8, h - startloc - 3)))
        for ii in range(startloc, startloc + hh):
            ln = shoot((ii, locj), (0, sgn))
            g_in = fill(g_in, bgc, ln)
            g_out = fill(g_out, bgc, ln - ofcolor(g_out, 3))
        base = locj + dim - 2 if sgn == -1 else locj + 1
        for ii in range(startloc + 1 if startloc > 0 else startloc,
                        startloc + hh - 1):
            ln = shoot((ii, base), (0, sgn))
            g_out = fill(g_out, 3, ln)
        return g_in, g_out

    if sgn_mode == "left":
        sides = (-1,)
    elif sgn_mode == "right":
        sides = (1,)
    elif sgn_mode == "both":
        sides = (-1, 1)
    else:
        sides = random.choice(((-1,), (1,), (-1, 1)))
        if len(sides) == 1 and unifint(diff_lb, diff_ub, (0, 1)) == 1:
            sides = (sides[0], -sides[0])
    for sgn in sides:
        gi, go = arm(gi, go, sgn)

    return {'input': gi, 'output': go}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec,
             sgn_mode=None, top_mode=None) -> dict:
    last = None
    for _ in range(400):
        out = _build(diff_lb, diff_ub, max_h, max_w, bgc, noisec,
                     sgn_mode, top_mode)
        last = out
        # keep only instances whose pipe can actually be read off the input:
        # noise sitting flush against a wall could otherwise make the pipe look
        # one row / one column bigger than it is, and no solver could tell.
        try:
            rects = _detect(out['input'])
        except Exception:
            continue
        chk = out['input']
        for (r0, c0, r1, c1) in rects:
            chk = fill(chk, 3, frozenset((r, c) for r in range(r0, r1 + 1)
                                         for c in range(c0, c1 + 1)))
        if chk == out['output']:
            return out
    return last


# ------------------------------------------------------------------ derive --
def derive_operations(I, O):
    """Pour 3 through the pipe.

    Two solvers doing this differ in exactly two places, and both are settled
    by the input: which channel they clear first, and which way they run the
    fill along a channel.  Here the widest channel goes first (the main run
    before the branches), and each channel is filled cross-section by
    cross-section starting at its closed end -- the band's top wall, an arm's
    junction with the band -- and advancing towards the open grid edge.  A
    cross-section an earlier channel already filled is simply not drawn again.
    Nothing else about the grid changes, so there is no background to lay down
    first, and no colour-0 cell is ever carried anywhere (only Color3 is used).
    """
    g = np.asarray(I, dtype=int)
    h, w = g.shape
    rects = _detect(g)                 # band + arms, measured from I alone

    channels = []
    for k, (r0, c0, r1, c1) in enumerate(rects):
        if k == 0:
            # the band: walled at the top (unless it starts at the edge),
            # open at the bottom edge -> the fill runs downward, a row at a time
            secs = [[(r, c) for c in range(c0, c1 + 1)]
                    for r in range(r0, r1 + 1)]
        else:
            cols = list(range(c0, c1 + 1))
            if c0 == 0:                # left arm: it leaves the band rightwards
                cols.reverse()
            secs = [[(r, c) for r in range(r0, r1 + 1)] for c in cols]
        channels.append((secs, (r1 - r0 + 1) * (c1 - c0 + 1), (r0, c0)))
    channels.sort(key=lambda t: (-t[1], t[2]))

    cur = g.copy()
    ops, sels = [], []
    for secs, _, _ in channels:
        for cells in secs:
            if all(cur[r, c] == 3 for (r, c) in cells):
                continue               # nothing left to draw here
            ops.append(3)              # Color3 over this slice of the channel
            sels.append(sel_of(cells))
            for (r, c) in cells:
                cur[r, c] = 3

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # full-grid rectangle: submit
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
