"""
ARC Task: a48eeaf7 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------------------
# episode-level colour roles
# ---------------------------------------------------------------------------
# The rule (dots slide onto the one-cell ring around the block) is colour
# independent, but ARCLE's object ops treat 0 as "nothing", so the colour that
# actually gets GRABBED and MOVED (the dots) must not be 0.
VARIANTS = [{"ncorn": 0}, {"ncorn": 2}, {"ncorn": 4}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    dotc = random.choice([c for c in cols if c not in (bgc, sqc) and c != 0])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "sqc": sqc, "dotc": dotc, "instance_plan": plan}


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, dotc, ncorn=None) -> dict:
    hub = min(30, max(8, max_h))
    wub = min(30, max(8, max_w))
    h = unifint(diff_lb, diff_ub, (8, hub))
    w = unifint(diff_lb, diff_ub, (8, wub))
    ih = unifint(diff_lb, diff_ub, (2, h // 2))
    iw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = randint(2, h - ih - 2)
    locj = randint(2, w - iw - 2)

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    sq = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))

    A = [(x, locj - 1) for x in interval(loci, loci + ih, 1)]
    Ap = [(x, randint(0, locj - 2)) for x in interval(loci, loci + ih, 1)]
    B = [(x, locj + iw) for x in interval(loci, loci + ih, 1)]
    Bp = [(x, randint(locj + iw + 1, w - 1)) for x in interval(loci, loci + ih, 1)]
    C = [(loci - 1, x) for x in interval(locj, locj + iw, 1)]
    Cp = [(randint(0, loci - 2), x) for x in interval(locj, locj + iw, 1)]
    D = [(loci + ih, x) for x in interval(locj, locj + iw, 1)]
    Dp = [(randint(loci + ih + 1, h - 1), x) for x in interval(locj, locj + iw, 1)]

    srarr = Ap + Bp + Cp + Dp
    dearr = A + B + C + D
    inds = interval(0, len(srarr), 1)
    num = unifint(diff_lb, diff_ub, (1, len(srarr)))
    locs = sample(inds, num)
    srarr = [e for j, e in enumerate(srarr) if j in locs]
    dearr = [e for j, e in enumerate(dearr) if j in locs]

    gi = fill(gi, sqc, sq)
    go = fill(go, sqc, sq)
    for s, d in zip(srarr, dearr):
        gi = fill(gi, dotc, {s})
        go = fill(go, dotc, {d})

    if ncorn is None:
        ncorn = unifint(diff_lb, diff_ub, (0, 4))
    fullinds = asindices(gi)
    if ncorn > 0:
        go = fill(go, dotc, {(loci - 1, locj - 1)})
        cands = shoot((loci - 2, locj - 2), (-1, -1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 1:
        go = fill(go, dotc, {(loci - 1, locj + iw)})
        cands = shoot((loci - 2, locj + iw + 1), (-1, 1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 2:
        go = fill(go, dotc, {(loci + ih, locj - 1)})
        cands = shoot((loci + ih + 1, locj - 2), (1, -1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 3:
        go = fill(go, dotc, {(loci + ih, locj + iw)})
        cands = shoot((loci + ih + 1, locj + iw + 1), (1, 1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})

    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------
def derive_operations(I, O, examples=None):
    """
    Rule (read off I alone):
      one solid rectangular block plus scattered single dots of a second colour.
      Every dot slides onto the nearest cell of the one-cell-wide ring that hugs
      the block -- straight in, or around one corner when the dot sits on a
      diagonal.  Each dot is GRABBED and MOVED cell by cell; the cell it left is
      repainted with the background.

      The rule is fixed, the ROUTE is not: where you start round the ring, which
      way round you go, whether you pull the near ones or the far ones in first,
      which leg of an L you walk first, and whether you tidy the vacated cells as
      you go or all at the end.  Every one of those choices is read off the input
      (block position, block shape, the two object colours, the dot count), so a
      given input always yields the same route and the variety comes from the
      instances differing.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # ---- colour roles ------------------------------------------------------
    cells_of = {}
    for r in range(h):
        for c in range(w):
            cells_of.setdefault(int(I[r, c]), []).append((r, c))

    bgc = max(cells_of, key=lambda c: len(cells_of[c]))
    cand = [c for c in cells_of if c != bgc]
    if len(cand) < 2:                      # no dots to move
        return [34], [[0, 0, h - 1, w - 1]]

    def solid_rect(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        hh = max(rs) - min(rs) + 1
        ww = max(cs) - min(cs) + 1
        return hh >= 2 and ww >= 2 and len(cells) == hh * ww

    rects = [c for c in cand if solid_rect(cells_of[c])]
    sqc = rects[0] if len(rects) == 1 else max(cand, key=lambda c: len(cells_of[c]))
    dotc = max([c for c in cand if c != sqc], key=lambda c: len(cells_of[c]))

    blk = cells_of[sqc]
    r0 = min(p[0] for p in blk); r1 = max(p[0] for p in blk)
    c0 = min(p[1] for p in blk); c1 = max(p[1] for p in blk)
    bh, bw = r1 - r0 + 1, c1 - c0 + 1

    # ---- the ring the dots are attracted to, walked as one closed loop -----
    tr0, tr1, tc0, tc1 = r0 - 1, r1 + 1, c0 - 1, c1 + 1
    loop = []
    for c in range(tc0, tc1 + 1):
        loop.append((tr0, c))
    for r in range(tr0 + 1, tr1 + 1):
        loop.append((r, tc1))
    for c in range(tc1 - 1, tc0 - 1, -1):
        loop.append((tr1, c))
    for r in range(tr1 - 1, tr0, -1):
        loop.append((r, tc0))
    loop = [p for p in loop if 0 <= p[0] < h and 0 <= p[1] < w]
    pidx = {p: i for i, p in enumerate(loop)}
    P = len(loop)

    def nearest_ring(cell):
        sr, sc = cell
        return min(loop, key=lambda t: (abs(t[0] - sr) + abs(t[1] - sc), t[0], t[1]))

    dots = [(p, nearest_ring(p)) for p in cells_of[dotc]]
    dots = [(s, t) for s, t in dots if s != t]
    if not dots:
        return [34], [[0, 0, h - 1, w - 1]]

    # ---- route choices, all read off the input ----------------------------
    clockwise = ((sqc + dotc) % 2 == 0)                     # which way round
    corners = [(tr0, tc0), (tr0, tc1), (tr1, tc1), (tr1, tc0)]
    start = corners[(r0 + c0) % 4]                          # where to start
    s = pidx.get(start, 0)
    mode = (len(dots) + c0) % 3                             # sweep / near / far
    if bh != bw:                                            # walk the long axis first
        vertical_first = bh > bw
    else:
        vertical_first = (dotc % 2 == 0)
    tidy_at_end = ((bgc + len(dots)) % 2 == 0)              # tidy now or later

    def seq_rank(t):
        i = pidx[t]
        return (i - s) % P if clockwise else (s - i) % P

    def sort_key(item):
        (sr, sc), (tr, tc) = item
        d = abs(tr - sr) + abs(tc - sc)
        rk = seq_rank((tr, tc))
        if mode == 0:
            return (rk, d)
        if mode == 1:
            return (d, rk)
        return (-d, rk)

    dots.sort(key=sort_key)

    # ---- emit --------------------------------------------------------------
    ops, sels = [], []
    holes = []
    for (sr, sc), (tr, tc) in dots:
        dr, dc = tr - sr, tc - sc
        vleg = [20 if dr < 0 else 21] * abs(dr)
        hleg = [23 if dc < 0 else 22] * abs(dc)
        seq = (vleg + hleg) if vertical_first else (hleg + vleg)
        first = True
        for op in seq:
            ops.append(op)
            # first step GRABS this dot; every further step of the same slide
            # carries an empty selection so ARCLE keeps it grabbed and restores
            # the cells it glides over.
            sels.append(sel_of([(sr, sc)]) if first else sel_of([]))
            first = False
        if bgc != 0:
            if tidy_at_end:
                holes.append((sr, sc))
            else:
                ops.append(int(bgc))
                sels.append(sel_of([(sr, sc)]))
    if bgc != 0 and holes:
        ops.append(int(bgc))
        sels.append(sel_of(sorted(set(holes))))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox == whole grid, intentional
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
                        f"num_examples+1 ({num_examples + 1}) for task a48eeaf7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a48eeaf7"
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
                                f"for task a48eeaf7"
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
                    f"Failed to build a complete episode for task a48eeaf7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a48eeaf7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
