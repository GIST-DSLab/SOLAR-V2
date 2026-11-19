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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc, dotc = random.sample(cols, 3)
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, dotc) -> dict:
    max_h = max(8, min(int(max_h), 30))
    max_w = max(8, min(int(max_w), 30))

    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
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


def derive_operations(I, O, examples=None):
    """
    Reading: the dots of the dot colour form ONE set that is pulled onto the
    outbox (one-cell ring) of the solid block. Each side's dots travel as a
    single group: all dots left of the block slide right together, those
    above slide down, etc. A group is grabbed once; when its nearest member
    lands on the ring, the remaining members are re-grabbed and keep going.
    Diagonal (corner) dots slide in two legs onto their ring corner.
    Every cell a dot was lifted from (its start, and where a side group was
    re-grabbed) is then covered with the background in one go.
    If the dot colour is 0 (a Move can't carry 0), the set is covered with the
    background and the ring cells nearest each dot are painted 0.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    cells_of = {}
    for r in range(h):
        for c in range(w):
            cells_of.setdefault(int(I[r, c]), []).append((r, c))

    def is_rect(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    rect_cols = [c for c, cl in cells_of.items() if is_rect(cl)]
    sqc = max(rect_cols, key=lambda c: len(cells_of[c]))
    others = [c for c in cells_of if c != sqc]
    bgc = max(others, key=lambda c: len(cells_of[c]))
    dotc = max([c for c in others if c != bgc], key=lambda c: len(cells_of[c]))

    block = cells_of[sqc]
    r0 = min(p[0] for p in block); r1 = max(p[0] for p in block)
    c0 = min(p[1] for p in block); c1 = max(p[1] for p in block)

    ring = [(r, c)
            for r in range(r0 - 1, r1 + 2)
            for c in range(c0 - 1, c1 + 2)
            if (r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1))
            and 0 <= r < h and 0 <= c < w]
    corners = {(r0 - 1, c0 - 1), (r0 - 1, c1 + 1),
               (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)}

    def nearest_ring(cell):
        sr, sc = cell
        return min(ring, key=lambda t: (abs(t[0] - sr) + abs(t[1] - sc), t[0], t[1]))

    dot_cells = sorted(cells_of[dotc])
    pairs = [(p, nearest_ring(p)) for p in dot_cells]

    ops, sels = [], []
    lifted = set()   # every cell a dot was grabbed from (ARCLE leaves 0 there)

    if dotc == 0:
        ops.append(int(bgc)); sels.append(sel_of(dot_cells))
        targets = sorted({t for _, t in pairs})
        ops.append(0); sels.append(sel_of(targets))
        ops.append(34); sels.append([0, 0, h - 1, w - 1])  # whole grid
        return ops, sels

    # side groups: (move op, unit step, predicate on target)
    sides = [
        (21, (1, 0), lambda t: t[0] == r0 - 1),   # above -> slide down
        (23, (0, -1), lambda t: t[1] == c1 + 1),  # right -> slide left
        (20, (-1, 0), lambda t: t[0] == r1 + 1),  # below -> slide up
        (22, (0, 1), lambda t: t[1] == c0 - 1),   # left  -> slide right
    ]
    for op, (dr, dc), pred in sides:
        group = [(s, t) for s, t in pairs if t not in corners and pred(t)]
        if not group:
            continue
        cur = {s: s for s, _ in group}
        dist = {s: abs(t[0] - s[0]) + abs(t[1] - s[1]) for s, t in group}
        prev_active = None
        for step in range(1, max(dist.values()) + 1):
            active = [s for s, _ in group if dist[s] >= step]
            if active != prev_active:
                ops.append(op); sels.append(sel_of([cur[s] for s in active]))
                lifted.update(cur[s] for s in active)
            else:
                ops.append(op); sels.append(sel_of([]))
            prev_active = active
            for s in active:
                cur[s] = (cur[s][0] + dr, cur[s][1] + dc)

    for s, t in sorted([(s, t) for s, t in pairs if t in corners], key=lambda x: x[1]):
        dr, dc = t[0] - s[0], t[1] - s[1]
        cur = s
        lifted.add(s)
        grabbed = False
        for _ in range(abs(dr)):
            ops.append(20 if dr < 0 else 21)
            sels.append(sel_of([cur]) if not grabbed else sel_of([]))
            grabbed = True
            cur = (cur[0] + (-1 if dr < 0 else 1), cur[1])
        for _ in range(abs(dc)):
            ops.append(23 if dc < 0 else 22)
            sels.append(sel_of([cur]) if not grabbed else sel_of([]))
            grabbed = True
            cur = (cur[0], cur[1] + (-1 if dc < 0 else 1))

    # every grab left a 0 where the dot was lifted (its start cell, and the
    # cell where the rest of a side group was picked up again); cover them all
    if bgc != 0:
        ops.append(int(bgc)); sels.append(sel_of(sorted(lifted)))

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
