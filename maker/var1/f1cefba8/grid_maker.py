"""
ARC Task: f1cefba8 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, ringc, inc = random.sample(cols, 3)
    return {"bgc": bgc, "ringc": ringc, "inc": inc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, ringc: int, inc: int) -> dict:
    mh = max(7, int(max_h))
    mw = max(7, int(max_w))
    h = unifint(diff_lb, diff_ub, (7, mh))
    w = unifint(diff_lb, diff_ub, (7, mw))
    ih = unifint(diff_lb, diff_ub, (6, h - 1))
    iw = unifint(diff_lb, diff_ub, (6, w - 1))
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    obj = frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)})
    ring1 = box(obj)
    ring2 = inbox(obj)
    bd = backdrop(obj)
    c = canvas(bgc, (h, w))
    c = fill(c, inc, bd)
    c = fill(c, ringc, ring1 | ring2)
    cands = totuple(ring2 - corners(ring2))
    numc = unifint(diff_lb, diff_ub, (1, len(cands) // 2))
    locs = sample(cands, numc)
    gi = fill(c, inc, locs)
    lm = lowermost(ring2)
    hori = sfilter(locs, lambda ij: ij[0] > loci + 1 and ij[0] < lm)
    verti = difference(locs, hori)
    hlines = mapply(hfrontier, hori)
    vlines = mapply(vfrontier, verti)
    fulllocs = set(hlines) | set(vlines)
    topaintinc = fulllocs & ofcolor(c, bgc)
    topaintringc = fulllocs & ofcolor(c, inc)
    go = fill(c, inc, topaintinc)
    go = fill(go, ringc, topaintringc)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # ---- roles, read off bounding-box extent only (never from O):
    # bgc spans the whole grid, ringc spans the frame box, inc lives inside it
    areas = {}
    for col in np.unique(I):
        rs_, cs_ = np.where(I == col)
        areas[int(col)] = (int(rs_.max()) - int(rs_.min()) + 1) * \
                          (int(cs_.max()) - int(cs_.min()) + 1)
    order = sorted(areas, key=lambda k: areas[k])
    inc, ringc, bgc = order[0], order[1], order[-1]

    rs, cs = np.where(I == ringc)
    r0, r1 = int(rs.min()), int(rs.max())      # outer frame rows
    c0, c1 = int(cs.min()), int(cs.max())      # outer frame cols

    # ---- marks ("pokes"): inc cells sitting on the inner frame ring, corners excluded
    pokes = set()
    for j in range(c0 + 2, c1 - 1):
        for i in (r0 + 1, r1 - 1):
            if I[i, j] == inc:
                pokes.add((i, j))
    for i in range(r0 + 2, r1 - 1):
        for j in (c0 + 1, c1 - 1):
            if I[i, j] == inc:
                pokes.add((i, j))

    row_lines = sorted({i for (i, j) in pokes if j in (c0 + 1, c1 - 1)})
    col_lines = sorted({j for (i, j) in pokes if i in (r0 + 1, r1 - 1)})

    n_r, n_c = len(row_lines), len(col_lines)

    # ---- route choices, all read off the input's own geometry / counts,
    #      so one instance always gives one route and routes differ between instances
    rows_first  = (n_r > n_c) if n_r != n_c else ((r0 + c0) % 2 == 0)
    down        = (r0 <= h - 1 - r1)              # box hugs the top -> work downward
    rightw      = (c0 <= w - 1 - c1)              # box hugs the left -> work rightward
    erase_first = ((len(pokes) + r0 + c1) % 2 == 1)   # clear the marks first, or let beams swallow them
    outward     = ((n_r + n_c + r1 + c0) % 2 == 1)    # draw from the middle out, or sweep straight across
    bulk        = (n_r + n_c) >= 5                # busy grid -> paint each band in one gesture

    ops, sels = [], []
    G = I.copy()

    def paint(color, cells):
        color = int(color)
        sel = [(int(r), int(c)) for (r, c) in cells if G[r, c] != color]
        if not sel:
            return
        ops.append(color)
        sels.append(sel_of(sel))
        for (r, c) in sel:
            G[r, c] = color

    # segment builders: inside the box a beam is ring-coloured, outside it is inc-coloured
    def row_inner(i):  return [(i, j) for j in range(c0 + 1, c1)]
    def row_left(i):   return [(i, j) for j in range(0, c0)]
    def row_right(i):  return [(i, j) for j in range(c1 + 1, w)]
    def col_inner(j):  return [(i, j) for i in range(r0 + 1, r1)]
    def col_top(j):    return [(i, j) for i in range(0, r0)]
    def col_bot(j):    return [(i, j) for i in range(r1 + 1, h)]

    def emit_rows():
        if not row_lines:
            return
        lines = row_lines if down else row_lines[::-1]
        if bulk:
            inner = [p for i in lines for p in row_inner(i)]
            bl = [p for i in lines for p in row_left(i)]
            br = [p for i in lines for p in row_right(i)]
            first, second = (bl, br) if rightw else (br, bl)
            if outward:
                paint(ringc, inner); paint(inc, first); paint(inc, second)
            else:
                paint(inc, first); paint(ringc, inner); paint(inc, second)
            return
        for i in lines:
            inner, bl, br = row_inner(i), row_left(i), row_right(i)
            if outward:
                lp = (i, c0 + 1) in pokes
                rp = (i, c1 - 1) in pokes
                if lp and not rp:
                    first, second = bl, br
                elif rp and not lp:
                    first, second = br, bl
                else:
                    first, second = (bl, br) if rightw else (br, bl)
                paint(ringc, inner); paint(inc, first); paint(inc, second)
            else:
                first, second = (bl, br) if rightw else (br, bl)
                paint(inc, first); paint(ringc, inner); paint(inc, second)

    def emit_cols():
        if not col_lines:
            return
        lines = col_lines if rightw else col_lines[::-1]
        if bulk:
            inner = [p for j in lines for p in col_inner(j)]
            bt = [p for j in lines for p in col_top(j)]
            bb = [p for j in lines for p in col_bot(j)]
            first, second = (bt, bb) if down else (bb, bt)
            if outward:
                paint(ringc, inner); paint(inc, first); paint(inc, second)
            else:
                paint(inc, first); paint(ringc, inner); paint(inc, second)
            return
        for j in lines:
            inner, bt, bb = col_inner(j), col_top(j), col_bot(j)
            if outward:
                tp = (r0 + 1, j) in pokes
                bp = (r1 - 1, j) in pokes
                if tp and not bp:
                    first, second = bt, bb
                elif bp and not tp:
                    first, second = bb, bt
                else:
                    first, second = (bt, bb) if down else (bb, bt)
                paint(ringc, inner); paint(inc, first); paint(inc, second)
            else:
                first, second = (bt, bb) if down else (bb, bt)
                paint(inc, first); paint(ringc, inner); paint(inc, second)

    # some solvers wipe every mark into the frame first, then shoot the beams;
    # others let each beam swallow its own mark on the way through
    if erase_first and pokes:
        paint(ringc, sorted(pokes))

    if rows_first:
        emit_rows(); emit_cols()
    else:
        emit_cols(); emit_rows()

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
                        f"num_examples+1 ({num_examples + 1}) for task f1cefba8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f1cefba8"
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
                                f"for task f1cefba8"
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
                    f"Failed to build a complete episode for task f1cefba8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f1cefba8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
