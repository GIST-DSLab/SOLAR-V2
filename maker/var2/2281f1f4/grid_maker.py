"""
ARC Task: 2281f1f4 (RE-ARC) — LLM-generated grid_maker
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
    """bgc = canvas colour, dc = the marker colour on the two border lines.
    2 is the colour the rule paints with, so it is excluded from both roles
    (exactly as the original generator's `colopts` does).
    The four rigid rotations are a discrete structural variant (which two
    borders carry the lines), so they are planned per instance."""
    colopts = [c for c in range(10) if c != 2]
    bgc = random.choice(colopts)
    dc = random.choice([c for c in colopts if c != bgc])

    variants = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(variants, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "dc": dc, "instance_plan": plan}


# -------------------------------------------------------------- generator ---
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, dc=None, rot=None) -> dict:
    colopts = remove(2, interval(0, 10, 1))
    if bgc is None:
        bgc = choice(colopts)
    if dc is None:
        dc = choice(remove(bgc, colopts))
    if rot is None:
        rot = choice((0, 1, 2, 3))

    rotf = (identity, rot90, rot180, rot270)[rot]
    swap = rot in (1, 3)                       # rot90/rot270 transpose the shape
    hub = min(max_h, max_w) if swap else max_h
    wub = min(max_h, max_w) if swap else max_w

    h = unifint(diff_lb, diff_ub, (3, max(3, hub)))
    w = unifint(diff_lb, diff_ub, (3, max(3, wub)))
    card_h_bounds = (1, h // 2 + 1)
    card_w_bounds = (1, w // 2 + 1)
    numtop = unifint(diff_lb, diff_ub, card_w_bounds)
    numright = unifint(diff_lb, diff_ub, card_h_bounds)
    if numtop == numright == 1:
        numtop, numright = sample([1, 2], 2)
    tp = sample(interval(0, w - 1, 1), numtop)
    rp = sample(interval(1, h, 1), numright)
    res = combine(apply(lbind(astuple, 0), tp), apply(rbind(astuple, w - 1), rp))
    gi = fill(canvas(bgc, (h, w)), dc, res)
    go = fill(gi, 2, product(rp, tp))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ------------------------------------------------------------ derivation ----
def derive_operations(I, O, examples=None):
    """
    Rule (read off I alone):
      one marker colour sits on two perpendicular border lines -- a border ROW
      and a border COLUMN.  Their meeting grid-corner is the background corner
      sharing its row/column with the most markers.  Every marker on the corner's
      column projects the whole border-row marker pattern into its own row; those
      intersections become 2.

    The ROUTE through that rule (which line first, which way the sweep runs,
    whole line vs contiguous blocks vs stamp-once-then-copy) is decided from
    features of I only, so the same instance always gives the same route and the
    variety comes from the instances differing.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    submit = ([0, 0, h - 1, w - 1])

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]                      # canvas colour
    fg = [c for c in cnt if c != bgc]
    if not fg:
        ops.append(34); sels.append(submit)
        return ops, sels
    mc = min(fg, key=lambda c: (cnt[c], c))             # marker colour (leastcolor)
    marks = [(r, c) for r in range(h) for c in range(w) if I[r, c] == mc]

    # -- the corner where the two marker lines meet ---------------------------
    best, corner = -1, None
    for cr in (0, h - 1):
        for cc in (0, w - 1):
            if I[cr, cc] != bgc:
                continue
            k = sum(1 for (r, c) in marks if r == cr or c == cc)
            if k > best:
                best, corner = k, (cr, cc)
    if corner is None:
        ops.append(34); sels.append(submit)
        return ops, sels
    cr, cc = corner

    proj_rows = sorted({r for (r, c) in marks if c == cc and r != cr})
    pat_cols = sorted({c for (r, c) in marks if r == cr and c != cc})
    if not proj_rows or not pat_cols:
        ops.append(34); sels.append(submit)
        return ops, sels
    rmin, rmax = proj_rows[0], proj_rows[-1]
    cmin, cmax = pat_cols[0], pat_cols[-1]

    # -- route, decided from the picture -------------------------------------
    sig = (h * 31 + w * 17 + int(mc) * 7 + int(bgc) * 13 + len(marks) * 5
           + len(pat_cols) * 3 + len(proj_rows) * 11
           + (23 if cr else 0) + (29 if cc else 0))
    axis = sig % 2              # 0: work row by row      1: work column by column
    rev = (sig // 2) % 2        # 0: sweep away from the corner   1: sweep toward it
    style = (sig // 4) % 3      # 0: whole line  1: block by block  2: stamp & copy

    row_order = sorted(proj_rows, key=lambda r: abs(r - cr))
    col_order = sorted(pat_cols, key=lambda c: abs(c - cc))
    if axis == 0:
        lines = list(row_order)
        anchor, cross = cc, col_order
        def cells_of(l, poss):
            return [(l, p) for p in poss]
    else:
        lines = list(col_order)
        anchor, cross = cr, row_order
        def cells_of(l, poss):
            return [(p, l) for p in poss]
    if rev:
        lines.reverse()

    G = I.copy()

    def paint(cells, color):
        cells = [(r, c) for (r, c) in cells if G[r, c] != color]
        if not cells:
            return
        ops.append(color)
        sels.append(sel_of(cells))
        for (r, c) in cells:
            G[r, c] = color

    if style == 2 and len(lines) >= 2:
        # stamp the first line by hand, then carry that stamp to the others
        first = lines[0]
        paint(cells_of(first, cross), 2)
        if axis == 0:
            strip = [(first, c) for c in range(cmin, cmax + 1)]   # full 1-row rect
            src = (first, cmin)
            def origin_of(l):
                return (l, cmin)
        else:
            strip = [(r, first) for r in range(rmin, rmax + 1)]   # full 1-col rect
            src = (rmin, first)
            def origin_of(l):
                return (rmin, l)
        ops.append(29)                                   # CopyO the stamped line
        sels.append(sel_of(strip))
        clip = [(r - src[0], c - src[1], int(G[r, c])) for (r, c) in strip]
        for l in lines[1:]:
            orow, ocol = origin_of(l)
            landing = [(orow + dr, ocol + dc, v) for (dr, dc, v) in clip
                       if v != 0 and G[orow + dr, ocol + dc] != v]
            if landing:
                ops.append(30)
                sels.append(sel_of([(orow, ocol)]))
                for (r, c, v) in landing:
                    G[r, c] = v
            # Paste is blind to colour 0: lay any 0 of the stamp down explicitly
            zeros = [(orow + dr, ocol + dc) for (dr, dc, v) in clip
                     if v == 0 and G[orow + dr, ocol + dc] != 0]
            if zeros:
                ops.append(0)
                sels.append(sel_of(zeros))
                for (r, c) in zeros:
                    G[r, c] = 0
    elif style == 1:
        # block by block: contiguous stretches of the pattern, one op each
        blocks = []
        run = [cross_p for cross_p in sorted(cross)]
        cur = [run[0]]
        for p in run[1:]:
            if p == cur[-1] + 1:
                cur.append(p)
            else:
                blocks.append(cur); cur = [p]
        blocks.append(cur)
        blocks.sort(key=lambda b: min(abs(p - anchor) for p in b), reverse=bool(rev))
        for l in lines:
            for b in blocks:
                paint(cells_of(l, b), 2)
    else:
        # one op per projected line: the whole pattern stamped at once
        for l in lines:
            paint(cells_of(l, cross), 2)

    ops.append(34)
    sels.append(submit)
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
                        f"num_examples+1 ({num_examples + 1}) for task 2281f1f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2281f1f4"
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
                                f"for task 2281f1f4"
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
                    f"Failed to build a complete episode for task 2281f1f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2281f1f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
