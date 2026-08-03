"""
ARC Task: 673ef223 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, barc, dotc = random.sample(cols, 3)
    return {"bgc": bgc, "barc": barc, "dotc": dotc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, barc=None, dotc=None) -> dict:
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    swapping = (dmirror, cmirror, rot90, rot270)
    nmfs = choice((1, 2))
    fns = sample(mfs, nmfs)
    swap = (sum(1 for fn in fns if fn in swapping) % 2) == 1
    h_ub = max(5, max_w if swap else max_h)
    w_ub = max(5, max_h if swap else max_w)
    h = unifint(diff_lb, diff_ub, (5, h_ub))
    w = unifint(diff_lb, diff_ub, (5, w_ub))
    barh = unifint(diff_lb, diff_ub, (2, (h - 1) // 2))
    ncells = unifint(diff_lb, diff_ub, (1, barh))
    sg = canvas(bgc, (barh, w))
    topsgi = fill(sg, barc, connect((0, 0), (barh - 1, 0)))
    botsgi = vmirror(topsgi)
    topsgo = tuple(e for e in topsgi)
    botsgo = tuple(e for e in botsgi)
    iloccands = interval(0, barh, 1)
    ilocs = sample(iloccands, ncells)
    for k in ilocs:
        jloc = randint(2, w - 2)
        topsgi = fill(topsgi, dotc, {(k, jloc)})
        topsgo = fill(topsgo, 4, {(k, jloc)})
        topsgo = fill(topsgo, dotc, connect((k, 1), (k, jloc - 1)))
        botsgo = fill(botsgo, dotc, connect((k, 0), (k, w - 2)))
    outpi = (topsgi, botsgi)
    outpo = (topsgo, botsgo)
    rr = canvas(bgc, (1, w))
    while len(merge(outpi)) < h:
        idx = randint(0, len(outpi) - 1)
        outpi = outpi[:idx] + (rr,) + outpi[idx:]
        outpo = outpo[:idx] + (rr,) + outpo[idx:]
    gi = merge(outpi)
    go = merge(outpo)
    for fn in fns:
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read off I alone):
      * two straight bars of one colour sit on opposite grid borders (all their
        cells are on the border, each bar has >= 2 cells).
      * single cells of the other colour ("dots") lie inside the span of ONE of
        the bars (the "dot bar").
      * each dot shoots a beam back to its bar: every cell strictly between the
        bar and the dot becomes the dot colour, and the dot itself becomes 4.
      * the twin bar answers: the dot, carried over by the offset between the two
        bars' upper-left corners, emits a FULL line across the grid (everything
        except bar cells) at that position.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- connected components of non-background cells (4-conn, single colour) --
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((col, cells))

    if not comps:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    def on_border(p):
        return p[0] == 0 or p[1] == 0 or p[0] == hi - 1 or p[1] == wi - 1

    # --- bar colour: exactly two objects, all cells on the border, none a single cell
    barc = None
    for col in sorted({c for c, _ in comps}):
        groups = [cells for c2, cells in comps if c2 == col]
        if (len(groups) == 2 and all(len(g) >= 2 for g in groups)
                and all(on_border(p) for g in groups for p in g)):
            barc = col
            break
    if barc is None:
        barc = max(comps, key=lambda t: len(t[1]))[0]

    others = [c for c in sorted({c for c, _ in comps}) if c != barc]
    if not others:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    dotc = others[0]

    bars = [cells for col, cells in comps if col == barc]
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]
    if len(bars) != 2 or not dots:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    bar_rows = {p[0] for b in bars for p in b}
    bar_cols = {p[1] for b in bars for p in b}
    vertical = len(bar_rows) > len(bar_cols)   # bars run down columns

    def ul(b):
        return (min(p[0] for p in b), min(p[1] for p in b))

    if vertical:
        dotbar = next((b for b in bars if {p[0] for p in b} & {d[0] for d in dots}), bars[0])
    else:
        dotbar = next((b for b in bars if {p[1] for p in b} & {d[1] for d in dots}), bars[0])
    otherbar = bars[1] if dotbar is bars[0] else bars[0]
    shift = (ul(otherbar)[0] - ul(dotbar)[0]) if vertical else (ul(otherbar)[1] - ul(dotbar)[1])

    def runs(vals):
        out = []
        for v in sorted(vals):
            if out and v == out[-1][1] + 1:
                out[-1][1] = v
            else:
                out.append([v, v])
        return out

    # dots taken in order along the bar; each dot's beam, its head, and the twin
    # bar's answering line are emitted together.
    dots.sort(key=(lambda d: d[0]) if vertical else (lambda d: d[1]))

    if vertical:
        cb = ul(dotbar)[1]
        for (r, c) in dots:
            lo, hi_ = (cb + 1, c - 1) if c > cb else (c + 1, cb - 1)
            if lo <= hi_:
                ops.append(int(dotc)); sels.append([r, lo, 0, hi_ - lo])
            ops.append(4); sels.append([r, c, 0, 0])
            rr = r + shift
            if 0 <= rr < hi:
                free = [cc for cc in range(wi) if I[rr, cc] != barc]
                for a, b in runs(free):
                    ops.append(int(dotc)); sels.append([rr, a, 0, b - a])
    else:
        rb = ul(dotbar)[0]
        for (r, c) in dots:
            lo, hi_ = (rb + 1, r - 1) if r > rb else (r + 1, rb - 1)
            if lo <= hi_:
                ops.append(int(dotc)); sels.append([lo, c, hi_ - lo, 0])
            ops.append(4); sels.append([r, c, 0, 0])
            cc0 = c + shift
            if 0 <= cc0 < wi:
                free = [rr for rr in range(hi) if I[rr, cc0] != barc]
                for a, b in runs(free):
                    ops.append(int(dotc)); sels.append([a, cc0, b - a, 0])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 673ef223"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 673ef223"
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
                                f"for task 673ef223"
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
                    f"Failed to build a complete episode for task 673ef223 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"673ef223-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
