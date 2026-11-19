"""
ARC Task: e179c5f4 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    from dsl import (canvas, fill, add, identity, dmirror, cmirror, vmirror,
                     hmirror, rot90, rot180, rot270, ofcolor, replace)
    from utils import unifint
    cols = [c for c in range(10) if c != 8]
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    lim = min(30, max_h, max_w)
    w = unifint(diff_lb, diff_ub, (2, min(10, lim - 1)))
    h = unifint(diff_lb, diff_ub, (w + 1, lim))
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    go = tuple(e for e in gi)
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        go2 = fill(go, linc, {sp})
        if go2 == go:
            break
        go = go2
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = random.choice((1, 2))
    for fn in random.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    gix = tuple(e for e in gi)
    gox = tuple(e for e in go)
    numlins = unifint(diff_lb, diff_ub, (1, 4))
    if numlins > 1:
        gi = fill(gi, linc, ofcolor(hmirror(gix), linc))
        go = fill(go, linc, ofcolor(hmirror(gox), linc))
    if numlins > 2:
        gi = fill(gi, linc, ofcolor(vmirror(gix), linc))
        go = fill(go, linc, ofcolor(vmirror(gox), linc))
    if numlins > 3:
        gi = fill(gi, linc, ofcolor(hmirror(vmirror(gix)), linc))
        go = fill(go, linc, ofcolor(hmirror(vmirror(gox)), linc))
    go = replace(go, bgc, 8)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    from collections import Counter
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    cset = set(corners)
    noncorner = [(r, c) for r in range(h) for c in range(w) if (r, c) not in cset]
    bgc = Counter(int(I[r, c]) for r, c in noncorner).most_common(1)[0][0]
    pal = set(int(v) for v in I.flatten().tolist()) - {bgc}
    linc = pal.pop() if pal else None
    grid = I.copy()
    ops, sels = [], []

    # 1. background -> 8 (one connected region)
    seed = noncorner[0]
    ops.append(18); sels.append(sel_of([seed]))
    grid[grid == bgc] = 8

    if linc is None:
        ops.append(34); sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
        return ops, sels

    portrait = h > w

    def trace(r0, c0):
        # bouncing line from a marked corner along the long axis; list of runs
        # (first run = the start corner alone, then one run per diagonal segment)
        if portrait:
            L, S, a, b = h, w, r0, c0
        else:
            L, S, a, b = w, h, c0, r0
        da = -1 if a == L - 1 else 1
        db = 1 if b == 0 else -1
        runs = [[(a, b)]]
        cur = []
        while True:
            a += da; b += db
            if a < 0 or a >= L:
                break
            cur.append((a, b))
            if b == 0 or b == S - 1:
                db = -db
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        if not portrait:
            runs = [[(y, x) for (x, y) in run] for run in runs]
        return runs

    marked = [p for p in corners if I[p] == linc]
    lines = []
    for p in marked:
        runs = trace(*p)
        cells = frozenset(x for run in runs for x in run)
        if cells not in [l[1] for l in lines]:   # identical lines drawn once only
            lines.append((runs, cells))

    def paint_runs(runs):
        for run in runs:
            todo = [p for p in run if grid[p] != linc]
            if todo:
                ops.append(linc); sels.append(sel_of(todo))
                for p in todo:
                    grid[p] = linc

    # 2. draw the first line segment by segment
    base_runs, base_cells = lines[0]
    paint_runs(base_runs)
    drawn = [base_cells]

    def mirror(cells, op):
        if op == 26:
            return frozenset((r, w - 1 - c) for r, c in cells)
        return frozenset((h - 1 - r, c) for r, c in cells)

    # 3. every other distinct line = mirror of an already drawn line
    for runs, cells in lines[1:]:
        done = False
        if linc != 0:
            for src in drawn:
                for fop in (27, 26):
                    if mirror(src, fop) == cells:
                        # flip the drawn line onto its mirror, then restore the source line
                        ops.append(fop); sels.append(sel_of(sorted(src)))
                        for p in src:
                            grid[p] = 0
                        for p in cells:
                            grid[p] = linc
                        hole = sorted(p for p in src if grid[p] != linc)
                        ops.append(linc); sels.append(sel_of(hole))
                        for p in hole:
                            grid[p] = linc
                        done = True
                        break
                if done:
                    break
        if not done:
            paint_runs(runs)
        drawn.append(cells)

    ops.append(34); sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
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
                                f"for task e179c5f4"
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
                    f"Failed to build a complete episode for task e179c5f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e179c5f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
