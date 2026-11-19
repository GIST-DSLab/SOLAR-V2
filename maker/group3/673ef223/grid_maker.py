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
from maker.sel_helpers import sel_of
try:
    from dsl import *
    from utils import *
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, barc, dotc = random.sample(cols, 3)
    return {"bgc": bgc, "barc": barc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, dotc) -> dict:
    lim = min(max_h, max_w)   # mirrors/rotations may swap h and w
    h = unifint(diff_lb, diff_ub, (5, lim))
    w = unifint(diff_lb, diff_ub, (5, lim))
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
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Same rule as the reference maker, carved by PHASE over the whole grid instead
    of dot by dot:
      1. every beam at once  (all cells between the dot bar and each dot)  -> dotc
      2. every dot at once   (all cells of the dot colour in I)            -> 4
      3. every twin line at once: the full rows/cols opposite the twin bar
         (the dot positions carried over by the bar-to-bar offset)         -> dotc
      4. the twin bar's cells that step 3 painted over are restored        -> barc
    Nothing is read from O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    full = [0, 0, hi - 1, wi - 1]

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c]); seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                a, b = stack.pop(); cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True; stack.append((na, nb))
            comps.append((col, cells))

    def on_border(p):
        return p[0] in (0, hi - 1) or p[1] in (0, wi - 1)

    colors = sorted({c for c, _ in comps})
    barc = None
    for col in colors:
        g = [cells for c2, cells in comps if c2 == col]
        if len(g) == 2 and all(len(x) >= 2 for x in g) and all(on_border(p) for x in g for p in x):
            barc = col
            break
    others = [c for c in colors if c != barc]
    if barc is None or not others:
        ops.append(34); sels.append(full)
        return ops, sels
    dotc = others[0]
    bars = [cells for c2, cells in comps if c2 == barc]
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]

    vertical = len({p[0] for b in bars for p in b}) > len({p[1] for b in bars for p in b})
    ax = 0 if vertical else 1          # axis along which the bars run
    dot_axis = {d[ax] for d in dots}
    dotbar = next((b for b in bars if {p[ax] for p in b} & dot_axis), bars[0])
    twin = bars[1] if dotbar is bars[0] else bars[0]
    shift = min(p[ax] for p in twin) - min(p[ax] for p in dotbar)
    bar_line = dotbar[0][1 - ax]       # the column (or row) the dot bar sits on

    def cell(along, across):
        return (along, across) if vertical else (across, along)

    # 1. all beams, one stroke
    beams = []
    for d in dots:
        a, x = d[ax], d[1 - ax]
        lo, hi_ = (bar_line + 1, x - 1) if x > bar_line else (x + 1, bar_line - 1)
        beams += [cell(a, k) for k in range(lo, hi_ + 1)]
    if beams:
        ops.append(dotc); sels.append(sel_of(beams))

    # 2. all dot heads -> 4 (all cells of the dot colour in I)
    ops.append(4); sels.append(sel_of(dots))

    # 3. all twin lines, full width (base layer)
    n_across = wi if vertical else hi
    n_along = hi if vertical else wi
    lines = sorted({d[ax] + shift for d in dots if 0 <= d[ax] + shift < n_along})
    line_cells = [cell(a, k) for a in lines for k in range(n_across)]
    if line_cells:
        ops.append(dotc); sels.append(sel_of(line_cells))
        # 4. restore the twin bar where those lines crossed it
        lineset = set(lines)
        crossed = [p for b in bars for p in b if p[ax] in lineset]
        if crossed:
            ops.append(barc); sels.append(sel_of(crossed))

    ops.append(34); sels.append(full)
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
