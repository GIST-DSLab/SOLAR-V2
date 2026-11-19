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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, barc, dotc = random.sample(cols, 3)
    return {"bgc": bgc, "barc": barc, "dotc": dotc}


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    lo = a + int(round((b - a) * min(lb, d)))
    hi = a + int(round((b - a) * max(lb, d)))
    return random.randint(min(lo, b), min(max(lo, hi), b))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, barc=1, dotc=2, **kw) -> dict:
    m = max(5, min(max_h, max_w, 30))   # transposing mirrors swap h/w
    h = _unif(diff_lb, diff_ub, (5, m))
    w = _unif(diff_lb, diff_ub, (5, m))
    barh = _unif(diff_lb, diff_ub, (2, (h - 1) // 2))
    ncells = _unif(diff_lb, diff_ub, (1, barh))
    top_i = np.full((barh, w), bgc, dtype=int); top_i[:, 0] = barc
    bot_i = top_i[:, ::-1].copy()
    top_o = top_i.copy(); bot_o = bot_i.copy()
    for k in random.sample(range(barh), ncells):
        j = random.randint(2, w - 2)
        top_i[k, j] = dotc
        top_o[k, j] = 4
        top_o[k, 1:j] = dotc
        bot_o[k, 0:w - 1] = dotc
    pi = [top_i, bot_i]; po = [top_o, bot_o]
    rr = np.full((1, w), bgc, dtype=int)
    while sum(x.shape[0] for x in pi) < h:
        idx = random.randint(0, len(pi) - 1)
        pi.insert(idx, rr); po.insert(idx, rr)
    gi = np.vstack(pi); go = np.vstack(po)
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T,
           lambda g: g[:, ::-1], lambda g: g[::-1, :],
           lambda g: np.rot90(g, -1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi); go = fn(go)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O=None, examples=None):
    """
    Reading taken from the golf entry, which goes through the grid one bar-line
    at a time: first do the dot bar's half. Each dot sends a beam (dot colour)
    back to its bar and the dot itself becomes 4. Only after that, do the twin
    bar's half: the line at the same offset along the twin bar as a dot's line
    along the dot bar is filled completely (except the bar cell) with the dot colour.
    Where the two readings agree, nothing changes: same detection, and the twin
    line is still placed by its offset from the start of the bar. The difference
    is the order. The maker handles one dot at a time (beam, head, twin line).
    This reading makes two passes, one per bar: all beams and heads first, then
    all answering lines.
    Every selection comes from I alone (a dot, the cells between bar and dot, a
    full grid line minus bar cells).
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((hi, wi), bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c]); seen[r, c] = True
            st = [(r, c)]; cells = []
            while st:
                a, b = st.pop(); cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True; st.append((na, nb))
            comps.append((col, cells))

    def done():
        ops.append(34); sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
        return ops, sels

    if not comps:
        return done()
    border = lambda p: p[0] in (0, hi - 1) or p[1] in (0, wi - 1)
    barc = None
    for col in sorted({c for c, _ in comps}):
        g = [cl for c2, cl in comps if c2 == col]
        if len(g) == 2 and all(len(x) >= 2 for x in g) and all(border(p) for x in g for p in x):
            barc = col; break
    if barc is None:
        return done()
    others = sorted({c for c, _ in comps} - {barc})
    if not others:
        return done()
    dotc = others[0]
    bars = [cl for c2, cl in comps if c2 == barc]
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]
    vertical = len({p[0] for b in bars for p in b}) > len({p[1] for b in bars for p in b})
    ax = 0 if vertical else 1                     # axis running along the bars
    dotbar = next((b for b in bars if {p[ax] for p in b} & {d[ax] for d in dots}), bars[0])
    twin = bars[1] if dotbar is bars[0] else bars[0]
    start = lambda b: min(p[ax] for p in b)
    shift = start(twin) - start(dotbar)
    bpos = dotbar[0][1 - ax]                      # the dot bar's fixed cross coordinate
    dots.sort(key=lambda d: d[ax])

    # pass 1: the dot bar's half -- each dot's beam back to the bar, then its head -> 4
    for d in dots:
        along, cross = d[ax], d[1 - ax]
        lo, hi_ = (bpos + 1, cross - 1) if cross > bpos else (cross + 1, bpos - 1)
        beam = [((along, x) if vertical else (x, along)) for x in range(lo, hi_ + 1)]
        if beam:
            ops.append(int(dotc)); sels.append(sel_of(beam))
        ops.append(4); sels.append(sel_of([d]))

    # pass 2: the twin bar's half -- full answering line at the same bar offset
    n_cross = wi if vertical else hi
    n_along = hi if vertical else wi
    for d in dots:
        t = d[ax] + shift
        if not (0 <= t < n_along):
            continue
        line = [((t, x) if vertical else (x, t)) for x in range(n_cross)]
        line = [p for p in line if I[p] != barc]
        if line:
            ops.append(int(dotc)); sels.append(sel_of(line))
    return done()


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
