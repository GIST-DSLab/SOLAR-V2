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


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, barc=None, dotc=None, **kw) -> dict:
    def unifint(lo, hi):
        lo, hi = min(lo, hi), max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))
    if bgc is None or barc is None or dotc is None:
        bgc, barc, dotc = random.sample([c for c in range(10) if c != 4], 3)
    # the final mirror/rotation may swap h and w, so both must fit either bound
    lim = max(5, min(max_h, max_w, 30))
    h = unifint(5, lim)
    w = unifint(5, lim)
    barh = unifint(2, (h - 1) // 2)
    ncells = unifint(1, barh)
    topi = np.full((barh, w), bgc, dtype=int)
    topi[:, 0] = barc
    boti = topi[:, ::-1].copy()
    topo = topi.copy()
    boto = boti.copy()
    for k in random.sample(range(barh), ncells):
        j = random.randint(2, w - 2)
        topi[k, j] = dotc
        topo[k, j] = 4
        topo[k, 1:j] = dotc
        boto[k, 0:w - 1] = dotc
    pi = [topi, boti]
    po = [topo, boto]
    rr = np.full((1, w), bgc, dtype=int)
    while sum(p.shape[0] for p in pi) < h:
        idx = random.randint(0, len(pi) - 1)
        pi.insert(idx, rr)
        po.insert(idx, rr)
    gi = np.vstack(pi)
    go = np.vstack(po)
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T,
           lambda g: g[:, ::-1], lambda g: g[::-1, :],
           lambda g: np.rot90(g, -1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Reading taken from the golf solution, which splits the work into three layers
    across the whole grid instead of finishing one dot at a time:
      layer 1 - for every line that holds a dot, fill the gap between the bar and
                that dot with the dot colour (the cells strictly between the two
                marks on that line);
      layer 2 - turn all the dots into 4 together, as one set of cells of one colour;
      layer 3 - the twin bar answers: move each dot line by the offset between the
                two bars, then fill that line across the whole grid with the dot
                colour, leaving the bar cells alone.
    The maker works per dot: beam, then head, then twin line, then the next dot.
    Both readings agree on bar/dot detection, the bar-to-bar offset, and leaving
    bar cells untouched. Only the grouping and order of the ops differ.
    Every selection is read from I alone (a line segment, the set of dot cells,
    a line minus the bar's cells).
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c]); seen[r, c] = True
            st = [(r, c)]; cells = []
            while st:
                a, b = st.pop(); cells.append((a, b))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + dr, b + dc
                    if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True; st.append((na, nb))
            comps.append((col, cells))

    def fin():
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    if not comps:
        return fin()

    def on_border(p):
        return p[0] == 0 or p[1] == 0 or p[0] == hi - 1 or p[1] == wi - 1

    palette = sorted({c for c, _ in comps})
    barc = None
    for col in palette:
        g = [cells for c2, cells in comps if c2 == col]
        if len(g) == 2 and all(len(x) >= 2 for x in g) and all(on_border(p) for x in g for p in x):
            barc = col; break
    if barc is None:
        barc = max(comps, key=lambda t: len(t[1]))[0]
    others = [c for c in palette if c != barc]
    if not others:
        return fin()
    dotc = others[0]
    bars = [cells for col, cells in comps if col == barc]
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]
    if len(bars) != 2 or not dots:
        return fin()

    vertical = len({p[0] for b in bars for p in b}) > len({p[1] for b in bars for p in b})

    def ul(b):
        return (min(p[0] for p in b), min(p[1] for p in b))
    k = 0 if vertical else 1
    dotbar = next((b for b in bars if {p[k] for p in b} & {d[k] for d in dots}), bars[0])
    twin = bars[1] if dotbar is bars[0] else bars[0]
    shift = ul(twin)[k] - ul(dotbar)[k]
    dots.sort(key=lambda d: d[k])

    # layer 1: the gap between the dot bar and each dot, one line at a time
    for (r, c) in dots:
        if vertical:
            cb = ul(dotbar)[1]
            rng = range(cb + 1, c) if c > cb else range(c + 1, cb)
            gap = [(r, x) for x in rng]
        else:
            rb = ul(dotbar)[0]
            rng = range(rb + 1, r) if r > rb else range(r + 1, rb)
            gap = [(y, c) for y in rng]
        if gap:
            ops.append(int(dotc)); sels.append(sel_of(gap))

    # layer 2: every dot (all cells of the dot colour) becomes 4 at once
    ops.append(4); sels.append(sel_of(dots))

    # layer 3: the twin bar answers with a full line per dot line, bar cells kept
    for (r, c) in dots:
        if vertical:
            rr = r + shift
            if not (0 <= rr < hi):
                continue
            line = [(rr, x) for x in range(wi) if I[rr, x] != barc]
        else:
            cc = c + shift
            if not (0 <= cc < wi):
                continue
            line = [(y, cc) for y in range(hi) if I[y, cc] != barc]
        if line:
            ops.append(int(dotc)); sels.append(sel_of(line))
    return fin()


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
