"""
ARC Task: db3e9e38 (RE-ARC) — LLM-generated grid_maker
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

# How the two readings differ:
# - The maker treats each diagonal copy of the bar (a "stripe") as one unit and
#   paints stripe by stripe.
# - The code-golf entry treats each LINE PARALLEL TO THE EDGE the bar touches as
#   one unit. Each line is the line beside it shifted one step diagonally, with
#   the stripe colours swapped (fgc <-> 8).
# So derive_operations works line by line. It starts at the line just past the
# bar's free tip and walks toward the edge. On each line it paints all
# odd-numbered stripes 8, then all new even-numbered stripes in the bar colour.
# The staircase therefore grows one line at a time and gets wider toward the edge.
# Both readings agree on what stays fixed: the colour 8, the bar's thickness as
# the sideways step, and one step toward the edge per stripe.

ROTS = [0, 1, 2, 3]  # number of CW quarter turns applied to the canonical (top-edge) layout


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    return random.randint(max(a, lo_i), min(b, max(lo_i, hi_i)))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, fgc=None, rot=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    if fgc is None:
        fgc = random.choice([c for c in cols if c != bgc])
    if rot is None:
        rot = random.choice(ROTS)
    lim_h, lim_w = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = _unifint(diff_lb, diff_ub, (3, max(3, lim_h)))
    w = _unifint(diff_lb, diff_ub, (3, max(3, lim_w)))
    barth = _unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    loci = _unifint(diff_lb, diff_ub, (1, h - 2))
    locj = random.randint(1, w - barth - 1)
    bar = {(i, j) for i in range(0, loci + 1) for j in range(locj, locj + barth)}
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for (i, j) in bar:
        gi[i][j] = fgc

    def fill(g, col, cells):
        for (i, j) in cells:
            if 0 <= i < h and 0 <= j < w:
                g[i][j] = col

    for k in range(16):
        for sgn in (1, -1):
            sh = [(i - 2 * k, j + sgn * 2 * k * barth) for (i, j) in bar]
            fill(go, fgc, sh)
        for sgn in (1, -1):
            sh = [(i - (2 * k + 1), j + sgn * (2 * k + 1) * barth) for (i, j) in bar]
            fill(go, 8, sh)

    def rot90(g):  # clockwise
        return [list(r) for r in zip(*g[::-1])]

    for _ in range(rot):
        gi = rot90(gi)
        go = rot90(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # The bar touches exactly one edge and never reaches a corner -> corners are background.
    bgc = int(I[0, 0])
    ys, xs = np.where(I != bgc)
    ops, sels = [], []
    if len(ys) == 0:
        return [34], [sel_of([(r, c) for r in range(hi) for c in range(wi)])]
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    fgc = int(I[r0, c0])
    STRIPE = 8  # hardcoded by the task: alternating stripes are always colour 8

    if r0 == 0:
        edge = 'top'
    elif r1 == hi - 1:
        edge = 'bottom'
    elif c0 == 0:
        edge = 'left'
    else:
        edge = 'right'

    # Lines run parallel to the touched edge. On the line at distance d from the
    # bar's tip, stripes k = 1..d sit on both sides of the bar, each stripe
    # shifted k*barth sideways. Odd k -> 8, even k -> bar colour.
    if edge in ('top', 'bottom'):
        barth = c1 - c0 + 1
        lines = list(range(r1 - 1, -1, -1)) if edge == 'top' else list(range(r0 + 1, hi))
        tip = r1 if edge == 'top' else r0

        def stripe_cells(line, k):
            cells = []
            for sgn in (1, -1):
                for c in range(c0 + sgn * k * barth, c1 + sgn * k * barth + 1):
                    if 0 <= c < wi:
                        cells.append((line, c))
            return cells
    else:
        barth = r1 - r0 + 1
        lines = list(range(c1 - 1, -1, -1)) if edge == 'left' else list(range(c0 + 1, wi))
        tip = c1 if edge == 'left' else c0

        def stripe_cells(line, k):
            cells = []
            for sgn in (1, -1):
                for r in range(r0 + sgn * k * barth, r1 + sgn * k * barth + 1):
                    if 0 <= r < hi:
                        cells.append((r, line))
            return cells

    # Walk from the line just past the bar's tip toward the edge; each line is wider.
    for line in lines:
        kmax = abs(tip - line)
        odd, even = [], []
        for k in range(1, kmax + 1):
            (odd if k % 2 == 1 else even).extend(stripe_cells(line, k))
        if odd:
            ops.append(STRIPE); sels.append(sel_of(sorted(set(odd))))
        if even:
            ops.append(fgc); sels.append(sel_of(sorted(set(even))))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
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
                                f"for task db3e9e38"
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
                    f"Failed to build a complete episode for task db3e9e38 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"db3e9e38-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
