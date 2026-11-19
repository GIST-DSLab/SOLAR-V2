"""
ARC Task: e21d9049 (RE-ARC) — LLM-generated grid_maker
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
    # The rule depends only on the geometry/pattern of the two bars, not on their
    # colours, so only the background colour needs fixing for the episode.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    hlo = min(10, max_h)
    wlo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))
    ph = unifint(diff_lb, diff_ub, (2, max(2, min(9, h - 1))))
    pw = unifint(diff_lb, diff_ub, (2, max(2, min(9, w - 1))))
    remcols = remove(bgc, cols)
    hbar = frozenset({(choice(remcols), (k, 0)) for k in range(ph)})
    wbar = frozenset({(choice(remcols), (0, k)) for k in range(pw)})

    # place the bars; reject placements where one bar's stray cell lands exactly
    # adjacent to (but not on) the other bar, which would make the two bars
    # indistinguishable from a single longer bar in the input
    locih = randint(0, h - ph)
    locjh = randint(0, w - 1)
    lociw = randint(0, h - 1)
    locjw = randint(0, w - pw)
    for _ in range(500):
        inV = locih <= lociw <= locih + ph - 1
        inW = locjw <= locjh <= locjw + pw - 1
        if inV == inW:
            break
        if inW and (lociw < locih - 1 or lociw > locih + ph):
            break
        if inV and (locjh < locjw - 1 or locjh > locjw + pw):
            break
        locih = randint(0, h - ph)
        locjh = randint(0, w - 1)
        lociw = randint(0, h - 1)
        locjw = randint(0, w - pw)

    loch = (locih, locjh)
    locw = (lociw, locjw)
    canv = canvas(bgc, (h, w))
    hbar = shift(hbar, loch)
    wbar = shift(wbar, locw)
    col = choice(remcols)
    hbard = extract(hbar, lambda cij: abs(cij[1][0] - lociw) % ph == 0)[1]
    hbar = sfilter(hbar, lambda cij: abs(cij[1][0] - lociw) % ph != 0) | {(col, hbard)}
    wbard = extract(wbar, lambda cij: abs(cij[1][1] - locjh) % pw == 0)[1]
    wbar = sfilter(wbar, lambda cij: abs(cij[1][1] - locjh) % pw != 0) | {(col, wbard)}
    gi = paint(canv, hbar | wbar)
    go = paint(canv, hbar | wbar)
    for k in range(h // ph + 1):
        go = paint(go, shift(hbar, (k * ph, 0)))
        go = paint(go, shift(hbar, (-k * ph, 0)))
    for k in range(w // pw + 1):
        go = paint(go, shift(wbar, (0, k * pw)))
        go = paint(go, shift(wbar, (0, -k * pw)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (read entirely from I): the input holds one short vertical bar and one
    short horizontal bar on a plain background.  The vertical bar's colour sequence
    is repeated periodically (period = its length) down its whole column, and the
    horizontal bar's colour sequence is repeated periodically (period = its length)
    across its whole row.  O is only used for its shape (same as I)."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    G = I.copy()

    # background = the colour the canvas was painted with (bars are <= 18 cells
    # on a grid of >= 100 cells, so it is the overwhelming majority colour)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    # the vertical bar's column is the column holding the most foreground cells
    # (>= 2), every other column holds at most one; likewise for the row.
    colcnt = Counter(c for _, c in fg)
    rowcnt = Counter(r for r, _ in fg)
    jh = max(sorted(colcnt), key=lambda c: colcnt[c])
    iw = max(sorted(rowcnt), key=lambda r: rowcnt[r])

    def longest_run(vals):
        runs, cur = [], [vals[0]]
        for v in vals[1:]:
            if v == cur[-1] + 1:
                cur.append(v)
            else:
                runs.append(cur)
                cur = [v]
        runs.append(cur)
        return max(runs, key=len)

    vrows = longest_run(sorted(r for r, c in fg if c == jh))
    hcols = longest_run(sorted(c for r, c in fg if r == iw))
    ih, ph = vrows[0], len(vrows)
    jw, pw = hcols[0], len(hcols)
    vpat = [int(I[r, jh]) for r in vrows]   # vertical bar colour sequence
    hpat = [int(I[iw, c]) for c in hcols]   # horizontal bar colour sequence

    ops, sels = [], []

    # 1. extend the vertical bar periodically over its whole column,
    #    one Color op per colour of the bar's sequence (top-down order)
    vtarget = [vpat[(r - ih) % ph] for r in range(hi)]
    order = []
    for r in range(hi):
        if vtarget[r] not in order:
            order.append(vtarget[r])
    for v in order:
        cells = [(r, jh) for r in range(hi) if vtarget[r] == v and G[r, jh] != v]
        if cells:
            ops.append(int(v))
            sels.append(sel_of(cells))
            for (r, c) in cells:
                G[r, c] = v

    # 2. extend the horizontal bar periodically over its whole row
    htarget = [hpat[(c - jw) % pw] for c in range(wi)]
    order = []
    for c in range(wi):
        if htarget[c] not in order:
            order.append(htarget[c])
    for v in order:
        cells = [(iw, c) for c in range(wi) if htarget[c] == v and G[iw, c] != v]
        if cells:
            ops.append(int(v))
            sels.append(sel_of(cells))
            for (r, c) in cells:
                G[r, c] = v

    # Submit: selection is the whole (unchanged-size) grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task e21d9049"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e21d9049"
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
                                f"for task e21d9049"
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
                    f"Failed to build a complete episode for task e21d9049 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e21d9049-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
