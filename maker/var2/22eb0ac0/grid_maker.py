"""
ARC Task: 22eb0ac0 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback if helper module is unavailable
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for r, c in cells]}


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    """Only the background is a real colour role here: the rule compares the two
    endpoint colours of a line to each other, never to a fixed palette entry.
    The one structural variant is the orientation (rows vs columns, i.e. the
    generator's final dmirror), so it is planned per instance."""
    bgc = random.choice(list(range(10)))

    VARIANTS = [{"transpose": False}, {"transpose": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# -------------------------------------------------------------- generator ----
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, transpose=None) -> dict:
    colopts = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(colopts)
    if transpose is None:
        transpose = choice((True, False))

    # the grid is transposed at the end, so budget the dimensions accordingly
    hlim = max(4, max_w if transpose else max_h)
    wlim = max(4, max_h if transpose else max_w)

    h = unifint(diff_lb, diff_ub, (4, hlim))
    w = unifint(diff_lb, diff_ub, (4, wlim))

    remcols = remove(bgc, colopts)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))

    nlocs = unifint(diff_lb, diff_ub, (1, h))
    locs = sample(interval(0, h, 1), nlocs)
    while set(locs).issubset({0, h - 1}):
        locs = sample(interval(0, h, 1), nlocs)

    mp = nlocs // 2
    nbarsdev = unifint(diff_lb, diff_ub, (0, mp))
    nbars = choice((nbarsdev, h - nbarsdev))
    nbars = max(1, min(nbars, nlocs))          # always at least one line to draw
    barlocs = sample(locs, nbars)
    nonbarlocs = difference(locs, barlocs)

    barcols = [choice(remcols) for j in range(nbars)]
    acols = [choice(remcols) for j in range(len(nonbarlocs))]
    bcols = [choice(remove(acols[j], remcols)) for j in range(len(nonbarlocs))]

    for bc, bl in zip(barcols, barlocs):
        gi = fill(gi, bc, ((bl, 0), (bl, w - 1)))
        go = fill(go, bc, connect((bl, 0), (bl, w - 1)))

    for (a, b), loc in zip(zip(acols, bcols), nonbarlocs):
        gi = fill(gi, a, {(loc, 0)})
        go = fill(go, a, {(loc, 0)})
        gi = fill(gi, b, {(loc, w - 1)})
        go = fill(go, b, {(loc, w - 1)})

    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)

    return {'input': gi, 'output': go}


# ------------------------------------------------------------- operations ----
def derive_operations(I, O=None, examples=None):
    """Rule: a pair of border dots that face each other across the grid and share
    a colour is a line -- draw it.  Mismatched pairs stay as dots.

    The rule is fixed; the *route* is read off the input, so the same instance
    always yields the same trajectory while different instances are solved in
    visibly different ways:
      * where the sweep starts (top/left edge, bottom/right edge, or the middle
        outward) -- decided by where the first/last line sits,
      * whether a line is painted end to end or only the gap between its two
        dots is joined -- decided by the line count and the line colours,
      * whether repeated colours are drawn one by one or the first one is
        copied onto its twins -- decided by whether a colour repeats.
    Colour 0 is never carried by a Copy/Paste (ARCLE treats 0 as empty); those
    lines are always laid down with an explicit Color0.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- background: the inside of the frame is untouched canvas -------------
    inner = I[1:hi - 1, 1:wi - 1] if (hi > 2 and wi > 2) else I
    vals, counts = np.unique(inner, return_counts=True)
    bgc = int(vals[int(np.argmax(counts))])

    # --- orientation: dots on the side columns => lines run across rows ------
    nz = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) != bgc]
    horizontal = len(nz) > 0 and all(c == 0 or c == wi - 1 for _, c in nz)

    if horizontal:
        L, span_len = hi, wi
        ends = [(int(I[i, 0]), int(I[i, wi - 1])) for i in range(L)]
    else:
        L, span_len = wi, hi
        ends = [(int(I[0, i]), int(I[hi - 1, i])) for i in range(L)]

    # --- the lines to draw ---------------------------------------------------
    bars = [(i, a) for i, (a, b) in enumerate(ends) if a != bgc and a == b]
    color_of = {i: c for i, c in bars}
    positions = [i for i, _ in bars]

    if positions:
        # route choice 1: where to start sweeping, read from the line positions
        if positions[0] == 0:
            order = sorted(positions)
        elif positions[-1] == L - 1:
            order = sorted(positions, reverse=True)
        else:
            mid = (L - 1) / 2.0
            order = sorted(positions, key=lambda p: (abs(p - mid), p))

        # route choice 2: paint the whole line, or only join the two dots
        gap_only = (span_len >= 3 and
                    (len(bars) + sum(color_of.values())) % 2 == 1)
        s0 = 1 if gap_only else 0
        s1 = span_len - 2 if gap_only else span_len - 1

        def span(idx):
            if horizontal:
                return [(idx, k) for k in range(s0, s1 + 1)]
            return [(k, idx) for k in range(s0, s1 + 1)]

        # route choice 3: repeated colours handled together (copy the first
        # line onto its twins) or each line drawn on its own
        groups = []
        seen = {}
        for p in order:
            c = color_of[p]
            if c in seen:
                groups[seen[c]][1].append(p)
            else:
                seen[c] = len(groups)
                groups.append((c, [p]))
        repeated = any(len(g[1]) > 1 for g in groups)

        if repeated:
            for c, members in groups:
                head = members[0]
                cells = span(head)
                ops.append(int(c))
                sels.append(sel_of(cells))               # draw the first line
                rest = members[1:]
                if not rest:
                    continue
                if c == 0:
                    # 0 does not travel through the clipboard: lay it down直接
                    for p in rest:
                        ops.append(0)
                        sels.append(sel_of(span(p)))
                else:
                    ops.append(29)                        # CopyO the drawn line
                    sels.append(sel_of(cells))
                    for p in rest:
                        tgt = span(p)
                        ops.append(30)                    # Paste onto its twin
                        sels.append(sel_of([tgt[0]]))
        else:
            for p in order:
                ops.append(int(color_of[p]))
                sels.append(sel_of(span(p)))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])                   # full grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 22eb0ac0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22eb0ac0"
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
                                f"for task 22eb0ac0"
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
                    f"Failed to build a complete episode for task 22eb0ac0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22eb0ac0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
