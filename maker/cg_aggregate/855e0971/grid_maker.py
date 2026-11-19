"""
ARC Task: 855e0971 (RE-ARC) — LLM-generated grid_maker
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

# Reading taken from the golf entry: each dot grows outward, step by step, from
# its own position until it reaches its bar's edge on each side. The bar edge is
# where the colour pair (bar colour + dot colour) changes. So the unit of work is
# a "ray": the cells from the dot toward one edge of its bar.
# The maker treats the whole bar-wide line as a single unit instead.

ORIENTATIONS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    dotc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTATIONS):
        examples = [dict(v) for v in ORIENTATIONS]
        examples += [dict(random.choice(ORIENTATIONS)) for _ in range(n_ex - len(ORIENTATIONS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ORIENTATIONS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, dotc=None, transpose=None) -> dict:
    from random import choice, randint, sample

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    if dotc is None:
        dotc = random.choice(range(10))
    if transpose is None:
        transpose = choice((True, False))
    # bars stacked along rows of the un-mirrored grid
    Hmax = max_w if transpose else max_h
    Wmax = max_h if transpose else max_w
    cols = list(range(10))
    nbarsd = unifint(diff_lb, diff_ub, (1, 4))
    nbars = choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    nbars = min(nbars, max(3, Hmax // 2))
    h = unifint(diff_lb, diff_ub, (2 * nbars, max(2 * nbars, Hmax)))
    w = unifint(diff_lb, diff_ub, (3, max(3, Wmax)))
    barsizes = [2] * nbars
    while sum(barsizes) < h:
        barsizes[randint(0, nbars - 1)] += 1
    remcols = [c for c in cols if c != dotc]
    nloclbs = [choice((0, 1)) for _ in barsizes]
    if sum(nloclbs) < 2:
        l1, l2 = sample(range(len(nloclbs)), 2)
        nloclbs[l1] = 1
        nloclbs[l2] = 1
    gi_rows, go_rows = [], []
    lastcol = -1
    for bs, nloclb in zip(barsizes, nloclbs):
        col = choice([c for c in remcols if c != lastcol])
        gim = [[col] * w for _ in range(bs)]
        gom = [[col] * w for _ in range(bs)]
        nl = unifint(diff_lb, diff_ub, (nloclb, max(nloclb, w // 2)))
        for jj in sample(range(w), nl):
            r = randint(0, bs - 1)
            gim[r][jj] = dotc
            for rr in range(bs):
                gom[rr][jj] = dotc
        lastcol = col
        gi_rows += gim
        go_rows += gom
    gi = np.array(gi_rows, dtype=int)
    go = np.array(go_rows, dtype=int)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _dot_color(I, examples):
    if examples:
        votes = {}
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            for v in eo[ei != eo].tolist():
                votes[v] = votes.get(v, 0) + 1
        if votes:
            c = max(votes, key=votes.get)
            if c in set(I.flatten().tolist()):
                return c
    # fallback (verifier's reading): the colour with the most isolated cells
    hi, wi = I.shape
    best, bestn = None, -1
    for c in sorted(set(I.flatten().tolist())):
        n = 0
        for r in range(hi):
            for k in range(wi):
                if I[r, k] != c:
                    continue
                iso = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, k + dc
                        if 0 <= rr < hi and 0 <= cc < wi and I[rr, cc] == c:
                            iso = False
                if iso:
                    n += 1
        if n > bestn:
            best, bestn = c, n
    return best


def _bars(G, dotc):
    # rows of G belong to bars; returns list of (r0, r1) or None if inconsistent
    labels = []
    for row in G:
        others = set(row.tolist()) - {dotc}
        if len(others) > 1:
            return None
        labels.append(next(iter(others)) if others else None)
    # rows made entirely of dots join the previous bar
    for i in range(len(labels)):
        if labels[i] is None:
            labels[i] = labels[i - 1] if i > 0 else None
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] is None and i + 1 < len(labels):
            labels[i] = labels[i + 1]
    bars, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            bars.append((start, i - 1))
            start = i
    return bars


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    dotc = _dot_color(I, examples)
    ops, sels = [], []

    # orientation from I alone: bars run along rows if every row holds one bar colour
    bars = _bars(I, dotc)
    horizontal = bars is not None
    if not horizontal:
        bars = _bars(I.T, dotc) or [(0, wi - 1)]
    G = I if horizontal else I.T
    L = G.shape[1]

    def cell(a, b):  # (index along bar-stack, index along bar) -> grid coords
        return (a, b) if horizontal else (b, a)

    for (a0, a1) in bars:
        for b in range(L):
            dots = [a for a in range(a0, a1 + 1) if G[a, b] == dotc]
            if not dots:
                continue
            top, bot = min(dots), max(dots)
            # ray toward the bar's near edge
            ray1 = [cell(a, b) for a in range(a0, top)]
            # ray toward the bar's far edge
            ray2 = [cell(a, b) for a in range(bot + 1, a1 + 1)]
            # any gap between dots in the same bar segment (if ever present)
            gap = [cell(a, b) for a in range(top + 1, bot) if G[a, b] != dotc]
            for ray in (ray1, gap, ray2):
                if ray:
                    ops.append(int(dotc))
                    sels.append(sel_of(ray))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 855e0971"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 855e0971"
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
                                f"for task 855e0971"
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
                    f"Failed to build a complete episode for task 855e0971 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"855e0971-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
