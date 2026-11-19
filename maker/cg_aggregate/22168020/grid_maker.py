"""
ARC Task: 22168020 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


# ── helpers for the generator ────────────────────────────────────────────────
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * min(diff_lb, d)))
    hi = a + int(round((b - a) * max(diff_ub, d)))
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def _connect(a, b):
    (ai, aj), (bi, bj) = a, b
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    if ai == bi:
        return {(ai, j) for j in range(sj, ej + 1)}
    if aj == bj:
        return {(i, aj) for i in range(si, ei + 1)}
    if bi - ai == bj - aj:
        return {(i, j) for i, j in zip(range(si, ei + 1), range(sj, ej + 1))}
    if bi - ai == aj - bj:
        return {(i, j) for i, j in zip(range(si, ei + 1), range(ej, sj - 1, -1))}
    return set()


def _bbox(p):
    rs = [i for i, _ in p]
    cs = [j for _, j in p]
    return min(rs), max(rs), min(cs), max(cs)


def _hmirror(p):
    u, d, _, _ = _bbox(p)
    return {(u + d - i, j) for i, j in p}


def _vmirror(p):
    _, _, l, r = _bbox(p)
    return {(i, l + r - j) for i, j in p}


def _dmirror(p):
    a, _, b, _ = _bbox(p)
    return {(j - b + a, i - a + b) for i, j in p}


def _cmirror(p):
    return _vmirror(_dmirror(_vmirror(p)))


def _identity(p):
    return set(p)


# ── 1. sample_colors ─────────────────────────────────────────────────────────
def sample_colors(num_examples=None) -> dict:
    # Only the background matters for the rule; shape colours are free.
    return {"bgc": random.choice(range(10))}


# ── 2. generate ──────────────────────────────────────────────────────────────
def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    h = _unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = _unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    num = _unifint(diff_lb, diff_ub, (1, min(9, (h * w) // 10)))
    remcols = [c for c in range(10) if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    succ, tr, maxtr = 0, 0, 6 * num
    inds = {(i, j) for i in range(h) for j in range(w)}
    while tr < maxtr and succ < num:
        d = _unifint(diff_lb, diff_ub, (2, 5))
        ow = 2 * d
        if len(inds) == 0:
            tr += 1
            continue
        loci, locj = random.choice(sorted(inds))
        io1 = _connect((loci, locj), (loci + d - 1, locj + d - 1))
        io2 = _connect((loci, locj + ow - 1), (loci + d - 1, locj + d))
        io = io1 | io2 | {(loci + d, locj + d - 1), (loci + d, locj + d)}
        oo = set()
        iol = list(io)
        for a in iol:
            for b in iol:
                seg = _connect(a, b)
                if seg and len({i for i, _ in seg}) == 1:
                    oo |= seg
        mf = random.choice((_identity, _dmirror, _cmirror, _hmirror, _vmirror))
        io = mf(io)
        oo = mf(oo)
        col = random.choice(remcols)
        if oo.issubset(inds):
            for (i, j) in io:
                gi[i, j] = col
            for (i, j) in oo:
                go[i, j] = col
            succ += 1
            inds = inds - oo
            remcols = [c for c in remcols if c != col]
        tr += 1
    return {"input": gi.tolist(), "output": go.tolist()}


# ── 3. derive_operations ─────────────────────────────────────────────────────
def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): each funnel shape is a set of
    # facing cell pairs along one axis; the rule fills the straight gap
    # between the two same-coloured cells in each row (or in each column for
    # a transposed funnel). Unlike the maker, which works pair by pair over
    # every connectable pair, the unit here is one row's (or one column's)
    # span of one colour. Every selection comes from I alone: the cells
    # between the leftmost and rightmost cell of colour c in row r.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    colors = sorted({int(v) for v in I.flatten().tolist() if int(v) != bgc})

    grid = I.copy()
    ops, sels = [], []

    for col in colors:
        cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col]
        # row spans (horizontal pairs)
        rows = sorted({r for r, _ in cells})
        for r in rows:
            cs = [c for rr, c in cells if rr == r]
            if len(cs) < 2:
                continue
            span = [(r, c) for c in range(min(cs), max(cs) + 1)]
            if all(grid[p] == col for p in span):
                continue  # gap already closed: painting would be a no-op
            ops.append(col)
            sels.append(sel_of(span))
            for p in span:
                grid[p] = col
        # column spans (vertical pairs, for transposed funnels)
        colsx = sorted({c for _, c in cells})
        for c in colsx:
            rs = [r for r, cc in cells if cc == c]
            if len(rs) < 2:
                continue
            span = [(r, c) for r in range(min(rs), max(rs) + 1)]
            if all(grid[p] == col for p in span):
                continue
            ops.append(col)
            sels.append(sel_of(span))
            for p in span:
                grid[p] = col

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 22168020"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22168020"
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
                                f"for task 22168020"
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
                    f"Failed to build a complete episode for task 22168020 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22168020-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
