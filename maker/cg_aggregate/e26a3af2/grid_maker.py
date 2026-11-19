"""
ARC Task: e26a3af2 (RE-ARC) — LLM-generated grid_maker
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
    # Each band's colour is sampled independently in the generator, and the rule
    # (restore every line to its stripe's colour) works for any colours. There is
    # no background and no colour role to fix, so nothing is shared across the episode.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))

    cols = list(range(10))
    transpose = random.choice((True, False))
    H = max_w if transpose else max_h   # budget for the number of lines stacked as bands
    W = max_h if transpose else max_w   # budget for the band length
    nr = unifint(diff_lb, diff_ub, (1, max(1, min(10, H // 2))))
    w = unifint(diff_lb, diff_ub, (4, max(4, W)))
    scols = random.sample(cols, nr)
    sgs = [[[c] * w for _ in range(2)] for c in scols]
    numexp = unifint(diff_lb, diff_ub, (0, max(0, H - 2 * nr)))
    for _ in range(numexp):
        idx = random.randint(0, nr - 1)
        sgs[idx].append(list(sgs[idx][-1]))
    gi, go = [], []
    for idx, col in enumerate(scols):
        sg = sgs[idx]
        a, b = len(sg), w
        ub = (a * b) // 2 - 1
        nnoise = unifint(diff_lb, diff_ub, (0, ub))
        inds = [(i, j) for i in range(a) for j in range(b)]
        sg2 = [list(r) for r in sg]
        oc = [c for c in cols if c != col]
        for (i, j) in random.sample(inds, nnoise):
            sg2[i][j] = random.choice(oc)
        for ri in (0, a - 1):
            while sum(e == col for e in sg2[ri]) < w // 2:
                locs = [j for j, e in enumerate(sg2[ri]) if e != col]
                sg2[ri][random.choice(locs)] = col
        gi += sg2
        go += sg
    gi = np.array(gi)
    go = np.array(go)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _band_lines(G):
    """Split the lines (rows of G) into stripes of >= 2 lines, each with one colour,
    so that as few cells as possible differ from their stripe's colour.
    Returns (colour assigned to each line, number of differing cells)."""
    n, w = G.shape
    INF = float("inf")
    dp = [INF] * (n + 1)
    dp[0] = 0
    back = [None] * (n + 1)
    for i in range(2, n + 1):
        for j in range(0, i - 1):
            if dp[j] == INF:
                continue
            cnt = np.bincount(G[j:i].ravel(), minlength=10)
            c = int(cnt.argmax())
            cost = dp[j] + (i - j) * w - int(cnt[c])
            if cost < dp[i]:
                dp[i] = cost
                back[i] = (j, c)
    line_col = [0] * n
    i = n
    while i > 0:
        j, c = back[i]
        for k in range(j, i):
            line_col[k] = c
        i = j
    return line_col, dp[n]


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of
    # Reading taken from the golf entry: each cell is restored from the LINE it lies on,
    # so the unit of work is one row (or one column), not a whole band. Every noisy line
    # gets one Color op, painted with its stripe's colour. A single row's majority is
    # not used directly, because noise can take over an inner row of a band. Instead the
    # stripe colour comes from the best stripe split of I. The orientation is the axis
    # whose split leaves fewer cells that disagree with their stripe colour.
    # Everything here is measured from I alone; O is never consulted.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    if hi < 2:
        vertical = True
    elif wi < 2:
        vertical = False
    else:
        _, cost_h = _band_lines(I)
        _, cost_v = _band_lines(I.T)
        vertical = cost_v < cost_h

    G = I.T if vertical else I
    line_col, _ = _band_lines(G)

    # Go through the lines in order and repaint each one in full with its stripe colour.
    for k, col in enumerate(line_col):
        if np.all(G[k] == col):
            continue                      # this line is already clean: no op needed
        ops.append(int(col))
        if vertical:
            sels.append(sel_of([(r, k) for r in range(hi)]))   # the whole column k
        else:
            sels.append(sel_of([(k, c) for c in range(wi)]))   # the whole row k

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
                        f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
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
                                f"for task e26a3af2"
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
                    f"Failed to build a complete episode for task e26a3af2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e26a3af2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
