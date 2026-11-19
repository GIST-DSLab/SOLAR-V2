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

# Bars stacked as rows (False) or as columns (True)
ORIENTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    dotc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        ex = [dict(v) for v in ORIENTS]
        ex += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"dotc": dotc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    return int(round(random.uniform(lo, hi)))


def generate(diff_lb, diff_ub, max_h, max_w, dotc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([False, True])
    lim_h = max_w if transpose else max_h   # bar-stack dimension
    lim_w = max_h if transpose else max_w
    cols = list(range(10))
    nbarsd = _unifint(diff_lb, diff_ub, (1, 4))
    nbars = random.choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    nbars = min(nbars, max(3, lim_h // 2))
    h = _unifint(diff_lb, diff_ub, (max(nbars * 2, nbars), max(lim_h, 2 * nbars)))
    w = _unifint(diff_lb, diff_ub, (3, max(3, lim_w)))
    barsizes = [2] * nbars
    while sum(barsizes) < h:
        barsizes[random.randint(0, nbars - 1)] += 1
    remcols = [c for c in cols if c != dotc]
    gi, go = [], []
    lastcol = -1
    nloclbs = [random.choice((0, 1)) for _ in barsizes]
    if sum(nloclbs) < 2:
        a, b = random.sample(range(len(nloclbs)), 2)
        nloclbs[a] = 1
        nloclbs[b] = 1
    for bs, nloclb in zip(barsizes, nloclbs):
        col = random.choice([c for c in remcols if c != lastcol])
        gim = [[col] * w for _ in range(bs)]
        gom = [[col] * w for _ in range(bs)]
        nl = _unifint(diff_lb, diff_ub, (nloclb, max(nloclb, w // 2)))
        for jj in random.sample(range(w), nl):
            gim[random.randint(0, bs - 1)][jj] = dotc
            for r in range(bs):
                gom[r][jj] = dotc
        lastcol = col
        gi += gim
        go += gom
    gi = np.array(gi)
    go = np.array(go)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _dot_color_from_examples(examples):
    """The marker colour is the one the demonstrations spread (the colour cells change TO)."""
    if not examples:
        return None
    for ex in examples:
        a, b = (ex[0], ex[1]) if not isinstance(ex, dict) else (ex["input"], ex["output"])
        a = np.asarray(a, dtype=int)
        b = np.asarray(b, dtype=int)
        diff = b[a != b]
        if diff.size:
            return int(diff[0])
    return None


def _analyze(G, dotc=None):
    """Bars are rows of G: each row holds one bar colour, plus possibly dots."""
    rowsets = [set(np.unique(row).tolist()) for row in G]
    if any(len(s) > 2 for s in rowsets):
        return None
    two = [s for s in rowsets if len(s) == 2]
    if not two:
        return None
    cand = set.intersection(*two)
    if dotc is None:
        if not cand:
            return None
        if len(cand) == 1:
            dotc = next(iter(cand))
        else:
            cnt = {c: sum(int((row == c).sum()) for row, s in zip(G, rowsets) if len(s) == 2)
                   for c in cand}
            dotc = min(cnt, key=cnt.get)
    elif dotc not in cand:
        return None
    barcol = [next(iter(s - {dotc})) for s in rowsets]
    bands = []
    s = 0
    for r in range(1, len(barcol) + 1):
        if r == len(barcol) or barcol[r] != barcol[s]:
            bands.append((s, r - 1))
            s = r
    return dotc, bands


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # orientation + marker colour, from I (and the demonstrations for the colour)
    horizontal = True
    known = _dot_color_from_examples(examples)
    res = _analyze(I, known)
    if res is None:
        res = _analyze(I.T, known)
        horizontal = False
    dotc, bands = res
    G = (I if horizontal else I.T).copy()   # working copy in bar-major coords
    W = G.shape[1]
    ops, sels = [], []

    def emit(cells):
        real = [(r, c) if horizontal else (c, r) for r, c in cells]
        ops.append(dotc)
        sels.append(sel_of(real))
        for r, c in cells:
            G[r, c] = dotc

    # Propagation reading: inside each bar, the marker colour spreads one line
    # at a time. Sweep forward (down / right): each line takes on the marker
    # wherever the previous line already holds it. Then sweep back the other
    # way. A sweep stops at the bar's edges, where the bar colour changes.
    # Each selection = "cells of this line next to a marker cell in the
    # neighbouring line" in the current grid.
    for (r0, r1) in bands:
        for r in range(r0 + 1, r1 + 1):
            cells = [(r, c) for c in range(W) if G[r - 1, c] == dotc and G[r, c] != dotc]
            if cells:
                emit(cells)
        for r in range(r1 - 1, r0 - 1, -1):
            cells = [(r, c) for c in range(W) if G[r + 1, c] == dotc and G[r, c] != dotc]
            if cells:
                emit(cells)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # full-grid rectangle: submit
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
