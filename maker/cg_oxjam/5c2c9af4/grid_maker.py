"""
ARC Task: 5c2c9af4 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * min(diff_lb, d)))
    hi = a + int(round((b - a) * d))
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.randint(0, 9)
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c != bgc])
    mirror = random.choice((True, False))
    # if transposed at the end, the generated h becomes the width
    lim_h, lim_w = (max_w, max_h) if mirror else (max_h, max_w)
    lim_h = max(5, min(30, lim_h))
    lim_w = max(5, min(30, lim_w))
    h = _unifint(diff_lb, diff_ub, (5, lim_h))
    w = _unifint(diff_lb, diff_ub, (5, lim_w))
    boxhd = _unifint(diff_lb, diff_ub, (0, h // 2))
    boxwd = _unifint(diff_lb, diff_ub, (0, w // 2))
    boxh = random.choice((boxhd, h - boxhd))
    boxw = random.choice((boxwd, w - boxwd))
    if boxh % 2 == 0:
        boxh = random.choice((boxh - 1, boxh + 1))
    if boxw % 2 == 0:
        boxw = random.choice((boxw - 1, boxw + 1))
    boxh = min(max(1, boxh), h if h % 2 == 1 else h - 1)
    boxw = min(max(1, boxw), w if w % 2 == 1 else w - 1)
    loci = random.randint(0, h - boxh)
    locj = random.randint(0, w - boxw)
    cpi = loci + boxh // 2
    cpj = locj + boxw // 2
    A = (loci, locj)
    B = (loci + boxh - 1, locj + boxw - 1)
    cp = (cpi, cpj)
    gi = np.full((h, w), bgc, dtype=int)
    for (r, c) in {A, B, cp}:
        gi[r, c] = fgc
    go = gi.copy()
    pts = {A, B, cp}
    rows = {p[0] for p in pts}
    colset = {p[1] for p in pts}
    cond = True
    if len(pts) == 3 and len(rows) == 1:
        go[cpi, :] = fgc
        cond = False
    if len(pts) == 3 and len(colset) == 1:
        go[:, cpj] = fgc
        cond = False
    k = 1
    while cond:
        f1 = k * (boxh // 2)
        f2 = k * (boxw // 2)
        t, b, l, r = cpi - f1, cpi + f1, cpj - f2, cpj + f2
        prev = go.copy()
        for rr in (t, b):
            if 0 <= rr < h:
                for cc in range(max(0, l), min(w - 1, r) + 1):
                    go[rr, cc] = fgc
        for cc in (l, r):
            if 0 <= cc < w:
                for rr in range(max(0, t), min(h - 1, b) + 1):
                    go[rr, cc] = fgc
        cond = not np.array_equal(prev, go)
        k += 1
    if mirror:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading: the three markers fix a centre and a step (f1, f2). The output is the
    # family of concentric rings around the centre at scaled Chebyshev distance
    # k = 1, 2, 3, ... . Each ring is ONE unit: one Color(fgc) on that ring's on-grid
    # cells, emitted from the centre outward until a ring falls entirely off the grid.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    fgc = min(cnt.keys(), key=lambda c: (cnt[c], c))
    pts = [(int(r), int(c)) for r, c in zip(*np.where(I == fgc))]

    G = I.copy()

    def paint(cells):
        cells = sorted(set(cells))
        if not cells:
            return
        if all(G[r, c] == fgc for r, c in cells):
            return
        for r, c in cells:
            G[r, c] = fgc
        ops.append(int(fgc))
        sels.append(sel_of(cells))

    if len(pts) >= 2:
        ul_i = min(p[0] for p in pts); lr_i = max(p[0] for p in pts)
        ul_j = min(p[1] for p in pts); lr_j = max(p[1] for p in pts)
        cpi, cpj = (ul_i + lr_i) // 2, (ul_j + lr_j) // 2
        f1, f2 = (lr_i - ul_i) // 2, (lr_j - ul_j) // 2

        if f1 == 0 and f2 == 0:
            pass
        elif f1 == 0:
            # markers on one row: the ring family degenerates to that whole row
            paint([(cpi, c) for c in range(w)])
        elif f2 == 0:
            # markers on one column: whole column
            paint([(r, cpj) for r in range(h)])
        else:
            k = 1
            while True:
                t, b = cpi - k * f1, cpi + k * f1
                l, r = cpj - k * f2, cpj + k * f2
                ring = set()
                for rr in (t, b):
                    if 0 <= rr < h:
                        for cc in range(max(0, l), min(w - 1, r) + 1):
                            ring.add((rr, cc))
                for cc in (l, r):
                    if 0 <= cc < w:
                        for rr in range(max(0, t), min(h - 1, b) + 1):
                            ring.add((rr, cc))
                if not ring:
                    break
                paint(ring)
                k += 1

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5c2c9af4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5c2c9af4"
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
                                f"for task 5c2c9af4"
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
                    f"Failed to build a complete episode for task 5c2c9af4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5c2c9af4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
