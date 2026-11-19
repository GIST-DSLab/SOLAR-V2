"""
ARC Task: f1cefba8 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, ringc, inc = random.sample(cols, 3)
    return {"bgc": bgc, "ringc": ringc, "inc": inc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ringc, inc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    H = max(7, min(30, max_h))
    W = max(7, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (7, H))
    w = unifint(diff_lb, diff_ub, (7, W))
    ih = unifint(diff_lb, diff_ub, (6, h - 1))
    iw = unifint(diff_lb, diff_ub, (6, w - 1))
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    r1, c1 = loci + ih - 1, locj + iw - 1

    c = np.full((h, w), bgc, dtype=int)
    c[loci:r1 + 1, locj:c1 + 1] = inc
    for (a, b, cc, d) in [(loci, locj, r1, c1), (loci + 1, locj + 1, r1 - 1, c1 - 1)]:
        c[a, b:d + 1] = ringc
        c[cc, b:d + 1] = ringc
        c[a:cc + 1, b] = ringc
        c[a:cc + 1, d] = ringc

    ra, ca, rb, cb = loci + 1, locj + 1, r1 - 1, c1 - 1
    ring2 = set()
    for j in range(ca, cb + 1):
        ring2.add((ra, j)); ring2.add((rb, j))
    for i in range(ra, rb + 1):
        ring2.add((i, ca)); ring2.add((i, cb))
    corners = {(ra, ca), (ra, cb), (rb, ca), (rb, cb)}
    cands = sorted(ring2 - corners)
    numc = unifint(diff_lb, diff_ub, (1, len(cands) // 2))
    locs = random.sample(cands, numc)

    gi = c.copy()
    for (i, j) in locs:
        gi[i, j] = inc

    lm = rb
    fulllocs = set()
    for (i, j) in locs:
        if i > loci + 1 and i < lm:
            for jj in range(w):
                fulllocs.add((i, jj))
        else:
            for ii in range(h):
                fulllocs.add((ii, j))
    go = c.copy()
    for (i, j) in fulllocs:
        if c[i, j] == bgc:
            go[i, j] = inc
        elif c[i, j] == inc:
            go[i, j] = ringc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background: some full row or column lies entirely outside the rectangle
    bgc = None
    for r in range(h):
        if np.all(I[r] == I[r, 0]):
            bgc = int(I[r, 0]); break
    if bgc is None:
        for c in range(w):
            if np.all(I[:, c] == I[0, c]):
                bgc = int(I[0, c]); break

    pts = np.argwhere(I != bgc)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    ringc = int(I[r0, c0])
    inc = int(I[r0 + 2, c0 + 2])

    # markers on inner ring (non-corner) holding interior colour
    hlines = {}  # row -> marker cells (left/right sides)
    vlines = {}  # col -> marker cells (top/bottom sides)
    for c in range(c0 + 2, c1 - 1):
        for r in (r0 + 1, r1 - 1):
            if I[r, c] == inc:
                vlines.setdefault(c, []).append((r, c))
    for r in range(r0 + 2, r1 - 1):
        for c in (c0 + 1, c1 - 1):
            if I[r, c] == inc:
                hlines.setdefault(r, []).append((r, c))

    G = I.copy()
    ops, sels = [], []

    def do_line(line_cells, markers):
        outside = [(r, c) for (r, c) in line_cells
                   if not (r0 <= r <= r1 and c0 <= c <= c1) and G[r, c] != inc]
        if outside:
            ops.append(inc); sels.append(sel_of(outside))
            for (r, c) in outside:
                G[r, c] = inc
        inner = [(r, c) for (r, c) in line_cells
                 if r0 + 2 <= r <= r1 - 2 and c0 + 2 <= c <= c1 - 2 and G[r, c] != ringc]
        cells = [m for m in markers if G[m] != ringc] + inner
        if cells:
            ops.append(ringc); sels.append(sel_of(cells))
            for (r, c) in cells:
                G[r, c] = ringc

    for r in sorted(hlines):
        do_line([(r, c) for c in range(w)], hlines[r])
    for c in sorted(vlines):
        do_line([(r, c) for r in range(h)], vlines[c])

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f1cefba8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f1cefba8"
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
                                f"for task f1cefba8"
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
                    f"Failed to build a complete episode for task f1cefba8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f1cefba8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
