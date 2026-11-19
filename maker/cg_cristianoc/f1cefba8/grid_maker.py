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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    v = int(round(a + (b - a) * d))
    return max(a, min(b, v))


def sample_colors(num_examples=None) -> dict:
    bgc, ringc, inc = random.sample(range(10), 3)
    return {"bgc": bgc, "ringc": ringc, "inc": inc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ringc=None, inc=None, **kwargs) -> dict:
    if bgc is None or ringc is None or inc is None:
        bgc, ringc, inc = random.sample(range(10), 3)
    max_h = max(7, min(30, max_h))
    max_w = max(7, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (7, max_h))
    w = _unifint(diff_lb, diff_ub, (7, max_w))
    ih = _unifint(diff_lb, diff_ub, (6, h - 1))
    iw = _unifint(diff_lb, diff_ub, (6, w - 1))
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    r0, r1 = loci, loci + ih - 1
    c0, c1 = locj, locj + iw - 1

    c = np.full((h, w), bgc, dtype=int)
    c[r0:r1 + 1, c0:c1 + 1] = inc
    # ring1 (box) and ring2 (inbox)
    c[r0, c0:c1 + 1] = ringc; c[r1, c0:c1 + 1] = ringc
    c[r0:r1 + 1, c0] = ringc; c[r0:r1 + 1, c1] = ringc
    c[r0 + 1, c0 + 1:c1] = ringc; c[r1 - 1, c0 + 1:c1] = ringc
    c[r0 + 1:r1, c0 + 1] = ringc; c[r0 + 1:r1, c1 - 1] = ringc

    ring2 = set()
    for j in range(c0 + 1, c1):
        ring2.add((r0 + 1, j)); ring2.add((r1 - 1, j))
    for i in range(r0 + 1, r1):
        ring2.add((i, c0 + 1)); ring2.add((i, c1 - 1))
    corners2 = {(r0 + 1, c0 + 1), (r0 + 1, c1 - 1), (r1 - 1, c0 + 1), (r1 - 1, c1 - 1)}
    cands = sorted(ring2 - corners2)
    numc = _unifint(diff_lb, diff_ub, (1, len(cands) // 2))
    locs = random.sample(cands, numc)

    gi = c.copy()
    for (i, j) in locs:
        gi[i, j] = inc

    lm = r1 - 1
    hori = [ij for ij in locs if r0 + 1 < ij[0] < lm]
    verti = [ij for ij in locs if ij not in hori]
    full = set()
    for (i, j) in hori:
        for jj in range(w):
            full.add((i, jj))
    for (i, j) in verti:
        for ii in range(h):
            full.add((ii, j))
    go = c.copy()
    for (i, j) in full:
        if c[i, j] == bgc:
            go[i, j] = inc
        elif c[i, j] == inc:
            go[i, j] = ringc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # Roles by bounding-box area: inc (innermost) < ringc (frame box) < bgc (whole grid)
    areas = {}
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        areas[int(col)] = (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)
    order = sorted(areas, key=lambda k: areas[k])
    inc, ringc = order[0], order[1]

    rs, cs = np.where(I == ringc)
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    # Pokes: inc cells sitting on the inner ring
    beam_rows, beam_cols = set(), set()
    for j in range(c0 + 2, c1 - 1):
        if I[r0 + 1, j] == inc or I[r1 - 1, j] == inc:
            beam_cols.add(j)
    for i in range(r0 + 2, r1 - 1):
        if I[i, c0 + 1] == inc or I[i, c1 - 1] == inc:
            beam_rows.add(i)
    beam_rows = sorted(beam_rows)
    beam_cols = sorted(beam_cols)

    ops, sels = [], []

    # 1. The whole box turns frame-coloured: every inc cell (interior + pokes) -> ringc
    inc_cells = [(int(r), int(c)) for r, c in zip(*np.where(I == inc))]
    ops.append(ringc); sels.append(sel_of(inc_cells))

    # 2. The interior is tiled into blocks cut apart by the beam rows/cols;
    #    each block is restored to inc as one rectangle.
    def runs(lo, hi, cut):
        out, cur = [], []
        for x in range(lo, hi + 1):
            if x in cut:
                if cur:
                    out.append(cur); cur = []
            else:
                cur.append(x)
        if cur:
            out.append(cur)
        return out

    row_runs = runs(r0 + 2, r1 - 2, set(beam_rows))
    col_runs = runs(c0 + 2, c1 - 2, set(beam_cols))
    for rr in row_runs:
        for cc in col_runs:
            block = [(r, c) for r in rr for c in cc]
            ops.append(inc); sels.append(sel_of(block))

    # 3. Beams escape the box: inc-coloured lines across the background
    for i in beam_rows:
        if c0 > 0:
            ops.append(inc); sels.append(sel_of([(i, c) for c in range(0, c0)]))
        if c1 < w - 1:
            ops.append(inc); sels.append(sel_of([(i, c) for c in range(c1 + 1, w)]))
    for j in beam_cols:
        if r0 > 0:
            ops.append(inc); sels.append(sel_of([(r, j) for r in range(0, r0)]))
        if r1 < h - 1:
            ops.append(inc); sels.append(sel_of([(r, j) for r in range(r1 + 1, h)]))

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
