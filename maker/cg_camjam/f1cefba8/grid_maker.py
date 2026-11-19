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
    return max(a, min(b, a + int(round((b - a) * d))))


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

    c = [[bgc] * w for _ in range(h)]
    for i in range(r0, r1 + 1):
        for j in range(c0, c1 + 1):
            c[i][j] = inc

    def box(a0, a1, b0, b1):
        s = set()
        for j in range(b0, b1 + 1):
            s.add((a0, j)); s.add((a1, j))
        for i in range(a0, a1 + 1):
            s.add((i, b0)); s.add((i, b1))
        return s

    ring1 = box(r0, r1, c0, c1)
    ring2 = box(r0 + 1, r1 - 1, c0 + 1, c1 - 1)
    for (i, j) in ring1 | ring2:
        c[i][j] = ringc
    corners2 = {(r0 + 1, c0 + 1), (r0 + 1, c1 - 1), (r1 - 1, c0 + 1), (r1 - 1, c1 - 1)}
    cands = sorted(ring2 - corners2)
    numc = _unifint(diff_lb, diff_ub, (1, len(cands) // 2))
    locs = random.sample(cands, numc)

    gi = [row[:] for row in c]
    for (i, j) in locs:
        gi[i][j] = inc

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
    go = [row[:] for row in c]
    for (i, j) in full:
        if c[i][j] == bgc:
            go[i][j] = inc
        elif c[i][j] == inc:
            go[i][j] = ringc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # roles from bbox area: background spans the grid, the frame spans the box,
    # the inner colour lives strictly inside the frame
    areas = {}
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        areas[int(col)] = (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)
    order = sorted(areas, key=lambda k: areas[k])
    inc, ringc, bgc = order[0], order[1], order[-1]

    rs, cs = np.where(I == ringc)
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    # pokes: inner-colour cells sitting on the inner ring
    row_lines, col_lines = set(), set()
    for j in range(c0 + 2, c1 - 1):
        for i in (r0 + 1, r1 - 1):
            if I[i, j] == inc:
                col_lines.add(j)
    for i in range(r0 + 2, r1 - 1):
        for j in (c0 + 1, c1 - 1):
            if I[i, j] == inc:
                row_lines.add(i)

    G = I.copy()
    ops, sels = [], []

    # Each poke fires a full grid line (row or column). Along that line, swap colours
    # one step: inner colour -> frame colour, then background -> inner colour.
    def do_line(cells):
        inc_cells = [(r, c) for r, c in cells if G[r, c] == inc]
        if inc_cells:
            ops.append(ringc); sels.append(sel_of(inc_cells))
            for r, c in inc_cells:
                G[r, c] = ringc
        bg_cells = [(r, c) for r, c in cells if G[r, c] == bgc]
        if bg_cells:
            ops.append(inc); sels.append(sel_of(bg_cells))
            for r, c in bg_cells:
                G[r, c] = inc

    for i in sorted(row_lines):
        do_line([(i, j) for j in range(w)])
    for j in sorted(col_lines):
        do_line([(i, j) for i in range(h)])

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
