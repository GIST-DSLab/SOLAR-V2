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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + round((b - a) * diff_lb)
    hi = a + round((b - a) * diff_ub)
    return random.randint(min(lo, hi), max(lo, hi))


def sample_colors(num_examples=None) -> dict:
    bgc, ringc, inc = random.sample(range(10), 3)
    return {"bgc": bgc, "ringc": ringc, "inc": inc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ringc=1, inc=2) -> dict:
    h = _unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = _unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    ih = _unifint(diff_lb, diff_ub, (6, h - 1))
    iw = _unifint(diff_lb, diff_ub, (6, w - 1))
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    r0, c0, r1, c1 = loci, locj, loci + ih - 1, locj + iw - 1
    c = [[bgc] * w for _ in range(h)]
    for i in range(r0, r1 + 1):
        for j in range(c0, c1 + 1):
            ring = (i in (r0, r1) or j in (c0, c1)
                    or i in (r0 + 1, r1 - 1) or j in (c0 + 1, c1 - 1))
            c[i][j] = ringc if ring else inc
    ring2 = set()
    for j in range(c0 + 1, c1):
        ring2.add((r0 + 1, j)); ring2.add((r1 - 1, j))
    for i in range(r0 + 1, r1):
        ring2.add((i, c0 + 1)); ring2.add((i, c1 - 1))
    corners = {(r0 + 1, c0 + 1), (r0 + 1, c1 - 1), (r1 - 1, c0 + 1), (r1 - 1, c1 - 1)}
    cands = sorted(ring2 - corners)
    numc = _unifint(diff_lb, diff_ub, (1, len(cands) // 2))
    locs = random.sample(cands, numc)
    gi = [row[:] for row in c]
    for i, j in locs:
        gi[i][j] = inc
    lm = r1 - 1
    hori = [(i, j) for i, j in locs if r0 + 1 < i < lm]
    verti = [l for l in locs if l not in hori]
    full = set()
    for i, _ in hori:
        full |= {(i, j) for j in range(w)}
    for _, j in verti:
        full |= {(i, j) for i in range(h)}
    go = [row[:] for row in c]
    for i, j in full:
        if c[i][j] == bgc:
            go[i][j] = inc
        elif c[i][j] == inc:
            go[i][j] = ringc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading taken from the golf entry: a beam is ONE WHOLE grid line (the full
    # row/column through a poke), treated as a single unit. Along that line a
    # colour shift is applied: inc -> ringc, then bgc -> inc. The maker cuts each
    # beam into an inside segment and two outside segments using the box
    # geometry. Here the line is never split. Each step selects "the cells of
    # colour X on this line" in the current grid.
    import numpy as np
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    cols = [int(x) for x in np.unique(I)]
    border = (set(I[0].tolist()) | set(I[-1].tolist())
              | set(I[:, 0].tolist()) | set(I[:, -1].tolist()))

    def nbrs(col):
        out = set()
        for r, c in zip(*np.where(I == col)):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and I[rr, cc] != col:
                    out.add(int(I[rr, cc]))
        return out

    # Roles: inc is enclosed by two rings, so it never reaches the border and never
    # touches bgc. ringc is the colour next to inc. bgc is the colour left over.
    inner = [x for x in cols if x not in border]
    if len(inner) == 1:          # frame touches an edge: only inc stays off the border
        inc = inner[0]
    else:                        # only bgc is on the border; inc never touches it
        inc = [x for x in inner if not (nbrs(x) & border)][0]
    ringc = nbrs(inc).pop()
    bgc = [x for x in cols if x not in (inc, ringc)][0]

    rs, cs = np.where(I == ringc)
    r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())

    # Pokes: inc cells sitting on the inner ring.
    pokes = [(int(r), int(c)) for r, c in zip(*np.where(I == inc))
             if r in (r0 + 1, r1 - 1) or c in (c0 + 1, c1 - 1)]
    row_lines = sorted({r for r, c in pokes if r0 + 1 < r < r1 - 1})   # side pokes
    col_lines = sorted({c for r, c in pokes if r in (r0 + 1, r1 - 1)})  # top/bottom pokes

    G = I.copy()
    ops, sels = [], []

    def shift_line(cells):
        cells = list(cells)
        a = [(r, c) for r, c in cells if G[r, c] == inc]      # inc on this line -> ringc
        if a:
            ops.append(ringc); sels.append(sel_of(a))
            for r, c in a:
                G[r, c] = ringc
        b = [(r, c) for r, c in cells if G[r, c] == bgc]      # bgc on this line -> inc
        if b:
            ops.append(inc); sels.append(sel_of(b))
            for r, c in b:
                G[r, c] = inc

    for r in row_lines:
        shift_line((r, c) for c in range(w))
    for c in col_lines:
        shift_line((r, c) for r in range(h))

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
