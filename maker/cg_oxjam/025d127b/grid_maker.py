"""
ARC Task: 025d127b (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    def connect(a, b):
        (i1, j1), (i2, j2) = a, b
        if i1 == i2:
            return {(i1, j) for j in range(min(j1, j2), max(j1, j2) + 1)}
        if j1 == j2:
            return {(i, j1) for i in range(min(i1, i2), max(i1, i2) + 1)}
        if abs(i2 - i1) == abs(j2 - j1):
            di = 1 if i2 > i1 else -1
            dj = 1 if j2 > j1 else -1
            return {(i1 + k * di, j1 + k * dj) for k in range(abs(i2 - i1) + 1)}
        return set()

    def shift(p, d):
        return {(i + d[0], j + d[1]) for i, j in p}

    if bgc is None:
        bgc = random.randint(0, 9)
    if ccols is None:
        rem = [c for c in range(10) if c != bgc]
        ccols = random.sample(rem, random.randint(1, 9))
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    succ, tr, maxtr = 0, 0, 5 * nobjs
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(3, 6)
        ow = random.randint(3, 6)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        topl = connect((0, 0), (0, ow - 1))
        leftl = connect((1, 0), (oh - 2, oh - 3))
        rightl = connect((1, ow), (oh - 2, ow + oh - 3))
        botl = connect((oh - 1, oh - 2), (oh - 1, oh - 3 + ow))
        inobj = topl | leftl | rightl | botl
        outobj = (shift(topl, (0, 1)) | botl | shift(leftl, (0, 1))
                  | connect((1, ow + 1), (oh - 3, ow + oh - 3)) | {(oh - 2, ow + oh - 3)})
        rm = max(j for _, j in inobj)
        outobj = {ij for ij in outobj if ij[1] <= rm}
        fullobj = shift(inobj | outobj, loc)
        inobj = shift(inobj, loc)
        outobj = shift(outobj, loc)
        if fullobj.issubset(inds):
            nb = set()
            for (i, j) in fullobj:
                nb |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
            inds = (inds - fullobj) - nb
            succ += 1
            col = random.choice(ccols)
            for (i, j) in inobj:
                gi[i][j] = col
            for (i, j) in outobj:
                go[i][j] = col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour; the shapes are thin outlines, so it is the
    # clear majority colour in every grid this generator makes.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- whole shapes: same colour, 8-connected (the outlines step diagonally)
    seen = np.zeros((hi, wi), dtype=bool)
    shapes = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                and int(I[nr, nc]) == col:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            shapes.append((min(cells), col, cells))
    shapes.sort()

    ops, sels = [], []
    # Reading (row-scan, as in the golf entry): go through each shape row by
    # row, top to bottom. In each row the shape is a set of horizontal runs.
    # Each run moves one cell right, except the shape's right-edge corner: the
    # two runs whose right ends sit one directly above the other (the end of
    # the bottom edge and the last cell of the right side). That corner is
    # pinned and stays where it is.
    def runs_of(cells):
        rows = {}
        for (r, c) in cells:
            rows.setdefault(r, []).append(c)
        out = []
        for r in sorted(rows):
            cs = sorted(rows[r])
            cur = [cs[0]]
            for c in cs[1:]:
                if c == cur[-1] + 1:
                    cur.append(c)
                else:
                    out.append((r, cur)); cur = [c]
            out.append((r, cur))
        return out

    for _, col, cells in shapes:
        runs = runs_of(cells)
        right_ends = {(r, run[-1]) for r, run in runs}
        pinned = set()
        for (r, c) in right_ends:
            if (r + 1, c) in right_ends:          # vertical pair of right ends
                pinned.add((r, c)); pinned.add((r + 1, c))
        for r, run in runs:
            if (r, run[-1]) in pinned:
                continue
            run_cells = [(r, c) for c in run]
            lead = (r, run[-1] + 1)               # cell the run steps into
            tail = (r, run[0])                    # cell the run leaves
            if col != 0:
                ops.append(22); sels.append(sel_of(run_cells))   # MoveR the run
                if bgc != 0:
                    ops.append(bgc); sels.append(sel_of([tail]))
            else:
                # colour 0 is invisible to ARCLE's object buffer: draw the step
                ops.append(0); sels.append(sel_of([lead]))
                ops.append(bgc); sels.append(sel_of([tail]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 025d127b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 025d127b"
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
                                f"for task 025d127b"
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
                    f"Failed to build a complete episode for task 025d127b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"025d127b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
