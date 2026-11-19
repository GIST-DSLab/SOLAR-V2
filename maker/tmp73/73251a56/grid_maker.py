"""
ARC Task: 73251a56 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

import numpy as np

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback keeps the mask contract
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for r, c in cells]}


# ---------------------------------------------------------------- colors ----
# The task rule is "restore the diagonal symmetry that the noise rectangles
# broke".  Colors that matter as ROLES: the noise color (must be a color that
# never occurs in the clean pattern, otherwise noise is undetectable) and the
# ordered palette the diagonal bands are drawn from.  The only discrete
# structural variant is the final rotation: rot 0/2 leave the pattern symmetric
# about the MAIN diagonal, rot 1/3 about the ANTI diagonal.

_ROT_VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    noisec = random.choice(cols)
    colorder = [c for c in cols if c != noisec]
    random.shuffle(colorder)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(_ROT_VARIANTS):
        examples = [dict(v) for v in _ROT_VARIANTS]
        examples += [dict(random.choice(_ROT_VARIANTS)) for _ in range(n_ex - len(_ROT_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(_ROT_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"noisec": noisec, "colorder": colorder, "instance_plan": plan}


# -------------------------------------------------------------- generator ----
def generate(diff_lb, diff_ub, max_h, max_w, noisec, colorder, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    def _uni(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    def _rot90(g):  # CCW, like np.rot90(k=1)
        return [list(r) for r in list(zip(*g))[::-1]]

    upper = min(30, max_h, max_w)
    lower = min(10, upper)

    while True:
        d = _uni(diff_lb, diff_ub, (lower, upper))
        h = w = d
        nsl = _uni(diff_lb, diff_ub, (2, min(9, h // 2)))
        nsl = max(2, min(nsl, len(colorder)))
        slopes = [0] + sorted(random.sample(list(range(1, h - 1)), nsl - 1))
        ccols = list(colorder[:nsl])

        grid = [[0] * w for _ in range(h)]
        for col, hdelt in zip(ccols, slopes):
            slope = hdelt / w
            for i in range(h):
                for j in range(w):
                    if slope * j <= i:
                        grid[i][j] = col

        # diagonal line, then mirror the upper triangle onto the lower one
        diagcol = ccols[-2]
        for i in range(h):
            grid[i][i] = diagcol
        for i in range(h):
            for j in range(i, w):
                grid[j][i] = grid[i][j]

        go = [row[:] for row in grid]

        ndist = _uni(diff_lb, diff_ub, (1, max(1, (h * w) // 15)))
        maxtr = 10 * ndist
        tr = 0
        succ = 0
        while tr < maxtr and succ < ndist:
            tr += 1
            oh = random.randint(1, 5)
            ow = random.randint(1, 5)
            if h - oh - 1 < 1 or w - ow - 1 < 1:
                continue
            loci = random.randint(1, h - oh - 1)
            locj = random.randint(1, w - ow - 1)
            trial = [row[:] for row in grid]
            for r in range(loci, loci + oh):
                for c in range(locj, locj + ow):
                    trial[r][c] = noisec
            # cf1: the diagonal must keep at least one of its own cells visible
            if not any(trial[i][i] == diagcol for i in range(h)):
                continue
            # cf2: no off-diagonal mirrored pair may be noise on BOTH sides
            nz = set()
            for r in range(h):
                for c in range(w):
                    if trial[r][c] == noisec:
                        nz.add((r, c))
            bad = False
            for (r, c) in nz:
                if r != c and (c, r) in nz:
                    bad = True
                    break
            if bad:
                continue
            grid = trial
            succ += 1

        if grid != go:
            break

    gi = grid
    for _ in range(rot % 4):
        gi = _rot90(gi)
        go = _rot90(go)

    return {
        "input": tuple(tuple(int(v) for v in row) for row in gi),
        "output": tuple(tuple(int(v) for v in row) for row in go),
    }


# ------------------------------------------------------------ derivation ----
def _components(cells, hi, wi):
    """4-connected blobs of a set of cells (one stamped rectangle each)."""
    todo = set(cells)
    blobs = []
    for cell in sorted(todo):
        if cell not in todo:
            continue
        todo.discard(cell)
        dq = deque([cell])
        comp = [cell]
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in todo:
                    todo.discard(nb)
                    dq.append(nb)
                    comp.append(nb)
        blobs.append(sorted(comp))
    blobs.sort(key=lambda comp: comp[0])
    return blobs


def derive_operations(I, O):
    """Rule (read off I alone):

    The picture is its own mirror image across one of the two diagonals;
    opaque rectangles of a single 'noise' colour were stamped on top of it.

    So the repair IS the reflection: mirror the whole grid across that
    diagonal (Rotate270 + FlipV = transpose for the main diagonal,
    Rotate270 + FlipH = anti-transpose for the anti diagonal).  Because the
    picture is symmetric everywhere except at the stamps, that single
    reflection changes nothing but the stamps: every damaged cell is now
    showing the intact colour from the far side of the diagonal, and the
    damage itself has been carried across to the mirror positions.  What the
    reflection leaves to paint is exactly that: restore each carried-over
    blob to the colours it has in I (they are intact there), and redraw the
    diagonal line itself where a stamp sat on top of it, since cells on the
    axis are their own mirror and the reflection cannot heal them.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    n = min(hi, wi)
    ops, sels = [], []

    # --- 1. which diagonal is the symmetry axis, and what is the noise? -----
    # Both candidate axes are scored the same way: collect the mirrored pairs
    # that disagree, then ask whether ONE colour can account for all of them
    # (a stamp on exactly one side of every disagreeing pair).  The true axis
    # is explained completely by the stamp colour; the wrong axis leaves
    # structural disagreements no single colour can cover.  Counting raw
    # disagreements is not enough — with many stamps the two counts come out
    # nearly equal and the wrong axis can win.
    def _axis_info(kind):
        if kind == "main":
            mir_f = lambda r, c: (c, r)
            on_f = lambda r, c: r == c
            cells = [(i, i) for i in range(n)]
        else:
            mir_f = lambda r, c: (n - 1 - c, n - 1 - r)
            on_f = lambda r, c: r + c == n - 1
            cells = [(i, n - 1 - i) for i in range(n)]
        pairs = []
        for r in range(n):
            for c in range(n):
                if on_f(r, c):
                    continue
                mr, mc = mir_f(r, c)
                if (mr, mc) < (r, c):
                    continue
                a, b = int(I[r, c]), int(I[mr, mc])
                if a != b:
                    pairs.append((a, b))
        cand = set()
        best = 1.0
        if pairs:
            best = 0.0
            for col in range(10):
                cnt = sum(1 for a, b in pairs if (a == col) != (b == col))
                frac = cnt / len(pairs)
                if frac > best:
                    best = frac
                if cnt == len(pairs):
                    cand.add(col)
        vals = [int(I[r, c]) for (r, c) in cells]
        pure = Counter(vals).most_common(1)[0][1] / float(n)
        return {"mir": mir_f, "on": on_f, "cells": cells, "vals": vals,
                "pairs": pairs, "cand": cand, "score": best, "pure": pure}

    info_main = _axis_info("main")
    info_anti = _axis_info("anti")
    if (info_main["score"], info_main["pure"]) >= (info_anti["score"], info_anti["pure"]):
        axis, info = "main", info_main
        # transpose = np.flipud(np.rot90(g, 1)) -> Rotate270(op24) then FlipV(op27)
        mirror_ops = [24, 27]
    else:
        axis, info = "anti", info_anti
        # anti-transpose = np.fliplr(np.rot90(g, 1)) -> Rotate270(op24) then FlipH(op26)
        mirror_ops = [24, 26]

    mir = info["mir"]
    on_axis = info["on"]
    axis_cells = info["cells"]
    axis_vals = info["vals"]

    # --- 2. which colour is the noise? --------------------------------------
    cand = set(info["cand"])
    if not cand:
        dom = Counter(axis_vals).most_common(1)[0][0]
        cand = {v for v in axis_vals if v != dom}

    if len(cand) > 1:
        # a stamp colour never occurs in the clean pattern, so it cannot be one
        # of the band colours that run out to the border
        border = set(I[0, :].tolist()) | set(I[n - 1, :].tolist())
        border |= set(I[:, 0].tolist()) | set(I[:, n - 1].tolist())
        inner = {c for c in cand if c not in border}
        if inner:
            cand = inner

    if not cand:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    noisec = min(cand, key=lambda c: int((I == c).sum()))

    clean_axis = [v for v in axis_vals if v != noisec]
    axis_col = Counter(clean_axis).most_common(1)[0][0] if clean_axis else noisec

    # --- 3. the damage, split into what the reflection can and cannot heal ---
    # off-axis stamp cells: their mirror is intact, so the reflection heals them
    broken = set()
    for r in range(n):
        for c in range(n):
            if on_axis(r, c):
                continue
            if int(I[r, c]) == noisec and int(I[mir(r, c)]) != noisec:
                broken.add((r, c))
    # the reflection carries that same damage over to the mirror positions
    carried = {mir(r, c) for (r, c) in broken}
    # cells sitting ON the axis are their own mirror — only the line redraws them
    on_line = [p for p in axis_cells if int(I[p[0], p[1]]) != axis_col]

    # --- 4. reflect the whole grid across its symmetry diagonal --------------
    # full-rectangle selection: the reflection acts on the entire canvas.
    if broken and hi == wi:
        full = [0, 0, hi - 1, wi - 1]
        for op in mirror_ops:
            ops.append(op)
            sels.append(full)

        # --- 5. put back the blobs the reflection carried across ------------
        for comp in _components(carried, hi, wi):
            groups = {}
            for (r, c) in comp:
                groups.setdefault(int(I[r, c]), []).append((r, c))
            for col in sorted(groups, key=lambda k: min(groups[k])):
                ops.append(int(col))
                sels.append(sel_of(sorted(groups[col])))
    else:
        # degenerate: every stamp sits on the axis, so there is nothing for the
        # reflection to do — the line redraw below is the whole repair.
        for comp in _components({p for p in broken}, hi, wi):
            groups = {}
            for (r, c) in comp:
                groups.setdefault(int(I[mir(r, c)]), []).append((r, c))
            for col in sorted(groups, key=lambda k: min(groups[k])):
                ops.append(int(col))
                sels.append(sel_of(sorted(groups[col])))

    # --- 6. redraw the diagonal line where a stamp covered it ---------------
    if on_line:
        ops.append(int(axis_col))
        sels.append(sel_of(on_line))

    # Submit: bbox covers the whole (unchanged-size) canvas.
    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 73251a56"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 73251a56"
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
                                f"for task 73251a56"
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
                    f"Failed to build a complete episode for task 73251a56 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"73251a56-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
