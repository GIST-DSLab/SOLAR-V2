"""
ARC Task: c909285e (RE-ARC) — LLM-generated grid_maker
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
from collections import defaultdict

from maker.sel_helpers import sel_of  # noqa: F401  (selections below are full rectangles)


# ── helpers ───────────────────────────────────────────────────────────────────
def _hfrontier(loc, w):
    return {(loc[0], j) for j in range(w)}


def _vfrontier(loc, h):
    return {(i, loc[1]) for i in range(h)}


def _box_cells(r0, c0, r1, c1):
    cells = set()
    for j in range(c0, c1 + 1):
        cells.add((r0, j)); cells.add((r1, j))
    for i in range(r0, r1 + 1):
        cells.add((i, c0)); cells.add((i, c1))
    return cells


def _frame_colour(I):
    """The colour whose cells are exactly a rectangle outline (smallest such, h,w>1)."""
    hi, wi = len(I), len(I[0])
    by_col = defaultdict(set)
    for r in range(hi):
        for c in range(wi):
            by_col[int(I[r][c])].add((r, c))
    best = None
    for col, cells in by_col.items():
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r1 == r0 or c1 == c0:
            continue
        if cells != _box_cells(r0, c0, r1, c1):
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if best is None or area < best[0]:
            best = (area, col)
    return best[1]


# ── 1. sample_colors ──────────────────────────────────────────────────────────
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    boxcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "boxcol": boxcol}


# ── 2. generate ───────────────────────────────────────────────────────────────
def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxcol, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))

    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    nfronts = unifint(diff_lb, diff_ub, (1, (h + w) // 2))
    remcols = [c for c in range(10) if c != bgc and c != boxcol]
    gi = [[bgc] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    for _ in range(nfronts):
        if not inds:
            break
        loc = random.choice(inds)
        inds.remove(loc)
        col = random.choice(remcols)
        cells = _hfrontier(loc, w) if random.random() < 0.5 else _vfrontier(loc, h)
        for (i, j) in cells:
            gi[i][j] = col
    oh = unifint(diff_lb, diff_ub, (3, max(3, (h - 2) // 2)))
    ow = unifint(diff_lb, diff_ub, (3, max(3, (w - 2) // 2)))
    loci = random.randint(1, h - oh - 1)
    locj = random.randint(1, w - ow - 1)
    for (i, j) in _box_cells(loci, locj, loci + oh - 1, locj + ow - 1):
        gi[i][j] = boxcol
    go = [row[locj:locj + ow] for row in gi[loci:loci + oh]]
    return {"input": gi, "output": go}


# ── 3. derive_operations ──────────────────────────────────────────────────────
def derive_operations(I, O=None, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The frame colour: the one colour whose cells form a closed rectangle outline.
    fc = _frame_colour(I.tolist())
    rows = [r for r in range(hi) if (I[r, :] == fc).any()]
    cols = [c for c in range(wi) if (I[:, c] == fc).any()]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    nh, nw = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []
    # Step 1: keep only the rows that contain the frame colour
    # (full-width band of rows — the intended cells ARE this whole rectangle).
    ops.append(33); sels.append([r0, 0, nh - 1, wi - 1])
    # Step 2: on that band, keep only the columns that contain the frame colour
    # (full-height band of columns — whole rectangle intended).
    ops.append(33); sels.append([0, c0, nh - 1, nw - 1])
    ops.append(34); sels.append([0, 0, nh - 1, nw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c909285e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c909285e"
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
                                f"for task c909285e"
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
                    f"Failed to build a complete episode for task c909285e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c909285e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
