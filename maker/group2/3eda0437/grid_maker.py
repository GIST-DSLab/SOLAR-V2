"""
ARC Task: 3eda0437 (RE-ARC) — LLM-generated grid_maker
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
    # Background (0) and highlight (6) are fixed by the generator.
    # Only the noise colors are random. The rule depends only on where the zeros are,
    # but a fixed pool keeps the palette the same across the episode.
    cols = [c for c in range(1, 10) if c != 6]
    pool = random.sample(cols, 8)
    return {"fgc_pool": pool}


def _max_zero_rects(g):
    g = np.asarray(g, dtype=int)
    hi, wi = g.shape
    Z = (g == 0).astype(int)
    P = np.zeros((hi + 1, wi + 1), dtype=int)
    P[1:, 1:] = Z.cumsum(0).cumsum(1)

    def zsum(r, c, a, b):
        return int(P[r + a, c + b] - P[r, c + b] - P[r + a, c] + P[r, c])

    maxpossw = max(int(Z[r, :].sum()) for r in range(hi))
    maxpossh = max(int(Z[:, c].sum()) for c in range(wi))
    best, rects = -1, []
    for a in range(2, min(maxpossh, hi) + 1):
        for b in range(2, min(maxpossw, wi) + 1):
            area = a * b
            if area < best:
                continue
            occ = [(r, c, a, b) for r in range(hi - a + 1)
                   for c in range(wi - b + 1) if zsum(r, c, a, b) == area]
            if not occ:
                continue
            if area > best:
                best, rects = area, []
            rects.extend(occ)
    return rects


def generate(diff_lb, diff_ub, max_h, max_w, fgc_pool=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi_ = a + int(round((b - a) * ub))
        return random.randint(min(lo, hi_), max(lo, hi_))

    if fgc_pool is None:
        fgc_pool = random.sample([c for c in range(1, 10) if c != 6], 8)
    for _ in range(100):
        h = unifint(diff_lb, diff_ub, (3, 8))
        w = unifint(diff_lb, diff_ub, (3, 30))
        if random.choice((True, False)):
            h, w = w, h
        if h <= max_h and w <= max_w:
            break
    else:
        h, w = min(h, max_h), min(w, max_w)
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    fgcs = fgc_pool[:ncols]
    gi = [[random.choice(fgcs) for _ in range(w)] for _ in range(h)]
    spac = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 3 * 2)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    for (i, j) in random.sample(inds, min(spac, len(inds))):
        gi[i][j] = 0
    li, lj = random.randint(0, h - 1), random.randint(0, w - 1)
    for (i, j) in [(li, lj), (li, lj + 1), (li + 1, lj), (li + 1, lj + 1)]:
        if 0 <= i < h and 0 <= j < w:
            gi[i][j] = 0
    go = [row[:] for row in gi]
    for (r, c, a, b) in _max_zero_rects(gi):
        for i in range(r, r + a):
            for j in range(c, c + b):
                go[i][j] = 6
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Rule (read from I alone): find the largest area among all-zero rectangles
    # a x b with a, b >= 2. Every occurrence of every shape with that area
    # becomes 6.
    # Unit of work: one grid ROW at a time. Each Color6 paints the cells of that
    # row that are covered by the maximal zero rectangles (a horizontal slice
    # of those blocks, worked out from I).
    rects = _max_zero_rects(I)
    row_cells = {}
    for (r, c, a, b) in rects:
        for i in range(r, r + a):
            row_cells.setdefault(i, set()).update(range(c, c + b))

    ops, sels = [], []
    for i in sorted(row_cells):
        cells = sorted((i, j) for j in row_cells[i])
        ops.append(6)
        sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3eda0437"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3eda0437"
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
                                f"for task 3eda0437"
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
                    f"Failed to build a complete episode for task 3eda0437 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3eda0437-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
