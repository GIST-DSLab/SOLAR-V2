"""
ARC Task: 7447852a (RE-ARC) — LLM-generated grid_maker
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
from collections import deque

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Colour 4 is fixed as the fill colour, so it is never used for the
    # background, the line or the noise.
    cols = [c for c in range(10) if c != 4]
    bgc, linc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, linc)]
    return {"bgc": bgc, "linc": linc, "ccols_pool": remcols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, ccols_pool=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    cols = [c for c in range(10) if c != 4]
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    if ccols_pool is None:
        ccols_pool = [c for c in cols if c not in (bgc, linc)]

    # The canvas is (H, W) before rot90 and (W, H) after it,
    # so W is bounded by max_h and H by max_w.
    W = unifint(diff_lb, diff_ub, (2, min(8, max_h)))
    H = unifint(diff_lb, diff_ub, (W + 1, max(W + 1, min(30, max_w))))
    g = [[bgc] * W for _ in range(H)]
    r, c = H - 1, 0
    g[r][c] = linc
    direc = 1
    while True:
        r, c = r - 1, c + direc
        if c == W - 1 or c == 0:
            direc *= -1
        if r < 0 or c < 0 or c >= W:
            break
        if g[r][c] == linc:
            break
        g[r][c] = linc
    gi = np.array(np.rot90(np.array(g), -1))  # DSL rot90 (clockwise)
    h, w = gi.shape

    # The pockets of background cut out by the zigzag, ordered by leftmost column.
    seen = np.zeros((h, w), bool)
    regs = []
    for i in range(h):
        for j in range(w):
            if seen[i, j] or gi[i, j] != bgc:
                continue
            q = deque([(i, j)])
            seen[i, j] = True
            cs = []
            while q:
                y, x = q.popleft()
                cs.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and gi[ny, nx] == bgc:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            regs.append(cs)
    regs.sort(key=lambda cs: min(x for _, x in cs))

    # Noise is painted only on background cells, never on the line.
    inds = [(i, j) for i in range(h) for j in range(w) if gi[i, j] == bgc]
    numcols = unifint(diff_lb, diff_ub, (1, min(7, len(ccols_pool))))
    ccols = random.sample(ccols_pool, numcols)
    ncells = unifint(diff_lb, diff_ub, (0, len(inds)))
    for (i, j) in random.sample(inds, ncells):
        gi[i, j] = random.choice(ccols)

    go = gi.copy()
    for cs in regs[0::3]:
        for (i, j) in cs:
            go[i, j] = 4
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # The zigzag line always starts at the top-left corner, and no noise uses its colour.
    linc = int(I[0, 0])

    # Comparing the two readings:
    #  - Where they agree: both use the same set of cells. The zigzag cuts the
    #    grid into triangular pockets, and every third pocket from the left
    #    (index 0, 3, 6, ...) becomes 4. The golf table, with period 12 for
    #    3 rows, is that same pattern written out per column: one cycle of
    #    the zigzag is 2*(rows-1) columns, and three cycles cover six pockets.
    #  - Where they differ: the golf entry works one column at a time and
    #    fills that column's cells. The maker fills one whole pocket per op.
    #    Following the column reading, we scan the columns left to right and
    #    fill, in each column, the cells that lie in a chosen pocket.
    #    Each selection is "this column's part of pocket k", which can be
    #    found from the input alone.
    label = -np.ones((h, w), dtype=int)
    regions = []
    for r in range(h):
        for c in range(w):
            if label[r, c] >= 0 or I[r, c] == linc:
                continue
            k = len(regions)
            q = deque([(r, c)])
            label[r, c] = k
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and label[ny, nx] < 0 and I[ny, nx] != linc:
                        label[ny, nx] = k
                        q.append((ny, nx))
            regions.append(cells)

    order = sorted(range(len(regions)),
                   key=lambda k: (min(c for _, c in regions[k]), min(r for r, _ in regions[k])))
    rank = {k: i for i, k in enumerate(order)}
    painted = {k for k in order if rank[k] % 3 == 0}

    ops, sels = [], []
    for c in range(w):
        col_cells = [(r, c) for r in range(h) if label[r, c] in painted]
        if col_cells:
            ops.append(4)
            sels.append(sel_of(col_cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 7447852a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7447852a"
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
                                f"for task 7447852a"
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
                    f"Failed to build a complete episode for task 7447852a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7447852a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
