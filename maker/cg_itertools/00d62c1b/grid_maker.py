"""
ARC Task: 00d62c1b (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int(round((b - a) * max(0.0, min(1.0, diff_lb))))
    hi = a + int(round((b - a) * max(0.0, min(1.0, diff_ub))))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _components(grid, color, h, w):
    seen = [[False] * w for _ in range(h)]
    comps = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] or grid[sr][sc] != color:
                continue
            comp = []
            stack = [(sr, sc)]
            seen[sr][sc] = True
            borders = False
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    borders = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps.append((comp, borders))
    return comps


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    if bgc is None or fgc is None:
        cols = [c for c in range(10) if c != 4]
        bgc, fgc = random.sample(cols, 2)
    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    nblocks = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    succ = 0
    tr = 0
    maxtr = 5 * nblocks
    inds = {(i, j) for i in range(h) for j in range(w)}
    while succ < nblocks and tr < maxtr:
        tr += 1
        oh = random.randint(3, 8)
        ow = random.randint(3, 8)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        bx = set()
        for j in range(locj, locj + ow):
            bx.add((loci, j)); bx.add((loci + oh - 1, j))
        for i in range(loci, loci + oh):
            bx.add((i, locj)); bx.add((i, locj + ow - 1))
        corners = [(loci, locj), (loci, locj + ow - 1),
                   (loci + oh - 1, locj), (loci + oh - 1, locj + ow - 1)]
        bx -= set(random.sample(corners, random.randint(0, 4)))
        if bx.issubset(inds) and len(inds - bx) > (h * w) // 2 + 1:
            for (i, j) in bx:
                gi[i][j] = fgc
            succ += 1
            inds = inds - bx
    nfg = sum(row.count(fgc) for row in gi)
    maxnnoise = max(0, (h * w) // 2 - 1 - nfg)
    namt = _unifint(diff_lb, diff_ub, (0, maxnnoise))
    namt = min(namt, len(inds))
    for (i, j) in random.sample(sorted(inds), namt):
        gi[i][j] = fgc
    go = [row[:] for row in gi]
    for comp, borders in _components(gi, bgc, h, w):
        if not borders:
            for (r, c) in comp:
                go[r][c] = 4
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading taken from the golf entry: treat ALL background as "enclosed" first
    # (paint the whole bgc colour class 4), then let the outside leak back in from
    # the grid border: each 4-region that touches an edge is flood-filled back to bgc.
    # What remains 4 is exactly the enclosed background.
    grid = [list(map(int, row)) for row in I]
    h = len(grid)
    w = len(grid[0])

    # generator guarantees background is strictly the majority colour
    bgc = Counter(v for row in grid for v in row).most_common(1)[0][0]

    ops, sels = [], []
    comps = _components(grid, bgc, h, w)
    enclosed = [comp for comp, borders in comps if not borders]

    if enclosed:
        # 1. every background cell (all cells of one colour in I) -> 4
        bg_cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == bgc]
        ops.append(4)
        sels.append(sel_of(bg_cells))
        # 2. outside leaks back in: one flood from a border seed per edge-touching region
        #    (4 never occurs in I, so 4-regions coincide with I's bgc components)
        for comp, borders in comps:
            if not borders:
                continue
            seed = next((r, c) for (r, c) in comp
                        if r == 0 or r == h - 1 or c == 0 or c == w - 1)
            ops.append(10 + bgc)
            sels.append(sel_of([seed]))
    # if nothing is enclosed, the grid is already the answer; painting and
    # restoring would just bring the grid back to a state it already had.

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
                        f"num_examples+1 ({num_examples + 1}) for task 00d62c1b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 00d62c1b"
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
                                f"for task 00d62c1b"
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
                    f"Failed to build a complete episode for task 00d62c1b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"00d62c1b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
