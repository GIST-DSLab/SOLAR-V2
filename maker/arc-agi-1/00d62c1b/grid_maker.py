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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]      # 4 reserved as fill color
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    if hi < lo:
        lo, hi = hi, lo
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    return random.randint(lo, hi)


def _enclosed_bg_cells(grid, bgc):
    """4-connected bgc components that do NOT touch any grid edge -> set of cells."""
    h = len(grid)
    w = len(grid[0])
    seen = [[False] * w for _ in range(h)]
    result = set()
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] or grid[sr][sc] != bgc:
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
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == bgc:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if not borders:
                result.update(comp)
    return result


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    h = _unifint(diff_lb, diff_ub, (5, max_h))
    w = _unifint(diff_lb, diff_ub, (5, max_w))

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    total = h * w
    nblocks = _unifint(diff_lb, diff_ub, (1, max(1, total // 20)))

    inds = {(r, c) for r in range(h) for c in range(w)}
    succ = 0
    tr = 0
    maxtr = 5 * nblocks
    while succ < nblocks and tr < maxtr:
        tr += 1
        oh = random.randint(3, 8)
        ow = random.randint(3, 8)
        cands = [(r, c) for (r, c) in inds if r <= h - oh and c <= w - ow]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        r0, c0 = loci, locj
        r1, c1 = loci + oh - 1, locj + ow - 1
        bx = set()
        for c in range(c0, c1 + 1):
            bx.add((r0, c)); bx.add((r1, c))
        for r in range(r0, r1 + 1):
            bx.add((r, c0)); bx.add((r, c1))
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        rm = random.sample(corners, random.randint(0, 4))
        bx = bx - set(rm)
        if bx.issubset(inds) and len(inds - bx) > total // 2 + 1:
            for (r, c) in bx:
                gi[r][c] = fgc
            inds = inds - bx
            succ += 1

    fgc_count = sum(1 for row in gi for v in row if v == fgc)
    maxnnoise = max(0, total // 2 - 1 - fgc_count)
    namt = _unifint(diff_lb, diff_ub, (0, maxnnoise))
    namt = min(namt, len(inds))
    noise = random.sample(list(inds), namt) if namt > 0 else []
    for (r, c) in noise:
        gi[r][c] = fgc

    go = [row[:] for row in gi]
    for (r, c) in _enclosed_bg_cells(gi, bgc):
        go[r][c] = 4

    return {"input": gi, "output": go}


def derive_operations(I, O):
    grid = [list(map(int, row)) for row in I]
    h = len(grid)
    w = len(grid[0])
    ho, wo = len(O), len(O[0])

    # background = majority color of I (canvas fill of generator)
    bgc = Counter(v for row in grid for v in row).most_common(1)[0][0]

    ops = []
    sels = []

    # measure enclosure FROM I: bgc components that touch no edge get filled.
    seen = [[False] * w for _ in range(h)]
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] or grid[sr][sc] != bgc:
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
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == bgc:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if not borders:
                # enclosed region -> flood-fill its connected bgc area with 4 from one seed
                r, c = comp[0]
                ops.append(14)            # FloodFill4
                sels.append([r, c, 0, 0])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
