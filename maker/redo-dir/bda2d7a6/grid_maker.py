"""
ARC Task: bda2d7a6 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}

try:
    unifint
except NameError:
    def unifint(diff_lb, diff_ub, bounds):
        a, b = bounds
        if a >= b:
            return a
        lo = a + int((b - a) * diff_lb)
        hi = a + int((b - a) * diff_ub)
        lo = max(a, min(b, lo)); hi = max(a, min(b, hi))
        if lo > hi:
            lo, hi = hi, lo
        return random.randint(lo, hi)


# Discrete structural variant: whether innermost band colour == outermost band colour
# (that case merges them into one object, so the outermost takes the SECOND-innermost colour).
VARIANTS = [{"special": False}, {"special": True}]


def sample_colors(num_examples=None) -> dict:
    ncols = random.randint(2, 10)
    cols = random.sample(list(range(10)), ncols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"cols": cols, "instance_plan": plan}


def _bands(g):
    """4-connected same-colour components, ordered ASCENDING by max(bbox_h, bbox_w)."""
    H, W = len(g), len(g[0])
    seen = [[False] * W for _ in range(H)]
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r][c]:
                continue
            col = g[r][c]
            st = [(r, c)]; seen[r][c] = True; cells = []
            while st:
                rr, cc = st.pop(); cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] and g[nr][nc] == col:
                        seen[nr][nc] = True; st.append((nr, nc))
            comps.append(cells)

    def maxdim(cells):
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        return max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    comps.sort(key=maxdim)
    return comps


def generate(diff_lb, diff_ub, max_h, max_w, cols=None, special=None) -> dict:
    if cols is None:
        ncols = unifint(diff_lb, diff_ub, (2, 10))
        cols = random.sample(list(range(10)), ncols)
    cols = list(cols)
    if special is None:
        special = random.choice([True, False])

    hub = max(2, min(14, max_h // 2))
    wub = max(2, min(14, max_w // 2))
    lo = 3 if special else 2
    if hub < lo or wub < lo:
        special = False
        lo = 2
    hub = max(hub, lo); wub = max(wub, lo)

    h = unifint(diff_lb, diff_ub, (lo, hub))
    w = unifint(diff_lb, diff_ub, (lo, wub))
    n = min(h, w)

    while True:
        colord = [random.choice(cols) for _ in range(n)]
        if special:
            colord[-1] = colord[0]
            if len(set(colord)) >= 2:
                break
        else:
            if colord[0] != colord[-1]:
                break

    H, W = 2 * h, 2 * w
    gi = [[0] * W for _ in range(H)]
    for idx in range(n):
        ci = colord[idx]
        r0, c0, r1, c1 = idx, idx, H - 1 - idx, W - 1 - idx
        for c in range(c0, c1 + 1):
            gi[r0][c] = ci; gi[r1][c] = ci
        for r in range(r0, r1 + 1):
            gi[r][c0] = ci; gi[r][c1] = ci

    comps = _bands(gi)                       # ascending: innermost first
    m = len(comps)
    cl = [gi[cells[0][0]][cells[0][1]] for cells in comps]
    new = [cl[(i + 1) % m] for i in range(m)]
    if m >= 2 and cl[0] == cl[-1]:
        new[-1] = cl[1]
    go = [row[:] for row in gi]
    for i, cells in enumerate(comps):
        for (r, c) in cells:
            go[r][c] = new[i]

    return {'input': tuple(tuple(r) for r in gi), 'output': tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    g = [[int(I[r, c]) for c in range(wi)] for r in range(hi)]
    comps = _bands(g)[::-1]          # outermost concentric band first, then inward
    grid = [row[:] for row in g]

    def flood_region(gr, seed):
        col = gr[seed[0]][seed[1]]
        seen = {seed}; st = [seed]
        while st:
            rr, cc = st.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < hi and 0 <= nc < wi and (nr, nc) not in seen and gr[nr][nc] == col:
                    seen.add((nr, nc)); st.append((nr, nc))
        return seen

    for cells in comps:
        # this band's new colour: the colour of the band just outside it (cyclic)
        target = Counter(int(O[r, c]) for r, c in cells).most_common(1)[0][0]
        if all(grid[r][c] == target for r, c in cells):
            continue                                   # band already holds it: no op
        seed = min(cells)
        if flood_region(grid, seed) == set(cells):
            ops.append(10 + target)                    # FloodFill: recolour this whole band
            sels.append([seed[0], seed[1], 0, 0])      # exactly 1 seed cell
        else:
            # an already-recoloured neighbouring band now shares this band's colour:
            # a flood would leak into it, so paint this band's own cells.
            ops.append(target)
            sels.append(sel_of(sorted(cells)))
        for (r, c) in cells:
            grid[r][c] = target

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
                        f"num_examples+1 ({num_examples + 1}) for task bda2d7a6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bda2d7a6"
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
                                f"for task bda2d7a6"
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
                    f"Failed to build a complete episode for task bda2d7a6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bda2d7a6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
