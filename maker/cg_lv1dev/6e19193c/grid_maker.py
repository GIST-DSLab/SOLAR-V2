"""
ARC Task: 6e19193c (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

# Comparison with the maker's reading:
#   Both readings agree on the rule. Each object is a 2x2 block with one corner
#   missing. A ray of the object's colour starts just past the missing corner,
#   runs diagonally away from it, and continues to the grid edge.
#   The code-golf entry splits the work differently. It turns the whole grid four
#   times and handles ONE diagonal direction per turn, doing every object that
#   faces that way together. It also treats each ray as a whole line of
#   background cells, not one cell at a time.
#   So the operations here come in four direction passes. Inside a pass, each ray
#   is painted with ONE Color op on the entire diagonal line, not one op per cell.
#   Like the golf regex, a ray stops at the first cell that is not background.

def sample_colors(num_examples=None) -> dict:
    # The rule depends only on the shape of each object, not on its colour.
    # Only the background has to stay the same across the episode.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(max(a, round(a + (b - a) * d)), b)

    def shoot(start, direction, h, w):
        out = set()
        r, c = start
        dr, dc = direction
        for _ in range(42):
            out.add((r, c))
            r += dr
            c += dc
        return out

    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    dirs = (
        ((0, 0), (-1, -1)),
        ((0, 1), (-1, 1)),
        ((1, 0), (1, -1)),
        ((1, 1), (1, 1)),
    )
    base = {(0, 0), (1, 0), (0, 1), (1, 1)}
    candsi = [base - {dr[0]} for dr in dirs]
    candso = [(base | shoot(dr[0], dr[1], 0, 0)) - {dr[0]} for dr in dirs]
    cands = list(zip(candsi, candso))
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 8)))
    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    fullinds = {(i, j) for i in range(h) for j in range(w)}
    inds = set(fullinds)
    kk, tr = 0, 0
    maxtrials = num * 4
    while kk < num and tr < maxtrials:
        if len(inds) == 0:
            break
        loc = random.choice(tuple(inds))
        obji, objo = random.choice(cands)
        obji = {(r + loc[0], c + loc[1]) for r, c in obji}
        objo = {(r + loc[0], c + loc[1]) for r, c in objo} & fullinds
        if objo.issubset(inds) and obji.issubset(objo):
            col = random.choice(remcols)
            for r, c in obji:
                gi[r][c] = col
            for r, c in objo:
                go[r][c] = col
            dn = set()
            for r, c in obji:
                dn |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
            inds = (inds - objo) - dn
            kk += 1
        tr += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The generator paints the background first, and objects can cover at most
    # 3/8 of the grid, so the background is the most common colour in I.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Find every object: a 3-cell, single-colour, 4-connected component whose
    # bounding box is 2x2. Work out its ray direction from the missing corner.
    seen = np.zeros((hi, wi), dtype=bool)
    objs = []  # (dy, dx, start_y, start_x, colour)
    for sr in range(hi):
        for sc in range(wi):
            if seen[sr, sc] or I[sr, sc] == bgc:
                continue
            col = int(I[sr, sc])
            stack = [(sr, sc)]
            seen[sr, sc] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for ddy, ddx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + ddy, x + ddx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) != 3:
                continue
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, c0 = min(rs), min(cs)
            if max(rs) - r0 != 1 or max(cs) - c0 != 1:
                continue
            cells = set(comp)
            missing = [(y, x) for y in (r0, r0 + 1) for x in (c0, c0 + 1) if (y, x) not in cells]
            if len(missing) != 1:
                continue
            my, mx = missing[0]
            dy = -1 if my == r0 else 1
            dx = -1 if mx == c0 else 1
            objs.append((dy, dx, my + dy, mx + dx, col))

    ops, sels = [], []
    # One pass per diagonal direction. In each pass, every object facing that way
    # gets its whole ray painted as one line.
    for (pdy, pdx) in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        for dy, dx, y, x, col in objs:
            if (dy, dx) != (pdy, pdx):
                continue
            ray = []
            while 0 <= y < hi and 0 <= x < wi and I[y, x] == bgc:
                ray.append((y, x))
                y += dy
                x += dx
            if ray:
                ops.append(col)
                sels.append(sel_of(ray))

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
                        f"num_examples+1 ({num_examples + 1}) for task 6e19193c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6e19193c"
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
                                f"for task 6e19193c"
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
                    f"Failed to build a complete episode for task 6e19193c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6e19193c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
