"""
ARC Task: 83302e8f (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, hi_i))


def _bg_components(grid, bgc):
    H, W = len(grid), len(grid[0])
    seen = [[False] * W for _ in range(H)]
    comps = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] != bgc or seen[i][j]:
                continue
            seen[i][j] = True
            q = deque([(i, j)])
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] and grid[nr][nc] == bgc:
                        seen[nr][nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    if bgc is None or linc is None:
        cols = [c for c in range(10) if c not in (3, 4)]
        bgc, linc = random.sample(cols, 2)
    h = _unifint(diff_lb, diff_ub, (2, 5))
    w = _unifint(diff_lb, diff_ub, (2, 5))
    while h > 2 and 3 * (h + 1) - 1 > max_h:
        h -= 1
    while w > 2 and 3 * (w + 1) - 1 > max_w:
        w -= 1
    nh = _unifint(diff_lb, diff_ub, (3, max(3, (max_h + 1) // (h + 1))))
    nw = _unifint(diff_lb, diff_ub, (3, max(3, (max_w + 1) // (w + 1))))
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = [[bgc] * fullw for _ in range(fullh)]
    for iloc in range(h, fullh, h + 1):
        for j in range(fullw):
            gi[iloc][j] = linc
    for jloc in range(w, fullw, w + 1):
        for i in range(fullh):
            gi[i][jloc] = linc
    ofc = {(i, j) for i in range(fullh) for j in range(fullw) if gi[i][j] == linc}

    def dn(i, j):
        return {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}

    dots = {p for p in ofc if dn(*p).issubset(ofc)}
    tmp = [row[:] for row in gi]
    for (i, j) in dots:
        tmp[i][j] = bgc
    # line segments of linc in tmp
    seen = set()
    lns = []
    for p in sorted(ofc - dots):
        if p in seen:
            continue
        seen.add(p)
        q = [p]
        comp = []
        while q:
            cur = q.pop()
            comp.append(cur)
            for nb in dn(*cur):
                if nb in ofc and nb not in dots and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        lns.append(tuple(sorted(comp)))
    cands = lns + [(d,) for d in sorted(dots)]
    nbreaks = _unifint(diff_lb, diff_ub, (0, len(cands) // 2))
    breakobjs = random.sample(cands, nbreaks)
    for bo in breakobjs:
        i, j = random.choice(bo)
        gi[i][j] = bgc
    go = [row[:] for row in gi]
    for comp in _bg_components(gi, bgc):
        col = 3 if len(comp) == h * w else 4
        for (i, j) in comp:
            go[i][j] = col
    return {"input": gi, "output": go}


def _is_full_rect(comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    hh = max(rs) - min(rs) + 1
    ww = max(cs) - min(cs) + 1
    return len(comp) > 1 and len(comp) == hh * ww


def derive_operations(I, O, examples=None):
    # The golf reading and the maker agree: each 4-connected background region is
    # one unit, and its colour depends only on whether it is a solid rectangle with
    # more than one cell (an intact lattice cell gets 3, a merged region or a lone
    # crossing hole gets 4). The golf entry also requires H == W, but that only
    # holds because the original ARC cells are square. Here h and w are sampled
    # independently, so that test is dropped.
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = int(I[0, 0])  # (0,0) is always inside a lattice cell, never on a line
    grid = I.tolist()

    ops, sels = [], []
    for comp in _bg_components(grid, bgc):
        if _is_full_rect(comp):
            ops.append(3)                       # intact cell -> paint the region
            sels.append(sel_of(comp))
        else:
            ops.append(14)                      # broken/merged region -> FloodFill4
            sels.append(sel_of([comp[0]]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 83302e8f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 83302e8f"
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
                                f"for task 83302e8f"
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
                    f"Failed to build a complete episode for task 83302e8f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"83302e8f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
