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
import numpy as np
from collections import deque


def sample_colors(num_examples=None) -> dict:
    # generator samples bgc, linc from colors excluding 3 and 4 (output colors)
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def _components(g, color):
    hh, ww = len(g), len(g[0])
    seen = [[False] * ww for _ in range(hh)]
    comps = []
    for r in range(hh):
        for c in range(ww):
            if g[r][c] != color or seen[r][c]:
                continue
            q = deque([(r, c)])
            seen[r][c] = True
            comp = []
            while q:
                i, j = q.popleft()
                comp.append((i, j))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < hh and 0 <= nj < ww and not seen[ni][nj] and g[ni][nj] == color:
                        seen[ni][nj] = True
                        q.append((ni, nj))
            comps.append(comp)
    return comps


def _is_full_cell(comp):
    # rule (verifier x3/x4/x6): component is a solid rectangle AND has size > 1
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
    return len(comp) == area and len(comp) > 1


def _classify(g, bgc):
    out = [row[:] for row in g]
    for comp in _components(g, bgc):
        col = 3 if _is_full_cell(comp) else 4
        for (i, j) in comp:
            out[i][j] = col
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc) -> dict:
    hub = max(2, min(5, max_h // 3 - 1))
    wub = max(2, min(5, max_w // 3 - 1))
    h = _unifint(diff_lb, diff_ub, (2, hub))
    w = _unifint(diff_lb, diff_ub, (2, wub))
    nh_ub = max(3, max_h // (h + 1))
    nw_ub = max(3, max_w // (w + 1))
    nh = _unifint(diff_lb, diff_ub, (3, nh_ub))
    nw = _unifint(diff_lb, diff_ub, (3, nw_ub))
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1

    base = [[bgc] * fullw for _ in range(fullh)]
    for iloc in range(h, fullh, h + 1):
        for j in range(fullw):
            base[iloc][j] = linc
    for jloc in range(w, fullw, w + 1):
        for i in range(fullh):
            base[i][jloc] = linc

    linecells = set()
    for i in range(fullh):
        for j in range(fullw):
            if base[i][j] == linc:
                linecells.add((i, j))
    dots = set()
    for (i, j) in linecells:
        if {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)} <= linecells:
            dots.add((i, j))

    tmp = [row[:] for row in base]
    for (i, j) in dots:
        tmp[i][j] = bgc
    cands = [frozenset(c) for c in _components(tmp, linc)]
    cands += [frozenset([d]) for d in dots]

    nbreaks = _unifint(diff_lb, diff_ub, (0, len(cands) // 2))
    nbreaks = max(1, min(nbreaks, len(cands)))

    best = None
    for _ in range(20):
        gi = [row[:] for row in base]
        for obj in random.sample(cands, nbreaks):
            i, j = random.choice(sorted(obj))
            gi[i][j] = bgc
        go = _classify(gi, bgc)
        vals = {v for row in go for v in row}
        if best is None:
            best = (gi, go)
        if 3 in vals and 4 in vals:
            best = (gi, go)
            break
    gi, go = best
    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    bgc = int(I[0, 0])  # canvas color: (0,0) is never a line cell

    ops, sels = [], []
    grid = I.tolist()
    # every 4-connected background region: solid rectangle of >1 cell (an intact
    # lattice cell) -> 3, anything else (cells merged through a broken line, or a
    # lone hole in a line crossing) -> 4.  Colour comes from I's geometry only.
    for comp in _components(grid, bgc):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if _is_full_cell(comp):
            ops.append(3)
            sels.append([r0, c0, r1 - r0, c1 - c0])   # bbox == the whole region
        else:
            ops.append(14)                            # FloodFill4 from one seed
            sels.append([comp[0][0], comp[0][1], 0, 0])

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
