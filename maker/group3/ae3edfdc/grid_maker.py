"""
ARC Task: ae3edfdc (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Only the canvas colour is random; 1, 2, 3 and 7 are fixed roles.
    cols = [c for c in range(10) if c not in (1, 2, 3, 7)]
    return {"bgc": random.choice(cols)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    import random as _r

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi = a + int(round((b - a) * ub))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return _r.randint(lo, hi)

    def neighbors(loc):
        i, j = loc
        return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)}

    def sgn(v):
        return (v > 0) - (v < 0)

    if bgc is None:
        bgc = _r.choice([c for c in range(10) if c not in (1, 2, 3, 7)])

    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    go = [[bgc] * w for _ in range(h)]
    rd = (_r.randint(1, h - 2), _r.randint(1, w - 2))
    reminds = [(i, j) for i in range(1, h - 1) for j in range(1, w - 1)
               if (i, j) != rd and (i, j) not in neighbors(rd)]
    bd = _r.choice(reminds)
    go[rd[0]][rd[1]] = 2
    go[bd[0]][bd[1]] = 1
    ngd = unifint(diff_lb, diff_ub, (1, 8))
    gd = _r.sample(sorted(neighbors(rd)), ngd)
    nod = unifint(diff_lb, diff_ub, (1, 8))
    od = _r.sample(sorted(neighbors(bd)), nod)
    for (i, j) in gd:
        go[i][j] = 3
    for (i, j) in od:
        go[i][j] = 7
    mpr = {}
    for d in gd:
        mpr[d] = (3, (sgn(d[0] - rd[0]), sgn(d[1] - rd[1])))
    for d in od:
        mpr[d] = (7, (sgn(d[0] - bd[0]), sgn(d[1] - bd[1])))
    ub = (len(gd) + len(od)) * ((h + w) // 5)
    ndist = unifint(diff_lb, diff_ub, (1, max(1, ub)))
    gi = [row[:] for row in go]
    for _ in range(ndist):
        options = []
        for loc, (col, direc) in mpr.items():
            ii, jj = loc[0] + direc[0], loc[1] + direc[1]
            if 0 <= ii < h and 0 <= jj < w and gi[ii][jj] == bgc:
                options.append((loc, col, direc))
        if not options:
            break
        loc, col, direc = _r.choice(options)
        del mpr[loc]
        newloc = (loc[0] + direc[0], loc[1] + direc[1])
        mpr[newloc] = (col, direc)
        gi[loc[0]][loc[1]] = bgc
        gi[newloc[0]][newloc[1]] = col
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    """Whole-grid / colour-at-a-time route.
    Each 3-dot sits on a ray from the 2 and each 7-dot sits on a ray from the 1.
    Every dot belongs on the cell right next to its anchor, along its own ray.
      1. One stroke clears every dot that is not yet next to its anchor
         (both colours together).
      2. One Color3 paints all the home cells of the 3-dots that were cleared.
      3. One Color7 paints all the home cells of the 7-dots that were cleared.
    Every selection comes from I alone."""
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    cands = [c for c in cnt if c not in (1, 2, 3, 7)]
    bgc = max(cands, key=lambda c: cnt[c]) if cands else 0

    def sgn(v):
        return (v > 0) - (v < 0)

    def anchor(col):
        p = np.argwhere(I == col)
        return (int(p[0][0]), int(p[0][1])) if len(p) else None

    displaced = []          # dots not yet next to their anchor (both colours)
    homes = {3: [], 7: []}  # where each displaced dot belongs
    for dot_col, anc_col in ((3, 2), (7, 1)):
        a = anchor(anc_col)
        if a is None:
            continue
        for cell in np.argwhere(I == dot_col):
            r, c = int(cell[0]), int(cell[1])
            home = (a[0] + sgn(r - a[0]), a[1] + sgn(c - a[1]))
            if home != (r, c):
                displaced.append((r, c))
                homes[dot_col].append(home)

    ops, sels = [], []
    if displaced:
        # 1. clear every displaced dot in a single stroke
        ops.append(bgc)
        sels.append(sel_of(sorted(set(displaced))))
        # 2./3. one colour at a time, paint the home cells next to each anchor
        # (3 first, then 7: the generator draws 7 last where neighbourhoods overlap)
        for col in (3, 7):
            if homes[col]:
                ops.append(col)
                sels.append(sel_of(sorted(set(homes[col]))))

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
                        f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
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
                                f"for task ae3edfdc"
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
                    f"Failed to build a complete episode for task ae3edfdc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ae3edfdc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
