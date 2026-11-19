"""
ARC Task: 11852cab (RE-ARC) — LLM-generated grid_maker
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
    # Only the background matters; ring colours are free (the rule is
    # colour-agnostic: each ring keeps whatever colour it already shows).
    return {"bgc": random.randint(0, 9)}


def _unif(lb, ub, rng):
    a, b = rng
    lo = a + round((b - a) * lb)
    hi = a + round((b - a) * ub)
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kw) -> dict:
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = [(i, j) for i in range(5) for j in range(5)]
    h = _unif(diff_lb, diff_ub, (7, max(7, max_h)))
    w = _unif(diff_lb, diff_ub, (7, max(7, max_w)))
    remcols = [c for c in range(10) if c != bgc]
    numc = _unif(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, numc)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(1, h - 1) for j in range(1, w - 1)}
    nobjs = _unif(diff_lb, diff_ub, (1, max(1, (h * w) // 36)))
    succ = tr = 0
    maxtr = 10 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = [ij for ij in inds if ij[0] <= h - 5 and ij[0] <= w - 5]
        if not cands:
            break
        loc = random.choice(sorted(cands))
        plcd = {(loc[0] + i, loc[1] + j) for i, j in bx}
        if not plcd.issubset(inds):
            continue
        outb = {(loc[0] + i, loc[1] + j) for i in range(-1, 6) for j in range(-1, 6)} - plcd
        ringcols = [random.choice(ccols) for _ in range(4)]
        plr = [[(loc[0] + i, loc[1] + j) for i, j in r] for r in rings]
        idx = random.randint(1, 3)
        remr = plr[1:idx] + plr[idx + 1:]
        remc = ringcols[1:idx] + ringcols[idx + 1:]
        numrs = _unif(diff_lb, diff_ub, (1, 2))
        locs = random.sample((0, 1), numrs)
        remr = [rr for j, rr in enumerate(remr) if j in locs]
        remc = [rr for j, rr in enumerate(remc) if j in locs]
        fgi, fgo = {}, {}
        for rr, col in zip(remr, remc):
            for p in random.sample(rr, 4 - _unif(diff_lb, diff_ub, (0, 3))):
                fgi[p] = col
            for p in rr:
                fgo[p] = col
        base = {p: ringcols[0] for p in plr[0]}
        base.update({p: ringcols[idx] for p in plr[idx]})
        if not fgi:
            continue
        rs = [p[0] for p in fgi]
        cs = [p[1] for p in fgi]
        if min(max(rs) - min(rs), max(cs) - min(cs)) + 1 == 5:
            inds = inds - plcd - outb
            succ += 1
            for g, extra in ((gi, fgi), (go, fgo)):
                for (i, j), v in list(base.items()) + list(extra.items()):
                    g[i][j] = v
    return {"input": gi, "output": go}


_R1 = ((0, 0), (0, 4), (4, 0), (4, 4))
_R2 = ((2, 0), (0, 2), (4, 2), (2, 4))
_R3 = ((1, 1), (3, 1), (1, 3), (3, 3))


def _is_block(g, bgc, r0, c0):
    h, w = len(g), len(g[0])
    if r0 < 1 or c0 < 1 or r0 + 6 > h or c0 + 6 > w:
        return False
    if g[r0 + 2][c0 + 2] == bgc:
        return False
    for i in range(-1, 6):
        for j in range(-1, 6):
            v = g[r0 + i][c0 + j]
            inside = 0 <= i < 5 and 0 <= j < 5
            if (not inside or (i + j) % 2) and v != bgc:
                return False
    full = False
    for ring in (_R1, _R2, _R3):
        vals = [g[r0 + i][c0 + j] for i, j in ring]
        fg = set(v for v in vals if v != bgc)
        if len(fg) > 1:
            return False
        if len(fg) == 1 and bgc not in vals:
            full = True
    if not full:
        return False
    cells = [(i, j) for i in range(5) for j in range(5) if g[r0 + i][c0 + j] != bgc]
    rows = {i for i, _ in cells}
    cols = {j for _, j in cells}
    return 0 in rows and 4 in rows and 0 in cols and 4 in cols


def _find_blocks(g, bgc):
    # Candidate 5x5 windows that look like a ring block; then choose the set of
    # candidates covering every non-background cell exactly once (exact cover),
    # which discards windows straddling two neighbouring blocks.
    h, w = len(g), len(g[0])
    cands = [(r, c) for r in range(h) for c in range(w) if _is_block(g, bgc, r, c)]
    cov = {}
    for k, (r, c) in enumerate(cands):
        cov[k] = {(r + i, c + j) for i in range(5) for j in range(5) if g[r + i][c + j] != bgc}
    todo = {(r, c) for r in range(h) for c in range(w) if g[r][c] != bgc}

    def solve(left, used, chosen):
        if not left:
            return chosen
        cell = min(left)
        for k in range(len(cands)):
            if k in used or cell not in cov[k] or not cov[k] <= left:
                continue
            res = solve(left - cov[k], used | {k}, chosen + [k])
            if res is not None:
                return res
        return None

    res = solve(todo, set(), [])
    if res is None:
        res, taken = [], set()
        for k in range(len(cands)):
            if not (cov[k] & taken):
                res.append(k)
                taken |= cov[k]
    return sorted(cands[k] for k in res)


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    cnt = {}
    for v in I.flatten().tolist():
        cnt[v] = cnt.get(v, 0) + 1
    bgc = max(sorted(cnt), key=lambda c: cnt[c])   # the canvas colour
    grid = [list(map(int, row)) for row in I.tolist()]
    ops, sels = [], []
    # Reading: each 5x5 block is concentric one-colour rings, and every ring is
    # a single orbit of the quarter-turn.  So the block is completed by turning
    # it a quarter (CCW) at a time and redrawing the marks it carried just
    # before the turn; after at most three turns it holds every rotated copy of
    # itself.  Blocks are located in I alone; O is never consulted.
    for (r0, c0) in _find_blocks(grid, bgc):
        rect = [r0, c0, 4, 4]          # exactly the whole square block, bg included
        for _ in range(3):
            sub = [row[c0:c0 + 5] for row in grid[r0:r0 + 5]]
            new = [[sub[j][4 - i] for j in range(5)] for i in range(5)]  # rot90 CCW
            if new == sub:
                break                   # already turn-symmetric: block complete
            ops.append(24)
            sels.append(rect)
            for i in range(5):
                for j in range(5):
                    grid[r0 + i][c0 + j] = new[i][j]
            bycol = {}
            for i in range(5):
                for j in range(5):
                    v = sub[i][j]
                    if v != bgc and grid[r0 + i][c0 + j] != v:
                        bycol.setdefault(v, []).append((r0 + i, c0 + j))
            for v in sorted(bycol):
                ops.append(int(v))
                sels.append(sel_of(bycol[v]))
                for (r, c) in bycol[v]:
                    grid[r][c] = v
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
                        f"num_examples+1 ({num_examples + 1}) for task 11852cab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 11852cab"
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
                                f"for task 11852cab"
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
                    f"Failed to build a complete episode for task 11852cab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"11852cab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
