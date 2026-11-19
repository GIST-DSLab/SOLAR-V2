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
    # The rule depends only on where the marks sit, not on their colours,
    # so only the background colour is fixed for the episode.
    return {"bgc": random.choice(range(10))}


def _unif(lb, ub, rng):
    a, b = rng
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    return int(round(random.uniform(lo, hi)))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
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
        if plcd.issubset(inds):
            ob = {(loc[0] + i, loc[1] + j) for i in range(-1, 6) for j in range(-1, 6)}
            inds = inds - ob
            ringcols = [random.choice(ccols) for _ in range(4)]
            pr = [[(loc[0] + a, loc[1] + b) for a, b in r] for r in rings]
            idx = random.randint(1, 3)
            remr = pr[1:idx] + pr[idx + 1:]
            remc = ringcols[1:idx] + ringcols[idx + 1:]
            numrs = _unif(diff_lb, diff_ub, (1, 2))
            locs = random.sample((0, 1), numrs)
            remr = [rr for j, rr in enumerate(remr) if j in locs]
            remc = [rr for j, rr in enumerate(remc) if j in locs]
            fgi, fgo = {}, {}
            for rr, col in zip(remr, remc):
                k = 4 - _unif(diff_lb, diff_ub, (0, 3))
                for c in random.sample(rr, k):
                    fgi[c] = col
                for c in rr:
                    fgo[c] = col
            rs = [c[0] for c in fgi]
            cs = [c[1] for c in fgi]
            if min(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1) == 5:
                succ += 1
                for g in (gi, go):
                    for (a, b) in pr[0] + pr[idx]:
                        g[a][b] = ringcols[0] if (a, b) in pr[0] else ringcols[idx]
                for (a, b), v in fgi.items():
                    gi[a][b] = v
                for (a, b), v in fgo.items():
                    go[a][b] = v
    return {"input": gi, "output": go}


def _find_blocks(I, bgc):
    # Locate every 5x5 ring-block in I alone (nothing here looks at O).
    h, w = I.shape
    out = []
    rings = (((1, 1), (1, 3), (3, 1), (3, 3)),
             ((0, 2), (2, 0), (2, 4), (4, 2)),
             ((0, 0), (0, 4), (4, 0), (4, 4)))
    for r0 in range(1, h - 5):
        for c0 in range(1, w - 5):
            win = I[r0 - 1:r0 + 6, c0 - 1:c0 + 6]
            blk = win[1:6, 1:6]
            if blk[2, 2] == bgc:
                continue
            border = np.concatenate([win[0], win[6], win[:, 0], win[:, 6]])
            if (border != bgc).any():
                continue
            fg = blk != bgc
            if any(fg[i, j] for i in range(5) for j in range(5) if (i + j) % 2):
                continue
            if not (fg[0].any() and fg[4].any() and fg[:, 0].any() and fg[:, 4].any()):
                continue
            # one ring around the centre is always complete and single-coloured
            if not any(len({int(blk[p]) for p in rg}) == 1 and blk[rg[0]] != bgc
                       for rg in rings):
                continue
            if any(abs(r0 - a) < 5 and abs(c0 - b) < 5 for a, b in out):
                continue
            out.append((r0, c0))
    return out


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])
    if examples:
        from collections import Counter
        c = Counter()
        for ei, _ in examples:
            v, n = np.unique(np.asarray(ei), return_counts=True)
            c[int(v[np.argmax(n)])] += 1
        cand = c.most_common(1)[0][0]
        if (I == cand).sum() * 2 > I.size:
            bgc = cand

    grid = I.copy()
    ops, sels = [], []
    for (r0, c0) in _find_blocks(I, bgc):
        rect = [r0, c0, 4, 4]  # the whole square 5x5 block, background included
        # Rotate the block a quarter turn clockwise about its centre, then paint
        # back every mark it had just before the turn (grouped by colour). Each
        # mark is carried to its next rotated position. Stop as soon as a turn
        # would change nothing: the block is then 4-fold rotationally symmetric.
        for _ in range(3):
            before = grid[r0:r0 + 5, c0:c0 + 5].copy()
            new = np.rot90(before, 3)
            if (new == before).all():
                break
            ops.append(25)
            sels.append(rect)
            grid[r0:r0 + 5, c0:c0 + 5] = new
            bycol = {}
            for i in range(5):
                for j in range(5):
                    v = int(before[i, j])
                    if v != bgc and grid[r0 + i, c0 + j] != v:
                        bycol.setdefault(v, []).append((r0 + i, c0 + j))
            for v in sorted(bycol):
                ops.append(v)
                sels.append(sel_of(bycol[v]))
                for rc in bycol[v]:
                    grid[rc] = v

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
