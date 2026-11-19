"""
ARC Task: d0f5fe59 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    from random import choice

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + lb * (b - a)
        hi = a + ub * (b - a)
        return int(round(random.uniform(lo, hi)))

    if bgc is None:
        bgc = random.randint(0, 9)
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c != bgc])

    def dnb(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def nb(p):
        r, c = p
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}

    while True:
        h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        h = min(h, max_h)
        w = min(w, max_w)
        nobjs = unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 9)))
        gi = [[bgc] * w for _ in range(h)]
        inds = {(r, c) for r in range(h) for c in range(w)}
        nfound = 0
        trials = 0
        maxtrials = nobjs * 5
        while trials < maxtrials and nfound < nobjs:
            oh = unifint(diff_lb, diff_ub, (1, 5))
            ow = unifint(diff_lb, diff_ub, (1, 5))
            bx = {(r, c) for r in range(oh) for c in range(ow)}
            sp = choice(sorted(bx))
            shp = {sp}
            dev = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
            ncells = choice((dev, oh * ow - dev))
            ncells = min(max(1, ncells), oh * ow - 1)
            for _ in range(ncells):
                cand = set()
                for p in shp:
                    cand |= dnb(p)
                cand = (bx - shp) & cand
                if not cand:
                    break
                shp.add(choice(sorted(cand)))
            mr = min(r for r, c in shp)
            mc = min(c for r, c in shp)
            shp = {(r - mr, c - mc) for r, c in shp}
            if len(inds) == 0:
                break
            loc = choice(sorted(inds))
            plcd = {(r + loc[0], c + loc[1]) for r, c in shp}
            if plcd.issubset(inds):
                for r, c in plcd:
                    gi[r][c] = fgc
                ring = set()
                for p in plcd:
                    ring |= nb(p)
                inds = (inds - plcd) - ring
                nfound += 1
            trials += 1
        if nfound >= 1:
            break
    go = [[bgc] * nfound for _ in range(nfound)]
    for i in range(nfound):
        go[i][i] = fgc
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (shared by the maker and the golf entry): every object in I is one
    # tally mark; the answer is an n x n canvas of background with the main
    # diagonal drawn in the object colour, n = number of objects.  The two readings
    # agree, so the route is the same: size the canvas by the count, lay the
    # background, draw the diagonal as ONE line.
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    fg_cands = [c for c in cnt if c != bgc]
    fgc = min(fg_cands, key=lambda c: (cnt[c], c)) if fg_cands else bgc

    # count 4-connected objects of I (non-background)
    vis = np.zeros((hi, wi), bool)
    n = 0
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or vis[r, c]:
                continue
            n += 1
            stack = [(r, c)]
            vis[r, c] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not vis[ny, nx] and I[ny, nx] == I[y, x]:
                        vis[ny, nx] = True
                        stack.append((ny, nx))
    n = max(1, min(30, n))

    ops, sels = [], []

    # 1. canvas becomes n x n (transparent crop of the top-left of I)
    ops.append(33)
    sels.append([0, 0, n - 1, n - 1])
    state = np.zeros((n, n), dtype=int)
    rr, cc = min(n, hi), min(n, wi)
    state[:rr, :cc] = I[:rr, :cc]
    state[state == 0] = 0

    diag = [(i, i) for i in range(n)]
    diag_set = set(diag)

    # 2. lay the background
    if bgc != 0:
        # whole canvas: every cell is 0 or a leftover fragment, none is the base yet
        ops.append(bgc)
        sels.append(sel_of([(r, c) for r in range(n) for c in range(n)]))
        state[:, :] = bgc
    else:
        # zero padding already is background; clear the object-coloured cells
        # lying off the main diagonal (all fgc cells not on the diagonal)
        stray = [(r, c) for r in range(n) for c in range(n)
                 if state[r, c] != 0 and (r, c) not in diag_set]
        if stray:
            ops.append(0)
            sels.append(sel_of(stray))
            for r, c in stray:
                state[r, c] = 0

    # 3. draw the main diagonal as one line in the object colour
    if any(state[r, c] != fgc for r, c in diag):
        ops.append(fgc)
        sels.append(sel_of(diag))
        for r, c in diag:
            state[r, c] = fgc

    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d0f5fe59"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d0f5fe59"
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
                                f"for task d0f5fe59"
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
                    f"Failed to build a complete episode for task d0f5fe59 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d0f5fe59-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
