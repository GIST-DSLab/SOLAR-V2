"""
ARC Task: 3631a71a (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    patchcol = random.choice(cols)
    bgc = random.choice([c for c in cols if c != patchcol])
    return {"bgc": bgc, "patchcol": patchcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, patchcol=None) -> dict:
    from random import randint, choice, sample as rsample

    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi_)))))

    cols = list(range(10))
    if patchcol is None:
        patchcol = choice(cols)
    if bgc is None or bgc == patchcol:
        bgc = choice([c for c in cols if c != patchcol])
    hmax = max(6, min(15, (min(max_h, max_w) + 2) // 2))
    h = unifint(diff_lb, diff_ub, (6, hmax))
    w = h
    remcols = [c for c in cols if c not in (bgc, patchcol)]
    c = [[bgc] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w) if i >= j]
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = rsample(remcols, ncols)
    ncells = unifint(diff_lb, diff_ub, (1, len(inds)))
    cells = rsample(inds, ncells)
    obj = [(choice(ccols), ij) for ij in cells]
    for v, (i, j) in obj:
        c[i][j] = v
    c = [list(r) for r in zip(*c)]
    for v, (i, j) in obj:
        c[i][j] = v
    c = [r + r[::-1] for r in c]
    c = c + c[::-1]
    cutoff = 2
    go = [r[:-cutoff] for r in c[:-cutoff]]
    fullh, fullw = len(go), len(go[0])
    gi = [list(r) for r in go]
    n = fullh + 2

    def ok_grid(g):
        # every orbit must keep a non-patch cell; top-left 2x2 free of patch
        for i in range(2):
            for j in range(2):
                if g[i][j] == patchcol:
                    return False
        G = np.array(g)
        while True:
            prog = False
            for s in (lambda r, cc: (n - 1 - r, cc), lambda r, cc: (r, n - 1 - cc), lambda r, cc: (cc, r)):
                snap = G.copy()
                for r in range(fullh):
                    for cc in range(fullw):
                        if snap[r, cc] != patchcol:
                            continue
                        a, b = s(r, cc)
                        if 0 <= a < fullh and 0 <= b < fullw and snap[a, b] != patchcol:
                            G[r, cc] = snap[a, b]
                            prog = True
            if not prog:
                break
        return not (G == patchcol).any()

    ndist = unifint(diff_lb, diff_ub, (1, int((2 * h * 2 * w) ** 0.5)))
    succ, tr, maxtr = 0, 0, 10 * ndist
    while succ < ndist and tr < maxtr:
        tr += 1
        oh = randint(2, h // 2 + 1)
        ow = randint(2, w // 2 + 1)
        loci = randint(0, fullh - oh)
        locj = randint(0, fullw - ow)
        gi2 = [list(r) for r in gi]
        for i in range(loci, loci + oh):
            for j in range(locj, locj + ow):
                gi2[i][j] = patchcol
        if not ok_grid(gi2):
            continue
        succ += 1
        gi = gi2
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O=None, examples=None):
    """
    Reading (golf-style): the grid equals its own up-down mirror (row r <-> H+1-r,
    the +1 from the 2-cell crop), its left-right mirror, and its transpose.
    Restore it by overlaying the grid on each mirror image in turn, one axis per
    pass: every patch cell whose mirror cell (in the grid as it stands) is not
    patch takes that cell's colour.  Passes repeat until no patch is left.
    One Color op per (pass, colour).
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    n = hi + 2
    syms = [lambda r, c: (n - 1 - r, c), lambda r, c: (r, n - 1 - c)]
    if hi == wi:
        syms.append(lambda r, c: (c, r))

    def inside(a, b):
        return 0 <= a < hi and 0 <= b < wi

    # patch colour: the colour that is fully recoverable via mirrors and whose
    # removal leaves every mirror pair consistent (measured from I only)
    best_p, best_score = None, 0
    for p in sorted(set(I.flatten().tolist())):
        ok = True
        for s in syms:
            for r in range(hi):
                for c in range(wi):
                    a, b = s(r, c)
                    if inside(a, b) and I[r, c] != p and I[a, b] != p and I[r, c] != I[a, b]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        G = I.copy()
        while True:
            prog = False
            for s in syms:
                snap = G.copy()
                for r in range(hi):
                    for c in range(wi):
                        if snap[r, c] != p:
                            continue
                        a, b = s(r, c)
                        if inside(a, b) and snap[a, b] != p:
                            G[r, c] = snap[a, b]
                            prog = True
            if not prog:
                break
        if (G == p).any():
            continue
        sc = int((I == p).sum())
        if sc > best_score:
            best_p, best_score = p, sc

    ops, sels = [], []
    if best_p is not None:
        p = best_p
        G = I.copy()
        while (G == p).any():
            prog = False
            for s in syms:
                snap = G.copy()          # overlay this mirror of the current grid
                bycol = {}
                for r in range(hi):
                    for c in range(wi):
                        if snap[r, c] != p:
                            continue
                        a, b = s(r, c)
                        if inside(a, b) and snap[a, b] != p:
                            bycol.setdefault(int(snap[a, b]), []).append((r, c))
                for col in sorted(bycol):
                    ops.append(col)
                    sels.append(sel_of(bycol[col]))
                    for r, c in bycol[col]:
                        G[r, c] = col
                    prog = True
            if not prog:
                break

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
                        f"num_examples+1 ({num_examples + 1}) for task 3631a71a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3631a71a"
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
                                f"for task 3631a71a"
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
                    f"Failed to build a complete episode for task 3631a71a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3631a71a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
