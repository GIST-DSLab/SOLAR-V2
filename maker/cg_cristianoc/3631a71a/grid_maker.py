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


def _images(r, c, N):
    # the 8 symmetry images of (r,c) in the cropped (N x N) grid; mirror is x -> N+1-x
    out = []
    for t in (False, True):
        for a in (0, 1):
            for b in (0, 1):
                x, y = (c, r) if t else (r, c)
                if a:
                    x = N + 1 - x
                if b:
                    y = N + 1 - y
                out.append((x, y))
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, patchcol=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        return random.randint(min(lo, hi), max(lo, hi))

    cols = list(range(10))
    if patchcol is None:
        patchcol = random.choice(cols)
    if bgc is None or bgc == patchcol:
        bgc = random.choice([c for c in cols if c != patchcol])
    hmax = min(15, (min(max_h, max_w) + 2) // 2)
    h = unifint(diff_lb, diff_ub, (6, max(6, hmax)))
    w = h
    remcols = [c for c in cols if c not in (bgc, patchcol)]
    c = [[bgc] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w) if i >= j]
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, ncols)
    ncells = unifint(diff_lb, diff_ub, (1, len(inds)))
    cells = random.sample(inds, ncells)
    for (i, j) in cells:
        col = random.choice(ccols)
        c[i][j] = col
        c[j][i] = col
    c = [row + row[::-1] for row in c]
    c = c + c[::-1]
    N = 2 * h - 2
    go = [row[:N] for row in c[:N]]
    gi = [row[:] for row in go]
    ndist = unifint(diff_lb, diff_ub, (1, int((2 * h * 2 * w) ** 0.5)))
    succ, tr, maxtr = 0, 0, 10 * ndist
    while succ < ndist and tr < maxtr:
        tr += 1
        oh = random.randint(2, h // 2 + 1)
        ow = random.randint(2, w // 2 + 1)
        loci = random.randint(0, N - oh)
        locj = random.randint(0, N - ow)
        gi2 = [row[:] for row in gi]
        for i in range(loci, loci + oh):
            for j in range(locj, locj + ow):
                gi2[i][j] = patchcol
        if any(gi2[i][j] == patchcol for i in range(2) for j in range(2)):
            continue
        ok = True
        for i in range(N):
            for j in range(N):
                if gi2[i][j] != patchcol:
                    continue
                if not any(0 <= x < N and 0 <= y < N and gi2[x][y] != patchcol
                           for x, y in _images(i, j, N)):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        succ += 1
        gi = gi2
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared with the code-golf solution): the grid is a crop of a canvas
    symmetric under transpose and both mirrors (x -> N+1-x).  One colour occludes
    rectangles.  Restore by overlaying the grid's own mirror images onto the
    occluded cells, one symmetry at a time: transpose first, then the row mirror,
    the column mirror, and their compositions -- repeating the sweep until nothing
    is occluded.  Each pass reads the CURRENT grid, so cells restored by an
    earlier pass can feed later ones.
    Unit of action: (symmetry, source colour) -> one Color op on the occluded
    cells whose partner under that symmetry currently shows that colour.
    O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    N = hi
    ops, sels = [], []

    def submit():
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    if hi != wi:
        return submit()

    def orbit(r, c):
        return [(x, y) for x, y in _images(r, c, N) if 0 <= x < N and 0 <= y < N]

    # occluding colour: removing it leaves every orbit single-valued and determined
    best_p, best_score = None, 0
    for p in sorted(set(I.flatten().tolist())):
        ok, cnt = True, 0
        for r in range(N):
            for c in range(N):
                vals = {int(I[x, y]) for x, y in orbit(r, c) if I[x, y] != p}
                if len(vals) > 1 or (not vals and I[r, c] == p):
                    ok = False
                    break
                cnt += int(I[r, c] == p)
            if not ok:
                break
        if ok and cnt > best_score:
            best_p, best_score = p, cnt
    if best_p is None:
        return submit()
    p = best_p

    def sym(t, a, b):
        def f(r, c):
            x, y = (c, r) if t else (r, c)
            if a:
                x = N + 1 - x
            if b:
                y = N + 1 - y
            return x, y
        return f

    # golf reading's order: transpose overlay, then mirror overlays, then compositions
    passes = [sym(True, 0, 0), sym(False, 1, 0), sym(False, 0, 1), sym(False, 1, 1),
              sym(True, 1, 0), sym(True, 0, 1), sym(True, 1, 1)]

    G = I.copy()
    progress = True
    while progress and (G == p).any():
        progress = False
        for f in passes:
            bycol = {}
            for r in range(N):
                for c in range(N):
                    if G[r, c] != p:
                        continue
                    x, y = f(r, c)
                    if 0 <= x < N and 0 <= y < N and G[x, y] != p:
                        bycol.setdefault(int(G[x, y]), []).append((r, c))
            for col in sorted(bycol):
                ops.append(col)
                sels.append(sel_of(bycol[col]))
                for (r, c) in bycol[col]:
                    G[r, c] = col
                progress = True
    return submit()


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
