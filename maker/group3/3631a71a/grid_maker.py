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
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    patchcol = random.choice(cols)
    bgc = random.choice([c for c in cols if c != patchcol])
    return {"bgc": bgc, "patchcol": patchcol}


def _orbits(n):
    # symmetry orbits of the n x n crop of the (n+2) x (n+2) mirror-symmetric canvas
    R = n + 2
    owner = -np.ones((n, n), dtype=int)
    orbits = []
    for r in range(n):
        for c in range(n):
            if owner[r, c] >= 0:
                continue
            s = {(r, c)}
            st = [(r, c)]
            while st:
                a, b = st.pop()
                for x, y in ((R - 1 - a, b), (a, R - 1 - b), (b, a)):
                    if (x, y) not in s:
                        s.add((x, y))
                        st.append((x, y))
            cells = [(a, b) for a, b in s if a < n and b < n]
            for a, b in cells:
                owner[a, b] = len(orbits)
            orbits.append(cells)
    return orbits


def generate(diff_lb, diff_ub, max_h, max_w, bgc, patchcol, **kw) -> dict:
    def unifint(lo, hi):
        lo2 = lo + int(round(diff_lb * (hi - lo)))
        hi2 = lo + int(round(diff_ub * (hi - lo)))
        return random.randint(min(lo2, hi2), max(lo2, hi2))

    hmax = min(15, (min(max_h, max_w) + 2) // 2)
    h = unifint(6, hmax)
    w = h
    remcols = [c for c in range(10) if c not in (bgc, patchcol)]
    c = np.full((h, w), bgc, dtype=int)
    inds = [(i, j) for i in range(h) for j in range(w) if i >= j]
    ncols = unifint(1, 8)
    ccols = random.sample(remcols, ncols)
    ncells = unifint(1, len(inds))
    for (i, j) in random.sample(inds, ncells):
        col = random.choice(ccols)
        c[i, j] = col
        c[j, i] = col
    c = np.hstack([c, c[:, ::-1]])
    c = np.vstack([c, c[::-1, :]])
    go = c[:-2, :-2].copy()
    n = go.shape[0]
    orbits = _orbits(n)
    gi = go.copy()
    ndist = unifint(1, int((2 * h * 2 * w) ** 0.5))
    succ = tr = 0
    while succ < ndist and tr < 10 * ndist:
        tr += 1
        oh = random.randint(2, h // 2 + 1)
        ow = random.randint(2, w // 2 + 1)
        li = random.randint(0, n - oh)
        lj = random.randint(0, n - ow)
        g2 = gi.copy()
        g2[li:li + oh, lj:lj + ow] = patchcol
        if (g2[:2, :2] == patchcol).any():
            continue
        if any(all(g2[a, b] == patchcol for a, b in orb) for orb in orbits):
            continue
        gi = g2
        succ += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule (from I alone): the grid is a 2-cell crop of a canvas that is symmetric
    under left-right mirror c -> (n+1)-c, up-down mirror r -> (n+1)-r and
    transpose. One colour (absent from the picture) covers rectangles.
    Unit of work: ONE MIRROR AT A TIME. Pass 1 restores every covered cell whose
    left-right partner (in the current grid) can be seen, pass 2 does the same
    with the up-down partner, pass 3 with the transposed partner, and the passes
    repeat until nothing is covered. Within a pass there is one Color op per
    colour, selecting "covered cells whose mirror partner now shows that colour".
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    n, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []
    orbits = _orbits(n) if n == wi else []

    # covering colour: the one whose removal leaves every orbit single-valued
    best_p, best = None, 0
    for p in sorted(set(I.flatten().tolist())):
        ok, cnt = True, 0
        for cells in orbits:
            vals = {int(I[a, b]) for a, b in cells if I[a, b] != p}
            k = sum(1 for a, b in cells if I[a, b] == p)
            if len(vals) > 1 or (not vals and k):
                ok = False
                break
            cnt += k
        if ok and cnt > best:
            best_p, best = p, cnt
    if best_p is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    p = best_p

    R = n + 2
    mirrors = [lambda a, b: (a, R - 1 - b),   # left-right partner
               lambda a, b: (R - 1 - a, b),   # up-down partner
               lambda a, b: (b, a)]           # transposed partner
    G = I.copy()
    progress = True
    while progress and (G == p).any():
        progress = False
        for m in mirrors:
            bycol = {}
            for a in range(n):
                for b in range(n):
                    if G[a, b] != p:
                        continue
                    x, y = m(a, b)
                    if 0 <= x < n and 0 <= y < n and G[x, y] != p:
                        bycol.setdefault(int(G[x, y]), []).append((a, b))
            for col in sorted(bycol):
                cells = sorted(bycol[col])
                ops.append(col); sels.append(sel_of(cells))
                for a, b in cells:
                    G[a, b] = col
                progress = True
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
