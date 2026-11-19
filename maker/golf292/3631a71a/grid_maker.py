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


def _orbits(S):
    owner = -np.ones((S, S), dtype=int)
    orbits = []
    R = S + 2
    for r in range(S):
        for c in range(S):
            if owner[r, c] >= 0:
                continue
            s = {(r, c)}
            st = [(r, c)]
            while st:
                a, b = st.pop()
                for x, y in ((R - 1 - a, b), (a, R - 1 - b), (b, a)):
                    if 0 <= x < R and 0 <= y < R and (x, y) not in s:
                        s.add((x, y))
                        st.append((x, y))
            cells = sorted((a, b) for a, b in s if a < S and b < S)
            for a, b in cells:
                owner[a, b] = len(orbits)
            orbits.append(cells)
    return orbits


def generate(diff_lb, diff_ub, max_h, max_w, bgc, patchcol, **kw) -> dict:
    def unifint(lo, hi):
        a = lo + diff_lb * (hi - lo)
        b = lo + diff_ub * (hi - lo)
        return max(lo, min(hi, int(round(random.uniform(a, b)))))
    hmax = min(15, (min(max_h, max_w) + 2) // 2)
    h = unifint(6, max(6, hmax))
    h = min(h, hmax)
    remcols = [c for c in range(10) if c not in (bgc, patchcol)]
    c = np.full((h, h), bgc, dtype=int)
    inds = [(i, j) for i in range(h) for j in range(h) if i >= j]
    ncols = unifint(1, 8)
    ccols = random.sample(remcols, ncols)
    ncells = unifint(1, len(inds))
    for (i, j) in random.sample(inds, ncells):
        col = random.choice(ccols)
        c[i, j] = col
        c[j, i] = col
    c = np.hstack([c, np.fliplr(c)])
    c = np.vstack([c, np.flipud(c)])
    go = c[:-2, :-2].copy()
    S = go.shape[0]
    orbits = _orbits(S)
    gi = go.copy()
    ndist = unifint(1, int((2 * h * 2 * h) ** 0.5))
    succ, tr = 0, 0
    while (succ < ndist and tr < 10 * ndist) or succ == 0:
        tr += 1
        if tr > 1000:
            raise ValueError("no patch")
        oh = random.randint(2, h // 2 + 1)
        ow = random.randint(2, h // 2 + 1)
        li = random.randint(0, S - oh)
        lj = random.randint(0, S - ow)
        g2 = gi.copy()
        g2[li:li + oh, lj:lj + ow] = patchcol
        if (g2[:2, :2] == patchcol).any():
            continue
        if any(all(g2[a, b] == patchcol for a, b in o) for o in orbits):
            continue
        gi = g2
        succ += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    The grid is a 2-cell crop of a fully mirror-symmetric canvas (transpose,
    row-mirror r->S+1-r, col-mirror c->S+1-c).  One colour (patchcol) paints
    rectangular occluders over it.  Every occluded cell is restored to the
    colour its symmetric partners show in I.  Restoration is done in one Color
    stroke per restored colour: "all occluded cells whose mirror images read k"
    -- a selection computed from I alone.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    orbits = _orbits(hi)

    def consistent(p):
        n = 0
        for cells in orbits:
            vals = {int(I[a, b]) for a, b in cells} - {p}
            k = sum(1 for a, b in cells if I[a, b] == p)
            if len(vals) > 1 or (not vals and k):
                return -1
            n += k
        return n

    # occluder colour: the colour present in every example input but absent
    # from its output (episode convention); fall back to symmetry consistency
    patchcol = None
    pal = set(I.flatten().tolist())
    if examples:
        cand = None
        for ei, eo in examples:
            s = set(np.asarray(ei).flatten().tolist()) - set(np.asarray(eo).flatten().tolist())
            cand = s if cand is None else cand & s
        cand = (cand or set()) & pal
        if len(cand) == 1 and consistent(next(iter(cand))) > 0:
            patchcol = next(iter(cand))
    if patchcol is None:
        best = 0
        for p in sorted(pal):
            n = consistent(p)
            if n > best:
                best, patchcol = n, p

    if patchcol is not None:
        bycol = {}
        for cells in orbits:
            vals = [int(I[a, b]) for a, b in cells if I[a, b] != patchcol]
            if not vals:
                continue
            for a, b in cells:
                if I[a, b] == patchcol:
                    bycol.setdefault(vals[0], []).append((a, b))
        # one stroke per restored colour: all occluded cells whose mirrors show it
        for col in sorted(bycol, key=lambda k: -len(bycol[k])):
            ops.append(int(col))
            sels.append(sel_of(bycol[col]))

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
