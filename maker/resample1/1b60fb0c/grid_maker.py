"""
ARC Task: 1b60fb0c (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc, objc = random.sample(cols, 2)
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, objc=1) -> dict:
    def unifint(lb, ub, r):
        a, b = r
        if b <= a:
            return a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo = max(a, min(lo, b)); hi = max(lo, min(hi, b))
        return random.randint(lo, hi)

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    odh = unifint(diff_lb, diff_ub, (2, min(h, w) // 2))
    loci = random.randint(0, h - 2 * odh)
    locj = random.randint(0, w - 2 * odh)
    ncellsd = unifint(diff_lb, diff_ub, (0, odh ** 2 // 2))
    ncells = random.choice((ncellsd, odh ** 2 - ncellsd))
    ncells = min(max(1, ncells), odh ** 2 - 1)
    allc = [(i, j) for i in range(odh) for j in range(odh)]
    g1 = np.full((odh, odh), bgc, dtype=int)
    for (i, j) in random.sample(allc, ncells):
        g1[i, j] = objc
    g2 = np.rot90(g1, -1); g3 = np.rot90(g2, -1); g4 = np.rot90(g3, -1)

    def of(g):
        return {(int(i), int(j)) for i, j in zip(*np.nonzero(g == objc))}

    s = random.randint(0, odh)
    c1 = {(i, j + s) for i, j in of(g1)}
    c2 = {(i + s, j + odh) for i, j in of(g2)}
    c3 = {(i + odh, j + odh - s) for i, j in of(g3)}
    c4 = {(i + odh - s, j) for i, j in of(g4)}
    cs = [c1, c2, c3, c4]
    k = random.randrange(4)
    inobj = set().union(*[c for q, c in enumerate(cs) if q != k])
    rem = cs[k] - inobj
    gi = np.full((h, w), bgc, dtype=int)
    for i, j in inobj:
        gi[i + loci, j + locj] = objc
    go = gi.copy()
    for i, j in rem:
        go[i + loci, j + locj] = 2
    return {"input": gi.tolist(), "output": go.tolist()}


def _bg_from(I, examples):
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            if ei.shape == eo.shape and (eo == 2).any():
                return int(ei[eo == 2][0])
        for ei, _ in examples:
            ei = np.asarray(ei, dtype=int)
            return Counter(ei.flatten().tolist()).most_common(1)[0][0]
    return Counter(I.flatten().tolist()).most_common(1)[0][0]


def _missing_cells(I, bgc):
    # Rule: object + rotated copy (placed best) forms a 90°-rotation-symmetric figure;
    # cells of the rotated copy not already in the object are the missing quarter.
    x1 = {(int(r), int(c)) for r, c in zip(*np.nonzero(I != bgc))}
    if not x1:
        return []
    R = np.rot90(I, -1)
    x4 = [(int(r), int(c)) for r, c in zip(*np.nonzero(R != bgc))]
    mr = min(r for r, _ in x4); mc = min(c for _, c in x4)
    u0 = min(r for r, _ in x1); l0 = min(c for _, c in x1)
    x7 = [(r - mr + u0, c - mc + l0) for r, c in x4]
    H1 = max(r for r, _ in x1) - u0 + 1
    W1 = max(c for _, c in x1) - l0 + 1
    k = max(H1, W1) - min(H1, W1) + 1
    t = min(r for r, _ in x7); b = max(r for r, _ in x7)
    l = min(c for _, c in x7); rr = max(c for _, c in x7)
    cr = t + (b - t + 1) // 2; cc = l + (rr - l + 1) // 2
    best, bestn = None, -1
    for i in range(t - k, b + k + 1):
        for j in range(l - k, rr + k + 1):
            dr, dc = i - cr, j - cc
            sh = {(r + dr, c + dc) for r, c in x7}
            U = x1 | sh
            ut = min(r for r, _ in U); ul = min(c for _, c in U)
            N = {(r - ut, c - ul) for r, c in U}
            Hh = max(r for r, _ in N) + 1
            Rt = {(c, Hh - 1 - r) for r, c in N}
            if Rt != N:
                continue
            n = len(x1 & sh)
            if n > bestn:
                bestn, best = n, sh
    if best is None:
        return []
    h, w = I.shape
    return sorted((r, c) for r, c in best - x1 if 0 <= r < h and 0 <= c < w)


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = _bg_from(I, examples)
    ops, sels = [], []
    miss = _missing_cells(I, bgc)
    if miss:
        ops.append(2); sels.append(sel_of(miss))   # paint the missing quarter in colour 2
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1b60fb0c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1b60fb0c"
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
                                f"for task 1b60fb0c"
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
                    f"Failed to build a complete episode for task 1b60fb0c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1b60fb0c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
