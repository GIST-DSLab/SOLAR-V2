"""
ARC Task: 2bee17df (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cola = random.choice(rem)
    rem = [c for c in rem if c != cola]
    colb = random.choice(rem)
    return {"bgc": bgc, "cola": cola, "colb": colb}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, cola=None, colb=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    cols = [c for c in range(10) if c != 3]
    if bgc is None:
        bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    if cola is None or cola == bgc:
        cola = random.choice(rem)
    rem = [c for c in rem if c != cola]
    if colb is None or colb in (bgc, cola):
        colb = random.choice(rem)

    h = unifint(diff_lb, diff_ub, (7, min(30, max_h)))
    w = unifint(diff_lb, diff_ub, (7, min(30, max_w)))
    g = [[bgc] * w for _ in range(h)]

    indord = [(0, j) for j in range(w)]
    indord += [(i, w - 1) for i in range(1, h - 1)]
    indord += [(h - 1, j) for j in range(w - 1, 0, -1)]
    indord += [(i, 0) for i in range(h - 1, 0, -1)]
    k = len(indord)
    sp = random.randint(0, k)
    arr = indord[sp:] + indord[:sp]
    ep = random.randint(k // 2 - 3, k // 2 + 1)
    for (i, j) in arr[:ep]:
        g[i][j] = cola
    for (i, j) in arr[ep:]:
        g[i][j] = colb

    def ofcolor(c):
        return {(i, j) for i in range(h) for j in range(w) if g[i][j] == c}

    def dneigh(cells):
        out = set()
        for (i, j) in cells:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                out.add((i + di, j + dj))
        return out

    nr = unifint(diff_lb, diff_ub, (1, min(4, min(h, w) // 2)))
    for kk in range(nr):
        r0, c0, r1, c1 = 1 + kk, 1 + kk, h - 1 - kk, w - 1 - kk
        ring = set()
        for j in range(c0, c1 + 1):
            ring.add((r0, j)); ring.add((r1, j))
        for i in range(r0, r1 + 1):
            ring.add((i, c0)); ring.add((i, c1))
        for br in (cola, colb):
            bc = sorted(ring & ofcolor(bgc) & dneigh(ofcolor(br)))
            jj = len(bc)
            lo = max(0, jj // 2 - 2)
            hi_ = min(jj, jj // 2 + 1)
            jj2 = random.randint(lo, hi_) if hi_ >= lo else 0
            for (i, j) in random.sample(bc, jj2):
                g[i][j] = br

    gi = [row[:] for row in g]
    go = [row[:] for row in g]
    # frontiers of trimmed grid (uniform interior rows / cols), shifted by (1,1)
    for i in range(1, h - 1):
        vals = set(gi[i][1:w - 1])
        if len(vals) == 1:
            for j in range(1, w - 1):
                go[i][j] = 3
    for j in range(1, w - 1):
        vals = set(gi[i][j] for i in range(1, h - 1))
        if len(vals) == 1:
            for i in range(1, h - 1):
                go[i][j] = 3
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared by the maker and the golf entry): inside the 1-cell frame,
    # every line (row or column) whose interior is entirely background is a
    # "clear corridor" and becomes color 3. Each corridor is one unit -> one Color(3).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    interior = I[1:hi - 1, 1:wi - 1]
    bgc = Counter(interior.flatten().tolist()).most_common(1)[0][0]

    for r in range(1, hi - 1):
        if np.all(I[r, 1:wi - 1] == bgc):
            ops.append(3)
            sels.append(sel_of([(r, c) for c in range(1, wi - 1)]))

    for c in range(1, wi - 1):
        if np.all(I[1:hi - 1, c] == bgc):
            ops.append(3)
            sels.append(sel_of([(r, c) for r in range(1, hi - 1)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 2bee17df"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2bee17df"
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
                                f"for task 2bee17df"
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
                    f"Failed to build a complete episode for task 2bee17df "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2bee17df-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
