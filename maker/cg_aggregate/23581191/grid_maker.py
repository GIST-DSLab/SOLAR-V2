"""
ARC Task: 23581191 (RE-ARC) — LLM-generated grid_maker
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
    colopts = [c for c in range(10) if c != 2]
    bgc = random.choice(colopts)
    rem = [c for c in colopts if c != bgc]
    acol = random.choice(rem)
    bcol = random.choice([c for c in rem if c != acol])
    return {"bgc": bgc, "acol": acol, "bcol": bcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, acol=None, bcol=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    colopts = [c for c in range(10) if c != 2]
    if bgc is None:
        bgc = random.choice(colopts)
    rem = [c for c in colopts if c != bgc]
    if acol is None:
        acol = random.choice(rem)
    if bcol is None:
        bcol = random.choice([c for c in rem if c != acol])

    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    h = max(3, min(h, max_h))
    w = max(3, min(w, max_w))
    inds = [(i, j) for i in range(h) for j in range(w)]
    ub = max(1, (h * w) // 4)
    na = unifint(diff_lb, diff_ub, (1, ub))
    nb = unifint(diff_lb, diff_ub, (1, ub))
    na = max(1, min(na, ub))
    nb = max(1, min(nb, ub))
    a = random.sample(inds, na)
    aset = set(a)
    b = random.sample([x for x in inds if x not in aset], nb)

    gi = [[bgc] * w for _ in range(h)]
    for i, j in a:
        gi[i][j] = acol
    for i, j in b:
        gi[i][j] = bcol

    fa = {i for i, _ in a}; la = {j for _, j in a}
    fb = {i for i, _ in b}; lb = {j for _, j in b}
    go = [[bgc] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            ina = i in fa or j in la
            inb = i in fb or j in lb
            if ina and inb:
                go[i][j] = 2
            elif ina:
                go[i][j] = acol
            elif inb:
                go[i][j] = bcol
    for i, j in a:
        go[i][j] = acol
    for i, j in b:
        go[i][j] = bcol
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: strict majority (each point colour covers at most h*w//4 cells)
    cnt = Counter(I.flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])
    pcols = sorted(int(c) for c in cnt if c != bgc)

    # Per-cell reading: a background cell's fate is decided by the set of
    # point colours present anywhere in its row or its column (its "cross").
    row_sets = [set(int(v) for v in I[r, :]) - {bgc} for r in range(hi)]
    col_sets = [set(int(v) for v in I[:, c]) - {bgc} for c in range(wi)]

    classes = {}  # frozenset of cross colours -> list of background cells
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc:
                continue  # points keep their own colour
            s = frozenset(row_sets[r] | col_sets[c])
            if s:
                classes.setdefault(s, []).append((r, c))

    # single-colour crosses first, in colour order, then the crossing class
    singles = [frozenset([p]) for p in pcols]
    for s in singles:
        cells = classes.get(s)
        if cells:
            ops.append(next(iter(s)))
            sels.append(sel_of(cells))
    multi = []
    for s, cells in classes.items():
        if len(s) >= 2:
            multi.extend(cells)
    if multi:
        ops.append(2)   # where both families' crosses meet -> colour 2
        sels.append(sel_of(sorted(multi)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 23581191"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 23581191"
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
                                f"for task 23581191"
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
                    f"Failed to build a complete episode for task 23581191 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"23581191-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
