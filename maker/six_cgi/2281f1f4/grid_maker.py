"""
ARC Task: 2281f1f4 (RE-ARC) — LLM-generated grid_maker
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
    colopts = [c for c in range(10) if c != 2]
    bgc = random.choice(colopts)
    dc = random.choice([c for c in colopts if c != bgc])
    return {"bgc": bgc, "dc": dc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + d * (b - a)))))

    colopts = [c for c in range(10) if c != 2]
    if bgc is None:
        bgc = random.choice(colopts)
    if dc is None or dc == bgc:
        dc = random.choice([c for c in colopts if c != bgc])

    # the grid may be rotated, so both sides have to fit within both limits
    lim = min(max_h, max_w)
    h = unifint(diff_lb, diff_ub, (3, max(3, min(30, lim))))
    w = unifint(diff_lb, diff_ub, (3, max(3, min(30, lim))))
    numtop = unifint(diff_lb, diff_ub, (1, w // 2 + 1))
    numright = unifint(diff_lb, diff_ub, (1, h // 2 + 1))
    if numtop == 1 and numright == 1:
        numtop, numright = random.sample([1, 2], 2)
    numtop = min(numtop, w - 1)
    numright = min(numright, h - 1)
    tp = random.sample(range(0, w - 1), numtop)
    rp = random.sample(range(1, h), numright)

    gi = np.full((h, w), bgc, dtype=int)
    for c in tp:
        gi[0, c] = dc
    for r in rp:
        gi[r, w - 1] = dc
    go = gi.copy()
    for r in rp:
        for c in tp:
            go[r, c] = 2

    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule (from I alone): the marker colour lies on one border row and one
    border column. They meet at a corner that is background. Every marker on
    the border column takes the border row's marker pattern and copies it
    into its own row, painted in colour 2.

    The code-golf reading, [[x+r[-1] & r.pop(0)+2 for x in g[0]] for r in g],
    works the same way: it handles one row at a time and asks whether that
    row's border-column cell is a marker. If it is, the whole top-row pattern
    goes into that row. The unit (one row), the order and the fixed quantity
    (the border-row pattern) all match the maker, so nothing changes here:
    one Color(2) op per projected row.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    fg = [c for c in cnt if c != bgc]
    if not fg:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    mc = min(fg, key=lambda c: cnt[c])

    marks = [(r, c) for r in range(h) for c in range(w) if I[r, c] == mc]

    # corner where the two marker lines meet: the background corner whose
    # row or column holds the most markers
    best, corner = -1, (0, 0)
    for cr in (0, h - 1):
        for cc in (0, w - 1):
            if I[cr, cc] != bgc:
                continue
            k = sum(1 for (r, c) in marks if r == cr or c == cc)
            if k > best:
                best, corner = k, (cr, cc)
    cr, cc = corner

    proj_rows = sorted({r for (r, c) in marks if c == cc and r != cr})
    pat_cols = sorted({c for (r, c) in marks if r == cr and c != cc})

    for r in proj_rows:
        cells = [(r, c) for c in pat_cols if I[r, c] != 2]
        if cells:
            ops.append(2)
            sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 2281f1f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2281f1f4"
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
                                f"for task 2281f1f4"
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
                    f"Failed to build a complete episode for task 2281f1f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2281f1f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
