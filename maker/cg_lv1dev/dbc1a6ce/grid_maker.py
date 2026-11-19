"""
ARC Task: dbc1a6ce (RE-ARC) — LLM-generated grid_maker
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
    colopts = [c for c in range(10) if c != 8]
    bgc = random.choice(colopts)
    fgcol = random.choice([c for c in colopts if c != bgc])
    return {"bgc": bgc, "fgcol": fgcol}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, a + int(round((b - a) * d))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fgcol=1, **kwargs) -> dict:
    h = _unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = _unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    inds = [(r, c) for r in range(h) for c in range(w)]
    num = _unifint(diff_lb, diff_ub, (0, max(1, (h * w) // 4)))
    s = random.sample(inds, num)
    for r, c in s:
        gi[r][c] = fgcol
    go = [[bgc] * w for _ in range(h)]
    # horizontal segments
    for r in range(h):
        cols = [c for c in range(w) if gi[r][c] == fgcol]
        if len(cols) > 1:
            for c in range(cols[0], cols[-1] + 1):
                go[r][c] = 8
    # vertical segments
    for c in range(w):
        rows = [r for r in range(h) if gi[r][c] == fgcol]
        if len(rows) > 1:
            for r in range(rows[0], rows[-1] + 1):
                go[r][c] = 8
    for r, c in s:
        go[r][c] = fgcol
    return {"input": [list(row) for row in gi], "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared by the maker and the golf entry): the marks are the
    minority colour; every gap between two CONSECUTIVE marks in the same row,
    and every gap between two consecutive marks in the same column, becomes
    a bridge of 8. The golf entry treats "a gap between neighbouring marks"
    as its unit (row pass, then column pass on the transpose, only background
    cells changing). Here each bridge is one unit: the whole gap between two
    neighbouring marks, picked from I alone, painted in one Color8 stroke.
    Row bridges go first. Then each column bridge is painted as a whole gap,
    even where it crosses row bridges that are already 8 (that is layering).
    A column bridge is skipped only when every cell in it is already 8 on the
    current grid, because then the stroke would change nothing.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = None
    if examples:
        ex_cnt = Counter()
        for ei, _eo in examples:
            ex_cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
        bgc = ex_cnt.most_common(1)[0][0]
        if bgc not in cnt:
            bgc = None
    if bgc is None:
        bgc = cnt.most_common(1)[0][0]
    others = [c for c in cnt if c != bgc]

    if others:
        fgc = min(others, key=lambda c: (cnt[c], c))
        cur = I.copy()

        # --- row bridges: gap between each pair of neighbouring marks in a row ---
        for r in range(hi):
            marks = [c for c in range(wi) if I[r, c] == fgc]
            for a, b in zip(marks, marks[1:]):
                if b - a < 2:
                    continue
                cells = [(r, c) for c in range(a + 1, b)]
                if all(cur[p] == 8 for p in cells):
                    continue
                ops.append(8); sels.append(sel_of(cells))
                for p in cells:
                    cur[p] = 8

        # --- column bridges: gap between each pair of neighbouring marks in a column ---
        for c in range(wi):
            marks = [r for r in range(hi) if I[r, c] == fgc]
            for a, b in zip(marks, marks[1:]):
                if b - a < 2:
                    continue
                cells = [(r, c) for r in range(a + 1, b)]
                if all(cur[p] == 8 for p in cells):
                    continue
                ops.append(8); sels.append(sel_of(cells))
                for p in cells:
                    cur[p] = 8

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
                        f"num_examples+1 ({num_examples + 1}) for task dbc1a6ce"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task dbc1a6ce"
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
                                f"for task dbc1a6ce"
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
                    f"Failed to build a complete episode for task dbc1a6ce "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"dbc1a6ce-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
