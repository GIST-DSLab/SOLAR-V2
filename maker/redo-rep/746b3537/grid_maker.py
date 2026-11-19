"""
ARC Task: 746b3537 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    # Row colours are drawn randomly by the generator; the rule itself is colour-independent
    # (it is about runs of identical rows/columns), but we keep one palette per episode.
    palette = random.sample(list(range(10)), random.randint(4, 10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    if not palette or len(palette) < 2:
        palette = list(range(10))

    def uni(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + int((hi - lo) * diff_lb)
        b = lo + int((hi - lo) * diff_ub)
        if b < a:
            a, b = b, a
        return random.randint(a, b)

    # canonical layout: h distinct rows (stacked vertically), width w; transposed at the end
    lim_rows = min(30, max_w if transpose else max_h)
    lim_cols = min(30, max_h if transpose else max_w)
    lim_rows = max(3, lim_rows)
    lim_cols = max(1, lim_cols)

    hmax = min(15, lim_rows - 1)
    if hmax < 2:
        hmax = 2
    h = uni(2, hmax)
    h = max(2, min(h, lim_rows - 1))
    w = uni(1, lim_cols)
    w = max(1, min(w, lim_cols))

    cols = []
    lastc = -1
    for _ in range(h):
        choices = [c for c in palette if c != lastc]
        c = random.choice(choices)
        cols.append(c)
        lastc = c

    go = tuple((c,) for c in cols)
    gi = tuple(tuple([c] * w) for c in cols)

    numinserts = uni(1, max(1, lim_rows - h))
    numinserts = max(1, min(numinserts, lim_rows - h))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + gi[loc:loc + 1] + gi[loc + 1:]

    if transpose:
        gi = tuple(zip(*gi))
        go = tuple(zip(*go))

    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O):
    """The grid is a stack of identical rows (or identical columns).  Consecutive
    duplicates collapse to one line: for every maximal group of already-adjacent
    distinct lines we CopyI that block straight out of the input and Paste it at the
    row it must occupy in the compressed stack, then ResizeGrid down to the single
    column (row) the rule keeps."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    # transposed instance: output is a single ROW -> work in a canonical column space
    transposed = (ho == 1 and wo > 1)
    A = I.T.copy() if transposed else I.copy()
    H, W = A.shape
    line = A[:, 0].tolist()

    # run starts = index in the input of each distinct line, in order
    starts = [0]
    for r in range(1, H):
        if line[r] != line[r - 1]:
            starts.append(r)
    h = len(starts)

    # maximal blocks of distinct lines that are already adjacent in the input
    blocks = []
    a = 0
    for k in range(1, h + 1):
        if k == h or starts[k] != starts[k - 1] + 1:
            blocks.append((a, k - 1))
            a = k

    ops, sels = [], []

    def emit(op, cells):
        cs = [(c, r) for (r, c) in cells] if transposed else [(r, c) for (r, c) in cells]
        ops.append(op)
        sels.append(sel_of(cs))

    grid = A.copy()          # simulated working grid (canonical orientation)
    clip = None              # simulated clipboard contents

    for (a, b) in blocks:
        s = starts[a]
        L = b - a + 1
        if s == a:
            continue          # this block already sits at its final rows
        src = [line[s + i] for i in range(L)]

        # Paste is transparent for colour 0 -> clear those destination cells first
        zero_cells = [(a + i, 0) for i in range(L) if src[i] == 0 and grid[a + i, 0] != 0]
        if zero_cells:
            emit(0, zero_cells)
            for (r, c) in zero_cells:
                grid[r, c] = 0

        if any(src[i] != 0 and grid[a + i, 0] != src[i] for i in range(L)):
            if clip != src:
                emit(28, [(s + i, 0) for i in range(L)])   # CopyI from the untouched input
                clip = list(src)
            emit(30, [(a, 0)])                             # Paste at the compressed position
            for i in range(L):
                if src[i] != 0:
                    grid[a + i, 0] = src[i]

    keep = [(r, 0) for r in range(h)]
    emit(33, keep)   # ResizeGrid: keep exactly the compressed line
    emit(34, keep)
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
                        f"num_examples+1 ({num_examples + 1}) for task 746b3537"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 746b3537"
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
                                f"for task 746b3537"
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
                    f"Failed to build a complete episode for task 746b3537 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"746b3537-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
