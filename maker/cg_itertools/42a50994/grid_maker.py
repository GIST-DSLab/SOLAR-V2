"""
ARC Task: 42a50994 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background matters to the rule. The rule looks at whether
    # non-background cells are present, not at their colours, so the
    # foreground colours are still sampled per instance inside generate().
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _lone_cells(grid, bgc):
    """Non-background cells with no non-background cell among their 8 neighbours.
    These are exactly the size-1 multicolour 8-connected objects."""
    h, w = len(grid), len(grid[0])
    out = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bgc:
                continue
            alone = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != bgc:
                        alone = False
            if alone:
                out.append((r, c))
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    colopts = list(range(10))
    h = _unifint(diff_lb, diff_ub, (1, max_h))
    w = _unifint(diff_lb, diff_ub, (1, max_w))
    if bgc is None:
        bgc = random.choice(colopts)
    remcols = [c for c in colopts if c != bgc]
    card_bounds = (0, max(0, (h * w) // 2 - 1))
    num = _unifint(diff_lb, diff_ub, card_bounds)
    numcols = _unifint(diff_lb, diff_ub, (0, min(9, num)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    chosinds = random.sample(inds, num)
    choscols = random.sample(remcols, numcols)
    locs = list(range(len(chosinds)))
    choslocs = random.sample(locs, numcols)
    gi = [[bgc] * w for _ in range(h)]
    for col, endidx in zip(choscols, sorted(choslocs)[::-1]):
        for (i, j) in chosinds[:endidx]:
            gi[i][j] = col
    go = [row[:] for row in gi]
    for (i, j) in _lone_cells(gi, bgc):
        go[i][j] = bgc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # The code-golf version tests each cell on its own: a cell stays only if
    # its 3x3 neighbourhood holds some other non-background cell. Here that
    # gives the same result as removing the size-1 8-connected objects,
    # because an object has size 1 exactly when its cell has no non-background
    # neighbour. So the two readings agree. The thing removed is the isolated
    # single cell, and it is removed with one Color(bgc) per cell, using only I.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the generator scatters at most (h*w)//2 - 1 foreground cells,
    # so bgc is always the strict majority colour of I.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    for (r, c) in _lone_cells(I.tolist(), bgc):
        ops.append(int(bgc))
        sels.append(sel_of([(r, c)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 42a50994"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 42a50994"
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
                                f"for task 42a50994"
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
                    f"Failed to build a complete episode for task 42a50994 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"42a50994-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
