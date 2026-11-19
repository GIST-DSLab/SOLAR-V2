"""
ARC Task: bdad9b1f (RE-ARC) — LLM-generated grid_maker
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
    # The colour of a line has no effect on the rule. The crossings always become 4,
    # which is a fixed constant. Only the background is held fixed for the whole episode.
    cols = [c for c in range(10) if c != 4]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi_)))))

    cols = [c for c in range(10) if c != 4]
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    numh = unifint(diff_lb, diff_ub, (1, h // 2 - 1))
    numw = unifint(diff_lb, diff_ub, (1, w // 2 - 1))
    hlocs = random.sample(range(2, h - 1), numh)
    wlocs = random.sample(range(2, w - 1), numw)
    numcols = unifint(diff_lb, diff_ub, (2, 8))
    remcols = [c for c in cols if c != bgc]
    ccols = random.sample(remcols, numcols)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    fc = -1
    for ii in sorted(hlocs):
        col = random.choice([c for c in ccols if c != fc]); fc = col
        objw = random.randint(2, ii)
        gi[ii, 0:min(objw, w)] = col
        go[ii, :] = col
    fc = -1
    for jj in sorted(wlocs):
        col = random.choice([c for c in ccols if c != fc]); fc = col
        objh = random.randint(2, jj)
        gi[0:min(objh, h), jj] = col
        go[:, jj] = col
    for ii in hlocs:
        for jj in wlocs:
            go[ii, jj] = 4
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # This reading matches the maker's reading. Each straight stub stretches into a
    # full row or column of the grid, and every cell where an extended row crosses an
    # extended column turns 4. The golf entry's per-cell formula encodes the same rule.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # The generator guarantees that the corners are background under every rotation.
    bgc = int(I[0, 0])

    visited = np.zeros_like(I, dtype=bool)
    hlines, vlines = [], []
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or I[r, c] == bgc:
                continue
            color = int(I[r, c])
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                if 0 <= y < hi and 0 <= x < wi and not visited[y, x] and I[y, x] == color:
                    visited[y, x] = True
                    cells.append((y, x))
                    stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
            rows = {y for y, _ in cells}
            cs = {x for _, x in cells}
            if len(rows) == 1 and len(cs) >= 2:
                hlines.append((cells[0][0], color))
            elif len(cs) == 1 and len(rows) >= 2:
                vlines.append((cells[0][1], color))

    g = I.copy()
    ops, sels = [], []
    # Extend each horizontal stub across its entire row (the whole row is a rectangle).
    for r, col in hlines:
        if np.any(g[r, :] != col):
            ops.append(col); sels.append([r, 0, 0, wi - 1])
            g[r, :] = col
    # Extend each vertical stub down its entire column (the whole column is a rectangle).
    for j, col in vlines:
        if np.any(g[:, j] != col):
            ops.append(col); sels.append([0, j, hi - 1, 0])
            g[:, j] = col
    # Mark the crossings of the extended rows and columns with 4.
    cross = sorted({(r, j) for r, _ in hlines for j, _ in vlines})
    cross = [(r, j) for r, j in cross if g[r, j] != 4]
    if cross:
        ops.append(4); sels.append(sel_of(cross))
        for r, j in cross:
            g[r, j] = 4

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
                        f"num_examples+1 ({num_examples + 1}) for task bdad9b1f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bdad9b1f"
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
                                f"for task bdad9b1f"
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
                    f"Failed to build a complete episode for task bdad9b1f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bdad9b1f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
