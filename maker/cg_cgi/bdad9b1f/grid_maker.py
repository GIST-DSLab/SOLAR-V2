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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, a + int(round((b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c != 4]
    if bgc is None or bgc == 4:
        bgc = random.choice(cols)
    hmax = max(4, min(30, max_h))
    wmax = max(4, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (4, hmax))
    w = _unifint(diff_lb, diff_ub, (4, wmax))
    numh = _unifint(diff_lb, diff_ub, (1, max(1, h // 2 - 1)))
    numw = _unifint(diff_lb, diff_ub, (1, max(1, w // 2 - 1)))
    hpool = list(range(2, h - 1))
    wpool = list(range(2, w - 1))
    numh = max(1, min(numh, len(hpool)))
    numw = max(1, min(numw, len(wpool)))
    hlocs = random.sample(hpool, numh)
    wlocs = random.sample(wpool, numw)
    remcols = [c for c in cols if c != bgc]
    numcols = _unifint(diff_lb, diff_ub, (2, len(remcols)))
    ccols = random.sample(remcols, numcols)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    fc = -1
    for ii in sorted(hlocs):
        col = random.choice([c for c in ccols if c != fc])
        fc = col
        objw = random.randint(2, ii)
        gi[ii, 0:min(objw, w)] = col
        go[ii, :] = col
    fc = -1
    for jj in sorted(wlocs):
        col = random.choice([c for c in ccols if c != fc])
        fc = col
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


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Corners are never touched by a line in this generator -> background.
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
            colsx = {x for _, x in cells}
            if len(rows) == 1 and len(colsx) >= 2:
                hlines.append((cells[0][0], color))
            elif len(colsx) == 1 and len(rows) >= 2:
                vlines.append((cells[0][1], color))

    G = I.copy()
    ops, sels = [], []

    def paint(color, cells):
        cells = [(r, c) for r, c in cells if G[r, c] != color]
        if not cells:
            return
        for r, c in cells:
            G[r, c] = color
        ops.append(int(color))
        sels.append(sel_of(cells))

    # Each vertical segment is extended into its full column.
    for j, col in sorted(vlines):
        paint(col, [(r, j) for r in range(hi)])
    # Each horizontal segment is extended into its full row, crossing the columns.
    for r0, col in sorted(hlines):
        paint(col, [(r0, c) for c in range(wi)])
    # Every crossing of a line row with a line column becomes yellow.
    crossings = [(r0, j) for r0, _ in sorted(hlines) for j, _ in sorted(vlines)]
    paint(4, crossings)

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
