"""
ARC Task: 47c1f68c (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc, linc = random.sample(cols, 2)
    objc = random.choice([c for c in cols if c not in (bgc, linc)])
    return {"bgc": bgc, "linc": linc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, objc) -> dict:
    def unifint(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, int(round(lo + d * (hi - lo)))))
    h = unifint(2, min(14, (max_h - 1) // 2))
    w = unifint(2, min(14, (max_w - 1) // 2))
    nc = unifint(1, h * w - 1)
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    obj = {start}
    bx.discard(start)
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        ch = random.choice(sorted(bx & dns))
        obj.add(ch)
        bx.discard(ch)
    g = [[bgc] * w for _ in range(h)]
    for (i, j) in obj:
        g[i][j] = objc
    canv = [[bgc] * w for _ in range(h)]
    top = [g[i] + [linc] + canv[i] for i in range(h)]
    bot = [canv[i] + [linc] + canv[i] for i in range(h)]
    gi = top + [[linc] * (2 * w + 1)] + bot
    go = [row + row[::-1] for row in g]
    go = go + go[::-1]
    go = [[linc if v == objc else v for v in row] for row in go]
    scf = random.choice(["id", "h", "v", "hv"])

    def ap(x):
        if "h" in scf:
            x = x[::-1]
        if "v" in scf:
            x = [r[::-1] for r in x]
        return [list(r) for r in x]
    return {"input": ap(gi), "output": ap(go)}


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    h, w = (H - 1) // 2, (W - 1) // 2
    linc = int(I[h, 0])
    panels = [(0, 0), (0, w + 1), (h + 1, 0), (h + 1, w + 1)]
    bgc, src = None, None
    for (r0, c0) in panels:
        vals = set(I[r0:r0 + h, c0:c0 + w].flatten().tolist())
        if len(vals) == 1:
            bgc = vals.pop()
        else:
            src = (r0, c0)
    r0, c0 = src
    objc = [v for v in set(I[r0:r0 + h, c0:c0 + w].flatten().tolist()) if v != bgc][0]

    G = I.copy()
    ops, sels = [], []

    def rect(r, c, hh, ww):
        return [(y, x) for y in range(r, r + hh) for x in range(c, c + ww)]

    # 1. recolour the whole object colour at once to the divider colour
    cells = [(int(r), int(c)) for r, c in zip(*np.where(I == objc))]
    ops.append(linc); sels.append(sel_of(cells))
    for r, c in cells:
        G[r, c] = linc

    # 2. inside the original frame: copy the object panel into each sibling panel, mirror it there
    ops.append(29); sels.append(sel_of(rect(r0, c0, h, w)))
    panel = G[r0:r0 + h, c0:c0 + w].copy()
    for (pr, pc) in panels:
        if (pr, pc) == src:
            continue
        ops.append(30); sels.append(sel_of([(pr, pc)]))
        G[pr:pr + h, pc:pc + w] = panel
        if pc != c0:
            sub = G[pr:pr + h, pc:pc + w]
            if not np.array_equal(sub, sub[:, ::-1]):
                ops.append(26); sels.append(sel_of(rect(pr, pc, h, w)))
                G[pr:pr + h, pc:pc + w] = sub[:, ::-1].copy()
        if pr != r0:
            sub = G[pr:pr + h, pc:pc + w]
            if not np.array_equal(sub, sub[::-1, :]):
                ops.append(27); sels.append(sel_of(rect(pr, pc, h, w)))
                G[pr:pr + h, pc:pc + w] = sub[::-1, :].copy()

    # 3. close the vertical gutter: slide the right column of panels left over the divider
    ops.append(23); sels.append(sel_of(rect(0, w + 1, H, w)))
    # 4. close the horizontal gutter: slide the bottom row of panels up over the divider
    ops.append(20); sels.append(sel_of(rect(h + 1, 0, h, 2 * w)))
    # 5. trim the emptied last row/column (full output rectangle)
    ops.append(33); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
    ops.append(34); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
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
                                f"for task 47c1f68c"
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
                    f"Failed to build a complete episode for task 47c1f68c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"47c1f68c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
