"""
ARC Task: 1e32b0e9 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc, fgc = random.sample(range(10), 3)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, fgc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))
    for _ in range(100):
        h = unifint(diff_lb, diff_ub, (4, 6))
        w = unifint(diff_lb, diff_ub, (4, 6))
        nh = unifint(diff_lb, diff_ub, (1, 4))
        nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
        fullh = h * nh + (nh - 1)
        fullw = w * nw + (nw - 1)
        if fullh <= max_h and fullw <= max_w:
            break
    gi = np.full((fullh, fullw), linc, dtype=int)
    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    srcloc = random.choice(llocs)
    remlocs = [l for l in llocs if l != srcloc]
    ncells = unifint(diff_lb, diff_ub, (0, (h - 2) * (w - 2) - 1))
    inds = [(i, j) for i in range(h - 2) for j in range(w - 2)]
    sp = random.choice(inds)
    shp = {sp}
    for _ in range(ncells):
        cand = [p for p in inds if p not in shp and
                any(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in shp)]
        shp.add(random.choice(cand))
    shp = [(i + 1, j + 1) for i, j in shp]
    for (a, b) in llocs:
        gi[a:a + h, b:b + w] = bgc
    go = gi.copy()
    for (i, j) in shp:
        gi[srcloc[0] + i, srcloc[1] + j] = fgc
        go[srcloc[0] + i, srcloc[1] + j] = fgc
    for (a, b) in remlocs:
        nleft = random.randint(0, len(shp))
        sub = random.sample(shp, nleft)
        for (i, j) in shp:
            go[a + i, b + j] = linc
        for (i, j) in sub:
            gi[a + i, b + j] = fgc
            go[a + i, b + j] = fgc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = int(I[0, 0])  # cell corners are always background (shape is inset by 1)
    line_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1 and I[r, 0] != bgc]
    line_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1 and I[0, c] != bgc]
    linc = int(I[line_rows[0], 0]) if line_rows else int(I[0, line_cols[0]])

    def segs(lines, n):
        out, s = [], 0
        for l in sorted(lines) + [n]:
            if l > s:
                out.append((s, l))
            s = l + 1
        return out

    cells = [(r0, r1, c0, c1) for (r0, r1) in segs(line_rows, hi)
             for (c0, c1) in segs(line_cols, wi)]

    def fg(cell):
        r0, r1, c0, c1 = cell
        return {(r - r0, c - c0) for r in range(r0, r1) for c in range(c0, c1)
                if I[r, c] != bgc and I[r, c] != linc}

    # source cell holds the complete shape (every other cell holds a subset)
    src = max(cells, key=lambda cl: len(fg(cl)))
    shape = fg(src)
    ops, sels = [], []
    for cl in cells:
        if cl == src:
            continue
        have = fg(cl)
        # complete the template in this cell: shape positions still background -> line colour
        miss = sorted((cl[0] + i, cl[2] + j) for (i, j) in shape if (i, j) not in have)
        if miss:
            ops.append(linc); sels.append(sel_of(miss))
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
                        f"num_examples+1 ({num_examples + 1}) for task 1e32b0e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1e32b0e9"
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
                                f"for task 1e32b0e9"
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
                    f"Failed to build a complete episode for task 1e32b0e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1e32b0e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
