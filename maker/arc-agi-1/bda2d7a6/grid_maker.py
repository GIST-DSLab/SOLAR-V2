"""
ARC Task: bda2d7a6 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"same_ends": False}, {"same_ends": True}]


def sample_colors(num_examples=None) -> dict:
    # The task rule is a cyclic permutation of the ring colours; the palette itself is
    # arbitrary, but keep one palette per episode so all instances look consistent.
    ncols = random.randint(2, 10)
    cols = random.sample(range(10), ncols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"cols": cols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, cols, same_ends=None) -> dict:
    def _u(lb, ub, rng):
        a, b = rng
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        if hi < lo:
            lo, hi = hi, lo
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        return random.randint(lo, hi)

    hmax = max(2, min(14, max_h // 2))
    wmax = max(2, min(14, max_w // 2))
    if same_ends is None:
        same_ends = random.choice([True, False])
    if same_ends and (hmax < 3 or wmax < 3):
        same_ends = False
    lo_dim = 3 if same_ends else 2

    h = _u(diff_lb, diff_ub, (lo_dim, max(lo_dim, hmax)))
    w = _u(diff_lb, diff_ub, (lo_dim, max(lo_dim, wmax)))
    m = min(h, w)

    # ---- pick the per-ring colour list (outermost ring first) ----
    colord = None
    for _ in range(500):
        cand = [random.choice(cols) for _ in range(m)]
        if same_ends:
            cand[-1] = cand[0]
        else:
            if cand[-1] == cand[0]:
                alts = [c for c in cols if c != cand[0]]
                if not alts:
                    continue
                cand[-1] = random.choice(alts)
        nruns = 1 + sum(1 for i in range(1, m) if cand[i] != cand[i - 1])
        if nruns < 2:
            continue
        colord = cand
        break
    if colord is None:
        colord = [cols[i % len(cols)] for i in range(m)]

    H, W = 2 * h, 2 * w
    gi = [[0] * W for _ in range(H)]
    ring_cells = []
    for idx in range(m):
        r0, c0 = idx, idx
        r1, c1 = H - 1 - idx, W - 1 - idx
        cells = set()
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
            cells.add((r, c1))
        for (r, c) in cells:
            gi[r][c] = colord[idx]
        ring_cells.append(cells)

    # ---- runs of equal consecutive rings = connected objects ----
    runs = []  # (color, cellset), outermost first
    i = 0
    while i < m:
        j = i
        cs = set()
        while j < m and colord[j] == colord[i]:
            cs |= ring_cells[j]
            j += 1
        runs.append((colord[i], cs))
        i = j

    objs = list(reversed(runs))  # ascending by size: innermost first
    if len(objs) > 1 and objs[0][0] == objs[-1][0]:
        merged = (objs[0][0], objs[0][1] | objs[-1][1])
        objs = [merged] + objs[1:-1]

    n = len(objs)
    go = [row[:] for row in gi]
    for k in range(n):
        newc = objs[(k + 1) % n][0]
        for (r, c) in objs[k][1]:
            go[r][c] = newc

    return {"input": gi, "output": go}


def derive_operations(I, O):
    """Each nested object takes the colour of the next-LARGER object; the largest
    takes the colour of the smallest.  Everything is measured from I."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    # --- segment I into connected same-colour objects (4-connectivity, bg included) ---
    seen = np.zeros((H, W), dtype=bool)
    objs = []
    for r in range(H):
        for c in range(W):
            if seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            span = max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
            objs.append({"color": col, "cells": cells, "span": span})

    # --- order ascending by bounding-box extent (innermost first) ---
    objs.sort(key=lambda o: (o["span"], len(o["cells"])))

    # --- merge branch: smallest and largest share a colour -> single object, listed first ---
    if len(objs) > 1 and objs[0]["color"] == objs[-1]["color"]:
        merged = {"color": objs[0]["color"],
                  "cells": objs[0]["cells"] + objs[-1]["cells"],
                  "span": objs[-1]["span"]}
        objs = [merged] + objs[1:-1]

    n = len(objs)
    ops, sels = [], []

    # --- paint from the outermost object inward ---
    for k in range(n - 1, -1, -1):
        newc = objs[(k + 1) % n]["color"]
        if newc == objs[k]["color"]:
            continue  # object already holds this colour: painting it would do nothing
        ops.append(int(newc))
        sels.append(sel_of(objs[k]["cells"]))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task bda2d7a6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bda2d7a6"
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
                                f"for task bda2d7a6"
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
                    f"Failed to build a complete episode for task bda2d7a6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bda2d7a6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
