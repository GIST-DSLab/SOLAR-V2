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
from collections import deque
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule is colour-agnostic (each ring takes the colour of the next ring
    # outward, and the outermost takes the innermost colour). Fixing a palette
    # for the episode only keeps the instances visually consistent.
    ncols = random.randint(2, 10)
    cols = random.sample(range(10), ncols)
    return {"cols": cols}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    d = random.uniform(diff_lb, diff_ub)
    return int(round(lo + d * (hi - lo)))


def _objects(G):
    G = np.asarray(G)
    H, W = G.shape
    seen = np.zeros((H, W), bool)
    objs = []
    for r in range(H):
        for c in range(W):
            if seen[r, c]:
                continue
            col = G[r, c]
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                a, b = q.popleft()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < H and 0 <= nb < W and not seen[na, nb] and G[na, nb] == col:
                        seen[na, nb] = True
                        q.append((na, nb))
            rs = [x for x, _ in cells]
            cs = [y for _, y in cells]
            size = max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
            objs.append({"color": int(col), "cells": cells, "size": size})
    objs.sort(key=lambda o: o["size"])
    return objs


def _plan(G):
    objs = _objects(G)
    if len(objs) >= 2 and objs[0]["color"] == objs[-1]["color"]:
        merged = {"color": objs[0]["color"],
                  "cells": objs[0]["cells"] + objs[-1]["cells"],
                  "size": objs[0]["size"]}
        objs = [merged] + objs[1:-1]
    n = len(objs)
    # target object k = (last, 0, 1, ..., n-2)[i] receives colour of objs[i]
    targets = [objs[-1]] + objs[:-1]
    assign = []
    for i in range(n):
        assign.append((targets[i], objs[i]["color"]))
    return assign


def generate(diff_lb, diff_ub, max_h, max_w, cols=None) -> dict:
    if cols is None:
        cols = random.sample(range(10), random.randint(2, 10))
    hmax = max(2, min(14, max_h // 2))
    wmax = max(2, min(14, max_w // 2))
    h = _unifint(diff_lb, diff_ub, 2, hmax)
    w = _unifint(diff_lb, diff_ub, 2, wmax)
    colord = [random.choice(cols) for _ in range(min(h, w))]
    H, W = h * 2, w * 2
    gi = np.zeros((H, W), dtype=int)
    for idx, ci in enumerate(colord):
        r0, c0, r1, c1 = idx, idx, H - 1 - idx, W - 1 - idx
        gi[r0, c0:c1 + 1] = ci
        gi[r1, c0:c1 + 1] = ci
        gi[r0:r1 + 1, c0] = ci
        gi[r0:r1 + 1, c1] = ci
    go = gi.copy()
    for obj, col in _plan(gi):
        for r, c in obj["cells"]:
            go[r, c] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    assign = _plan(I)
    # paint from outermost object inward; each object is recoloured as a whole
    # with the colour of the next object inward-ordered rule (outer gets innermost)
    assign.sort(key=lambda t: -max(
        max(r for r, _ in t[0]["cells"]) - min(r for r, _ in t[0]["cells"]) + 1,
        max(c for _, c in t[0]["cells"]) - min(c for _, c in t[0]["cells"]) + 1))
    for obj, col in assign:
        if col == obj["color"]:
            continue
        ops.append(int(col))
        sels.append(sel_of(sorted(obj["cells"])))
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
