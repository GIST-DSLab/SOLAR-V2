"""
ARC Task: cce03e0d (RE-ARC) — LLM-generated grid_maker
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
    # Background (0) and the marker colour (2) are hardcoded in the generator.
    # The other colours only decorate the copied tile and do not change the rule,
    # so there is nothing to fix for the whole episode.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))

    # The output is h^2 x w^2, so h and w are limited by sqrt(max).
    hcap = max(2, min(5, int(max_h ** 0.5)))
    wcap = max(2, min(5, int(max_w ** 0.5)))
    cols = [c for c in range(10) if c not in (2, 8)]
    h = unifint(diff_lb, diff_ub, (2, hcap))
    w = unifint(diff_lb, diff_ub, (2, wcap))
    nred = unifint(diff_lb, diff_ub, (1, h * w - 1))
    ncols = unifint(diff_lb, diff_ub, (1, min(8, nred)))
    ncells = unifint(diff_lb, diff_ub, (1, h * w - nred))
    ccols = random.sample(cols, ncols)
    gi = [[0] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    reds = random.sample(inds, nred)
    for (i, j) in reds:
        gi[i][j] = 2
    reminds = [ij for ij in inds if ij not in set(reds)]
    rest = random.sample(reminds, ncells)
    for (i, j) in rest:
        gi[i][j] = random.choice(ccols)
    go = [[0] * (w * w) for _ in range(h * h)]
    for (ri, rj) in reds:
        for i in range(h):
            for j in range(w):
                if gi[i][j] != 0:
                    go[ri * h + i][rj * w + j] = gi[i][j]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Rule: canvas becomes (hi*hi, wi*wi); every red cell (r, c) of I places a copy
    # of I at block (r, c). Everything is computed from I alone.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi * hi, wi * wi
    red = (I == 2)

    def plan(axis):
        # axis 0: work in block-rows; axis 1: work in block-columns.
        # Block-rows (or columns) with the same red pattern look the same, so the
        # first one is built with pastes of I, then copied whole and pasted
        # into the others.
        n = hi if axis == 0 else wi
        pats = [tuple(red[i, :]) if axis == 0 else tuple(red[:, i]) for i in range(n)]
        groups = {}
        for i, p in enumerate(pats):
            if any(p):
                groups.setdefault(p, []).append(i)
        direct, dup = [], []
        for p, idx in groups.items():
            k = sum(p)
            if len(idx) > 1 and k >= 2:
                direct.append(idx[0])
                dup.append(idx)
            else:
                direct += idx
        ops, sels = [], []
        # The clipboard holds I: paste one tile at each red cell of the rows built directly.
        for i in sorted(direct):
            for j in range(wi if axis == 0 else hi):
                r, c = (i, j) if axis == 0 else (j, i)
                if red[r, c] and not (r == 0 and c == 0):
                    ops.append(30); sels.append([r * hi, c * wi, 0, 0])
        # Copy a finished block-row/column and paste it at its matching rows/columns.
        for idx in dup:
            src = idx[0]
            if axis == 0:
                # whole block-row strip: a full rectangle
                ops.append(29); sels.append([src * hi, 0, hi - 1, wo - 1])
            else:
                # whole block-column strip: a full rectangle
                ops.append(29); sels.append([0, src * wi, ho - 1, wi - 1])
            for i in idx[1:]:
                ops.append(30)
                sels.append([i * hi, 0, 0, 0] if axis == 0 else [0, i * wi, 0, 0])
        return ops, sels

    # 1. Grow the canvas to the output size (selection is the full rectangle).
    # 2. Put the whole input on the clipboard (full rectangle).
    ops = [33, 28]
    sels = [[0, 0, ho - 1, wo - 1], [0, 0, hi - 1, wi - 1]]
    # The resize left a copy of I in the top-left block. It only belongs there
    # if I[0,0] is red, so otherwise clear that block (a full rectangle).
    if not red[0, 0]:
        ops.append(0); sels.append([0, 0, hi - 1, wi - 1])

    a, b = plan(0), plan(1)
    po, ps = a if len(a[0]) <= len(b[0]) else b
    ops += po
    sels += ps

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task cce03e0d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task cce03e0d"
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
                                f"for task cce03e0d"
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
                    f"Failed to build a complete episode for task cce03e0d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"cce03e0d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
