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

VARIANTS = [{"transposed": False}, {"transposed": True}]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lb = a + int((b - a) * diff_lb)
    ub = a + int((b - a) * diff_ub)
    lb = max(a, min(lb, b))
    ub = max(lb, min(ub, b))
    return random.randint(lb, ub)


def sample_colors(num_examples=None) -> dict:
    # stripe colours: rule is colour-independent, but 0 is reserved (ARCLE treats 0 as
    # "nothing" for Move/Crop), so stripes never use it.
    palette = list(range(1, 10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None, transposed=None) -> dict:
    if palette is None:
        palette = list(range(1, 10))
    if transposed is None:
        transposed = random.choice([True, False])

    # limA = axis carrying the stripe sequence, limB = axis the stripes are extruded along
    if transposed:
        limA, limB = max_w, max_h
    else:
        limA, limB = max_h, max_w

    h = _unifint(diff_lb, diff_ub, (2, max(2, min(15, limA - 1))))
    w = _unifint(diff_lb, diff_ub, (1, max(1, limB)))

    cols = []
    lastc = -1
    for _ in range(h):
        c = random.choice([x for x in palette if x != lastc])
        cols.append(c)
        lastc = c

    go = tuple((c,) for c in cols)
    gi = tuple(tuple([c] * w) for c in cols)

    maxins = limA - h
    if maxins >= 1:
        numinserts = _unifint(diff_lb, diff_ub, (1, maxins))
        for _ in range(numinserts):
            loc = random.randint(0, len(gi) - 1)
            gi = gi[:loc + 1] + gi[loc:]

    if transposed:
        gi = tuple(zip(*gi))
        go = tuple(zip(*go))
        gi = tuple(tuple(int(v) for v in row) for row in gi)
        go = tuple(tuple(int(v) for v in row) for row in go)

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    ops, sels = [], []

    # --- measure the rule from I ---------------------------------------------
    # Stripes run along whole rows (first row uniform) or whole columns.
    row_mode = len(set(I[0].tolist())) == 1
    seq = [int(v) for v in (I[:, 0] if row_mode else I[0, :])]

    runs = []  # [colour, length] of consecutive identical stripes
    for v in seq:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    n = len(runs)

    # --- optional 0-stripe protection (Move/Crop ignore 0 cells) -------------
    present = set(int(v) for v in np.unique(I))
    zero_fix_cells = None
    if 0 in present:
        spare = next((c for c in range(1, 10) if c not in present), None)
        if spare is not None:
            zcells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 0]
            ops.append(spare)
            sels.append(sel_of(zcells))
            zero_fix_cells = [(i, 0) if row_mode else (0, i)
                              for i, (v, _l) in enumerate(runs) if v == 0]

    # --- collapse each duplicated stripe run by sliding the rest of the grid --
    # every selection below is exactly the full rectangle of live content that
    # must slide, background included -> bbox form is the intended cell set.
    clen = hi if row_mode else wi          # live extent along the stripe axis
    cur = 0                                # row/col where the current run now sits
    for (v, L) in runs:
        excess = L - 1
        if excess > 0:
            s = cur + L                    # start of the block below this run
            for _ in range(excess):
                if s > clen - 1:
                    break
                if row_mode:
                    ops.append(20)                       # MoveU
                    sels.append([s, 0, clen - 1 - s, wi - 1])
                else:
                    ops.append(23)                       # MoveL
                    sels.append([0, s, hi - 1, clen - 1 - s])
                s -= 1
                clen -= 1
            if s > clen - 1:               # trailing run: its copies are just dropped
                clen = cur + 1
        cur += 1

    # --- keep one cell per distinct stripe ------------------------------------
    if row_mode:
        ops.append(33)
        sels.append([0, 0, n - 1, 0])
        final = [0, 0, n - 1, 0]
    else:
        ops.append(33)
        sels.append([0, 0, 0, n - 1])
        final = [0, 0, 0, n - 1]

    if zero_fix_cells:
        ops.append(0)
        sels.append(sel_of(zero_fix_cells))

    ops.append(34)
    sels.append(final)
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
