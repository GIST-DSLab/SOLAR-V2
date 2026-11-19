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

VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    # One small stripe palette per episode (0 is an ordinary stripe colour here).
    palette = random.sample(range(10), random.randint(3, 6))
    return {"palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None, mirrored=None) -> dict:
    def unifint_(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if mirrored is None:
        mirrored = random.choice([True, False])
    len_cap = min(30, max_w if mirrored else max_h)   # stripe axis (rows before mirror)
    thick_cap = min(30, max_h if mirrored else max_w)
    len_cap = max(len_cap, 3)
    thick_cap = max(thick_cap, 1)
    fullcols = list(palette) if palette and len(palette) >= 2 else list(range(10))
    h = unifint_(diff_lb, diff_ub, (2, min(15, len_cap - 1)))
    w = unifint_(diff_lb, diff_ub, (1, thick_cap))
    cols = []
    lastc = -1
    for _ in range(h):
        c = random.choice([x for x in fullcols if x != lastc])
        cols.append(c)
        lastc = c
    go = [[c] for c in cols]
    gi = [[c] * w for c in cols]
    numinserts = unifint_(diff_lb, diff_ub, (1, len_cap - h))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + gi[loc:]
    gi = np.array(gi, dtype=int)
    go = np.array(go, dtype=int)
    if mirrored:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Orientation from I: if the first row is a single colour, the stripes are rows.
    rows_mode = len(set(I[0].tolist())) == 1
    A = I if rows_mode else I.T          # work in "stripes are rows" coordinates
    n_len, n_thick = A.shape

    def cell(k):                          # k = position along the stripe axis
        return (k, 0) if rows_mode else (0, k)

    MOVE_BACK = 20 if rows_mode else 23   # MoveU / MoveL

    ops, sels = [], []
    # 1. Collapse the thickness: every line across the stripes is identical,
    #    so keep only the first one (the whole first column / row rectangle).
    if n_thick > 1:
        ops.append(33)
        sels.append([0, 0, n_len - 1, 0] if rows_mode else [0, 0, 0, n_len - 1])

    # 2. Collapse each stripe to thickness 1, top-down, by sliding everything
    #    below it back over its duplicate lines.
    seq = A[:, 0].tolist()
    runs = []
    k = 0
    while k < n_len:
        j = k
        while j + 1 < n_len and seq[j + 1] == seq[k]:
            j += 1
        runs.append((seq[k], j - k + 1))
        k = j + 1

    cur = list(seq)                       # current content along the axis
    cur_len = n_len                       # meaningful length (rest is 0 padding)
    p = 0
    for color, L in runs:
        if L > 1:
            rest = cur[p + L:cur_len]
            # erase this stripe's duplicate lines (only those that stay inside
            # the collapsed canvas; the rest are cut by the final crop) ...
            dup = [cell(t) for t in range(p + 1, min(p + L, p + 1 + len(rest)))]
            if color != 0 and dup:
                ops.append(0)
                sels.append(sel_of(dup))
            # ... then close the gap: slide everything after the stripe back
            # by L-1 (one grab, then empty-selection continuation moves).
            if any(v != 0 for v in rest):
                src = [cell(t) for t in range(p + L, cur_len)]
                ops.append(MOVE_BACK); sels.append(sel_of(src))
                for _ in range(L - 2):
                    ops.append(MOVE_BACK); sels.append(sel_of([]))
            cur = cur[:p + 1] + rest + [0] * (L - 1)
            cur_len -= (L - 1)
        p += 1

    # 3. Crop the canvas to the collapsed length.
    ops.append(33)
    sels.append([0, 0, cur_len - 1, 0] if rows_mode else [0, 0, 0, cur_len - 1])
    ho, wo = (cur_len, 1) if rows_mode else (1, cur_len)
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
