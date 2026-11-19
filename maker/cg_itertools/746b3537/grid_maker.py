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

ORIENTATIONS = [{"transpose": False}, {"transpose": True}]


def _unif(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, a + int(round((b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    # Every stripe colour is random and the rule ignores colour, so the only
    # thing to plan is the orientation. Both orientations must show up.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTATIONS):
        examples = [dict(v) for v in ORIENTATIONS]
        examples += [dict(random.choice(ORIENTATIONS)) for _ in range(n_ex - len(ORIENTATIONS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ORIENTATIONS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, transpose=None, **kwargs) -> dict:
    if transpose is None:
        transpose = random.choice([False, True])
    # Stripe axis length is limited by the rows (or by the columns if transposed).
    lim_len = max_w if transpose else max_h
    lim_w = max_h if transpose else max_w
    lim_len = min(30, lim_len)
    lim_w = min(30, lim_w)
    fullcols = list(range(10))
    while True:
        h = _unif(diff_lb, diff_ub, (2, min(15, max(2, lim_len - 1))))
        w = _unif(diff_lb, diff_ub, (1, max(1, lim_w)))
        cols = []
        lastc = -1
        for _ in range(h):
            c = random.choice([x for x in fullcols if x != lastc])
            cols.append(c)
            lastc = c
        # 0-stripes need a free colour to survive Move/Crop, so all 10 colours cannot be used
        if 0 in cols and len(set(cols)) == 10:
            continue
        break
    go = [[c] for c in cols]
    gi = [[c] * w for c in cols]
    numinserts = _unif(diff_lb, diff_ub, (1, max(1, lim_len - h)))
    for _ in range(numinserts):
        if len(gi) >= lim_len:
            break
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + gi[loc:]
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Rule (axis-symmetric reading): merge consecutive identical ROWS into one,
    # then merge consecutive identical COLUMNS into one. No orientation test is
    # needed: along the uniform axis every line is identical, so that axis
    # collapses to a single line.
    def runs_of(lines):
        runs = []
        for v in lines:
            if runs and runs[-1][0] == v:
                runs[-1][1] += 1
            else:
                runs.append([v, 1])
        return runs

    row_runs = runs_of([tuple(I[r, :].tolist()) for r in range(hi)])
    col_runs = runs_of([tuple(I[:, c].tolist()) for c in range(wi)])
    nr, nc = len(row_runs), len(col_runs)

    # Move, Crop and Paste skip 0 cells, so a 0 stripe gets a temporary colour
    # (one that is not in I). After the crop it is painted back to 0.
    present = set(int(v) for v in np.unique(I))
    zero_fix_cells = []
    if 0 in present:
        spare = next((c for c in range(1, 10) if c not in present), None)
        if spare is not None:
            zcells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 0]
            ops.append(spare)
            sels.append(sel_of(zcells))
            row_cols = [set(run[0]) for run in row_runs]
            col_cols = [set(run[0]) for run in col_runs]
            for i in range(nr):
                for j in range(nc):
                    if row_cols[i] & col_cols[j] == {0}:
                        zero_fix_cells.append((i, j))

    # --- pass 1: merge duplicate rows by sliding everything below up ----------
    # selection = the full rectangle of live rows below the run (bbox = intended cells)
    rlen = hi
    cur = 0
    for (_v, L) in row_runs:
        s = cur + L
        for _ in range(L - 1):
            if s > rlen - 1:
                break
            ops.append(20)  # MoveU
            sels.append([s, 0, rlen - 1 - s, wi - 1])
            s -= 1
            rlen -= 1
        if s > rlen - 1:     # trailing run: the crop removes its extra copies
            rlen = cur + 1
        cur += 1

    # --- pass 2: merge duplicate columns by sliding everything to the right left --
    clen = wi
    cur = 0
    for (_v, L) in col_runs:
        s = cur + L
        for _ in range(L - 1):
            if s > clen - 1:
                break
            ops.append(23)  # MoveL
            sels.append([0, s, rlen - 1, clen - 1 - s])
            s -= 1
            clen -= 1
        if s > clen - 1:
            clen = cur + 1
        cur += 1

    # --- keep one cell per merged row/column ----------------------------------
    ops.append(33)
    sels.append([0, 0, nr - 1, nc - 1])

    if zero_fix_cells:
        ops.append(0)
        sels.append(sel_of(zero_fix_cells))

    ops.append(34)
    sels.append([0, 0, nr - 1, nc - 1])
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
