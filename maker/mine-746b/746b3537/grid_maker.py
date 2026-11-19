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

# The only structural branch: stripes are horizontal (answer is a column) or
# vertical (answer is a row, the generator's dmirror).
VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    # Rule is purely structural (collapse runs of repeated lines); the generator
    # samples stripe colours freely from all 10 and has no background, so no
    # colour role needs fixing.  Only the orientation is planned per instance.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    # One stripe palette per episode, and every instance uses all of it, so a
    # test colour is never one the examples did not show.  0 is an ordinary
    # stripe colour here (no background).
    palette = random.sample(range(10), random.randint(3, 6))
    return {"palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None, transposed=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if transposed is None:
        transposed = random.choice([True, False])
    # extent along the stripe axis (before the optional transpose = rows)
    len_cap = max(3, min(30, max_w if transposed else max_h))
    thick_cap = max(1, min(30, max_h if transposed else max_w))

    if not palette or len(palette) < 2:
        palette = list(range(10))
    np_ = min(len(palette), len_cap - 1)
    palette = list(palette)[:np_]
    h = unifint(diff_lb, diff_ub, (np_, max(np_, min(15, len_cap - 1))))
    h = max(np_, min(h, len_cap - 1))
    w = unifint(diff_lb, diff_ub, (1, thick_cap))
    w = max(1, min(w, thick_cap))

    for _ in range(200):
        cols, lastc = [], -1
        for _ in range(h):
            c = random.choice([x for x in palette if x != lastc])
            cols.append(c)
            lastc = c
        if set(cols) == set(palette):
            break
    else:
        cols = random.sample(palette, len(palette))
        while len(cols) < h:
            cols.append(random.choice([x for x in palette if x != cols[-1]]))
    go = [[c] for c in cols]
    gi = [[c] * w for c in cols]

    numinserts = unifint(diff_lb, diff_ub, (1, len_cap - h))
    numinserts = max(1, min(numinserts, len_cap - h))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + [list(gi[loc])] + gi[loc + 1:]

    if transposed:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Reading taken from the golf entry: "drop every line equal to the line
    before it", applied the same way to rows and then to columns -- no
    orientation test.  Each run of k identical lines is squeezed by sliding the
    whole block of lines after it back by k-1 (one grab, then empty-selection
    Moves); a run with nothing after it is simply left for the final crop.
    Along the axis where the stripes don't vary, everything is one run, so that
    pass reduces to the crop keeping its first line."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def runs_of(lines):
        out = []
        for ln in lines:
            if out and out[-1][0] == ln:
                out[-1][1] += 1
            else:
                out.append([ln, 1])
        return out

    row_runs = runs_of([tuple(r) for r in I.tolist()])
    col_runs = runs_of([tuple(c) for c in I.T.tolist()])
    nr, nc = len(row_runs), len(col_runs)

    # Move/Crop treat 0 as empty: protect 0-coloured stripes with an unused
    # colour while the grid is being squeezed, restore them at the end.
    present = set(int(v) for v in np.unique(I))
    zero_tmp = None
    if 0 in present:
        zero_tmp = next((c for c in range(1, 10) if c not in present), None)
        if zero_tmp is not None:
            ops.append(zero_tmp)
            sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 0]))

    # pass 1: rows.  Selection = full rectangle of the rows after the run
    # (whole-width block of stripes, intended exactly as that rectangle).
    live_h = hi
    cur = 0
    for _ln, L in row_runs:
        k = L - 1
        start = cur + L
        if k > 0 and start <= live_h - 1:
            ops.append(20); sels.append([start, 0, live_h - 1 - start, wi - 1])  # grab block
            for _ in range(k - 1):
                ops.append(20); sels.append(sel_of([]))                         # same block
            live_h -= k
        elif k > 0:
            live_h = cur + 1          # trailing run: copies are dropped by the crop
        cur += 1

    # pass 2: columns of the remaining nr rows.
    live_w = wi
    cur = 0
    for _ln, L in col_runs:
        k = L - 1
        start = cur + L
        if k > 0 and start <= live_w - 1:
            ops.append(23); sels.append([0, start, nr - 1, live_w - 1 - start])
            for _ in range(k - 1):
                ops.append(23); sels.append(sel_of([]))
            live_w -= k
        elif k > 0:
            live_w = cur + 1
        cur += 1

    # keep one line per run on each axis (rectangle crop)
    ops.append(33); sels.append([0, 0, nr - 1, nc - 1])

    # restore the black stripes in the collapsed picture
    if 0 in present:
        zr = [i for i, (ln, _l) in enumerate(row_runs) if ln[0] == 0 and len(set(ln)) == 1]
        zc = [j for j, (ln, _l) in enumerate(col_runs) if ln[0] == 0 and len(set(ln)) == 1]
        cells = sorted({(i, j) for i in zr for j in range(nc)} | {(i, j) for j in zc for i in range(nr)})
        if zero_tmp is not None and cells:
            ops.append(0); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, nr - 1, nc - 1])
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
