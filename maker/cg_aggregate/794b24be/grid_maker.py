"""
ARC Task: 794b24be (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of

# The output is a 3x3 board on the background colour.
# Each blue (colour-1) cell in the input adds ONE red (2) mark to it,
# placed in this order:
TALLY_ORDER = [(0, 0), (0, 1), (0, 2), (1, 1)]
NBLUE_VARIANTS = [1, 2, 3, 4]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(NBLUE_VARIANTS):
        ex = list(NBLUE_VARIANTS)
        ex += [random.choice(NBLUE_VARIANTS) for _ in range(n_ex - len(NBLUE_VARIANTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(NBLUE_VARIANTS, n_ex)
    plan = [{"nblue": k} for k in ex]
    plan.append({"nblue": random.choice(ex)})
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, nblue=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = [c for c in range(10) if c not in (1, 2)]
    if bgc is None:
        bgc = random.choice(cols)
    if nblue is None:
        nblue = random.randint(1, 4)
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))

    go = np.full((3, 3), bgc, dtype=int)
    for k in range(nblue):
        r, c = TALLY_ORDER[k]
        go[r, c] = 2

    gi = np.full((h, w), bgc, dtype=int)
    allidx = [(r, c) for r in range(h) for c in range(w)]
    for (r, c) in random.sample(allidx, nblue):
        gi[r, c] = 1
    remlocs = [(r, c) for r in range(h) for c in range(w) if gi[r, c] == bgc]
    namt = unifint(diff_lb, diff_ub, (0, max(0, len(remlocs) // 2 - 1)))
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (1, 7))
    ccols = random.sample(remcols, min(numc, len(remcols)))
    for (r, c) in random.sample(remlocs, namt):
        gi[r, c] = random.choice(ccols)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The rule, read from I only: count the blue cells.
    n = int((I == 1).sum())
    # Background = most frequent colour other than blue. The noise budget is
    # always less than half the free cells, so this is guaranteed.
    cnt = Counter(v for v in I.flatten().tolist() if v != 1)
    bgc = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    ops, sels = [], []

    # 1. Shrink the canvas to the 3x3 tally board (it is a full rectangle,
    #    so a bbox is the exact selection). Cropping a 3x3 input changes
    #    nothing, so skip it then.
    if (hi, wi) != (3, 3):
        ops.append(33); sels.append([0, 0, 2, 2])
    # Crop is a transparent copy, so zeros stay zero.
    board = {(r, c): int(I[r, c]) for r in range(3) for c in range(3)}

    # 2. Clear the board to background. This is a base layer: tally cells
    #    stay in it. Only cells that do not already show bgc are selected.
    dirty = sorted(rc for rc, v in board.items() if v != bgc)
    if dirty:
        ops.append(bgc); sels.append(sel_of(dirty))
        for rc in dirty:
            board[rc] = bgc

    # 3. Tally: each blue cell in I adds one red mark, in the fixed order
    #    across the top row and then the centre.
    for k in range(min(n, len(TALLY_ORDER))):
        rc = TALLY_ORDER[k]
        if board[rc] != 2:
            ops.append(2); sels.append(sel_of([rc]))
            board[rc] = 2

    ops.append(34); sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task 794b24be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 794b24be"
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
                                f"for task 794b24be"
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
                    f"Failed to build a complete episode for task 794b24be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"794b24be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
