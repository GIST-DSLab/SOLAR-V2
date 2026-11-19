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
    # Background (0) and marker (2) are hardcoded by the generator; the other
    # colours are only content that gets copied, so the rule does not depend on them.
    return {}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    return random.randint(lo_i, hi_i)


def generate(diff_lb, diff_ub, max_h, max_w, **kwargs) -> dict:
    cols = [c for c in range(10) if c not in (2, 8)]
    hmax = 5
    while hmax > 2 and hmax * hmax > max_h:
        hmax -= 1
    wmax = 5
    while wmax > 2 and wmax * wmax > max_w:
        wmax -= 1
    h = _unifint(diff_lb, diff_ub, (2, hmax))
    w = _unifint(diff_lb, diff_ub, (2, wmax))
    nred = _unifint(diff_lb, diff_ub, (1, h * w - 1))
    ncols = _unifint(diff_lb, diff_ub, (1, min(8, nred)))
    ncells = _unifint(diff_lb, diff_ub, (1, h * w - nred))
    ccols = random.sample(cols, ncols)
    gi = [[0] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    reds = random.sample(inds, nred)
    for (i, j) in reds:
        gi[i][j] = 2
    rem = [ij for ij in inds if ij not in set(reds)]
    rest = random.sample(rem, ncells)
    for (i, j) in rest:
        gi[i][j] = random.choice(ccols)
    go = [[0] * (w * w) for _ in range(h * h)]
    for (ri, rj) in reds:
        for i in range(h):
            for j in range(w):
                go[ri * h + i][rj * w + j] = gi[i][j]
    return {"input": [list(r) for r in gi], "output": go}


def derive_operations(I, O=None, examples=None):
    # Rule: output = h x w grid of blocks, block (i,j) = I if I[i,j]==2 else blank.
    # Decomposition here: tile I into EVERY block (first a block-row by pasting I,
    # then the whole block-row strip copied down), then erase, row by row, the
    # runs of blocks whose marker cell is not 2.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi * hi, wi * wi
    ops, sels = [], []

    # 1. expand canvas to output size (I lands in top-left block)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])  # full rectangle
    # 2. clipboard = whole input (full rectangle)
    ops.append(28); sels.append([0, 0, hi - 1, wi - 1])
    # 3. fill first block-row with copies of I
    for j in range(1, wi):
        ops.append(30); sels.append([0, j * wi, 0, 0])
    # 4. copy the full first block-row strip (full rectangle) and paste it down
    ops.append(29); sels.append([0, 0, hi - 1, wo - 1])
    for i in range(1, hi):
        ops.append(30); sels.append([i * hi, 0, 0, 0])
    # 5. per row of I, erase each contiguous run of non-2 marker cells as one
    #    rectangle of blocks
    for i in range(hi):
        j = 0
        while j < wi:
            if I[i, j] != 2:
                k = j
                while k + 1 < wi and I[i, k + 1] != 2:
                    k += 1
                cells = [(r, c)
                         for r in range(i * hi, (i + 1) * hi)
                         for c in range(j * wi, (k + 1) * wi)]
                ops.append(0); sels.append(sel_of(cells))
                j = k + 1
            else:
                j += 1

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
