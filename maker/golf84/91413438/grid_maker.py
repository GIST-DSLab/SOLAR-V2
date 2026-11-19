"""
ARC Task: 91413438 (RE-ARC) — LLM-generated grid_maker
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
    # Background is hard-coded to 0. The rule copies the input pattern whatever its
    # colour, so fgc is fixed per episode only to keep the instances consistent.
    fgc = random.choice(list(range(1, 10)))
    return {"fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, fgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        d = random.uniform(lb, ub)
        return a + round((b - a) * d)

    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    maxnb = min(h * w - 1, min(max_h // h, max_w // w))
    minnb = int(0.5 * ((4 * h * w + 1) ** 0.5 - 1)) + 1
    if maxnb < minnb:
        raise ValueError("grid too small for this h, w")
    nbi = unifint(diff_lb, diff_ub, (0, maxnb - minnb))
    nb = min(max(minnb, maxnb - nbi), maxnb)
    if fgc is None:
        fgc = random.randint(1, 9)
    cells = [(i, j) for i in range(h) for j in range(w)]
    obj = random.sample(cells, h * w - nb)
    gi = [[0] * w for _ in range(h)]
    for i, j in obj:
        gi[i][j] = fgc
    go = [[0] * (w * nb) for _ in range(h * nb)]
    for k in range(h * w - nb):
        br, bc = k // nb, k % nb
        for i, j in obj:
            go[br * h + i][bc * w + j] = fgc
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def rect(r, c, h, w):
        # a full rectangle, given by top-left corner and size
        return sel_of([(i, j) for i in range(r, r + h) for j in range(c, c + w)])

    # Background is 0. Everything is measured from I:
    #   n  = number of coloured cells = how many copies of I to place
    #   nb = number of zero cells     = blocks per row / column of the output
    n = int((I != 0).sum())
    nb = hi * wi - n
    ho, wo = hi * nb, wi * nb

    ops, sels = [], []
    # 1. Resize the canvas to nb x nb blocks. I stays at block (0,0) and the
    #    zero padding is already the background.
    ops.append(33); sels.append(rect(0, 0, ho, wo))

    # 2. Fill block-row 0 up to min(n, nb) copies by doubling: copy the blocks
    #    already built and paste them next to themselves.
    first = min(n, nb)
    k = 1
    clip = None
    while k < first:
        m = min(k, first - k)
        if m == 1:
            if clip != "I":
                ops.append(28); sels.append(rect(0, 0, hi, wi))      # CopyI: one block
                clip = "I"
        else:
            ops.append(29); sels.append(rect(0, 0, hi, m * wi))      # CopyO: m built blocks
            clip = ("row", m)
        ops.append(30); sels.append(rect(0, k * wi, 1, 1))           # paste where the run ends
        k += m

    # 3. Copy full block-rows down, doubling again.
    full = n // nb
    rem = n % nb
    k = 1
    while k < full:
        m = min(k, full - k)
        ops.append(29); sels.append(rect(0, 0, m * hi, wo))          # m full block-rows
        ops.append(30); sels.append(rect(k * hi, 0, 1, 1))
        k += m

    # 4. Final partial block-row: the first `rem` blocks of row 0, pasted in one go.
    if full >= 1 and rem:
        if rem == 1:
            ops.append(28); sels.append(rect(0, 0, hi, wi))
        else:
            ops.append(29); sels.append(rect(0, 0, hi, rem * wi))
        ops.append(30); sels.append(rect(full * hi, 0, 1, 1))

    ops.append(34); sels.append(rect(0, 0, ho, wo))
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
                        f"num_examples+1 ({num_examples + 1}) for task 91413438"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 91413438"
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
                                f"for task 91413438"
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
                    f"Failed to build a complete episode for task 91413438 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"91413438-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
