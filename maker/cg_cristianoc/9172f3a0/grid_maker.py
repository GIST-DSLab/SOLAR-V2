"""
ARC Task: 9172f3a0 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The rule (x3 upscale) is colour-independent; only the canvas colour is fixed per episode.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    from random import sample

    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    if bgc is None:
        bgc = random.choice(range(10))
    h = unifint(diff_lb, diff_ub, (1, max(1, min(10, max_h // 3))))
    w = unifint(diff_lb, diff_ub, (1, max(1, min(10, max_w // 3))))
    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in range(10) if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        num = min(num, len(inds))
        chos = sample(inds, num)
        for (i, j) in chos:
            gi[i][j] = col
        cs = set(chos)
        inds = [x for x in inds if x not in cs]
    go = [[gi[i // 3][j // 3] for j in range(3 * w)] for i in range(3 * h)]
    return {"input": [list(r) for r in gi], "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (from the code-golf factorisation): upscale x3 = two independent
    # stretches. First every input COLUMN becomes three columns, then every
    # (already widened) ROW becomes three rows. Nothing below reads O.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 3 * wi
    ops, sels = [], []
    cur = np.zeros((30, 30), dtype=int)
    cur[:hi, :wi] = I
    state = {"clip": None}

    def emit(op, sel):
        ops.append(op)
        sels.append(sel)

    # 1. Resize canvas to 3h x 3w (whole-canvas rectangle).
    emit(33, [0, 0, ho - 1, wo - 1])

    def paint(color, r0, c0, h, w):
        # Full rectangular strip; skipped only if it already holds `color` now.
        blk = cur[r0:r0 + h, c0:c0 + w]
        if np.all(blk == color):
            return
        emit(color, [r0, c0, h - 1, w - 1])
        cur[r0:r0 + h, c0:c0 + w] = color

    def paste(r0, c0):
        clip = state["clip"]
        ch, cw = clip.shape
        tgt = cur[r0:r0 + ch, c0:c0 + cw]
        new = np.where(clip != 0, clip, tgt)
        if np.array_equal(new, tgt):
            return
        emit(30, [r0, c0, 0, 0])
        cur[r0:r0 + ch, c0:c0 + cw] = new

    # 2. Horizontal stretch: input column c -> columns 3c..3c+2 (rows 0..hi-1).
    #    CopyI reads the untouched input, so overwriting the canvas is safe.
    for c in range(wi - 1, -1, -1):
        col = I[:, c:c + 1]
        if np.any(col == 0):
            paint(0, 0, 3 * c, hi, 3)          # 0 cells don't paste: lay a 0 base
        if np.any(col != 0):
            emit(28, [0, c, hi - 1, 0])        # CopyI the whole input column
            state["clip"] = col.copy()
            for k in range(3):
                paste(0, 3 * c + k)

    # 3. Vertical stretch: widened row r -> rows 3r..3r+2. Bottom-up so row r
    #    is still intact when copied from the working grid (CopyO).
    for r in range(hi - 1, -1, -1):
        row = cur[r:r + 1, :wo].copy()
        if np.any(row != 0):
            emit(29, [r, 0, 0, wo - 1])        # CopyO the whole widened row
            state["clip"] = row
        if np.any(row == 0):
            paint(0, 3 * r, 0, 3, wo)          # clear destination band (0 base)
        if np.any(row != 0):
            for k in range(3):
                paste(3 * r + k, 0)

    emit(34, [0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9172f3a0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9172f3a0"
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
                                f"for task 9172f3a0"
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
                    f"Failed to build a complete episode for task 9172f3a0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9172f3a0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
