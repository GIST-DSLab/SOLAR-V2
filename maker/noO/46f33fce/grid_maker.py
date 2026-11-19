"""
ARC Task: 46f33fce (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # Only the canvas/background colour matters for the rule; the object colours
    # are arbitrary and simply carried through the downscale/upscale.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]

    # output is 4h x 4w, input is 2h x 2w  -> output is the binding constraint
    h_ub = max(2, min(7, max_h // 4))
    w_ub = max(2, min(7, max_w // 4))
    h = _unifint(diff_lb, diff_ub, (2, h_ub))
    w = _unifint(diff_lb, diff_ub, (2, w_ub))
    nc = _unifint(diff_lb, diff_ub, (0, max(0, (h * w) // 2 - 1)))

    # small grid (h x w) holding the objects
    small = [[bgc for _ in range(w)] for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    nc = min(nc, len(inds))
    locs = random.sample(inds, nc)
    for (i, j) in locs:
        small[i][j] = random.choice(remcols)

    # input: 2h x 2w canvas, object cell (i, j) drawn at (2i + 1, 2j + 1)
    gi = [[bgc for _ in range(2 * w)] for _ in range(2 * h)]
    for (i, j) in locs:
        gi[2 * i + 1][2 * j + 1] = small[i][j]

    # output: small grid upscaled by 4  -> 4h x 4w
    go = [[bgc for _ in range(4 * w)] for _ in range(4 * h)]
    for i in range(h):
        for j in range(w):
            v = small[i][j]
            if v == bgc:
                continue
            for r in range(4):
                for c in range(4):
                    go[4 * i + r][4 * j + c] = v

    return {"input": gi, "output": go}


def derive_operations(I, O):
    """
    Rule (read from I only):
      I is a 2h x 2w canvas of background colour; every object pixel sits at an
      odd row and odd column, i.e. pixel (2i+1, 2j+1) is object cell (i, j) of an
      h x w lattice.  The answer is that h x w lattice upscaled by 4 -> 4h x 4w,
      so object cell (i, j) becomes the 4x4 block at rows 4i..4i+3, cols 4j..4j+3.
    Everything below is measured from I; O is never inspected.
    """
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    h, w = hi // 2, wi // 2
    ho, wo = 4 * h, 4 * w

    # the canvas colour: corner (0,0) is an even/even cell, always background
    bgc = int(I[0, 0])

    ops, sels = [], []

    # 1) grow the canvas to the upscaled size (full rectangle -> bbox selection)
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])

    # state after the resize: I's non-zero cells still sit in the top-left,
    # the newly added area is 0.
    cur = np.zeros((ho, wo), dtype=int)
    cur[:hi, :wi] = I

    # 2) lay the background base over the whole new canvas (clears the leftover
    #    single pixels and paints the padding).  Skipped only when the canvas
    #    already is entirely bgc, where the op would change nothing.
    if bool(np.any(cur != bgc)):
        ops.append(bgc)
        sels.append(sel_of([(r, c) for r in range(ho) for c in range(wo)]))

    # 3) draw each object as its own 4x4 block, lattice cell by lattice cell
    for i in range(h):
        for j in range(w):
            v = int(I[2 * i + 1, 2 * j + 1])
            if v == bgc:
                continue
            cells = [(4 * i + r, 4 * j + c) for r in range(4) for c in range(4)]
            ops.append(v)
            sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 46f33fce"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 46f33fce"
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
                                f"for task 46f33fce"
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
                    f"Failed to build a complete episode for task 46f33fce "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"46f33fce-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
