"""
ARC Task: c59eb873 (RE-ARC) — LLM-generated grid_maker
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
    # Rule (2x upscale) is colour-independent; only the background is fixed per episode.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    from utils import unifint
    if bgc is None:
        bgc = random.choice(range(10))
    h = unifint(diff_lb, diff_ub, (1, max(1, min(15, max_h // 2))))
    w = unifint(diff_lb, diff_ub, (1, max(1, min(15, max_w // 2))))
    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in range(10) if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(r, c) for r in range(h) for c in range(w)]
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = random.sample(inds, num)
        for r, c in chos:
            gi[r][c] = col
        chs = set(chos)
        inds = [x for x in inds if x not in chs]
    gi = tuple(tuple(row) for row in gi)
    go = tuple(tuple(v for v in row for _ in range(2)) for row in gi for _ in range(2))
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (golf): every row is written twice, and inside each row every
    # cell is written twice.  So: (1) double the columns -- spread each input
    # column out to its new slot 2c and repeat it at 2c+1; (2) double the rows
    # -- spread each widened row to slot 2r and repeat it at 2r+1.
    # Everything is derived from I alone.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    H2, W2 = 2 * h, 2 * w
    ops, sels = [], []

    # grow the canvas to twice the input in each direction (full rectangle)
    ops.append(33); sels.append([0, 0, H2 - 1, W2 - 1])
    cur = np.zeros((H2, W2), dtype=int)
    cur[:h, :w] = I

    clip = [None]

    def copy_paste(src_sel, block, dst):
        # CopyO only when the clipboard does not already hold this exact block
        key = (block.shape, block.tobytes())
        if clip[0] != key:
            ops.append(29); sels.append(src_sel)
            clip[0] = key
        ops.append(30); sels.append(dst)

    def slide(cells, dr, dc, steps):
        op = 21 if dr else 22
        ops.append(op); sels.append(sel_of(cells))        # grab once
        for _ in range(steps - 1):
            ops.append(op); sels.append(sel_of([]))       # keep sliding same object

    # ---- phase 1: each column becomes two columns ----
    # right-to-left so a column only ever slides into already-vacated space
    for c in range(w - 1, -1, -1):
        cells = [(r, c) for r in range(h) if cur[r, c] != 0]
        if not cells:
            continue  # all-zero column: canvas is already zero there
        if c > 0:
            slide(cells, 0, 1, c)                         # column c -> column 2c
            vals = [cur[r, c] for r, _ in cells]
            for (r, _) in cells:
                cur[r, c] = 0
            for (r, _), v in zip(cells, vals):
                cur[r, 2 * c] = v
        # repeat column 2c into column 2c+1
        copy_paste([0, 2 * c, h - 1, 0], cur[:h, 2 * c].copy(), [0, 2 * c + 1, 0, 0])
        cur[:h, 2 * c + 1] = np.where(cur[:h, 2 * c] != 0, cur[:h, 2 * c], cur[:h, 2 * c + 1])

    # ---- phase 2: each (widened) row becomes two rows ----
    # bottom-to-top so a row only ever slides into already-vacated space
    for r in range(h - 1, -1, -1):
        cells = [(r, c) for c in range(W2) if cur[r, c] != 0]
        if not cells:
            continue
        if r > 0:
            slide(cells, 1, 0, r)                         # row r -> row 2r
            row = cur[r].copy()
            cur[r] = 0
            cur[2 * r] = np.where(row != 0, row, cur[2 * r])
        # repeat row 2r into row 2r+1
        copy_paste([2 * r, 0, 0, W2 - 1], cur[2 * r].copy(), [2 * r + 1, 0, 0, 0])
        cur[2 * r + 1] = np.where(cur[2 * r] != 0, cur[2 * r], cur[2 * r + 1])

    ops.append(34); sels.append([0, 0, H2 - 1, W2 - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c59eb873"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c59eb873"
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
                                f"for task c59eb873"
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
                    f"Failed to build a complete episode for task c59eb873 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c59eb873-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
