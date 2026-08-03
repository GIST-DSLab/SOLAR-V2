"""
ARC Task: 90c28cc7 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np


def sample_colors(num_examples=None) -> dict:
    # Background (0) is hardcoded in this task; foreground colors are drawn at random
    # per cell and the rule (crop to content, then collapse duplicated rows/cols)
    # depends only on the pattern, never on which colors are used.
    return {}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, **color_kwargs) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, min(10, max_h)))
    w = unifint(diff_lb, diff_ub, (2, min(10, max_w)))
    nc = unifint(diff_lb, diff_ub, (2, 9))
    gi = canvas(-1, (h, w))
    inds = totuple(asindices(gi))
    colss = sample(cols, nc)
    for ij in inds:
        gi = fill(gi, choice(colss), {ij})
    gi = dmirror(dedupe(dmirror(dedupe(gi))))
    go = tuple(e for e in gi)
    h, w = shape(gi)
    fullh = unifint(diff_lb, diff_ub, (h, max_h))
    fullw = unifint(diff_lb, diff_ub, (w, max_w))
    inh = unifint(diff_lb, diff_ub, (h, fullh))
    inw = unifint(diff_lb, diff_ub, (w, fullw))
    while h < inh or w < inw:
        opts = []
        if h < inh:
            opts.append((h, identity))
        elif w < inw:
            opts.append((w, dmirror))
        dim, mirrf = choice(opts)
        idx = randint(0, dim - 1)
        gi = mirrf(gi)
        gi = gi[:idx + 1] + gi[idx:]
        gi = mirrf(gi)
        h, w = shape(gi)
    while h < fullh or w < fullw:
        opts = []
        if h < fullh:
            opts.append(identity)
        elif w < fullw:
            opts.append(dmirror)
        mirrf = choice(opts)
        gi = mirrf(gi)
        gi = merge(tuple(sample((((0,) * width(gi),), gi), 2)))
        gi = mirrf(gi)
        h, w = shape(gi)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule read off I: the picture sits inside a frame of empty (0) rows/cols, and inside it
    every distinct row is stretched into a run of identical rows, likewise for columns.
    So: keep one row per identical-row run, one col per identical-col run, then crop.

    Ops: pull each contiguous group of surviving rows up (CopyI+Paste), then each
    contiguous group of surviving columns left (CopyO+Paste on the row-collapsed strip),
    then CropGrid to the collapsed block.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- content region of I (0 is the frame color; the picture never contains 0) ---
    nz = np.argwhere(I != 0)
    r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
    r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
    core = I[r0:r1 + 1, c0:c1 + 1]
    ch, cw = core.shape

    # --- one representative per run of identical rows, then per run of identical cols ---
    row_reps = [0] + [i for i in range(1, ch) if not np.array_equal(core[i], core[i - 1])]
    row_collapsed = core[row_reps]
    col_reps = [0] + [j for j in range(1, cw)
                      if not np.array_equal(row_collapsed[:, j], row_collapsed[:, j - 1])]

    def runs(reps):
        # group representatives whose shift (rep - destination index) is constant:
        # such a group is contiguous in the source and moves as one block
        out, s = [], 0
        for k in range(1, len(reps) + 1):
            if k == len(reps) or reps[k] - k != reps[s] - s:
                out.append((s, k - 1))
                s = k
        return out

    # --- collapse rows: each block of kept rows slides up as a unit (source = input) ---
    for a, b in runs(row_reps):
        src = r0 + row_reps[a]
        dst = r0 + a
        if src == dst:
            continue  # this block is already where it belongs
        ops.append(28); sels.append([src, c0, b - a, cw - 1])   # CopyI block of kept rows
        ops.append(30); sels.append([dst, c0, 0, 0])            # Paste it higher up

    # --- collapse cols: same idea on the row-collapsed strip (source = working grid) ---
    for a, b in runs(col_reps):
        src = c0 + col_reps[a]
        dst = c0 + a
        if src == dst:
            continue
        ops.append(29); sels.append([r0, src, ho - 1, b - a])   # CopyO block of kept cols
        ops.append(30); sels.append([r0, dst, 0, 0])            # Paste it further left

    # --- keep only the collapsed block ---
    if (r0, c0, ho, wo) != (0, 0, hi, wi):
        ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
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
                                f"for task 90c28cc7"
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
                    f"Failed to build a complete episode for task 90c28cc7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"90c28cc7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
