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
import random
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The generator picks a random palette inside 1..9 for every instance, and the frame is
    # always the hardcoded 0. The rule depends only on structure (repeated lines, empty
    # lines), so no colour role needs to be fixed for the episode.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **color_kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi_)))))

    def dmirror(g):
        return tuple(zip(*g))

    def dedupe(g):
        out = []
        for r in g:
            if not out or out[-1] != r:
                out.append(r)
        return tuple(out)

    cap_h = max(2, min(30, max_h))
    cap_w = max(2, min(30, max_w))
    cols = list(range(1, 10))
    h = unifint(diff_lb, diff_ub, (2, min(10, cap_h)))
    w = unifint(diff_lb, diff_ub, (2, min(10, cap_w)))
    nc = unifint(diff_lb, diff_ub, (2, 9))
    colss = random.sample(cols, nc)
    gi = tuple(tuple(random.choice(colss) for _ in range(w)) for _ in range(h))
    gi = dmirror(dedupe(dmirror(dedupe(gi))))
    go = tuple(tuple(r) for r in gi)
    h, w = len(gi), len(gi[0])
    fullh = unifint(diff_lb, diff_ub, (h, cap_h))
    fullw = unifint(diff_lb, diff_ub, (w, cap_w))
    inh = unifint(diff_lb, diff_ub, (h, fullh))
    inw = unifint(diff_lb, diff_ub, (w, fullw))
    while h < inh or w < inw:
        if h < inh:
            dim, mirr = h, (lambda g: g)
        else:
            dim, mirr = w, dmirror
        idx = random.randint(0, dim - 1)
        gi = mirr(gi)
        gi = gi[:idx + 1] + gi[idx:]
        gi = mirr(gi)
        h, w = len(gi), len(gi[0])
    while h < fullh or w < fullw:
        if h < fullh:
            mirr = (lambda g: g)
        else:
            mirr = dmirror
        gi = mirr(gi)
        zero = ((0,) * len(gi[0]),)
        parts = [zero, gi]
        random.shuffle(parts)
        gi = parts[0] + parts[1]
        gi = mirr(gi)
        h, w = len(gi), len(gi[0])
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O=None, examples=None):
    """
    Reading (golf entry): handle one axis at a time, columns FIRST, then rows. In each
    pass, an empty (all-0) line is thrown away and a line is kept only if it differs
    from the line just before it. Frame removal and de-duplication happen in the same
    pass. Every line that is kept is packed against the top-left corner (column 0,
    then row 0). It is not cropped to the content box first.

    Column pass: each contiguous block of kept columns is a whole-height column strip
                 taken from the input (CopyI) and pasted further left (Paste).
    Row pass:    each contiguous block of kept rows is a strip taken from the packed
                 columns 0..wo-1 of the working grid (CopyO) and pasted further up.
    Finally CropGrid to the packed top-left block.
    (The golf entry drops duplicate lines globally. This generator only stretches
     lines next to each other, so lines that repeat further apart are real content
     and must stay. "Differs from the previous line" is the right test here.)
    Everything is measured from I alone.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    G = I.copy()  # simulated working grid (visible state)

    # kept columns: non-empty and different from the column just before it
    col_keep = [j for j in range(wi)
                if I[:, j].any() and (j == 0 or not np.array_equal(I[:, j], I[:, j - 1]))]
    # kept rows: non-empty and different from the row just before it
    row_keep = [i for i in range(hi)
                if I[i].any() and (i == 0 or not np.array_equal(I[i], I[i - 1]))]
    wo, ho = len(col_keep), len(row_keep)

    def runs(reps):
        # kept lines whose offset (source - destination) stays constant are contiguous
        # in the source, so they move as one block
        out, s = [], 0
        for k in range(1, len(reps) + 1):
            if k == len(reps) or reps[k] - k != reps[s] - s:
                out.append((s, k - 1))
                s = k
        return out

    # --- column pass: pack kept columns against the left edge ---
    for a, b in runs(col_keep):
        src = col_keep[a]
        if src == a:
            continue  # this block is already where it belongs
        blk = I[:, src:src + b - a + 1]
        new = G.copy()
        m = blk != 0
        new[:, a:b + 1][m] = blk[m]
        if np.array_equal(new, G):
            continue  # the paste would not change anything visible
        G = new
        # full-height column strip (a whole-column rectangle), copied from input
        ops.append(28); sels.append([0, src, hi - 1, b - a])
        ops.append(30); sels.append([0, a, 0, 0])

    # --- row pass: pack kept rows of the column-packed strip against the top edge ---
    for a, b in runs(row_keep):
        src = row_keep[a]
        if src == a:
            continue
        blk = G[src:src + b - a + 1, :wo].copy()
        new = G.copy()
        m = blk != 0
        new[a:b + 1, :wo][m] = blk[m]
        if np.array_equal(new, G):
            continue
        G = new
        # row strip across the packed columns 0..wo-1 (rectangle), copied from working grid
        ops.append(29); sels.append([src, 0, b - a, wo - 1])
        ops.append(30); sels.append([a, 0, 0, 0])

    # --- keep only the packed top-left block ---
    if (ho, wo) != (hi, wi):
        ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

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
