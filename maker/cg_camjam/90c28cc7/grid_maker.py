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
    # Background/frame colour is hard-coded to 0 and the picture colours are random
    # per instance; the rule ignores which colours are used, so nothing needs fixing.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    def dmirror(g):
        return tuple(zip(*g))

    def dedupe(g):
        out = []
        for row in g:
            if not out or out[-1] != row:
                out.append(row)
        return tuple(out)

    cols = list(range(1, 10))
    h = unifint(diff_lb, diff_ub, (2, min(10, max_h)))
    w = unifint(diff_lb, diff_ub, (2, min(10, max_w)))
    nc = unifint(diff_lb, diff_ub, (2, 9))
    colss = random.sample(cols, nc)
    gi = tuple(tuple(random.choice(colss) for _ in range(w)) for _ in range(h))
    gi = dmirror(dedupe(dmirror(dedupe(gi))))
    go = tuple(gi)
    h, w = len(gi), len(gi[0])
    fullh = unifint(diff_lb, diff_ub, (h, max_h))
    fullw = unifint(diff_lb, diff_ub, (w, max_w))
    fullh = max(h, min(fullh, max_h))
    fullw = max(w, min(fullw, max_w))
    inh = max(h, min(unifint(diff_lb, diff_ub, (h, fullh)), fullh))
    inw = max(w, min(unifint(diff_lb, diff_ub, (w, fullw)), fullw))
    ident = lambda g: g
    while h < inh or w < inw:
        if h < inh:
            dim, mirrf = h, ident
        else:
            dim, mirrf = w, dmirror
        idx = random.randint(0, dim - 1)
        gi = mirrf(gi)
        gi = gi[:idx + 1] + gi[idx:]
        gi = mirrf(gi)
        h, w = len(gi), len(gi[0])
    while h < fullh or w < fullw:
        mirrf = ident if h < fullh else dmirror
        gi = mirrf(gi)
        zrow = ((0,) * len(gi[0]),)
        parts = [zrow, gi]
        random.shuffle(parts)
        gi = parts[0] + parts[1]
        gi = mirrf(gi)
        h, w = len(gi), len(gi[0])
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O=None, examples=None):
    """
    Reading (shared with the golf entry): along one axis, a line is kept only if it
    tells you something new -- it is not blank (all 0) and it is not a repeat of the
    last line kept. A blank frame line and a stretched duplicate are the same kind of
    thing: a line that adds nothing. So there is no separate "find the frame, then
    collapse inside it" step. The surviving rows are packed against the TOP EDGE OF
    THE GRID (row 0), then the surviving columns are packed against the LEFT EDGE
    (col 0). The answer then sits in the top-left corner and is cropped there.

    Everything is measured from I alone; O is only used for its shape in Submit.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    grid = I.copy()

    def informative(lines):
        kept, last = [], None
        for i, ln in enumerate(lines):
            if ln.any() and (last is None or not np.array_equal(ln, last)):
                kept.append(i)
                last = ln
        return kept

    def blocks(kept):
        # group kept lines whose shift (src - dest) is constant: contiguous in source
        out, s = [], 0
        for k in range(1, len(kept) + 1):
            if k == len(kept) or kept[k] - k != kept[s] - s:
                out.append((s, k - 1))
                s = k
        return out

    # --- rows: pack informative rows of the whole grid up to row 0 (source = input) ---
    row_keep = informative([I[r] for r in range(hi)])
    ho = len(row_keep)
    for a, b in blocks(row_keep):
        src, dst, n = row_keep[a], a, b - a + 1
        if src == dst:
            continue
        block = I[src:src + n, :]
        new = grid.copy()
        region = new[dst:dst + n, :]
        region[block != 0] = block[block != 0]
        if np.array_equal(new, grid):
            continue
        ops.append(28); sels.append([src, 0, n - 1, wi - 1])   # CopyI full-width rows
        ops.append(30); sels.append([dst, 0, 0, 0])            # Paste at packed position
        grid = new

    # --- cols: pack informative columns of the packed strip left to col 0 (source = working grid) ---
    strip = grid[:ho, :]
    col_keep = informative([strip[:, c] for c in range(wi)])
    wo = len(col_keep)
    for a, b in blocks(col_keep):
        src, dst, n = col_keep[a], a, b - a + 1
        if src == dst:
            continue
        block = grid[:ho, src:src + n].copy()
        new = grid.copy()
        region = new[:ho, dst:dst + n]
        region[block != 0] = block[block != 0]
        if np.array_equal(new, grid):
            continue
        ops.append(29); sels.append([0, src, ho - 1, n - 1])   # CopyO strip columns
        ops.append(30); sels.append([0, dst, 0, 0])            # Paste at packed position
        grid = new

    # --- keep only the packed top-left block (full rectangle) ---
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
