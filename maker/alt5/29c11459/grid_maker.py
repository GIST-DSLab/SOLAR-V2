"""
ARC Task: 29c11459 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    pool = [c for c in range(10) if c != 5]
    bgc = random.choice(pool)
    rem = [c for c in pool if c != bgc]
    if bgc != 0:
        # 0 must never be a marker colour: 0 is "transparent" for CopyI/Paste,
        # which is how the line-replication is carried out.
        rem = [c for c in rem if c != 0]
    ncols = random.randint(2, len(rem))
    ccols = random.sample(rem, ncols)

    VARIANTS = [{"transpose": False}, {"transpose": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))

    # pre-mirror frame: h rows, w (odd) cols.  After dmirror the shape is (w, h).
    if transpose:
        hcap = max(5, min(30, max_w))
        wcap = max(5, min(29, max_h))
    else:
        hcap = max(5, min(30, max_h))
        wcap = max(5, min(29, max_w))

    h = unifint(diff_lb, diff_ub, (5, hcap))
    w = unifint(diff_lb, diff_ub, (5, wcap))
    if w % 2 == 0:
        opts = [max(5, w - 1)]
        if w + 1 <= wcap:
            opts.append(w + 1)
        w = random.choice(opts)
    if w % 2 == 0:
        w = max(5, w - 1)

    pool = [c for c in ccols if c != bgc]
    ncols = unifint(diff_lb, diff_ub, (2, len(pool)))
    cols = random.sample(pool, ncols)

    nlocs = unifint(diff_lb, diff_ub, (1, h))
    locs = random.sample(range(h), nlocs)
    while set(locs).issubset({0, h - 1}):
        locs = random.sample(range(h), nlocs)

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    cen = w // 2
    aforb = -1
    bforb = -1
    for loc in sorted(locs):
        a = random.choice([c for c in cols if c != aforb])
        aforb = a
        b = random.choice([c for c in cols if c != bforb])
        bforb = b
        gi[loc][0] = a
        gi[loc][w - 1] = b
        for j in range(0, cen):
            go[loc][j] = a
        for j in range(cen + 1, w):
            go[loc][j] = b
        go[loc][cen] = 5

    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []

    # Which way do the marker lines run?  The marked positions always include an
    # interior one, so an interior non-bg cell on the left/right edge column means
    # the markers sit on the left+right edges (lines run horizontally); otherwise
    # they sit on the top+bottom edge rows (lines run vertically).
    horizontal = any(I[r, 0] != bgc or I[r, W - 1] != bgc for r in range(1, H - 1))

    if horizontal:
        cen = W // 2
        marked = [r for r in range(H) if I[r, 0] != bgc]

        # The left edge column is the thing that repeats: copy it and lay it down
        # column by column, marching from the edge inward to just before the seam.
        ops.append(28); sels.append([0, 0, H - 1, 0])          # CopyI: full left column (whole rectangle)
        for j in range(1, cen):
            ops.append(30); sels.append([0, j, 0, 0])          # Paste column at (0, j)

        # Same for the right edge column, marching leftward to just after the seam.
        ops.append(28); sels.append([0, W - 1, H - 1, 0])      # CopyI: full right column (whole rectangle)
        for j in range(W - 2, cen, -1):
            ops.append(30); sels.append([0, j, 0, 0])

        # Guard: a marker coloured 0 is invisible to Copy/Paste — redraw those runs.
        if bgc != 0:
            zl = [r for r in marked if I[r, 0] == 0]
            if zl:
                ops.append(0)
                sels.append(sel_of([(r, j) for r in zl for j in range(1, cen)]))
            zr = [r for r in marked if I[r, W - 1] == 0]
            if zr:
                ops.append(0)
                sels.append(sel_of([(r, j) for r in zr for j in range(cen + 1, W - 1)]))

        # The seam: where the two halves meet on every marked line.
        ops.append(5); sels.append(sel_of([(r, cen) for r in marked]))

    else:
        cen = H // 2
        marked = [c for c in range(W) if I[0, c] != bgc]

        # The top edge row repeats downward to just above the seam.
        ops.append(28); sels.append([0, 0, 0, W - 1])          # CopyI: full top row (whole rectangle)
        for i in range(1, cen):
            ops.append(30); sels.append([i, 0, 0, 0])          # Paste row at (i, 0)

        # The bottom edge row repeats upward to just below the seam.
        ops.append(28); sels.append([H - 1, 0, 0, W - 1])      # CopyI: full bottom row (whole rectangle)
        for i in range(H - 2, cen, -1):
            ops.append(30); sels.append([i, 0, 0, 0])

        if bgc != 0:
            zt = [c for c in marked if I[0, c] == 0]
            if zt:
                ops.append(0)
                sels.append(sel_of([(i, c) for c in zt for i in range(1, cen)]))
            zb = [c for c in marked if I[H - 1, c] == 0]
            if zb:
                ops.append(0)
                sels.append(sel_of([(i, c) for c in zb for i in range(cen + 1, H - 1)]))

        ops.append(5); sels.append(sel_of([(cen, c) for c in marked]))

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 29c11459"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29c11459"
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
                                f"for task 29c11459"
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
                    f"Failed to build a complete episode for task 29c11459 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29c11459-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
