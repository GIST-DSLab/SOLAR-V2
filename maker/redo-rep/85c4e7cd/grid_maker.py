"""
ARC Task: 85c4e7cd (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# Task 85c4e7cd — concentric boxes (2h x 2w canvas, min(h,w) nested levels).
# Rule: the sequence of level colours is REVERSED (outermost <-> innermost).
# The finished figure is mirror-symmetric about both centre lines, so it is
# built as: draw the concentric quarter once, then replicate that quarter
# (Copy + Paste at each offset) and mirror each replica into place.
# ----------------------------------------------------------------------------


def sample_colors(num_examples=None) -> dict:
    # `cols` is the only randomly sampled colour source in the generator
    # (colord is drawn from it).  Fixing the pool keeps one palette per episode.
    ncols = random.randint(2, 6)
    cols = random.sample(range(10), ncols)
    return {"cols": cols}


def generate(diff_lb, diff_ub, max_h, max_w, cols=None) -> dict:
    if cols is None:
        cols = random.sample(range(10), random.randint(2, 6))
    cols = list(cols)

    hmax = max(1, max_h // 2)
    wmax = max(1, max_w // 2)
    hlo = 2 if hmax >= 2 else 1
    wlo = 2 if wmax >= 2 else 1

    h = unifint(diff_lb, diff_ub, (hlo, hmax))
    w = unifint(diff_lb, diff_ub, (wlo, wmax))
    n = min(h, w)

    # keep the palette non-palindromic so the reversal is actually visible
    colord = [random.choice(cols) for _ in range(n)]
    tries = 0
    while n >= 2 and colord == colord[::-1] and tries < 200:
        colord = [random.choice(cols) for _ in range(n)]
        tries += 1

    shp = (h * 2, w * 2)
    gi = canvas(0, shp)
    go = canvas(0, shp)
    for idx, (ci, co) in enumerate(zip(colord, colord[::-1])):
        ulc = (idx, idx)
        lrc = (h * 2 - 1 - idx, w * 2 - 1 - idx)
        bx = box(frozenset({ulc, lrc}))
        gi = fill(gi, ci, bx)
        go = fill(go, co, bx)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    h, w = H // 2, W // 2
    n = min(h, w)

    G = I.copy()
    ops, sels = [], []

    def rect(r0, c0, r1, c1):
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def emit(op, cells):
        ops.append(op)
        sels.append(sel_of(cells))

    # ---- Phase A -----------------------------------------------------------
    # Draw the concentric levels inside the TOP-LEFT QUARTER only, biggest
    # first, each smaller level layered on top.  Level k of the output carries
    # the colour of input level n-1-k, read straight off the diagonal.
    for k in range(n):
        col = int(O[k, k])
        if np.all(G[k:h, k:w] == col):
            continue                                   # already this colour
        emit(col, rect(k, k, h - 1, w - 1))            # exactly this rectangle
        G[k:h, k:w] = col

    # ---- Phase B -----------------------------------------------------------
    # Replicate the finished quarter downwards, then mirror it up-down.
    if h > 0 and w > 0 and not np.array_equal(G[h:H, 0:w], O[h:H, 0:w]):
        src = G[0:h, 0:w].copy()
        # Paste never writes colour-0 cells, so clear the destination first
        # whenever the quarter contains 0s.
        if (src == 0).any() and (G[h:H, 0:w] != 0).any():
            emit(0, rect(h, 0, H - 1, w - 1))
            G[h:H, 0:w] = 0
        pasted = G[h:H, 0:w].copy()
        m = src != 0
        pasted[m] = src[m]
        if not np.array_equal(pasted, G[h:H, 0:w]):
            emit(29, rect(0, 0, h - 1, w - 1))         # CopyO the quarter
            emit(30, [(h, 0)])                         # Paste at (h, 0)
            G[h:H, 0:w] = pasted
        flipped = np.flipud(G[h:H, 0:w])
        if not np.array_equal(flipped, G[h:H, 0:w]):
            emit(27, rect(h, 0, H - 1, W // 2 - 1))    # FlipV the replica
            G[h:H, 0:w] = flipped

    # ---- Phase C -----------------------------------------------------------
    # Replicate the finished left half to the right, then mirror it left-right.
    if not np.array_equal(G[0:H, w:W], O[0:H, w:W]):
        src = G[0:H, 0:w].copy()
        if (src == 0).any() and (G[0:H, w:W] != 0).any():
            emit(0, rect(0, w, H - 1, W - 1))
            G[0:H, w:W] = 0
        pasted = G[0:H, w:W].copy()
        m = src != 0
        pasted[m] = src[m]
        if not np.array_equal(pasted, G[0:H, w:W]):
            emit(29, rect(0, 0, H - 1, w - 1))         # CopyO the left half
            emit(30, [(0, w)])                         # Paste at (0, w)
            G[0:H, w:W] = pasted
        flipped = np.fliplr(G[0:H, w:W])
        if not np.array_equal(flipped, G[0:H, w:W]):
            emit(26, rect(0, w, H - 1, W - 1))         # FlipH the replica
            G[0:H, w:W] = flipped

    # ---- safety net: repair anything still off (normally nothing) ----------
    diff = np.argwhere(G != O)
    if len(diff):
        wanted = sorted({int(O[r, c]) for r, c in diff})
        for col in wanted:
            cells = [(int(r), int(c)) for r, c in diff if int(O[r, c]) == col]
            emit(col, cells)
            for r, c in cells:
                G[r, c] = col

    ops.append(34)
    sels.append(sel_of(rect(0, 0, H - 1, W - 1)))
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
                        f"num_examples+1 ({num_examples + 1}) for task 85c4e7cd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 85c4e7cd"
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
                                f"for task 85c4e7cd"
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
                    f"Failed to build a complete episode for task 85c4e7cd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"85c4e7cd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
