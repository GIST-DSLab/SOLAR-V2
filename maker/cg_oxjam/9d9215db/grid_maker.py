"""
ARC Task: 9d9215db (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    return {"bgc": random.choice(range(10))}


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    remcols = [c for c in range(10) if c != bgc]
    cap = max(5, min(14, (min(max_h, max_w) - 1) // 2))
    h = _unifint(diff_lb, diff_ub, (5, cap)) * 2 + 1
    w = _unifint(diff_lb, diff_ub, (5, cap)) * 2 + 1
    ub = min(h, w) // 4
    nrings = _unifint(diff_lb, diff_ub, (1, ub))
    onlinesbase = [(2 * k + 1, 2 * k + 1) for k in range(ub)]
    gi = np.full((h, w), bgc, dtype=int)
    for (i, j) in random.sample(onlinesbase, nrings):
        gi[i, j] = random.choice(remcols)
    linsbase = [(i, j + 2) for (i, j) in onlinesbase[:-1]]
    nlines = _unifint(diff_lb, diff_ub, (1, len(linsbase)))
    linesps = random.sample(linsbase, nlines)
    colors = [random.choice(remcols) for _ in range(nlines)]
    for col, (i, j) in zip(colors, linesps):
        gi[i, j] = col
        gi[j, i] = col
    gb = gi.copy()
    for col, (i, j) in zip(colors, linesps):
        for c in range(j, w - j):
            if c % 2 == 1:
                gb[i, c] = col
        for r in range(j, h - j):
            if r % 2 == 1:
                gb[r, i] = col
    go = gb.copy()
    for m in (np.fliplr(gb), np.flipud(gb), np.flipud(np.fliplr(gb))):
        go = np.where(m != bgc, m, go)
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    """
    Reading (golf-style): the board is closed under both mirrors, and every
    mark pair spans a straight dotted line from a mark to its own mirror image.
      1. symmetrise the sparse marks first: mirror the seed quadrant across
         the centre column, then the seed half across the centre row
         (CopyO -> Paste -> repair 0-coloured marks -> Flip);
      2. then, going round the four sides of the board (top, right, bottom,
         left - the four rotations), draw each pair's line as ONE segment
         between the mark and its mirrored twin.
    """
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ch, cw = (H - 1) // 2, (W - 1) // 2
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    ops, sels = [], []
    g = I.copy()

    fg = [(r, c) for r in range(H) for c in range(W) if I[r, c] != bgc]
    top = (max(r for r, _ in fg) < ch) if fg else True
    left = (max(c for _, c in fg) < cw) if fg else True

    def mirror_block(rs, cs, nr, nc, rd, cd, flip_op):
        # CopyO full rectangle (background included), Paste at (rd, cd)
        src = g[rs:rs + nr, cs:cs + nc].copy()
        ops.append(29); sels.append([rs, cs, nr - 1, nc - 1])
        before = g.copy()
        dst = g[rd:rd + nr, cd:cd + nc]
        dst[src != 0] = src[src != 0]
        if not np.array_equal(before, g):
            ops.append(30); sels.append([rd, cd, 0, 0])
        else:
            ops.pop(); sels.pop()   # nothing nonzero but background travels
        # 0-coloured cells of the copied block did not travel: paint them
        rep = [(rd + i, cd + j) for i in range(nr) for j in range(nc)
               if src[i, j] == 0 and g[rd + i, cd + j] != 0]
        if rep:
            ops.append(0); sels.append(sel_of(rep))
            for (r, c) in rep:
                g[r, c] = 0
        blk = g[rd:rd + nr, cd:cd + nc]
        fl = np.fliplr(blk) if flip_op == 26 else np.flipud(blk)
        if not np.array_equal(fl, blk):
            # full rectangle: the pasted block mirrored in place
            ops.append(flip_op); sels.append([rd, cd, nr - 1, nc - 1])
            g[rd:rd + nr, cd:cd + nc] = fl

    # ---- 1a. seed quadrant -> across the centre column ---------------------
    r0 = 0 if top else ch + 1
    mirror_block(r0, 0 if left else cw + 1, ch, cw,
                 r0, cw + 1 if left else 0, 26)
    # ---- 1b. seed half -> across the centre row ----------------------------
    mirror_block(0 if top else ch + 1, 0, ch, W,
                 ch + 1 if top else 0, 0, 27)

    # ---- 2. lines between each mark and its mirror twin, side by side ------
    def tor(rr):
        return rr if top else H - 1 - rr

    def toc(cc):
        return cc if left else W - 1 - cc

    ub = min(H, W) // 4
    pairs = []
    for k in range(max(0, ub - 1)):
        a, b = 2 * k + 1, 2 * k + 3
        col = int(I[tor(a), toc(b)])
        if col != bgc:
            pairs.append((a, b, col))

    sides = []
    sides.append(lambda a, b: [(a, c) for c in range(b + 2, W - 1 - b, 2)])          # top
    sides.append(lambda a, b: [(r, W - 1 - a) for r in range(b + 2, H - 1 - b, 2)])  # right
    sides.append(lambda a, b: [(H - 1 - a, c) for c in range(b + 2, W - 1 - b, 2)])  # bottom
    sides.append(lambda a, b: [(r, a) for r in range(b + 2, H - 1 - b, 2)])          # left
    for side in sides:
        for (a, b, col) in pairs:
            cells = [(r, c) for (r, c) in side(a, b) if g[r, c] != col]
            if cells:
                ops.append(col); sels.append(sel_of(cells))
                for (r, c) in cells:
                    g[r, c] = col

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
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
                                f"for task 9d9215db"
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
                    f"Failed to build a complete episode for task 9d9215db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9d9215db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
