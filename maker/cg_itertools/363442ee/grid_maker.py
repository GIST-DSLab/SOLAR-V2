"""
ARC Task: 363442ee (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barcol = random.choice(rem)
    rem = [c for c in rem if c != barcol]
    dotcol = random.choice(rem)
    return {"bgc": bgc, "barcol": barcol, "dotcol": dotcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, dotcol, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round(lb * (b - a))
        hi = a + round(ub * (b - a))
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    m = min(max_h, max_w)  # mirrors/rotations may swap dims
    h = unifint(diff_lb, diff_ub, (1, 3)) * 2 + 1
    w = unifint(diff_lb, diff_ub, (1, 3)) * 2 + 1
    nremh = unifint(diff_lb, diff_ub, (2, max(2, m // h)))
    nremw = unifint(diff_lb, diff_ub, (2, max(2, (m - w - 1) // w)))
    rsh, rsw = nremh * h, nremw * w

    remcols = [c for c in range(10) if c not in (bgc, barcol)]
    rsi = np.full((rsh, rsw), bgc, dtype=int)
    rso = np.full((rsh, rsw), bgc, dtype=int)
    nfull = unifint(diff_lb, diff_ub, (1, 8))
    fullremcols = random.sample(remcols, min(nfull, len(remcols)))
    ulc = np.array([[random.choice(fullremcols) for _ in range(w)] for _ in range(h)], dtype=int)

    dotcands = [(i, j) for i in range(0, rsh, h) for j in range(0, rsw, w)]
    dev = unifint(diff_lb, diff_ub, (1, max(1, len(dotcands) // 2)))
    ndots = random.choice((dev, len(dotcands) - dev))
    ndots = min(max(1, ndots), len(dotcands))
    dots = random.sample(dotcands, ndots)
    for (i, j) in dots:
        rsi[i + h // 2, j + w // 2] = dotcol
        rso[i:i + h, j:j + w] = ulc

    left = np.full((rsh, w), bgc, dtype=int)
    left[:h, :w] = ulc
    bar = np.full((rsh, 1), barcol, dtype=int)
    gi = np.hstack([left, bar, rsi])
    go = np.hstack([left, bar, rso])

    mfs = [
        lambda g: g,
        lambda g: g.T,
        lambda g: np.rot90(g, 2).T,
        lambda g: np.fliplr(g),
        lambda g: np.flipud(g),
        lambda g: np.rot90(g, 3),
        lambda g: np.rot90(g, 2),
        lambda g: np.rot90(g, 1),
    ]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (lattice view): the bar splits the grid into a narrow KEY side holding one
    solid block and a wide FIELD side. The field is exactly tiled by a lattice of
    block-sized cells, laid from the field's corner. Each lattice cell whose centre
    holds a marker dot becomes a copy of the key block. Every other cell stays as it is.
    The unit is the lattice cell (not the dot as a component). We visit the cells in
    lattice order and fill only the cells that are marked.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # locate the bar: a full uniform line of a non-background colour
    bar_row = bar_col = None
    for r in range(hi):
        if I[r, 0] != bgc and np.all(I[r, :] == I[r, 0]):
            bar_row = r
            break
    if bar_row is None:
        for c in range(wi):
            if I[0, c] != bgc and np.all(I[:, c] == I[0, c]):
                bar_col = c
                break

    # the key side is the narrower side of the bar; the field is the other side
    if bar_col is not None:
        lw, rw = bar_col, wi - bar_col - 1
        if lw <= rw:
            key = (0, 0, hi, lw)
            field = (0, bar_col + 1, hi, rw)
        else:
            key = (0, bar_col + 1, hi, rw)
            field = (0, 0, hi, lw)
    elif bar_row is not None:
        th, bh_ = bar_row, hi - bar_row - 1
        if th <= bh_:
            key = (0, 0, th, wi)
            field = (bar_row + 1, 0, bh_, wi)
        else:
            key = (bar_row + 1, 0, bh_, wi)
            field = (0, 0, th, wi)
    else:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    kr, kc, kh, kw = key
    sub = I[kr:kr + kh, kc:kc + kw]
    nz = np.argwhere(sub != bgc)
    if len(nz) == 0:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    br = kr + int(nz[:, 0].min()); bc = kc + int(nz[:, 1].min())
    bh = kr + int(nz[:, 0].max()) - br + 1
    bw = kc + int(nz[:, 1].max()) - bc + 1
    block = I[br:br + bh, bc:bc + bw]
    has_zero = bool(np.any(block == 0))
    has_nonzero = bool(np.any(block != 0))

    fr, fc, fh, fw = field
    marked = []
    for i in range(fh // bh):
        for j in range(fw // bw):
            top, left = fr + i * bh, fc + j * bw
            if I[top + bh // 2, left + bw // 2] != bgc:
                marked.append((top, left))
    if not marked:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # The key block is a full solid rectangle, so the bbox is exactly its cells.
    if has_nonzero:
        ops.append(28); sels.append([br, bc, bh - 1, bw - 1])

    for (top, left) in marked:
        # The lattice cell is a full rectangle. If the block has 0s, first lay a
        # Color0 base over it (Paste skips 0s), then paste the block on top.
        if has_zero:
            ops.append(0)
            sels.append(sel_of([(r, c) for r in range(top, top + bh)
                                for c in range(left, left + bw)]))
        if has_nonzero:
            ops.append(30); sels.append(sel_of([(top, left)]))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 363442ee"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 363442ee"
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
                                f"for task 363442ee"
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
                    f"Failed to build a complete episode for task 363442ee "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"363442ee-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
