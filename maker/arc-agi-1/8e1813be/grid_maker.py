"""
ARC Task: 8e1813be (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover — documented mask format fallback
    def sel_of(cells):
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = int(a + (b - a) * diff_lb)
    hi = int(a + (b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


# The one discrete structural variant: the generator's coin flip mirrors the whole
# instance diagonally, so the bars run as rows or as columns.  Both must be shown.
VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)          # background and marker-square colour
    barcols = [c for c in cols if c not in (bgc, sqc)]
    random.shuffle(barcols)                    # the palette the bars are drawn from
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "barcols": barcols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, barcols, mirrored=None):
    """RE-ARC generate_8e1813be with the colours fixed per episode.

    Kept from the original: nbars in 3..8, one full-width bar per colour, hmarg
    background rows spliced in among them, an nbars x nbars square isolated by a
    background ring, and the optional diagonal mirror.  Changed: 30 -> max_h/max_w,
    and w starts at 2*nbars+2 with locj placed so that nbars consecutive columns
    stay clear of the square — i.e. every bar is readable end to end somewhere,
    which is what the original leaves to chance at its smallest widths.
    """
    if mirrored is None:
        mirrored = random.choice([True, False])

    # dimensions of the grid BEFORE the optional mirror (bars are rows there)
    H = min(30, max_w if mirrored else max_h)      # bound for h2 = nbars + hmarg
    W = min(30, max_h if mirrored else max_w)      # bound for the bar length w
    nb_ub = min(8, len(barcols), H // 3, (W - 2) // 2)
    if nb_ub < 3:
        raise ValueError("grid bounds too small for this task")
    nbars = _unifint(diff_lb, diff_ub, (3, nb_ub))
    ccols = random.sample(list(barcols), nbars)

    w = _unifint(diff_lb, diff_ub, (2 * nbars + 2, W))
    hmarg = _unifint(diff_lb, diff_ub, (2 * nbars, H - nbars))
    h2 = nbars + hmarg

    gi = [[c] * w for c in ccols]
    bgrow = [bgc] * w
    for _ in range(hmarg):
        idx = random.randint(0, nbars - 1)         # as in the original
        gi = gi[:idx] + [list(bgrow)] + gi[idx:]

    loci = random.randint(1, h2 - nbars - 2)
    locj = random.choice([j for j in range(1, w - nbars - 1)
                          if (j - 1 >= nbars or w - j - nbars - 1 >= nbars)])
    for i in range(loci, loci + nbars):            # the square
        for j in range(locj, locj + nbars):
            gi[i][j] = sqc
    for i in range(loci - 1, loci + nbars + 1):    # its background ring
        for j in range(locj - 1, locj + nbars + 1):
            if i in (loci - 1, loci + nbars) or j in (locj - 1, locj + nbars):
                gi[i][j] = bgc

    go = [[c] * nbars for c in ccols]
    if mirrored:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    """Squeeze the bars together, then keep that block.  Everything is read off I.

    Read from I alone: a bar is a colour whose cells all lie in one row (or all in
    one column) — the square and the background span many of both.  Their count n
    is the answer's side, their orientation says whether the bars are rows or
    columns, their order along the grid is their order in the answer, and n
    consecutive lines across which every bar is unbroken (the square hides pieces
    of the ones it crosses) is the strip the answer is cut from.

    Each bar then has to travel from where it is to its own slot — bar k to line k.
    That move is done as a REFLECTION: mirroring the strip segment between slot k
    and bar k carries the bar onto slot k, and carries nothing else, because the
    bars before k are already parked outside the segment and the bars after k are
    still beyond its far end.  A reflection is also the only rigid motion that can
    carry a BLACK bar: ARCLE's Move keeps only non-zero cells of a selection, so a
    colour-0 bar cannot be dragged, while FlipH/FlipV mirror the whole region and
    leave its zeros as zeros.  Bars the square cut into are repaired first, in
    their own colour, on their own line — the only cells this ever paints.
    Finally the n x n block is cropped out.  O is never inspected.
    """
    I = np.asarray(I, dtype=int)

    # --- the bars: colours confined to a single row or a single column ---------
    n_row1 = n_col1 = 0
    found = []
    for v in np.unique(I):
        cells = np.argwhere(I == v)
        r0, r1 = int(cells[:, 0].min()), int(cells[:, 0].max())
        c0, c1 = int(cells[:, 1].min()), int(cells[:, 1].max())
        one_row, one_col = (r0 == r1), (c0 == c1)
        n_row1 += one_row
        n_col1 += one_col
        if one_row or one_col:
            found.append((r0, c0, int(v)))
    rowcase = n_row1 > n_col1            # bars run as rows, else as columns
    bars = sorted((r0 if rowcase else c0, v) for (r0, c0, v) in found)
    n = len(bars)

    # canonical view: bars as columns of G, bar k at column p, target column k
    G = I.T if rowcase else I

    # --- the strip: n consecutive lines crossing every bar (fewest gaps wins) ---
    best = None
    for b in range(G.shape[0] - n + 1):
        gaps = sum(1 for r in range(b, b + n) for p, c in bars if G[r, p] != c)
        if best is None or gaps < best[0]:
            best = (gaps, b)
    b = best[1]

    def back(r, c):                       # canonical cell -> cell of I
        return (c, r) if rowcase else (r, c)

    ops, sels = [], []

    # --- restore the pieces of a bar the square covers, inside the strip -------
    for p, c in bars:
        hidden = [back(r, p) for r in range(b, b + n) if G[r, p] != c]
        if hidden:
            ops.append(int(c))
            sels.append(sel_of(hidden))

    # --- fold each bar onto its slot: mirror the strip segment [slot k .. bar p]
    flip = 27 if rowcase else 26          # rows -> FlipV (flipud), cols -> FlipH
    for k, (p, c) in enumerate(bars):
        if p == k:                        # already standing in its slot
            continue
        # the selection IS exactly this full rectangle: the whole segment is
        # reflected, background and all, which is what moves the bar to slot k
        sels.append([k, b, p - k, n - 1] if rowcase else [b, k, n - 1, p - k])
        ops.append(flip)

    # --- keep the assembled n x n block ---------------------------------------
    ops.append(33)
    sels.append([0, b, n - 1, n - 1] if rowcase else [b, 0, n - 1, n - 1])
    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
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
                                f"for task 8e1813be"
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
                    f"Failed to build a complete episode for task 8e1813be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e1813be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
