"""
ARC Task: 150deff5 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of

# ---------------------------------------------------------------- colors

def sample_colors(num_examples=None) -> dict:
    # the generator samples bgc / fgc from 0..9 minus the two answer colors (2, 8)
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


# ---------------------------------------------------------------- shared rule
# The whole task is one rule: the foreground of I is an exact packing of
# 2x2 squares and 3-cell straight lines. Squares become 8, lines become 2.

_BO = ((0, 0), (0, 1), (1, 0), (1, 1))      # 2x2 square      -> 8
_RO1 = ((0, 0), (0, 1), (0, 2))             # horizontal line -> 2
_RO2 = ((0, 0), (1, 0), (2, 0))             # vertical line   -> 2


def _components(cells):
    """4-connected components of a cell set."""
    pool = set(cells)
    comps = []
    while pool:
        seed = min(pool)
        pool.discard(seed)
        stack, comp = [seed], {seed}
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (r + dr, c + dc)
                if p in pool:
                    pool.discard(p)
                    comp.add(p)
                    stack.append(p)
        comps.append(comp)
    return comps


def _tile(comp, limit=1):
    """Exact-cover `comp` with 2x2 squares and 3-cell lines.

    The uncovered cell first in raster order can only be the top-left corner of
    the piece covering it, so each step branches over just 3 candidates.
    Returns (first solution or None, number of solutions found, capped at limit).
    """
    sols = []

    def rec(rem, acc):
        if len(sols) >= limit:
            return
        if not rem:
            sols.append(list(acc))
            return
        r, c = min(rem)
        for col, piece in ((8, [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]),
                           (2, [(r, c), (r, c + 1), (r, c + 2)]),
                           (2, [(r, c), (r + 1, c), (r + 2, c)])):
            if all(p in rem for p in piece):
                acc.append((col, piece))
                rec(rem - set(piece), acc)
                acc.pop()

    rec(set(comp), [])
    return (sols[0] if sols else None), len(sols)


def _decompose(I, unique=False):
    """Split I's foreground into squares (8) and lines (2). Uses only I."""
    I = np.asarray(I, dtype=int)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = [(r, c) for r in range(I.shape[0]) for c in range(I.shape[1])
          if I[r, c] != bgc]
    pieces = []
    for comp in sorted(_components(fg), key=min):
        sol, n = _tile(comp, 2 if unique else 1)
        if sol is None or (unique and n != 1):
            return None
        pieces.extend(sol)
    return pieces


# ---------------------------------------------------------------- generator

def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    max_h = max(8, min(30, max_h))
    max_w = max(8, min(30, max_w))
    last = None
    for _ in range(60):
        h = _unifint(diff_lb, diff_ub, (8, max_h))
        w = _unifint(diff_lb, diff_ub, (8, max_w))
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        noccs = _unifint(diff_lb, diff_ub, (2, (h * w) // 10))
        inds = {(i, j) for i in range(h) for j in range(w)}
        boforb, reforb = set(), set()
        for _k in range(noccs):
            obj, col = random.choice([(_BO, 8),
                                      (random.choice([_RO1, _RO2]), 2)])
            oh = max(a for a, _ in obj) + 1
            ow = max(b for _, b in obj) + 1
            forb = boforb if col == 8 else reforb
            cands = [ij for ij in sorted(inds)
                     if ij[0] <= h - oh and ij[1] <= w - ow and ij not in forb
                     and all((ij[0] + a, ij[1] + b) in inds for a, b in obj)]
            if not cands:
                break
            loc = random.choice(cands)
            if col == 8:
                for d in ((-2, 0), (2, 0), (0, 2), (0, -2)):
                    boforb.add((loc[0] + d[0], loc[1] + d[1]))
            else:
                ds = ((0, 3), (0, -3)) if obj == _RO1 else ((1, 0), (-1, 0))
                for d in ds:
                    reforb.add((loc[0] + d[0], loc[1] + d[1]))
            plcd = [(loc[0] + a, loc[1] + b) for a, b in obj]
            for (r, c) in plcd:
                gi[r][c] = fgc
                go[r][c] = col
            inds -= set(plcd)
        last = {'input': gi, 'output': go}
        # keep only well-posed instances: both answer colors occur, and the
        # packing of the foreground into squares/lines is unique, so it can be
        # read off the input alone
        present = {v for row in go for v in row}
        if 8 not in present or 2 not in present:
            continue
        pieces = _decompose(gi, unique=True)
        if pieces is None:
            continue
        chk = np.array(gi)
        for col, cells in pieces:
            for r, c in cells:
                chk[r, c] = col
        if np.array_equal(chk, np.array(go)):
            return {'input': gi, 'output': go}
    return last


# ---------------------------------------------------------------- operations

def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # measured from I alone: which foreground piece is a 2x2 square and which
    # is a 3-cell line, via the unique exact packing of the foreground
    pieces = _decompose(I)
    if pieces is None:
        pieces = []

    ops, sels = [], []
    for col, cells in pieces:
        cells = [(r, c) for r, c in cells if 0 <= r < hi and 0 <= c < wi]
        if not cells:
            continue
        # one op per whole piece: Color8 for a square, Color2 for a line
        ops.append(col)
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 150deff5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 150deff5"
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
                                f"for task 150deff5"
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
                    f"Failed to build a complete episode for task 150deff5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"150deff5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
