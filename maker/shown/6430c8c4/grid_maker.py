"""
ARC Task: 6430c8c4 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for r, c in cells]}


# ----------------------------------------------------------------------------- #
# 1. sample_colors
# ----------------------------------------------------------------------------- #
_VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]          # 3 is reserved for the marks
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(_VARIANTS):
        examples = [dict(v) for v in _VARIANTS]
        examples += [dict(random.choice(_VARIANTS)) for _ in range(n_ex - len(_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(_VARIANTS, max(1, n_ex))]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "linc": linc, "acol": acol, "bcol": bcol,
            "instance_plan": plan}


# ----------------------------------------------------------------------------- #
# 2. generate
# ----------------------------------------------------------------------------- #
def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, acol, bcol, mirror=None) -> dict:
    if mirror is None:
        mirror = choice((True, False))

    # the assembled input is (h, 2w+1); after dmirror it is (2w+1, h)
    if mirror:
        h_cap = max_w
        w_cap = (max_h - 1) // 2
    else:
        h_cap = max_h
        w_cap = (max_w - 1) // 2
    h_ub = max(2, min(30, h_cap))
    w_ub = max(2, min(14, w_cap))

    h = unifint(diff_lb, diff_ub, (2, h_ub))
    w = unifint(diff_lb, diff_ub, (2, w_ub))

    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))

    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)

    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)

    gi = hconcat(hconcat(A, bar), B)
    res = (set(inds) - set(aset)) - set(bset)
    go = fill(c, 3, res)

    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------- #
# 3. derive_operations
# ----------------------------------------------------------------------------- #
def _split_halves(g):
    """Split an input grid on its single-colour separator line.

    Returns (first_half, second_half, orientation).  Orientation 'h' means the
    separator is a full row (halves are top/bottom), 'v' means it is a full
    column (halves are left/right).  Both facts are read from the INPUT only.
    """
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    # a full uniform row can only be the separator bar (the bar colour never
    # occurs inside either half, so no half row can be uniform across the bar)
    if h % 2 == 1 and h >= 3 and len(set(g[h // 2].tolist())) == 1:
        k = h // 2
        return g[:k, :], g[k + 1:, :], 'h'
    if w % 2 == 1 and w >= 3 and len(set(g[:, w // 2].tolist())) == 1:
        k = w // 2
        return g[:, :k], g[:, k + 1:], 'v'
    # defensive fallback (should not trigger for this task)
    if w % 2 == 1:
        k = w // 2
        return g[:, :k], g[:, k + 1:], 'v'
    k = h // 2
    return g[:k, :], g[k + 1:, :], 'h'


def _bg_of(first, second, whole):
    """The background is the one colour the two halves have in common."""
    common = set(first.flatten().tolist()) & set(second.flatten().tolist())
    if len(common) == 1:
        return int(next(iter(common)))
    if common:
        cnt = Counter(np.asarray(whole).flatten().tolist())
        return int(max(common, key=lambda c: cnt.get(c, 0)))
    return int(Counter(np.asarray(whole).flatten().tolist()).most_common(1)[0][0])


def _mark_colour_from_examples(examples):
    """The colour the rule paints the shared-background cells with.

    It is a convention of the task, not something the input shows, so it is
    read from the demonstrations (each demo output holds exactly its own
    background plus this colour).  Never read from the O being derived.
    """
    votes = []
    for ex in (examples or []):
        if isinstance(ex, dict):
            Ie, Oe = ex.get('input'), ex.get('output')
        else:
            try:
                Ie, Oe = ex[0], ex[1]
            except Exception:
                continue
        if Ie is None or Oe is None:
            continue
        try:
            Ie = np.asarray(Ie, dtype=int)
            Oe = np.asarray(Oe, dtype=int)
            fa, fb, _ = _split_halves(Ie)
            bg_e = _bg_of(fa, fb, Ie)
            others = set(Oe.flatten().tolist()) - {bg_e}
            if len(others) == 1:
                votes.append(int(next(iter(others))))
        except Exception:
            continue
    if votes:
        return Counter(votes).most_common(1)[0][0]
    return 3


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)

    first, second, _orient = _split_halves(I)
    bgc = _bg_of(first, second, I)
    mark = _mark_colour_from_examples(examples)

    kh, kw = first.shape                       # the kept half sits at the top-left

    # cells that are background in BOTH halves -> get the mark colour
    inter = [(r, c) for r in range(kh) for c in range(kw)
             if first[r, c] == bgc and second[r, c] == bgc]
    # the first half's own marks -> erased back to background
    marks = [(r, c) for r in range(kh) for c in range(kw) if first[r, c] != bgc]

    ops, sels = [], []

    if inter:
        ops.append(int(mark))
        sels.append(sel_of(inter))

    if marks:
        ops.append(int(bgc))
        sels.append(sel_of(marks))

    # keep only the first half (full rectangle -> bbox selection is exact)
    ops.append(33)
    sels.append([0, 0, kh - 1, kw - 1])

    ops.append(34)
    sels.append([0, 0, kh - 1, kw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6430c8c4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6430c8c4"
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
                                f"for task 6430c8c4"
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
                    f"Failed to build a complete episode for task 6430c8c4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6430c8c4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
