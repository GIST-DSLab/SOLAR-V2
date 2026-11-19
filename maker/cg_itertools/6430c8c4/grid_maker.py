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
from maker.sel_helpers import sel_of

ORIENTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        ex = [dict(v) for v in ORIENTS]
        ex += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "linc": linc, "acol": acol, "bcol": bcol,
            "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, acol, bcol,
             mirrored=None) -> dict:
    if mirrored is None:
        mirrored = random.choice([True, False])
    if mirrored:
        hcap, wcap = max_w, (max_h - 1) // 2
    else:
        hcap, wcap = max_h, (max_w - 1) // 2
    h = unifint(diff_lb, diff_ub, (2, min(30, hcap)))
    w = unifint(diff_lb, diff_ub, (2, min(14, wcap)))
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
    if mirrored:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def _split(I):
    """Find the divider line; return (horizontal_divider, half_len)."""
    H, W = I.shape
    if H % 2 == 1:
        m = H // 2
        row = I[m]
        rest = np.delete(I, m, axis=0)
        if len(set(row.tolist())) == 1 and row[0] not in set(rest.flatten().tolist()):
            return True, m
    m = W // 2
    return False, m


def _slide_is_meaningful(src, dst, bgc, toward_start):
    """src's empty cells (as 3s) slide onto dst; dst's marks are then erased.
    The slide only carries information if (a) some 3 survives, and (b) stopping
    one step short would give a different picture.  Both are facts about I."""
    full = (src == bgc) & (dst == bgc)
    if not full.any():
        return False
    # one step short: the pattern stops one cell before its place
    short = np.zeros_like(full)
    if toward_start:
        short[:, 1:] = (src[:, :-1] == bgc)
    else:
        short[:, :-1] = (src[:, 1:] == bgc)
    short = short & (dst == bgc)
    return not np.array_equal(full, short)


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    horiz, m = _split(I)
    if horiz:
        first, second = I[:m], I[m + 1:]
    else:
        first, second = I[:, :m], I[:, m + 1:]
    kh, kw = first.shape
    # background = the one colour both halves share
    bgc = (set(first.flatten().tolist()) & set(second.flatten().tolist())).pop()

    # work in "slide along columns" orientation for the degeneracy check
    def along(a):
        return a.T if horiz else a
    up_ok = _slide_is_meaningful(along(second), along(first), bgc, True)
    down_ok = _slide_is_meaningful(along(first), along(second), bgc, False)

    def cell(half, r, c):
        if half == 0:
            return (r, c)
        return (r + m + 1, c) if horiz else (r, c + m + 1)

    ops, sels = [], []
    if up_ok or down_ok:
        if up_ok:
            src, dst, s_idx, d_idx = second, first, 1, 0
            step = 20 if horiz else 23          # MoveU / MoveL
        else:
            src, dst, s_idx, d_idx = first, second, 0, 1
            step = 21 if horiz else 22          # MoveD / MoveR
        # 1. the source half's EMPTY cells become 3
        empty_src = [cell(s_idx, r, c) for r in range(kh) for c in range(kw)
                     if src[r, c] == bgc]
        ops.append(3); sels.append(sel_of(empty_src))
        # 2. slide that 3-pattern across the divider onto the other half
        ops.append(step); sels.append(sel_of(empty_src))   # grab the 3-cells
        for _ in range(m):
            ops.append(step); sels.append(sel_of([]))      # keep sliding it
        # 3. the destination half's own marks (as in I) return to background:
        #    a 3 survives only where that half was empty too
        marks = [cell(d_idx, r, c) for r in range(kh) for c in range(kw)
                 if dst[r, c] != bgc]
        ops.append(int(bgc)); sels.append(sel_of(marks))
        keep = cell(d_idx, 0, 0)
    else:
        # degenerate overlay (nothing survives / slide length immaterial):
        # paint the both-empty cells directly, erase the first half's marks
        inter = [(r, c) for r in range(kh) for c in range(kw)
                 if first[r, c] == bgc and second[r, c] == bgc]
        if inter:
            ops.append(3); sels.append(sel_of(inter))
        marks = [(r, c) for r in range(kh) for c in range(kw) if first[r, c] != bgc]
        ops.append(int(bgc)); sels.append(sel_of(marks))
        keep = (0, 0)

    # 4. keep only the overlaid half (full rectangle -> bbox is exact)
    ops.append(33); sels.append([keep[0], keep[1], kh - 1, kw - 1])
    ops.append(34); sels.append([0, 0, kh - 1, kw - 1])
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
