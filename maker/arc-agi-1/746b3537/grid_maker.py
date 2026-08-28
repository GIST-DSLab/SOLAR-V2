"""
ARC Task: 746b3537 (RE-ARC) — LLM-generated grid_maker
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

# ---------------------------------------------------------------------------
# Task 746b3537
#
# Rule (from the verifier):
#   The grid is a stack of solid colour BANDS, thickened by duplicated
#   rows/columns.  The answer is the strip of band colours with consecutive
#   duplicates removed.
#     * first row is a single colour  -> the bands are stacked VERTICALLY;
#       the verifier dmirrors the grid, collapses left-to-right, dmirrors back.
#       (ARCLE: Rotate90 turns the horizontal bands into vertical ones, the
#        duplicate columns are slid away with MoveL, Rotate270 puts the strip
#        back upright.)
#     * otherwise                     -> identity branch: the bands already run
#       left-to-right, so the duplicates are slid away directly.
#   Two structural variants (mirrored / not) => instance_plan.
# ---------------------------------------------------------------------------

VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    # No background exists in this task and the rule is purely structural
    # (collapse a constant strip + dedupe), so no colour role has to be fixed.
    # The one thing that MUST be covered is the orientation branch.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, mirrored=None, instance_plan=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = int(a + (b - a) * lb)
        hi = int(a + (b - a) * ub + 0.999999)
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    if mirrored is None:
        mirrored = random.choice([True, False])

    # strip length = number of bands + inserted duplicates, strip thickness = w
    len_lim = max(3, min(30, max_w if mirrored else max_h))
    thick_lim = max(1, min(30, max_h if mirrored else max_w))

    h = unifint(diff_lb, diff_ub, (2, max(2, min(15, len_lim - 1))))
    w = unifint(diff_lb, diff_ub, (1, thick_lim))

    cols = []
    lastc = -1
    for _ in range(h):
        c = random.choice([x for x in range(10) if x != lastc])
        cols.append(c)
        lastc = c

    go = [[c] for c in cols]
    gi = [[c] * w for c in cols]

    ni_ub = max(1, min(30 - h, len_lim - h))
    numinserts = unifint(diff_lb, diff_ub, (1, ni_ub))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + [list(gi[loc])] + gi[loc + 1:]

    if mirrored:                       # dmirror == transpose
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- which branch does the rule take? (verifier: size(dedupe(first(I))) == 1)
    mirror_branch = len(set(int(v) for v in I[0].tolist())) == 1

    # the strip of band colours, in reading order along the band axis
    if mirror_branch:                       # bands are horizontal -> read a column
        seq = [int(I[r][0]) for r in range(hi)]
    else:                                   # bands are vertical  -> read a row
        seq = [int(v) for v in I[0].tolist()]
    n = len(seq)

    # the answer strip: consecutive duplicates removed
    target = [seq[0]]
    for v in seq[1:]:
        if v != target[-1]:
            target.append(v)
    nb = len(target)

    # --- plan the duplicate removals -------------------------------------
    # Band position i is fixed by sliding everything from i onwards one step
    # toward the front, as often as position i still repeats position i-1.
    # Duplicates lying beyond position nb-1 are never touched: the final
    # crop drops them, so removing them would be an invisible action.
    cur = list(seq)
    edge = n - 1                 # right-most cell that still holds band content
    dels = []                    # (index to collapse, content edge at that time)
    for i in range(1, nb):
        while cur[i] == cur[i - 1]:
            dels.append((i, edge))
            cur = cur[:i] + cur[i + 1:] + [0]
            edge -= 1

    if mirror_branch:
        # ---- dmirror branch --------------------------------------------
        if dels:
            sq = max(hi, wi)
            if hi != wi:
                # square canvas so the rotation's position maths is exact
                ops.append(33); sels.append([0, 0, sq - 1, sq - 1])
            # Rotate90 (CCW): the horizontal bands become vertical bands,
            # in the same order.  This IS the rule's dmirror.
            # (full-rectangle selection: the whole canvas, background included)
            ops.append(24); sels.append([0, 0, sq - 1, sq - 1])
            r0 = sq - wi                      # rows the strip occupies now
            for (i, e) in dels:
                # slide the remaining bands one column left over the duplicate.
                # full-rectangle selection: the entire tail block, including
                # its 0-coloured bands, is what moves.
                ops.append(23); sels.append([r0, i, wi - 1, e - i])
            # Rotate270 (CW): put the collapsed strip back upright.
            ops.append(25); sels.append([0, 0, sq - 1, sq - 1])
        # keep the one column that carries the nb band colours
        ops.append(33); sels.append([0, 0, nb - 1, 0])
    else:
        # ---- identity branch: bands already run left to right -----------
        for (i, e) in dels:
            # full-rectangle selection: whole tail block (all rows), 0s included
            ops.append(23); sels.append([0, i, hi - 1, e - i])
        # keep the one row that carries the nb band colours
        ops.append(33); sels.append([0, 0, 0, nb - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 746b3537"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 746b3537"
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
                                f"for task 746b3537"
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
                    f"Failed to build a complete episode for task 746b3537 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"746b3537-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
