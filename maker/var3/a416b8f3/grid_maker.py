"""
ARC Task: a416b8f3 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = min(max(a, int(round(a + (b - a) * diff_lb))), b)
    hi = min(max(a, int(round(a + (b - a) * diff_ub))), b)
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # The rule (duplicate the grid side by side) does not depend on any colour role,
    # only on the pattern itself, so only the background needs to be fixed per episode.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)

    h_ub = max(1, min(30, int(max_h)))
    w_ub = max(1, min(15, int(max_w) // 2))

    h = _unifint(diff_lb, diff_ub, (1, h_ub))
    w = _unifint(diff_lb, diff_ub, (1, w_ub))

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    remcols = [c for c in cols if c != bgc]
    numc = _unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc) if numc > 0 else []

    inds = [(i, j) for i in range(h) for j in range(w)]
    for col in colsch:
        if not inds:
            break
        num = _unifint(diff_lb, diff_ub, (1, max(1, len(inds) // max(1, numc))))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        chosset = set(chos)
        for (i, j) in chos:
            gi[i][j] = col
        inds = [x for x in inds if x not in chosset]

    go = [row + list(row) for row in gi]
    gi = [list(row) for row in gi]
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape                      # ho == hi, wo == 2*wi

    ops, sels = [], []
    full_out = [0, 0, ho - 1, wo - 1]     # bbox: the whole (widened) canvas
    src_rect = [0, 0, hi - 1, wi - 1]     # bbox: exactly the original picture, background included

    nz = int(np.count_nonzero(I))
    if nz == 0:
        # Nothing on the sheet travels (ARCLE reads 0 as "nothing there"), and the
        # doubled sheet is 0 everywhere too: only the canvas has to grow.
        ops.append(33); sels.append(full_out)
        ops.append(34); sels.append(full_out)
        return ops, sels

    # Which of the equally valid routes this instance takes is read off the picture
    # itself (how full it is, how many colours it uses, how wide it is) -- never off O.
    distinct = len(set(I.flatten().tolist()))
    route = (nz + distinct + wi) % 3

    if route == 0:
        # Widen the sheet first, then stamp the copy into the empty right half.
        ops.append(33); sels.append(full_out)
        ops.append(28); sels.append(src_rect)          # CopyI: the whole picture rectangle
        ops.append(30); sels.append([0, wi, 0, 0])     # Paste at the right half's corner
        # cells of I holding 0 do not travel with the Paste, and the right half is
        # already 0 where they land, so no Color0 is needed here

    elif route == 1:
        # Take the picture onto the clipboard while it is still the only thing on the
        # sheet, then widen, then stamp the right half.
        ops.append(28); sels.append(src_rect)
        ops.append(33); sels.append(full_out)
        ops.append(30); sels.append([0, wi, 0, 0])

    else:
        # Widen the sheet, shove the picture across into the new right half one column
        # at a time, then stamp the original back into the emptied left half.
        ops.append(33); sels.append(full_out)
        ops.append(22); sels.append(src_rect)          # first MoveR grabs the whole picture
        for _ in range(wi - 1):
            ops.append(22); sels.append(sel_of([]))    # empty selection keeps the same grab
        ops.append(28); sels.append(src_rect)          # CopyI reads the untouched INPUT
        ops.append(30); sels.append([0, 0, 0, 0])      # stamp it back at the left corner
        # the vacated left half is 0 everywhere, which is already correct wherever I
        # holds 0, so again no Color0 is emitted

    ops.append(34); sels.append(full_out)
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
                        f"num_examples+1 ({num_examples + 1}) for task a416b8f3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a416b8f3"
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
                                f"for task a416b8f3"
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
                    f"Failed to build a complete episode for task a416b8f3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a416b8f3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
