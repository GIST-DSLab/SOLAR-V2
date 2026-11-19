"""
ARC Task: 623ea044 (RE-ARC) — LLM-generated grid_maker
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
    # Background is fixed to 0 for the whole episode: the rule is a transparent
    # union of shifted copies of the dot pattern, and ARCLE's Copy/Paste treats 0
    # as "nothing", which is exactly the stamping semantics the rule needs.
    bgc = 0
    fgc = random.choice([c for c in range(10) if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc=0, fgc=None) -> dict:
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    if fgc is None:
        fgc = choice(remove(bgc, interval(0, 10, 1)))
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    card_bounds = (1, max(int(h * w * 0.1), 1))
    numdots = unifint(diff_lb, diff_ub, card_bounds)
    numdots = max(1, min(numdots, len(inds)))
    dots = sample(inds, numdots)
    gi = fill(gi, fgc, dots)
    go = fill(gi, fgc, mapply(rbind(shoot, UP_RIGHT), dots))
    go = fill(go, fgc, mapply(rbind(shoot, DOWN_LEFT), dots))
    go = fill(go, fgc, mapply(rbind(shoot, UNITY), dots))
    go = fill(go, fgc, mapply(rbind(shoot, NEG_UNITY), dots))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Each dot shoots four diagonal rays.

    A ray is the dot pattern re-stamped one step further along that diagonal,
    again and again.  So for every diagonal direction and every step distance d
    we CopyI the (still original) dot pattern and Paste it displaced by d steps:
    Paste is transparent, so only the dots land and the rays grow.  All four ray
    families are grown from the INPUT stamp (op28), never from the accumulating
    grid, which is why the diagonals stay lines instead of smearing into wedges.
    """
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]

    ops, sels = [], []

    if not dots:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    fgc = int(I[dots[0][0], dots[0][1]])
    grid = I.copy()

    if bgc == 0:
        # ---- ray growth by repeated transparent stamping of the input dots ----
        for (dr, dc) in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            for d in range(1, max(h, w)):
                # rows/cols of the input whose d-step image is still on the grid
                if dr > 0:
                    r0, r1 = 0, h - 1 - d
                else:
                    r0, r1 = d, h - 1
                if dc > 0:
                    c0, c1 = 0, w - 1 - d
                else:
                    c0, c1 = d, w - 1
                if r0 > r1 or c0 > c1:
                    break

                clip = I[r0:r1 + 1, c0:c1 + 1]
                orr, occ = r0 + dr * d, c0 + dc * d

                target = grid[orr:orr + clip.shape[0], occ:occ + clip.shape[1]]
                mask = clip != 0
                if not np.any(mask & (target != clip)):
                    continue  # this step adds nothing visible -> emit nothing

                # CopyI: the source rectangle of the original dot pattern
                # (a genuine full rectangle, so bbox form is the intended mask)
                ops.append(28); sels.append([r0, c0, r1 - r0, c1 - c0])
                # Paste: only the top-left corner matters as the stamp origin
                ops.append(30); sels.append([orr, occ, 0, 0])

                target[mask] = clip[mask]

        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # ---- defensive fallback (non-zero background makes Paste unsafe):
    #      paint one whole X per dot, grouped by the dot that emits it ----
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        sel_of = None

    for (dr0, dc0) in [(0, 0)]:
        pass
    for (r, c) in dots:
        cells = []
        for (dr, dc) in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] != fgc:
                    cells.append((rr, cc))
                rr += dr; cc += dc
        if not cells:
            continue
        ops.append(fgc)
        if sel_of is not None:
            sels.append(sel_of(cells))
        else:
            sels.append({"cells": [[int(a), int(b)] for a, b in cells]})
        for (rr, cc) in cells:
            grid[rr, cc] = fgc

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
                        f"num_examples+1 ({num_examples + 1}) for task 623ea044"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 623ea044"
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
                                f"for task 623ea044"
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
                    f"Failed to build a complete episode for task 623ea044 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"623ea044-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
