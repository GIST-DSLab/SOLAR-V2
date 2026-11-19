"""
ARC Task: 68b16354 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    """The rule here is purely geometric (mirror top<->bottom), so no colour role
    carries meaning.  Still fix the palette once per episode so every instance is
    drawn on the same background with the same foreground colours."""
    cols = list(range(10))
    bgc = random.choice(cols)
    fgcols = [c for c in cols if c != bgc]
    random.shuffle(fgcols)
    return {"bgc": bgc, "fgcols": fgcols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgcols=None) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, max_h))
    w = unifint(diff_lb, diff_ub, (1, max_w))
    if bgc is None:
        bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = list(remove(bgc, cols))
    if fgcols:
        pal = [c for c in fgcols if c != bgc]
        if pal:
            remcols = pal
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    if fgcols:
        colsch = remcols[:numc]
    else:
        colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = hmirror(gi)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Rule: the grid is mirrored top<->bottom (each column's cells reversed in place).

    flipud acts on every column independently, so the mirror can honestly be carried
    out one full-height vertical band at a time — the whole grid in one stroke, or the
    left half then the right half, or three bands swept the other way.  Which
    decomposition is used, and in which order the bands are worked, is read off the
    input itself (its colour count, and its two opposite corner colours); handed a
    different answer grid this function returns exactly the same operations.

    Every selection is a FULL rectangle (all rows of a contiguous column range,
    background included), which is what FlipV needs: ARCLE clears the whole selected
    band before re-pasting the flipped content, so cells of colour 0 land correctly as
    0 without any extra Color(0) — there are none left over to place.
    """
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # --- how many vertical bands to work in: read from the input's colour count ---
    ncolors = len(set(I.flatten().tolist()))
    nbands = 1 + (ncolors % 3)              # 1, 2 or 3 bands
    nbands = max(1, min(nbands, wi))

    bands = []
    for b in range(nbands):
        c0 = (wi * b) // nbands
        c1 = (wi * (b + 1)) // nbands - 1
        if c1 >= c0:
            bands.append((c0, c1))

    # --- which side to start from: the corner colours of the input decide the sweep ---
    if len(bands) > 1 and int(I[0, 0]) > int(I[hi - 1, wi - 1]):
        bands.reverse()                     # sweep right -> left

    for (c0, c1) in bands:
        band = I[:, c0:c1 + 1]
        if np.array_equal(band, band[::-1, :]):
            continue                        # already its own mirror: flipping shows nothing
        # full-height, full-width rectangle of this band -> bbox selection is exact
        ops.append(27)                      # FlipV = flipud: mirror this band top<->bottom
        sels.append([0, c0, hi - 1, c1 - c0])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 68b16354"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 68b16354"
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
                                f"for task 68b16354"
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
                    f"Failed to build a complete episode for task 68b16354 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"68b16354-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
