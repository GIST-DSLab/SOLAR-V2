"""
ARC Task: 4258a5f9 (RE-ARC) — LLM-generated grid_maker
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
    # colour 1 is reserved by the rule for the halo, so it is never bgc/fgc
    colopts = [c for c in range(10) if c != 1]
    bgc = random.choice(colopts)
    fgc = random.choice([c for c in colopts if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    # keep 2 rows/cols of headroom: the derivation works on a 1-cell margin
    hcap = max(2, min(int(max_h) - 2, 28))
    wcap = max(2, min(int(max_w) - 2, 28))
    h = unifint(diff_lb, diff_ub, (2, hcap))
    w = unifint(diff_lb, diff_ub, (2, wcap))
    gi = canvas(bgc, (h, w))
    mp = ((h * w) // 2) if (h * w) % 2 == 1 else ((h * w) // 2 - 1)
    ndots = unifint(diff_lb, diff_ub, (1, mp))
    inds = totuple(asindices(gi))
    dots = sample(inds, ndots)
    go = fill(gi, 1, mapply(neighbors, frozenset(dots)))
    go = fill(go, fgc, dots)
    gi = fill(gi, fgc, dots)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Rule: every dot wears the same 3x3 halo ring.  Account of the build:
    give the picture a one-cell margin so no halo falls off the canvas, draw the
    ring once around the first dot, then stamp that same ring on every other dot,
    restore the dots that a neighbour's stamp covered, and crop the margin away."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background = most common colour, dot colour = the other one (dots < half the cells)
    counts = Counter(I.flatten().tolist()).most_common()
    bgc = counts[0][0]
    fgc = counts[-1][0] if len(counts) > 1 else bgc

    # halo colour: the colour the demonstrations introduce that their inputs never had
    halo = 1
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            new = set(eo.flatten().tolist()) - set(ei.flatten().tolist())
            if len(new) == 1:
                halo = int(new.pop())
                break

    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] == fgc]

    ops, sels = [], []

    # --- 1. give the picture a one-cell margin on every side -------------------
    H, W = h + 2, w + 2
    ops.append(33); sels.append([0, 0, H - 1, W - 1])      # bbox: the whole enlarged canvas

    # --- 2. slide the whole picture into that margin (down 1, then right 1) ----
    whole = [(r, c) for r in range(h) for c in range(w)]   # bbox: the entire picture, background included
    ops.append(21); sels.append(sel_of(whole))             # grab the picture, move down
    ops.append(22); sels.append(sel_of([]))                # same grab kept, move right

    G = np.zeros((H, W), dtype=int)
    G[1:h + 1, 1:w + 1] = I
    nd = [(r + 1, c + 1) for (r, c) in dots]               # dot positions on the padded canvas

    # --- 3. draw the halo ring once, around the first dot ----------------------
    dr_, dc_ = nd[0]
    ring = [(dr_ + a, dc_ + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    ops.append(halo); sels.append(sel_of(ring))
    for (r, c) in ring:
        G[r, c] = halo

    # --- 4. stamp that ring onto every other dot ------------------------------
    clip = G[dr_ - 1:dr_ + 2, dc_ - 1:dc_ + 2].copy()      # ring of halo + the dot at its centre
    nz = clip != 0                                         # 0 cells of the stamp are transparent
    paste_at = []
    for (r, c) in nd[1:]:
        tgt = G[r - 1:r + 2, c - 1:c + 2]
        if np.any(tgt[nz] != clip[nz]):                    # skip dots already fully haloed
            paste_at.append((r - 1, c - 1))
            tgt[nz] = clip[nz]
    if paste_at:
        stamp_cells = [(dr_ + a, dc_ + b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        ops.append(29); sels.append(sel_of(stamp_cells))   # bbox: the 3x3 ring just drawn
        for (pr, pc) in paste_at:
            ops.append(30); sels.append(sel_of([(pr, pc)]))

    # --- 5. dots that a neighbouring dot's halo painted over come back --------
    fix = [(r, c) for (r, c) in nd if G[r, c] != fgc]
    if fix:
        ops.append(fgc); sels.append(sel_of(fix))
        for (r, c) in fix:
            G[r, c] = fgc

    # --- 6. drop the margin again --------------------------------------------
    ops.append(33); sels.append([1, 1, h - 1, w - 1])
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4258a5f9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4258a5f9"
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
                                f"for task 4258a5f9"
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
                    f"Failed to build a complete episode for task 4258a5f9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4258a5f9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
