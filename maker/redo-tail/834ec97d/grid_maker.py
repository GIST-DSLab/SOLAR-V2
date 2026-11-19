"""
ARC Task: 834ec97d (RE-ARC) — LLM-generated grid_maker
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
    # Generator: cols = remove(4, interval(0,10,1)); bgc, fgc = sample(cols, 2)
    # 4 is reserved for the beams, so it is excluded from both roles.
    cols = [c for c in range(10) if c != 4]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    max_h = max(2, min(int(max_h), 30))
    max_w = max(2, min(int(max_w), 30))

    h = unifint(diff_lb, diff_ub, (2, max_h))
    w = unifint(diff_lb, diff_ub, (2, max_w))
    loci = unifint(diff_lb, diff_ub, (0, h - 2))
    locjd = unifint(diff_lb, diff_ub, (0, w // 2))
    locj = random.choice((locjd, w - locjd))
    locj = min(max(0, locj), w - 1)

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    gi[loci][locj] = fgc

    go = [[bgc for _ in range(w)] for _ in range(h)]
    go[loci + 1][locj] = fgc
    for c in range(w):
        if (c - locj) % 2 == 0:
            for r in range(loci + 1):
                go[r][c] = 4

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # Background: the canvas colour the generator paints before placing the single dot.
    # Exactly one cell differs from it, so the majority colour is reliable here.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    loci, locj = dots[0]
    fgc = int(I[loci, locj])

    ops, sels = [], []

    # 1) The dot drops one row.  ARCLE's object ops only carry NON-ZERO cells, so a
    #    dot whose colour is literally 0 cannot be grabbed/moved -- in that (and only
    #    that) case the dot has to be drawn at its new home with Color0.  Either way
    #    the dot's ORIGINAL cell needs no repair: the beam drawn in step 2 runs down
    #    column locj through row loci and paints it 4.
    if fgc != 0:
        ops.append(21)                          # MoveD: grab the dot, slide it down 1
        sels.append(sel_of([(loci, locj)]))
    else:
        ops.append(0)                           # Color0: dot colour is 0, cannot be grabbed
        sels.append(sel_of([(loci + 1, locj)]))

    # 2) Beams of 4 shoot UP from the dot's old row, on every second column outward
    #    from the dot's column (both directions), each spanning rows 0..loci.
    #    Emit one Color4 per beam, working outward from the dot.
    beam_cols = []
    d = 0
    while True:
        right, left = locj + 2 * d, locj - 2 * d
        added = False
        if 0 <= right < w:
            beam_cols.append(right)
            added = True
        if d > 0 and 0 <= left < w:
            beam_cols.append(left)
            added = True
        if not added and 2 * d > w:
            break
        d += 1
        if d > w:
            break

    for cc in beam_cols:
        ops.append(4)
        sels.append(sel_of([(r, cc) for r in range(loci + 1)]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 834ec97d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 834ec97d"
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
                                f"for task 834ec97d"
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
                    f"Failed to build a complete episode for task 834ec97d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"834ec97d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
