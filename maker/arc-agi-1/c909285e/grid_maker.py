"""
ARC Task: c909285e (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
import random
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    boxcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "boxcol": boxcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxcol) -> dict:
    max_h = max(5, max_h)
    max_w = max(5, max_w)
    h = random.randint(5, max_h)
    w = random.randint(5, max_w)
    nfronts = random.randint(1, max(1, (h + w) // 2))
    remcols = [c for c in range(10) if c != bgc and c != boxcol]

    gi = np.full((h, w), bgc, dtype=int)
    # frontiers: full horizontal or vertical lines (each dim spans grid -> a "line")
    for _ in range(nfronts):
        col = random.choice(remcols)
        if random.random() < 0.5:
            gi[random.randint(0, h - 1), :] = col
        else:
            gi[:, random.randint(0, w - 1)] = col

    # single hollow rectangular frame in boxcol
    oh = random.randint(3, max(3, (h - 2) // 2))
    ow = random.randint(3, max(3, (w - 2) // 2))
    loci = random.randint(1, h - oh - 1)
    locj = random.randint(1, w - ow - 1)
    r0, c0 = loci, locj
    r1, c1 = loci + oh - 1, locj + ow - 1
    for r in range(r0, r1 + 1):
        gi[r, c0] = boxcol
        gi[r, c1] = boxcol
    for c in range(c0, c1 + 1):
        gi[r0, c] = boxcol
        gi[r1, c] = boxcol

    go = gi[r0:r1 + 1, c0:c1 + 1].copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def _find_box(I):
    hi, wi = I.shape
    best = None
    for col in set(I.flatten().tolist()):
        coords = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col]
        rs = [p[0] for p in coords]
        cs = [p[1] for p in coords]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:  # frontiers (a full line) have a dim of 1 -> skip
            continue
        frame = set()
        for r in range(r0, r1 + 1):
            frame.add((r, c0))
            frame.add((r, c1))
        for c in range(c0, c1 + 1):
            frame.add((r0, c))
            frame.add((r1, c))
        if set(coords) == frame:  # this color's cells are exactly a hollow rectangle
            area = h * w
            if best is None or area < best[0]:
                best = (area, r0, c0, h, w)
    return best


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    box = _find_box(I)
    _, r0, c0, h, w = box  # bbox measured from I's frame, not from O

    ops, sels = [], []
    # crop working canvas down to the frame's bounding box
    ops.append(33)
    sels.append([r0, c0, h - 1, w - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task c909285e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c909285e"
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
                                f"for task c909285e"
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
                    f"Failed to build a complete episode for task c909285e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c909285e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
