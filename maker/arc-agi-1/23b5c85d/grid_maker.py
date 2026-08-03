"""
ARC Task: 23b5c85d (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # Rule is structural (smallest solid-rectangle region); output colour is
    # determined by which region is minimal, not fixed in advance. Only bgc must
    # be fixed across the episode.
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, **color_kwargs) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    colopts = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    oh = unifint(diff_lb, diff_ub, (2, h - 1))
    ow = unifint(diff_lb, diff_ub, (2, w - 1))
    num = unifint(diff_lb, diff_ub, (1, 8))
    cnt = 0
    go = canvas(bgc, (1, 1))
    while cnt < num:
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        col = choice(colopts)
        colopts = remove(col, colopts)
        obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        gi2 = fill(gi, col, obj)
        if color(argmin(sfilter(partition(gi2), fork(equality, size, fork(multiply, height, width))), fork(multiply, height, width))) != col:
            break
        else:
            gi = gi2
            go = canvas(col, shape(obj))
        oh = unifint(diff_lb, diff_ub, (max(0, oh - 4), oh - 1))
        ow = unifint(diff_lb, diff_ub, (max(0, ow - 4), ow - 1))
        if oh < 1 or ow < 1:
            break
        cnt += 1
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # Rule measured from I ONLY: among every colour region that exactly fills its
    # bounding box (a solid rectangle), pick the one with minimum area h*w.
    # Then crop I to that region's bounding box. O is never inspected.
    best = None  # (area, r0, c0, bh, bw)
    for c in set(I.flatten().tolist()):
        cells = [(r, cc) for r in range(hi) for cc in range(wi) if I[r, cc] == c]
        rs = [r for r, _ in cells]
        cs = [cc for _, cc in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if len(cells) != bh * bw:
            continue  # region has holes -> not a solid rectangle -> excluded
        area = bh * bw
        if best is None or area < best[0]:
            best = (area, r0, c0, bh, bw)

    _, r0, c0, ho, wo = best

    ops, sels = [], []
    # CropGrid to the minimal solid rectangle's bbox (full rectangle selection).
    ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 23b5c85d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 23b5c85d"
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
                                f"for task 23b5c85d"
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
                    f"Failed to build a complete episode for task 23b5c85d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"23b5c85d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
